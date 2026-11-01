"""FlyWire 전체 뇌로 전체 코스 운전 학습 (조향 + 속도·정지 + 방향지시등·비상점멸등).

입력 (포아송 발화율):
  카메라 → 차선 지도 15x20 (흰·노랑)        → 시각 투사뉴런(VPN) 70%, 대략적 망막지도
  카메라 → 색 지도 3x5 (신호 빨강·초록, 돌발 모형 파랑) → VPN 나머지 30%
  내비게이터 명령 (경로 계획·타이머: 카메라로 알 수 없는 것) → 상행뉴런(ascending) 9묶음
전체 뇌 LIF (FlyWire 연결 그대로, 고정) → 하강뉴런(DN) 1299개 발화 수 → 선형 읽기 → 속도, 조향, 좌·우 램프.
학습하는 것은 DN → 출력 읽기 가중치뿐. 교사 = sim.Expert (규칙 지키는 정답 운전).
신호 보고 서기·돌발 모형 보고 서기·비상점멸등·차로 유지는 카메라로만 알아야 함.

python train.py           수집(병렬) → 학습 → 뇌로 주행 시험 → data/brain_run.npz 저장
python train.py --dagger  뇌가 주행·교사가 정답 표시한 표본 추가 → 재학습 → 시험 (반복 가능)
python train.py --test    저장된 readout.npz 로 주행 시험만
python train.py --replay  저장된 뇌 주행을 3D 창에서 실제 속도로 재생
"""
import sys
import time
from multiprocessing import Pool

import numpy as np

import sim
from brain import DT as BRAIN_DT, Brain, load_annotations, load_weights

GRID_H, GRID_W = 15, 20          # 차선 지도
COL_H, COL_W = 3, 5              # 색 지도 (칸 = 80x64 픽셀). 칸이 적을수록 칸당 VPN이 많아 작은 신호등도 뇌에 강하게 들어감
SMOOTH = 0.7                     # DN 발화 수 지수평균 (제어 스텝마다 0.7 유지 ≈ 시정수 140ms). 50ms 발화는 잡음이 커서
COLOR_SHARE = 0.3                # VPN 중 색 지도에 쓰는 비율
MAX_RATE = 200.0                 # 입력 최대 발화율 [Hz]
STEPS_PER_CTRL = round(sim.DT * 1000 / BRAIN_DT)   # 제어 1회(50ms)당 뇌 스텝 수
CMDS = ("nav_left", "nav_right", "blink_left", "blink_right", "reverse", "rotate_left", "rotate_right", "hold", "accel")
OUTS = ("speed", "turn", "lamp_left", "lamp_right")
EPISODES = ((0, 0.0), (1, 0.12), (2, 0.0), (3, 0.12), (4, 0.06), (5, 0.06))   # 수집 (seed, 조향 잡음)
TEST_SEED = 100
RUN = "data/brain_run.npz"


def lane_marks(rgb):
    """차선 픽셀 (흰색 또는 노란색) → GRID_H x GRID_W 칸별 비율."""
    f = rgb.astype(np.float32) / 255
    white = (f.min(-1) > 0.85) & (abs(f[..., 2] - f[..., 0]) < 0.08)   # 하늘(푸른빛) 제외
    m = (white | sim.yellow(rgb)).astype(np.float32)
    return m.reshape(GRID_H, sim.CAM_H // GRID_H, GRID_W, sim.CAM_W // GRID_W).mean(axis=(1, 3))


def color_marks(rgb):
    """빛나는 빨강 / 초록 LED, 파란 사람 모형 → (3, COL_H, COL_W) 칸별 강도 (픽셀 4개면 꽉 참)."""
    f = rgb.astype(np.int16)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    planes = ((r > 180) & (g < 90) & (b < 90), (g > 180) & (r < 120) & (b < 170) & (g - b > 60),
              (b > 120) & (b - r > 60) & (b - g > 40))
    c = np.stack([p.reshape(COL_H, sim.CAM_H // COL_H, COL_W, sim.CAM_W // COL_W).sum(axis=(1, 3)) for p in planes])
    return np.minimum(c / 4, 1).astype(np.float32)


def commands(env, nav):
    """내비게이터 명령 벡터. nav = 이번 스텝 호출된 sim.Expert (경로 계획·타이머 상태만 씀)."""
    i, n = env.idx, env.n
    a = env.track[min(i + 10, n - 1)] - env.track[i]
    b = env.track[min(i + 60, n - 1)] - env.track[min(i + 30, n - 1)]
    turn = np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)    # 30~60cm 앞 경로 꺾임 (GPS 안내)
    return np.array([turn > 0.4, turn < -0.4, nav.blink & sim.LEFT, nav.blink & sim.RIGHT, nav.rev,
                     nav.rotate > 0, nav.rotate < 0, nav.hold, nav.accel], np.float32)


def labels(action, lamps):
    (l, r) = action
    return np.array([(l + r) / 2, (r - l) / 2, lamps & sim.LEFT > 0, lamps & sim.RIGHT > 0], np.float32)


class FlyDriver:
    def __init__(self, seed=0):
        self.W = load_weights()
        ann = load_annotations()
        self.rng = np.random.default_rng(seed)
        # VPN: 영상 왼쪽 절반 → 왼쪽 눈, 오른쪽 → 오른쪽.
        # ponytail: 대략적 망막지도 (등배 pos_y → 행, 전후 pos_z → 열 순위). 실제 시야 좌표 붙이면 정밀화
        vpn, rows, cols = [], [], []
        for side, c0 in (("left", 0), ("right", GRID_W // 2)):
            sel = ann.index[(ann.super_class == "visual_projection") & (ann.side == side)].to_numpy()
            row = np.argsort(np.argsort(ann.pos_y.to_numpy()[sel])) * GRID_H // len(sel)
            col = np.zeros(len(sel), int)
            for r in range(GRID_H):
                k = np.flatnonzero(row == r)
                z = ann.pos_z.to_numpy()[sel[k]]
                rank = np.argsort(np.argsort(z if side == "left" else -z))
                col[k] = c0 + rank * (GRID_W // 2) // len(k)
            vpn.append(sel), rows.append(row), cols.append(col)
        vpn, row, col = np.concatenate(vpn), np.concatenate(rows), np.concatenate(cols)
        rng = np.random.default_rng(1)                    # 입력 배치는 항상 같게
        color = rng.random(len(vpn)) < COLOR_SHARE
        plane = rng.integers(0, 3, len(vpn))
        lane_idx = row * GRID_W + col
        color_idx = (plane * COL_H + row * COL_H // GRID_H) * COL_W + col * COL_W // GRID_W
        asc = ann.index[ann.super_class == "ascending"].to_numpy()
        cmd_idx = rng.permutation(len(asc)) % len(CMDS)
        # 입력 뉴런 전부 한 배열로: [차선 VPN | 색 VPN | 상행뉴런], 특징 벡터도 같은 순서로 이어 붙임
        self.inputs = np.r_[vpn[~color], vpn[color], asc]
        self.feat_idx = np.r_[lane_idx[~color], GRID_H * GRID_W + color_idx[color],
                              GRID_H * GRID_W + 3 * COL_H * COL_W + cmd_idx]
        self.dn = ann.index[ann.super_class == "descending"].to_numpy()
        self.reset()

    def reset(self):
        self.brain = Brain(self.W)

    def sense(self, rgb, cmd):
        """영상 + 명령을 50ms 동안 넣고 DN 발화 수 반환."""
        feat = np.r_[lane_marks(rgb).ravel(), color_marks(rgb).ravel(), cmd]
        p = feat[self.feat_idx] * (MAX_RATE * BRAIN_DT / 1000)
        counts = np.zeros(self.brain.n, np.float32)
        for _ in range(STEPS_PER_CTRL):
            counts[self.brain.step(self.inputs[self.rng.random(len(self.inputs)) < p])] += 1
        return counts[self.dn]


def smooth(X, E):
    """에피소드(E) 안에서만 DN 발화 수 지수평균. BrainDriver 실시간 평균과 같은 식."""
    out = np.empty_like(X)
    for i in range(len(X)):
        m = X[i] if i == 0 or E[i] != E[i - 1] else SMOOTH * m + (1 - SMOOTH) * X[i]
        out[i] = m
    return out


def weigh(vstop):
    """카메라로만 알 수 있는 정지(신호·돌발)는 드물어서 가중치 5배.
    정지 후 출발 직후 15표본(초록불·사람 사라짐을 보고 막 떠나는 순간)은 교사가 바로 떠나 표본이 몇 개뿐이라 20배."""
    w, go = np.ones(len(vstop)), 0
    for i, v in enumerate(vstop):
        go = 15 if i and vstop[i - 1] and not v else max(go - 1, 0)
        w[i] = 5.0 if v else 20.0 if go else 1.0
    return w


def collect(args):
    """교사가 운전 (조향에 OU 잡음) 하며 (DN 발화, 교사 출력, 표본 가중치) 수집. 전체 코스 1회."""
    seed, noise = args
    fly, ex, env = FlyDriver(seed), sim.Expert(), sim.Env(seed=seed)
    rng = np.random.default_rng(seed)
    obs, done, ou = env.reset(), False, 0.0
    X, Y, V = [], [], []
    t0 = time.perf_counter()
    while not done:
        action, lamps = ex(env, obs)
        X.append(fly.sense(obs, commands(env, ex)))
        Y.append(labels(action, lamps))
        V.append(action == (0, 0) and not ex.hold)
        free = not (ex.rev or ex.rotate or ex.hold or env.park[env.idx])
        ou += -0.1 * ou + noise * 0.45 * rng.standard_normal()
        d = ou if free and action != (0, 0) else 0.0
        obs, _, done, info = env.step((action[0] - d, action[1] + d), lamps)
        if len(X) % 200 == 0:
            print(f"  [seed {seed}] t={env.t:.0f}s 진행 {info['progress']:.1f}/{env.length:.1f}m "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"  [seed {seed}] 끝: {info['result']} 점수 {info['score']} 표본 {len(X)}", flush=True)
    return np.array(X), np.array(Y), weigh(V), np.full(len(X), seed)


def fit(X, Y, Wt, E):
    """가중 릿지 회귀 (출력 여러 개 한꺼번에). λ는 뒤쪽 20% 검증으로 고름. 반환 (w, b, mu, sd, 출력별 검증 R²)."""
    X = smooth(X, E)
    mu, sd = X.mean(0), X.std(0) + 1e-3
    Z = (X - mu) / sd
    n = int(len(Y) * 0.8)

    def solve(Z, Y, Wt, lam):
        s = np.sqrt(Wt)[:, None]
        ym = np.average(Y, axis=0, weights=Wt)
        zm = np.average(Z, axis=0, weights=Wt)
        A, B = (Z - zm) * s, (Y - ym) * s
        w = np.linalg.solve(A.T @ A + lam * np.eye(Z.shape[1]), A.T @ B)
        return w, ym - zm @ w

    def r2(w, b):
        pred = Z[n:] @ w + b
        return 1 - ((Y[n:] - pred) ** 2).sum(0) / ((Y[n:] - Y[n:].mean(0)) ** 2).sum(0)

    scores = {lam: r2(*solve(Z[:n], Y[:n], Wt[:n], lam)) for lam in (10, 100, 1e3, 1e4, 1e5)}
    lam = max(scores, key=lambda k: scores[k].mean())
    print("  릿지 λ별 검증 R² (속도, 조향, 좌램프, 우램프):",
          {k: np.round(v, 2).tolist() for k, v in scores.items()}, flush=True)
    w, b = solve(Z, Y, Wt, lam)
    return w, b, mu, sd, scores[lam]


class BrainDriver:
    """뇌가 운전. 내비게이터(sim.Expert)는 경로 계획·타이머 명령만 주고, 그 행동은 버림."""

    def __init__(self, fly, readout):
        self.fly, self.nav, self.m = fly, sim.Expert(), None
        self.w, self.b, self.mu, self.sd = (readout[k] for k in ("w", "b", "mu", "sd"))

    def __call__(self, env, obs):
        self.label = self.nav(env, obs)
        self.x = self.fly.sense(obs, commands(env, self.nav))
        self.m = self.x if self.m is None else SMOOTH * self.m + (1 - SMOOTH) * self.x
        out = ((self.m - self.mu) / self.sd) @ self.w + self.b
        speed, turn = float(np.clip(out[0], -0.4, 1.0)), float(np.clip(out[1], -0.4, 0.4))
        if abs(speed) < 0.15:                             # 이 아래는 어차피 기어 마찰로 못 움직임
            speed = 0.0
        lamps = (sim.LEFT if out[2] > 0.5 else 0) | (sim.RIGHT if out[3] > 0.5 else 0)
        return (speed - turn, speed + turn), lamps


def dagger(seed):
    """DAgger: 뇌가 운전하고, 뇌가 간 (교사가 안 가본) 자리마다 교사 정답을 표본으로 수집."""
    fly, readout = FlyDriver(seed), np.load("readout.npz")
    env, driver = sim.Env(seed=seed), BrainDriver(fly, readout)
    obs, done = env.reset(), False
    X, Y, V = [], [], []
    while not done:
        action, lamps = driver(env, obs)
        X.append(driver.x)
        Y.append(labels(*driver.label))
        V.append(driver.label[0] == (0, 0) and not driver.nav.hold)
        obs, _, _, info = env.step(action, lamps)
        done = info["result"] in ("합격", "실격: 차로 이탈") or env.t >= sim.TIME_LIMIT
        if len(X) % 400 == 0:
            print(f"  [DAgger seed {seed}] t={env.t:.0f}s 진행 {info['progress']:.1f}m 점수 {info['score']}", flush=True)
    print(f"  [DAgger seed {seed}] {info['result']} 점수 {info['score']} 진행 {info['progress']:.1f}m 표본 {len(X)}", flush=True)
    part = np.array(X), np.array(Y), weigh(V), np.full(len(X), seed)
    np.savez(f"data/dagger_{seed}.npz", X=part[0], Y=part[1], Wt=part[2], E=part[3])   # 중간에 죽어도 끝난 주행은 남김
    return part


def drive(fly, readout, seed=TEST_SEED):
    """뇌만으로 전체 코스 1회 (교사 없음). 재생용 기록 저장. 반환 env, info."""
    env, driver = sim.Env(seed=seed), BrainDriver(fly, readout)
    obs, done = env.reset(), False
    fly.reset()
    rec = dict(qpos=[], mocap=[], t=[], lamps=[])
    t0 = time.perf_counter()
    while not done:
        action, lamps = driver(env, obs)
        obs, _, done, info = env.step(action, lamps)
        for k, v in (("qpos", env.data.qpos.copy()), ("mocap", env.data.mocap_pos.copy()), ("t", env.t), ("lamps", lamps)):
            rec[k].append(v)
        if len(rec["t"]) % 200 == 0:
            print(f"  뇌 주행 t={env.t:.0f}s 진행 {info['progress']:.1f}/{env.length:.1f}m 점수 {info['score']} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    np.savez(RUN, seed=seed, result=info["result"], score=info["score"], **{k: np.array(v) for k, v in rec.items()})
    return env, info


def replay(path=RUN):
    """저장된 뇌 주행을 3D 창에서 실제 속도로 재생."""
    import mujoco
    import mujoco.viewer
    r = np.load(path)
    env = sim.Env(seed=int(r["seed"]))
    env.reset()                                           # 같은 seed → 같은 신호 시점
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.type, v.cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, env.model.body("car").id
        v.cam.distance, v.cam.elevation = 1.0, -30
        for q, m, t, lamps in zip(r["qpos"], r["mocap"], r["t"], r["lamps"]):
            if not v.is_running():
                break
            t0 = time.time()
            env.data.qpos[:], env.data.mocap_pos[:], env.t = q, m, float(t)
            env._set_lamps(int(lamps))
            mujoco.mj_forward(env.model, env.data)
            v.sync()
            time.sleep(max(0, sim.DT - (time.time() - t0)))
        print("뇌 주행 재생:", r["result"], f"점수 {r['score']}")
        while v.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    if "--replay" in sys.argv:
        replay()
        sys.exit()
    if "--dagger" in sys.argv:
        d = np.load("data/drive_data.npz")
        print("DAgger: 뇌가 6회 병렬 주행, 교사가 정답 표시", flush=True)
        with Pool(6) as pool:
            parts = pool.map(dagger, range(d["E"].max() + 1, d["E"].max() + 7))   # 라운드마다 새 seed
        X, Y, Wt, E = (np.concatenate([d[k], *a]) for k, a in zip(("X", "Y", "Wt", "E"), zip(*parts)))
        np.savez("data/drive_data.npz", X=X, Y=Y, Wt=Wt, E=E)
        w, b, mu, sd, score = fit(X, Y, Wt, E)
        print("검증 R² (속도, 조향, 좌램프, 우램프):", np.round(score, 3).tolist(), flush=True)
        np.savez("readout.npz", w=w, b=b, mu=mu, sd=sd)
    elif "--test" not in sys.argv:
        print(f"수집: 전체 코스 {len(EPISODES)}회 병렬, 제어 1회당 뇌 {STEPS_PER_CTRL}스텝", flush=True)
        with Pool(len(EPISODES)) as pool:
            parts = pool.map(collect, EPISODES)
        X, Y, Wt, E = (np.concatenate(a) for a in zip(*parts))
        np.savez("data/drive_data.npz", X=X, Y=Y, Wt=Wt, E=E)
        w, b, mu, sd, score = fit(X, Y, Wt, E)
        print("검증 R² (속도, 조향, 좌램프, 우램프):", np.round(score, 3).tolist(), flush=True)
        np.savez("readout.npz", w=w, b=b, mu=mu, sd=sd)
        assert score.mean() > 0.3, "DN 발화로 운전 출력을 거의 예측 못 함"
    fly = FlyDriver()
    print(f"입력 뉴런 {len(fly.inputs)}개 (VPN + 상행), DN 출력 {len(fly.dn)}개", flush=True)
    env, info = drive(fly, np.load("readout.npz"))
    env.plot("drive.png")
    print("뇌 주행 결과:", info["result"], f"점수 {info['score']}  진행 {info['progress']:.1f}/{env.length:.1f}m "
          f"({info['progress'] / env.length:.0%})  t={info['t']:.0f}s", info["log"], flush=True)

"""대조 실험: 실제 FlyWire 뇌가 운전에 도움이 되는가?

같은 교사 주행에서 매 스텝 세 가지 입력을 동시에 기록하고, 똑같은 방법(지수평균 + 가중 릿지)으로 운전 출력을 학습해 비교.
  특징 직접   : 카메라 요약(차선·색) + 내비 명령 354개 → 바로 선형 읽기 (뇌 없음)
  실제 뇌     : FlyWire 연결 그대로 → DN 1299개 → 선형 읽기
  섞은 뇌     : 뉴런마다 시냅스 수·세기는 그대로, 받는 쪽만 무작위로 섞음 (연결 구조만 파괴) → DN → 선형 읽기
① 오프라인: 에피소드별 교차검증 R²   ② 폐루프: 교사 데이터만으로 학습한 읽기로 seed 101~106 주행

python compare.py          수집(병렬) → ① → ②
python compare.py --fit    저장된 data/compare_data.npz 로 ① → ②
"""
import sys
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp

import sim
import train

KINDS = ("특징 직접", "실제 뇌", "섞은 뇌")
DATA = "data/compare_data.npz"


def shuffled(W, seed=7):
    """시냅스 받는 쪽(post)만 무작위로 섞은 가중치. 보내는 뉴런의 시냅스 수·세기 분포는 유지."""
    c = W.tocoo()
    return sp.csr_matrix((c.data, (c.row, np.random.default_rng(seed).permutation(c.col))), shape=W.shape)


class Features:
    """뇌 없이 카메라 요약 + 명령을 그대로 내놓는 가짜 fly (BrainDriver에 그대로 끼움)."""

    def sense(self, rgb, cmd):
        return np.r_[train.lane_marks(rgb).ravel(), train.color_marks(rgb).ravel(), cmd]

    def reset(self):
        pass


def flies(seed):
    real, shuf = train.FlyDriver(seed), train.FlyDriver(seed)
    shuf.W = shuffled(shuf.W)
    shuf.reset()
    return Features(), real, shuf


def collect(args):
    """train.collect 와 같은 교사 주행 (OU 잡음), 세 가지 입력을 한꺼번에 기록."""
    seed, noise = args
    fs, ex, env = flies(seed), sim.Expert(), sim.Env(seed=seed)
    rng = np.random.default_rng(seed)
    obs, done, ou = env.reset(), False, 0.0
    X, Y, V = [[], [], []], [], []
    while not done:
        action, lamps = ex(env, obs)
        cmd = train.commands(env, ex)
        for x, f in zip(X, fs):
            x.append(f.sense(obs, cmd))
        Y.append(train.labels(action, lamps))
        V.append(action == (0, 0) and not ex.hold)
        free = not (ex.rev or ex.rotate or ex.hold or env.park[env.idx])
        ou += -0.1 * ou + noise * 0.45 * rng.standard_normal()
        d = ou if free and action != (0, 0) else 0.0
        obs, _, done, info = env.step((action[0] - d, action[1] + d), lamps)
        if len(Y) % 400 == 0:
            print(f"  [seed {seed}] t={env.t:.0f}s 진행 {info['progress']:.1f}m", flush=True)
    print(f"  [seed {seed}] 끝: 표본 {len(Y)}", flush=True)
    return (*map(np.array, X), np.array(Y), train.weigh(V), np.full(len(Y), seed))


def cv_r2(X, Y, Wt, E):
    """에피소드 하나씩 빼고 학습 → 뺀 에피소드로 R². λ는 격자 중 평균이 가장 좋은 것 (세 입력 모두 같은 규칙)."""
    Z = train.smooth(X, E)
    best = None
    for lam in (10, 100, 1e3, 1e4, 1e5):
        r = []
        for e in np.unique(E):
            tr, te = E != e, E == e
            w, b, mu, sd = solve(Z[tr], Y[tr], Wt[tr], lam)
            p = ((Z[te] - mu) / sd) @ w + b
            r.append(1 - ((Y[te] - p) ** 2).sum(0) / ((Y[te] - Y[te].mean(0)) ** 2).sum(0))
        r = np.mean(r, 0)
        if best is None or r.mean() > best[1].mean():
            best = lam, r
    return best


def solve(Z, Y, Wt, lam):
    """가중 릿지 (train.fit 과 같은 식). Z는 이미 지수평균된 입력."""
    mu, sd = Z.mean(0), Z.std(0) + 1e-3
    A0 = (Z - mu) / sd
    s = np.sqrt(Wt)[:, None]
    ym, zm = np.average(Y, 0, Wt), np.average(A0, 0, Wt)
    A, B = (A0 - zm) * s, (Y - ym) * s
    w = np.linalg.solve(A.T @ A + lam * np.eye(Z.shape[1]), A.T @ B)
    return w, ym - zm @ w, mu, sd


def drive_one(args):
    k, seed = args
    fly = flies(seed)[k]
    r = np.load(f"data/compare_readout_{k}.npz")
    env, info = train.drive(fly, {n: r[n] for n in ("w", "b", "mu", "sd")}, seed, path=None)
    print(f"  [{KINDS[k]} seed {seed}] {info['result']} 점수 {info['score']} 진행 {info['progress']:.1f}m", flush=True)
    return k, seed, info["result"], info["score"], info["progress"]


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn")    # Linux fork 는 MuJoCo/OpenGL 로드 뒤 멈춤 (Windows 는 원래 spawn)
    if "--fit" not in sys.argv:
        print("수집: 교사 6회 병렬, 스텝마다 특징·실제 뇌·섞은 뇌 동시 기록", flush=True)
        with Pool(len(train.EPISODES)) as pool:
            parts = pool.map(collect, train.EPISODES)
        cols = [np.concatenate(a) for a in zip(*parts)]
        np.savez(DATA, F=cols[0], R=cols[1], S=cols[2], Y=cols[3], Wt=cols[4], E=cols[5])
    d = np.load(DATA)
    print("\n① 오프라인 교차검증 R² (속도, 조향, 좌램프, 우램프)", flush=True)
    for k, key in enumerate("FRS"):
        lam, r = cv_r2(d[key], d["Y"], d["Wt"], d["E"])
        w, b, mu, sd = solve(train.smooth(d[key], d["E"]), d["Y"], d["Wt"], lam)
        np.savez(f"data/compare_readout_{k}.npz", w=w, b=b, mu=mu, sd=sd)
        print(f"  {KINDS[k]:6s} 입력 {d[key].shape[1]:5d}개  R² {np.round(r, 3).tolist()}  평균 {r.mean():.3f}  (λ={lam:g})", flush=True)

    print("\n② 폐루프: 교사 데이터만으로 학습 (DAgger 없음), seed 101~106", flush=True)
    jobs = [(k, s) for k in range(3) for s in range(train.TEST_SEED + 1, train.TEST_SEED + 7)]
    with Pool(6) as pool:
        res = pool.map(drive_one, jobs)
    for k in range(3):
        rs = [r for r in res if r[0] == k]
        print(f"  {KINDS[k]:6s} 합격 {sum(r[2] == '합격' for r in rs)}/6  평균 점수 {np.mean([r[3] for r in rs]):.0f}  "
              f"평균 진행 {np.mean([r[4] for r in rs]):.1f}/{28.1}m", flush=True)
    print("ALLDONE", flush=True)

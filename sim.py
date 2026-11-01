"""FlyDrive 3D 시뮬레이터 (MuJoCo): 기능시험장 코스 + Arduino 2WD 로봇 실물 모델 + 차량 카메라 + 채점.

단위는 m, s.
env = Env(); obs = env.reset(); obs, reward, done, info = env.step((left, right))
action = 좌/우 모터 PWM 듀티 (-1~1). obs = ESP32-CAM 영상 RGB (H, W, 3) uint8.
python sim.py --show 로 3D 창 재생.
"""
import time

import mujoco
import numpy as np

# --- 코스: 격자형 시험장, 왕복 2차로 (우측통행). 차로 폭 : 차 폭 = 실제 시험장 비율(약 0.57) ---
LANE_W = 0.32            # 차로 폭
LANE_HALF = LANE_W / 2
RH = LANE_W              # 도로 반폭 (2차로)
CORNER_R = 0.5           # 회전 구간 도로 중심선 반경
XS = (0.5, 2.55, 5.0)    # 세로 도로 x: 좌 / 중앙 / 우
YS = (0.5, 2.05, 3.9)    # 가로 도로 y: 하 / 중앙 / 상
XWALK = 0.25             # 교차로 앞 횡단보도 + 정지선 구간 길이
RAMP = (3.25, 0.45, 0.3, 0.0675)          # 경사로: 시작 x, 오르막 길이, 평지 길이, 높이 (경사 15%)
# 직각주차: 좌측 도로에서 동쪽으로 들어가는 막다른 통로 + 통로 북쪽의 주차칸. 후진 주차 후 전진 출차.
PARK_ALLEY = (2.6, 3.1, 1.1)              # 통로: 남쪽 y, 북쪽 y, 길이
PARK_BAY = (1.2, 1.5, 0.32)               # 주차칸: 서쪽 x, 동쪽 x, 깊이
PARK_X, ALLEY_Y = 1.35, 2.85              # 주차칸 중심 x, 통로 중심 y
PARK_Y = 3.25                             # 주차 완료 때 바퀴축 y (차 앞머리 통로 밖, 뒤는 칸 안)
# 주행 경로 = 구간 목록 (종류, 꼭짓점):
#   "road": 도로 중심선 꼭짓점, 실제 주행선은 오른쪽 차로로 자동 이동, 90도 회전은 원호
#   "rev" / "fwd": 주차용 직선 후진 / 전진 (꼭짓점 = 바퀴축 위치). 구간 사이 방향 전환은 제자리 회전
# 출발 → ①경사로 → 좌회전 → ②우측 북상 → 좌회전 → ④교차로 직진 → 서진 → 우회전(북)
# → ③직각주차 (통로 진입, 후진 주차, 전진 출차) → 우회전 → 우회전(좌상단) → 우회전(중앙 남하)
# → ④교차로 신호 지키고 좌회전 → 좌회전(우측 북상) → 좌회전(상단 서진) ⑤가속 → 좌회전(좌측 남하) → 도착
ROUTE = [("road", [(1.0, YS[0]), (XS[2], YS[0]), (XS[2], YS[1]), (XS[0], YS[1]),
                   (XS[0], ALLEY_Y + LANE_HALF), (PARK_X, ALLEY_Y + LANE_HALF)]),
         ("rev", [(PARK_X, ALLEY_Y), (PARK_X, PARK_Y)]),
         ("fwd", [(PARK_X, PARK_Y), (PARK_X, ALLEY_Y)]),
         ("road", [(PARK_X, ALLEY_Y - LANE_HALF), (XS[0], ALLEY_Y - LANE_HALF), (XS[0], YS[2]), (XS[1], YS[2]),
                   (XS[1], YS[1]), (XS[2], YS[1]), (XS[2], YS[2]), (XS[0], YS[2]), (XS[0], 1.2)])]
LOOP = [(1.0, YS[0]), (XS[2], YS[0]), (XS[2], YS[2]), (XS[0], YS[2]), (XS[0], YS[0]), (1.4, YS[0])]  # 바깥 순환 (교차로 회전 없음)
ACCEL_X = (XS[0] + CORNER_R + 0.6, XS[2] - CORNER_R)   # ⑤ 가속구간: 상단 도로 서진, 우측 모퉁이 ~ 중앙 삼거리 지나 (지도 파란 띠). 그 뒤 60cm 감속
ZONES = {"출발": (1.0, YS[0] - LANE_HALF), "① 경사로": (3.85, YS[0]), "② 우측": (XS[2], 1.3),
         "③ 직각주차": (1.35, 2.85), "④ 교차로": (XS[1], YS[1]), "⑤ 가속": (sum(ACCEL_X) / 2, YS[2]),
         "도착": (XS[0], 1.2)}
WHITE, YELLOW = "1 1 1 1", "1 .8 0 1"

# --- 차: Arduino 2WD 스마트카 키트 실측 기준 ---  실물 측정값으로 맞출 것
# 아크릴 215x135mm, TT 기어모터 1:48 + 65mm 바퀴, 뒤 캐스터, 18650x2, Arduino Uno, L298N, ESP32-CAM
WHEEL_R = 0.0325
WHEEL_HW = 0.013         # 바퀴 반폭
WHEEL_Y = 0.0825         # 바퀴 중심 좌우 위치
CAR_HALF_W = WHEEL_Y + WHEEL_HW          # 차 반폭 (바퀴 바깥면, 약 19cm 폭)
CAR_FRONT, CAR_REAR = 0.065, 0.1525      # 바퀴축 → 앞머리 / 뒤끝 (아크릴판)
W0 = 20.9                # TT 모터 무부하 속도 [rad/s] (약 200rpm @ 6V)
STALL = 0.08             # TT 모터 정지 토크 [N·m]
MOTOR_GAIN = (1.0, 1.0)  # 좌/우 모터 개체차 보정 (실물에서 직진 안 되면 조정)
V_MAX = W0 * WHEEL_R     # 무부하 최고속도 약 0.68 m/s
CAM_X, CAM_Z = 0.06, 0.15                # 카메라 위치: 바퀴축 앞, 지면 위 높이
CAM_PITCH = 30           # 카메라 하향 각도 [도]
CAM_DIAG_FOV = 120       # OV2640 광각 렌즈 대각 화각 [도]
CAM_H, CAM_W = 240, 320  # QVGA. 신호등 LED(지름 2cm)가 1m 앞에서 3~4픽셀 (80x60이면 1픽셀 미만이라 안 보임)
CAM_FOVY = 2 * np.degrees(np.arctan(0.6 * np.tan(np.radians(CAM_DIAG_FOV / 2))))
DT = 0.05                # 제어 주기 (20Hz)
PHYS_DT = 0.002

# --- 램프 (Arduino LED): step(action, lamps) 의 lamps 비트 ---
LEFT, RIGHT, HEAD = 1, 2, 4
HAZARD = LEFT | RIGHT    # 비상점멸등 = 좌우 동시 점멸

# --- 신호등 (도로용 Arduino + LED): 한국식 4색 가로형 (빨강·노랑·좌회전 화살표·초록), 4현시 ---
# (초, 동서 방향 진행 차로에 켜지는 등, 남북 방향 진행 차로에 켜지는 등). 마주 오는 두 방향은 같은 신호.
SIGNAL_PHASES = (
    (6.0, {"green"}, {"red"}),             # 동서 직진
    (2.0, {"yellow"}, {"red"}),
    (5.0, {"red", "arrow"}, {"red"}),      # 동서 좌회전
    (2.0, {"yellow"}, {"red"}),
    (6.0, {"red"}, {"green"}),             # 남북 직진
    (2.0, {"red"}, {"yellow"}),
    (5.0, {"red"}, {"red", "arrow"}),      # 남북 좌회전
    (2.0, {"red"}, {"yellow"}),
)
SIGNAL_LAMPS = ("red", "yellow", "arrow", "green")    # 보는 사람 기준 왼쪽부터
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))             # 진행 방향: 동, 서, 북, 남 (0·1 = 동서, 2·3 = 남북)


def can_go(lit, turn):
    """신호 lit 일 때 turn(0 직진, LEFT, RIGHT) 진행 가능? 우회전은 신호 무관 (보행자 주의)."""
    return turn == RIGHT or ("arrow" if turn == LEFT else "green") in lit

# --- 채점: 2종 보통 장내기능시험 (100점 시작, 80점 미만 불합격) ---
# 감점 점수는 공식 채점표 기준으로 넣은 값. 시험장 기준이 다르면 여기만 고칠 것.
PENALTY = {
    "차로 준수 위반 (검지선 접촉)": 15,
    "좌·우회전 검지선 접촉": 5,
    "지정속도 초과": 10,
    "출발 방향지시등 미작동": 5,
    "경사로 정지 안 함 (3초)": 10,
    "경사로 뒤로 밀림": 10,
    "경사로 30초 내 통과 못함": 10,
    "직각주차 검지선 접촉": 10,
    "직각주차 확인선 미접촉": 10,
    "교차로 방향지시등 미작동": 5,
    "교차로 30초 내 통과 못함": 5,
    "돌발 2초 내 정지 안 함": 10,
    "돌발 비상점멸등 미작동 (정지 후 3초)": 10,
    "돌발 후 비상점멸등 안 끄고 출발": 5,
    "가속구간 속도 미달": 10,
    "종료 방향지시등 미작동": 5,
}
PASS_SCORE = 80
TIME_LIMIT = 300.0
# 속도: 실제 20km/h 를 코스 축척(약 1/9.4)으로 줄인 값 근처. 실물 주행 보고 조정
SPEED_LIMIT = 0.45       # 지정속도 [m/s]
ACCEL_MIN = 0.45         # 가속구간 최소 도달 속도 [m/s]
ROLLBACK = 0.05          # 경사로 뒤로 밀림 허용 (실제 50cm 축척)
DUMMY_AHEAD = 0.7        # 돌발 사람 모형이 나타나는 거리 (차 앞)
DUMMY_HOLD = 5.0         # 차가 멈춘 뒤 모형이 치워질 때까지 [초] (비상점멸등 3초 판정보다 길게)


def make_track(route=ROUTE, step=0.01):
    """도로 중심선 꼭짓점 → 오른쪽 차로 중심 주행선 (N, 2), 점별 구간 번호, 회전 구간 여부.
    90도 회전은 도로 곡선과 같은 중심의 원호."""
    p = np.array(route, float)
    d = np.diff(p, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    right = np.c_[d[:, 1], -d[:, 0]]
    q = p.copy()
    q[0] += right[0] * LANE_HALF
    q[-1] += right[-1] * LANE_HALF
    q[1:-1] += (right[:-1] + right[1:]) * LANE_HALF
    pts, seg, arc, cur = [], [], [], q[0]
    for i in range(1, len(q)):
        a = d[i - 1]
        if i < len(q) - 1:
            b = d[i]
            left_turn = a[0] * b[1] - a[1] * b[0] > 0
            r = CORNER_R + LANE_HALF if left_turn else CORNER_R - LANE_HALF
            end = q[i] - a * r
        else:
            end = q[i]
        n = int(np.linalg.norm(end - cur) / step)
        pts.append(cur + np.outer(np.arange(n) * step, a))
        seg += [i - 1] * n
        arc += [False] * n
        if i < len(q) - 1:
            phi = np.arange(0, np.pi / 2, step / r)[:, None]
            pts.append(end + b * r + r * (-b * np.cos(phi) + a * np.sin(phi)))
            seg += [i - 1] * len(phi)
            arc += [True] * len(phi)
            cur = q[i] + b * r
    return np.vstack(pts), np.array(seg), np.array(arc)


def make_course(route, step=0.01):
    """구간 목록 → 주행선 (N, 2), 회전 여부, 구간 번호, 구간 종류 목록. 꼭짓점만 준 경로는 "road" 한 구간."""
    if not isinstance(route[0][0], str):
        route = [("road", route)]
    pts, arc, leg = [], [], []
    for k, (kind, v) in enumerate(route):
        if kind == "road":
            p, _, a = make_track(v, step)
        else:
            a0, a1 = np.array(v, float)
            p = a0 + np.outer(np.arange(int(np.linalg.norm(a1 - a0) / step) + 1) * step, (a1 - a0) / np.linalg.norm(a1 - a0))
            a = np.zeros(len(p), bool)
        pts.append(p)
        arc.append(a)
        leg += [k] * len(p)
    return np.vstack(pts), np.concatenate(arc), np.array(leg), [kind for kind, _ in route]


def resample(pts, step=0.05):
    """꼭짓점은 유지하고 각 변을 step 이하 간격으로 나눔."""
    pts = np.asarray(pts, float)
    out = [np.linspace(a, b, max(1, int(np.ceil(np.linalg.norm(b - a) / step))), endpoint=False)
           for a, b in zip(pts[:-1], pts[1:])]
    return np.vstack(out + [pts[-1:]])


def rrect(x0, y0, x1, y1, r, notch=None):
    """둥근 모서리 직사각형 둘레 (닫힌 선). notch = 서쪽 변에서 위→아래로 끼워 넣을 꼭짓점들."""
    arcs = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        a = np.radians(np.linspace(a0, a0 + 90, 10))
        arcs.append(np.c_[cx + r * np.cos(a), cy + r * np.sin(a)])
    if notch is not None:
        arcs.insert(3, notch)
    pts = np.vstack(arcs)
    return resample(np.vstack([pts, pts[:1]]))


def park_outline():
    """직각주차 통로 + 주차칸 테두리 (좌상단 블록 서쪽 변에서 위→아래 순서)."""
    x0 = XS[0] + RH
    ay0, ay1, alen = PARK_ALLEY
    bx0, bx1, bd = PARK_BAY
    return np.array([(x0, ay1), (bx0, ay1), (bx0, ay1 + bd), (bx1, ay1 + bd), (bx1, ay1),
                     (x0 + alen, ay1), (x0 + alen, ay0), (x0, ay0)])


def cut(line, boxes):
    """선에서 상자 (cx, cy, 반폭x, 반폭y) 안 부분을 잘라낸 조각들."""
    keep = np.ones(len(line), bool)
    for cx, cy, hx, hy in boxes:
        keep &= ~((abs(line[:, 0] - cx) < hx) & (abs(line[:, 1] - cy) < hy))
    runs = np.split(np.arange(len(line)), np.flatnonzero(np.diff(keep.astype(int))) + 1)
    return [line[r] for r in runs if keep[r[0]] and len(r) > 1]


def markings():
    """노면 표시 전부: [(점 배열, 폭, 색)]. 실물 코스에서는 테이프 한 줄 = 항목 하나."""
    m = []
    po = park_outline()
    for i in (0, 1):     # 흰 가장자리선: 잔디 블록 둘레 + 바깥 테두리. 주차장 테두리만 노란 검지선
        for j in (0, 1):
            m.append((rrect(XS[i] + RH, YS[j] + RH, XS[i + 1] - RH, YS[j + 1] - RH, CORNER_R - RH,
                            po if (i, j) == (0, 1) else None), 0.02, WHITE))
    lo, hi = po.min(axis=0), po.max(axis=0)
    white_tl = cut(m[1][0], [((lo[0] + hi[0]) / 2 + 0.01, (lo[1] + hi[1]) / 2, (hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2 + 0.01)])
    m[1:2] = [(piece, 0.02, WHITE) for piece in white_tl]
    m.append((resample(po), 0.02, YELLOW))
    bx0, bx1, bd = PARK_BAY
    yc = PARK_ALLEY[1] + bd * 0.7                                   # 주차칸 안 흰 확인선
    m.append(([(bx0, yc), (bx1, yc)], 0.03, WHITE))
    m.append((rrect(XS[0] - RH, YS[0] - RH, XS[2] + RH, YS[2] + RH, CORNER_R + RH), 0.02, WHITE))

    junctions = [(XS[1], YS[0], RH, RH), (XS[0], YS[1], RH, RH), (XS[2], YS[1], RH, RH), (XS[1], YS[2], RH, RH),
                 (XS[1], YS[1], RH + XWALK, RH + XWALK)]
    for line in (rrect(XS[0], YS[0], XS[2], YS[2], CORNER_R),          # 노란 중앙선, 교차로 안은 없음
                 resample([(XS[0], YS[1]), (XS[2], YS[1])]), resample([(XS[1], YS[0]), (XS[1], YS[2])])):
        m += [(piece, 0.02, YELLOW) for piece in cut(line, junctions)]

    c = np.array((XS[1], YS[1]))   # ④ 신호교차로: 진입 방향마다 정지선 + 횡단보도 + 마름모(횡단보도 예고)
    for a in np.array([(1, 0), (-1, 0), (0, 1), (0, -1)]):
        n = np.array((a[1], -a[0]))                  # 진입 차로 쪽 (우측)
        s = c - a * (RH + XWALK - 0.02)
        m.append(([s, s + n * RH], 0.04, WHITE))
        for o in np.arange(-RH + 0.035, RH, 0.07):
            m.append(([c - a * (RH + 0.02) + n * o, c - a * (RH + 0.17) + n * o], 0.035, WHITE))
        dc, k = c - a * (RH + XWALK + 0.35) + n * LANE_HALF, 0.06
        dia = [dc + a * k, dc + n * k * 0.6, dc - a * k, dc - n * k * 0.6, dc + a * k]
        m.append((dia, 0.012, WHITE))
    h = 0.22                       # 교차로 가운데 노란 빗금 (정차금지지대)
    m.append(([c + (-h, -h), c + (h, -h), c + (h, h), c + (-h, h), c + (-h, -h)], 0.015, YELLOW))
    for o in np.arange(-2 * h + 0.07, 2 * h, 0.07):
        p0 = c + (max(-h, o - h), max(-h, -o - h))
        p1 = c + (min(h, o + h), min(h, h - o))
        m.append(([p0, p1], 0.015, YELLOW))

    y_lane = YS[0] - LANE_HALF     # 출발선, 직진 화살표, 경사로 정지선, 도착 정지선
    m.append(([(1.0, YS[0] - RH), (1.0, YS[0])], 0.04, WHITE))
    for seg in ([(1.25, y_lane), (1.55, y_lane)], [(1.55, y_lane), (1.47, y_lane + 0.05)],
                [(1.55, y_lane), (1.47, y_lane - 0.05)]):
        m.append((seg, 0.025, WHITE))
    xr = RAMP[0] + RAMP[1] + RAMP[2] / 2
    m.append(([(xr, YS[0] - RH), (xr, YS[0])], 0.04, WHITE))
    m.append(([(XS[0] - RH, 1.2), (XS[0], 1.2)], 0.04, WHITE))
    return m


def on_park_area(p):
    """점이 좌측 도로 / 주차 통로 / 주차칸 안인가 (노란 검지선 안쪽)."""
    x, y = p
    ay0, ay1, alen = PARK_ALLEY
    bx0, bx1, bd = PARK_BAY
    return (abs(x - XS[0]) < RH or (XS[0] + RH <= x < XS[0] + RH + alen and ay0 < y < ay1)
            or (bx0 < x < bx1 and ay1 <= y < ay1 + bd))


def ramp_z(x, y):
    x0, up, flat, hgt = RAMP
    if abs(y - YS[0]) > RH + 0.02:
        return 0.0
    return hgt * float(np.clip(min(x - x0, x0 + 2 * up + flat - x) / up, 0, 1))


CAR_XML = """
    <body name="car" pos="0 0 {z0}">
      <freejoint/>
      <!-- 1층 아크릴판 215x135x3 (충돌 있음) -->
      <geom type="box" size=".1075 .0675 .0015" pos="-.045 0 .0115" mass=".09" rgba=".8 .9 1 .45"/>
      <!-- TT 기어모터 (노랑) -->
      <geom type="box" {vis} size=".035 .0095 .009" pos="-.02 .05 .001" mass=".03" rgba="1 .8 .1 1"/>
      <geom type="box" {vis} size=".035 .0095 .009" pos="-.02 -.05 .001" mass=".03" rgba="1 .8 .1 1"/>
      <!-- 뒤 볼 캐스터 -->
      <geom type="cylinder" {vis} fromto="-.125 0 .01 -.125 0 {cz}" size=".008" rgba=".7 .7 .7 1"/>
      <geom type="sphere" size=".012" pos="-.125 0 {cz}" mass=".02" friction=".01 .001 .001" priority="1" rgba=".8 .8 .8 1"/>  <!-- priority: 바닥 마찰 대신 캐스터 마찰 적용 -->
      <!-- 18650 x2 배터리 홀더 -->
      <geom type="box" {vis} size=".0385 .0205 .0105" pos="-.095 0 .0235" mass=".11" rgba=".1 .1 .1 1"/>
      <!-- L298N 모터드라이버 (빨간 기판 + 검은 방열판) -->
      <geom type="box" {vis} size=".0215 .0215 .0008" pos=".035 0 .0138" mass=".02" rgba=".8 .1 .1 1"/>
      <geom type="box" {vis} size=".0075 .0115 .0115" pos=".03 0 .026" mass=".01" rgba=".05 .05 .05 1"/>
      <!-- 황동 서포트 x4 + 2층 아크릴판 -->
      <geom type="cylinder" {vis} fromto="-.14 .055 .013 -.14 .055 .043" size=".0025" rgba=".8 .65 .2 1"/>
      <geom type="cylinder" {vis} fromto="-.14 -.055 .013 -.14 -.055 .043" size=".0025" rgba=".8 .65 .2 1"/>
      <geom type="cylinder" {vis} fromto="-.01 .055 .013 -.01 .055 .043" size=".0025" rgba=".8 .65 .2 1"/>
      <geom type="cylinder" {vis} fromto="-.01 -.055 .013 -.01 -.055 .043" size=".0025" rgba=".8 .65 .2 1"/>
      <geom type="box" {vis} size=".075 .0675 .0015" pos="-.075 0 .0445" mass=".06" rgba=".8 .9 1 .45"/>
      <!-- Arduino Uno 68.6x53.4 (청록 기판 + USB-B + MCU) -->
      <geom type="box" {vis} size=".0343 .0267 .0008" pos="-.085 0 .0468" mass=".025" rgba="0 .45 .55 1"/>
      <geom type="box" {vis} size=".008 .006 .0055" pos="-.115 .012 .053" rgba=".75 .75 .75 1"/>
      <geom type="box" {vis} size=".018 .004 .002" pos="-.08 0 .049" rgba=".05 .05 .05 1"/>
      <!-- 카메라 기둥 + ESP32-CAM (기판 27x40 + OV2640 렌즈) -->
      <geom type="box" {vis} size=".003 .02 .056" pos=".05 0 .069" mass=".01" rgba=".15 .15 .15 1"/>
      <geom type="box" {vis} size=".0023 .0135 .02" pos=".055 0 {bz}" mass=".01" rgba=".2 .15 .1 1"/>
      <geom type="cylinder" {vis} size=".005 .003" pos="{lx} 0 {lz}" euler="0 {lens} 0" rgba=".05 .05 .05 1"/>
      <!-- LED: 앞뒤 좌우 노랑 (방향지시등·비상점멸등) + 앞 흰색 (전조등) -->
      <geom name="led_fl" type="sphere" {vis} size=".004" pos=".06 .06 .016"/>
      <geom name="led_fr" type="sphere" {vis} size=".004" pos=".06 -.06 .016"/>
      <geom name="led_rl" type="sphere" {vis} size=".004" pos="-.15 .06 .016"/>
      <geom name="led_rr" type="sphere" {vis} size=".004" pos="-.15 -.06 .016"/>
      <geom name="head_l" type="sphere" {vis} size=".004" pos=".062 .025 .016"/>
      <geom name="head_r" type="sphere" {vis} size=".004" pos=".062 -.025 .016"/>
      <camera name="cam" pos="{cx} 0 {cam_z}" xyaxes="0 -1 0 {s} 0 {c}" fovy="{fovy}"/>
      <body name="wl" pos="0 {wy} 0">
        <joint name="wl" type="hinge" axis="0 1 0" damping=".0002" armature=".0002" frictionloss=".015"/>  <!-- 기어 관성·마찰 (경사로에서 안 밀림) -->
        <geom type="cylinder" size="{wr} {whw}" euler="90 0 0" mass=".025" friction="1.2" rgba=".1 .1 .1 1"/>
        <geom type="cylinder" {vis} size=".02 {hub}" euler="90 0 0" rgba="1 .85 .2 1"/>
      </body>
      <body name="wr" pos="0 -{wy} 0">
        <joint name="wr" type="hinge" axis="0 1 0" damping=".0002" armature=".0002" frictionloss=".015"/>
        <geom type="cylinder" size="{wr} {whw}" euler="90 0 0" mass=".025" friction="1.2" rgba=".1 .1 .1 1"/>
        <geom type="cylinder" {vis} size=".02 {hub}" euler="90 0 0" rgba="1 .85 .2 1"/>
      </body>
    </body>"""


def build_xml():
    g = []
    vis = 'contype="0" conaffinity="0"'
    asphalt = 'rgba=".38 .38 .4 1"'

    def box(x0, y0, x1, y1, z):   # 도로 면 (시각용, 차는 바닥 평면 위를 달림)
        g.append(f'<geom type="box" {vis} {asphalt} pos="{(x0 + x1) / 2} {(y0 + y1) / 2} {z}" '
                 f'size="{(x1 - x0) / 2} {(y1 - y0) / 2} {z}"/>')
    for y in YS:
        box(XS[0] - RH, y - RH, XS[2] + RH, y + RH, 0.0005)
    for x in XS:
        box(x - RH, YS[0] - RH, x + RH, YS[2] + RH, 0.0004)   # 높이 살짝 달리해 겹침 깜빡임 방지
    ay0, ay1, alen = PARK_ALLEY
    bx0, bx1, bd = PARK_BAY
    box(XS[0] + RH - 0.01, ay0, XS[0] + RH + alen, ay1, 0.0003)
    box(bx0, ay1 - 0.01, bx1, ay1 + bd, 0.0003)

    x0, up, flat, hgt = RAMP       # 경사로 (충돌 있음): 오르막, 평지, 내리막 판
    ang = np.arctan2(hgt, up)
    hl, t, rw = np.hypot(up, hgt) / 2, 0.005, RH + 0.02
    for cx, a in ((x0 + up / 2, -ang), (x0 + up + flat + up / 2, ang)):
        cz = hgt / 2 - t * np.cos(ang)
        g.append(f'<geom type="box" {asphalt} pos="{cx} {YS[0]} {cz}" euler="0 {np.degrees(a)} 0" '
                 f'size="{hl} {rw} {t}"/>')
    g.append(f'<geom type="box" {asphalt} pos="{x0 + up + flat / 2} {YS[0]} {hgt - t}" size="{flat / 2} {rw} {t}"/>')

    for pts, width, color in markings():    # 노면 표시: 납작한 띠 조각들
        pts = np.asarray(pts, float)
        z = np.array([ramp_z(x, y) for x, y in pts]) + 0.0015
        p = np.c_[pts, z]
        for a, b in zip(p[:-1], p[1:]):
            d = b - a
            L = np.linalg.norm(d)
            if L < 1e-6:          # 겹친 꼭짓점
                continue
            d /= L
            yax = np.array([-d[1], d[0], 0]) / np.hypot(d[0], d[1])
            c = (a + b) / 2
            g.append(f'<geom type="box" {vis} rgba="{color}" pos="{c[0]} {c[1]} {c[2]}" '
                     f'xyaxes="{d[0]} {d[1]} {d[2]} {yax[0]} {yax[1]} 0" size="{L / 2 + width / 4} {width / 2} 0.0005"/>')

    # 신호등 (도로용 Arduino Nano + 10mm LED): ④교차로 네 모퉁이 기둥 + 도로 위로 뻗은 팔 + 4색 가로형 등.
    # 진행 방향마다 건너편 오른쪽 모퉁이 기둥에서 팔이 그 차로 위까지 나와 차를 마주봄 (실제 교차로 배치).
    # 높이 30cm: 실제 5.5m 를 축척하면 58cm 지만 차 카메라(15cm, 30도 하향)에 정지선에서 보이게 낮춤
    c, zh, sp = np.array((XS[1], YS[1])), 0.3, 0.028
    for k, d in enumerate(np.array(DIRS)):
        n = np.array((d[1], -d[0]))                           # 진행 방향 오른쪽
        pole = c + d * (RH + 0.1) + n * (RH + 0.1)
        h = c + d * (RH + 0.1) + n * LANE_HALF                # 등 중심: 진입 차로 위
        g.append(f'<geom type="cylinder" {vis} rgba=".55 .55 .55 1" fromto="{pole[0]} {pole[1]} 0 {pole[0]} {pole[1]} {zh + .03}" size=".007"/>')
        g.append(f'<geom type="cylinder" {vis} rgba=".55 .55 .55 1" fromto="{pole[0]} {pole[1]} {zh + .03} {h[0]} {h[1]} {zh + .03}" size=".004"/>')
        hx, hy = abs(d) * .01 + abs(n) * (2 * sp + .012)
        g.append(f'<geom type="box" {vis} rgba=".08 .08 .08 1" pos="{h[0]} {h[1]} {zh}" size="{hx} {hy} .016"/>')
        for j, name in enumerate(SIGNAL_LAMPS):
            p = h - d * .011 + n * (j - 1.5) * sp            # 왼쪽(-n)부터 빨강·노랑·화살표·초록
            if name != "arrow":
                g.append(f'<geom name="sig_{name}{k}" type="cylinder" {vis} material="lamp" pos="{p[0]} {p[1]} {zh}" '
                         f'size=".011 .001" zaxis="{d[0]} {d[1]} 0"/>')
                continue
            g.append(f'<geom type="cylinder" {vis} rgba=".02 .02 .02 1" pos="{p[0]} {p[1]} {zh}" size=".011 .001" zaxis="{d[0]} {d[1]} 0"/>')
            q = p - d * .0015                                 # 검은 등 위에 ← 모양 (보는 사람 기준 왼쪽 = -n)
            for m, (a, b) in enumerate((((-.008, 0), (.008, 0)), ((-.008, 0), (-.002, .006)), ((-.008, 0), (-.002, -.006)))):
                u = np.r_[n * (b[0] - a[0]), b[1] - a[1]]
                L = np.linalg.norm(u)
                mid = np.r_[q + n * (a[0] + b[0]) / 2, zh + (a[1] + b[1]) / 2]
                g.append(f'<geom name="sig_arrow{k}_{m}" type="box" {vis} material="lamp" pos="{mid[0]} {mid[1]} {mid[2]}" '
                         f'xyaxes="{u[0] / L} {u[1] / L} {u[2] / L} {d[0]} {d[1]} 0" size="{L / 2 + .002} .0003 .0028"/>')
    # 남서 모퉁이 기둥 아래 도로용 Arduino 제어함: 회색 함 + Nano(파랑) + 브레드보드 + 기둥 따라 올라가는 LED 배선
    # (다른 세 기둥까지 배선은 바닥 밑으로)
    sx, sy, sz = XS[1] - RH - 0.1, YS[1] - RH - 0.1, zh
    bx, by = sx - .05, sy - .03
    g.append(f'<geom type="box" {vis} rgba=".55 .55 .55 1" pos="{bx} {by} .0125" size=".04 .03 .0125"/>')
    g.append(f'<geom type="box" {vis} rgba=".95 .95 .92 1" pos="{bx} {by} .026" size=".03 .02 .002"/>')
    g.append(f'<geom type="box" {vis} rgba=".1 .3 .7 1" pos="{bx} {by} .029" size=".0225 .009 .001"/>')
    g.append(f'<geom type="cylinder" {vis} rgba=".05 .05 .05 1" fromto="{bx + .03} {by + .02} .02 {sx} {sy - .008} .02" size=".002"/>')
    g.append(f'<geom type="cylinder" {vis} rgba=".05 .05 .05 1" fromto="{sx} {sy - .008} .02 {sx} {sy - .008} {sz}" size=".002"/>')
    # 돌발용 사람 모형 (키 18cm = 170cm 축척). 평소엔 코스 밖에 숨김
    g.append('<body name="dummy" mocap="true" pos="-3 -3 -1">'
             '<geom type="capsule" fromto="0 0 .025 0 0 .14" size=".025" rgba=".2 .3 .8 1"/>'
             '<geom type="sphere" pos="0 0 .165" size=".02" rgba=".95 .75 .6 1"/></body>')

    s, c = np.sin(np.radians(CAM_PITCH)), np.cos(np.radians(CAM_PITCH))
    car = CAR_XML.format(z0=WHEEL_R + 0.003, vis=vis, cz=0.012 - WHEEL_R, bz=CAM_Z - WHEEL_R - 0.008,
                         lx=CAM_X - 0.002, lz=CAM_Z - WHEEL_R, lens=90 + CAM_PITCH, cx=CAM_X, cam_z=CAM_Z - WHEEL_R,
                         s=s, c=c, fovy=CAM_FOVY, wy=WHEEL_Y, wr=WHEEL_R, whw=WHEEL_HW, hub=WHEEL_HW + 0.0005)
    return f"""
<mujoco>
  <option timestep="{PHYS_DT}" integrator="implicitfast"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <asset>
    <material name="lamp" emission="1"/>  <!-- LED: 조명과 상관없이 제 색으로 빛남 -->
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".6 .8 1" rgb2="1 1 1" width="64" height="64"/>
  </asset>
  <worldbody>
    <light pos="2.7 2.2 4" dir="0 0 -1" directional="true"/>
    <geom type="plane" size="5 5 .1" pos="2.75 2.2 0" rgba=".55 .75 .4 1"/>
    {chr(10).join(g)}
    {car}
  </worldbody>
  <actuator>
    <!-- DC 모터: 토크 = 정지토크 x (듀티 - 속도/무부하속도) -->
    <velocity joint="wl" kv="{STALL / W0}" ctrlrange="{-W0} {W0}" forcerange="{-STALL} {STALL}"/>
    <velocity joint="wr" kv="{STALL / W0}" ctrlrange="{-W0} {W0}" forcerange="{-STALL} {STALL}"/>
  </actuator>
</mujoco>"""


class Env:
    """route: 주행 경로. dummy: 돌발 사람 모형 사용. accel: ⑤가속구간 채점. 기본값은 전체 코스(ROUTE)일 때만 켬."""

    def __init__(self, route=ROUTE, dummy=None, accel=None, seed=None):
        full = route is ROUTE
        self.dummy = full if dummy is None else dummy
        accel = full if accel is None else accel
        self.model = mujoco.MjModel.from_xml_string(build_xml())
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, CAM_H, CAM_W)
        self.track, self.arc, self.leg, self.kinds = make_course(route)
        self.leg_start = np.searchsorted(self.leg, np.arange(len(self.kinds)))
        self.leg_end = np.r_[self.leg_start[1:], len(self.leg)] - 1
        self.park = np.array([self.kinds[k] != "road" for k in self.leg])
        self.n = len(self.track)
        self.length = self.n * 0.01
        self.substeps = round(DT / PHYS_DT)
        self.rng = np.random.default_rng(seed)
        g = self.model.geom
        self.leds = [g(k).id for k in ("led_fl", "led_rl")], [g(k).id for k in ("led_fr", "led_rr")]
        self.heads = [g("head_l").id, g("head_r").id]
        self.sig_geoms = [{name: [g(f"sig_{name}{k}").id] if name != "arrow" else [g(f"sig_arrow{k}_{m}").id for m in range(3)]
                           for name in SIGNAL_LAMPS} for k in range(len(DIRS))]
        self.dummy_mocap = self.model.body("dummy").mocapid[0]

        # ④ 교차로 정지선을 지나는 지점 (경로 인덱스), 거기서 할 회전 (= 켜야 할 방향지시등), 진행 방향 번호
        tang = np.gradient(self.track, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9      # 주차 후진→전진 꼭짓점은 0
        self.stoplines = []
        c = np.array((XS[1], YS[1]))
        for k, d in enumerate(np.array(DIRS)):
            n = np.array((d[1], -d[0]))
            s0 = c - d * (RH + XWALK - 0.02)
            rel = self.track - s0
            hit = np.flatnonzero((tang @ d > 0.9) & (abs(rel @ d) < 0.006) & (rel @ n > 0) & (rel @ n < RH))
            if len(hit):
                i = hit[0]
                t1 = tang[min(i + 200, self.n - 1)]
                cross = tang[i, 0] * t1[1] - tang[i, 1] * t1[0]
                self.stoplines.append((i, LEFT if cross > 0.5 else RIGHT if cross < -0.5 else 0, k))
        tx, ty = self.track.T
        acc = np.flatnonzero((tang[:, 0] < -0.9) & (abs(ty - YS[2] - LANE_HALF) < 0.05)
                             & (tx > ACCEL_X[0]) & (tx < ACCEL_X[1])) if accel else []
        self.accel = (acc[0], acc[-1] - 30) if len(acc) else None      # 끝 30cm 전까지 가속, 이후 감속

    # --- 상태 조회 (교사 제어기·디버깅용) ---
    def pose(self):
        """(x, y, heading)"""
        b = self.data.body("car")
        return b.xpos[0], b.xpos[1], np.arctan2(b.xmat[3], b.xmat[0])

    def corners(self):
        """차 네 모서리 (앞왼, 앞오, 뒤왼, 뒤오) 바닥 좌표."""
        x, y, th = self.pose()
        lx = np.array([CAR_FRONT, CAR_FRONT, -CAR_REAR, -CAR_REAR])
        ly = np.array([1, -1, 1, -1]) * CAR_HALF_W
        c, s = np.cos(th), np.sin(th)
        return np.c_[x + c * lx - s * ly, y + s * lx + c * ly]

    def speed(self):
        return float(np.hypot(*self.data.qvel[:2]))

    def signal(self, k):
        """진행 방향 k (DIRS 번호) 차로의 신호등에 켜진 등 집합."""
        t = (self.t + self.sig_offset) % sum(ph[0] for ph in SIGNAL_PHASES)
        for d, ew, ns in SIGNAL_PHASES:
            if t < d:
                return ew if k < 2 else ns
            t -= d

    def in_ramp_zone(self):
        x, y, th = self.pose()
        x0, up, flat, _ = RAMP
        return x0 + up - 0.05 <= x <= x0 + up + flat and abs(y - YS[0]) < RH and np.cos(th) > 0.7

    def in_accel(self, margin=0):
        return self.accel is not None and self.accel[0] <= self.idx <= self.accel[1] + margin

    def dummy_pos(self):
        return self.data.mocap_pos[self.dummy_mocap, :2]

    def _obs(self):
        self.renderer.update_scene(self.data, camera="cam")
        return self.renderer.render()

    def _nearest(self, x, y):
        # 경로가 자기 자신과 겹치므로(상단·좌측 도로 두 번 주행, 주차 후진·전진) 현재 구간의 근처에서만 찾고,
        # 구간 끝에 닿아야 다음 구간으로 넘어감
        k = self.leg[self.idx]
        end = self.leg_end[k]
        if self.idx < end - 2:              # 구간 끝 2cm 안이면 다음 구간도 후보 (Expert 와 같은 기준)
            lo, hi = max(self.idx - 30, self.leg_start[k]), min(self.idx + 100, end + 1)
        else:
            lo, hi = self.idx, min(self.idx + 100, self.leg_end[min(k + 1, len(self.kinds) - 1)] + 1)
        d = np.hypot(*(self.track[lo:hi] - (x, y)).T)
        i = d.argmin()
        return d[i], lo + i

    def _penalize(self, name):
        if name in self.given:          # 구간 과제는 한 번만 (차로·속도는 매 위반마다)
            return 0
        if not name.startswith(("차로", "좌·우회전", "지정속도")):
            self.given.add(name)
        self.score -= PENALTY[name]
        self.log.append((round(self.t, 1), name, -PENALTY[name]))
        return PENALTY[name]

    def _set_lamps(self, lamps):
        on = int(self.t * 3) % 2 == 0                   # 1.5Hz 점멸
        rgba = self.model.geom_rgba
        for bit, ids in zip((LEFT, RIGHT), self.leds):
            rgba[ids] = (1, .55, 0, 1) if lamps & bit and on else (.35, .25, .1, 1)
        rgba[self.heads] = (1, 1, .9, 1) if lamps & HEAD else (.5, .5, .5, 1)
        for k, geoms in enumerate(self.sig_geoms):
            sig = self.signal(k)
            for name, ids in geoms.items():
                lit = {"red": (1, 0, 0, 1), "yellow": (1, .8, 0, 1), "arrow": (0, 1, .3, 1), "green": (0, 1, .3, 1)}[name]
                off = (.02, .02, .02, 1) if name == "arrow" else tuple(v * .25 for v in lit[:3]) + (1,)
                rgba[ids] = lit if name in sig else off

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        d = self.track[1] - self.track[0]
        th = np.arctan2(d[1], d[0])
        self.data.qpos[:2] = self.track[0]
        self.data.qpos[3:7] = (np.cos(th / 2), 0, 0, np.sin(th / 2))
        self.data.mocap_pos[self.dummy_mocap] = (-3, -3, -1)
        mujoco.mj_forward(self.model, self.data)
        self.t, self.idx, self.progress = 0.0, 0, 0.0
        self.score, self.touching, self.speeding = 100, False, False
        self.given, self.log = set(), []
        self.lamps = 0
        self.started = False
        self.sig_offset = self.rng.uniform(0, sum(ph[0] for ph in SIGNAL_PHASES))
        self.ramp = dict(inside=False, t_in=0.0, still=0.0, stopped=False, stop_x=None)
        self.cross = {}                                  # 교차로 통과 시간 (신호 대기 뺀 값)
        self.acc_max, self.acc_was = 0.0, False
        self.parking = dict(touched=False, rear=-np.inf)
        self.dum = dict(state="wait", trigger=None)
        if self.dummy:   # 돌발: 코스 사이 직선 구간 아무 데서나 (교차로·경사로·주차장·가속구간 근처 제외)
            ok = ~self.arc & ~self.park & (np.array([ramp_z(*p) for p in self.track]) == 0)
            pk = np.flatnonzero(self.park)
            if len(pk):
                ok[max(pk[0] - 150, 0):pk[-1] + 150] = False
            if self.accel:
                ok[self.accel[0] - 100:self.accel[1] + 100] = False
            for sl, _, _ in self.stoplines:
                ok[max(sl - 150, 0):sl + 150] = False
            ahead = round(DUMMY_AHEAD / 0.01)
            cand = [i for i in range(int(self.n * 0.15), int(self.n * 0.6)) if ok[i] and ok[min(i + ahead, self.n - 1)]]
            self.dum["trigger"] = int(self.rng.choice(cand))
        self._set_lamps(0)
        self.traj = [self.pose()]
        return self._obs()

    def step(self, action, lamps=0):
        """action: (좌, 우) 모터 듀티 -1~1.  lamps: LED 비트 (LEFT | RIGHT | HEAD, 비상점멸등 = HAZARD)."""
        self.lamps = lamps
        self._set_lamps(lamps)
        self.data.ctrl[:] = np.clip(action, -1, 1) * MOTOR_GAIN * W0
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        self.t += DT
        x, y, th = self.pose()
        v = self.speed()
        self.traj.append((x, y, th))

        dist, idx = self._nearest(x, y)
        prev, di = self.idx, idx - self.idx
        self.idx = idx
        self.progress += di * 0.01
        pen = 0
        result = None

        # 차로 준수: 바퀴가 자기 차로 밖 선(중앙선·가장자리선)을 밟는 순간마다
        touch = dist + CAR_HALF_W > LANE_HALF and not self.park[idx]
        if touch and not self.touching:
            pen += self._penalize("좌·우회전 검지선 접촉" if self.arc[idx] else "차로 준수 위반 (검지선 접촉)")
        self.touching = touch

        pk, p = np.flatnonzero(self.park), self.parking   # ③ 직각주차: 차 네 모서리가 노란 검지선 밖으로, 후진 때 확인선
        if len(pk) and pk[0] - 60 <= idx <= pk[-1] + 60:
            if not all(map(on_park_area, self.corners())) and not p["touched"]:
                p["touched"] = True                       # ponytail: 모서리 4점만 검사, 안쪽 꺾인 모서리 스침은 놓침
                pen += self._penalize("직각주차 검지선 접촉")
            if self.kinds[self.leg[idx]] == "rev":
                p["rear"] = max(p["rear"], y - np.sin(th) * CAR_REAR)
            elif self.kinds[self.leg[idx]] == "fwd" and p["rear"] < PARK_ALLEY[1] + PARK_BAY[2] * 0.7:
                pen += self._penalize("직각주차 확인선 미접촉")

        if not self.started and self.progress > 0.02:      # 출발: 좌측 방향지시등
            self.started = True
            if not lamps & LEFT:
                pen += self._penalize("출발 방향지시등 미작동")

        over = v > SPEED_LIMIT and not self.in_accel(margin=30)   # 가속구간 (+감속 30cm) 밖 속도 제한
        if over and not self.speeding:
            pen += self._penalize("지정속도 초과")
        self.speeding = over

        r = self.ramp                                      # ① 경사로: 정지구간에서 3초 정지, 밀림 금지, 30초 내 통과
        if self.in_ramp_zone():
            if not r["inside"]:
                r.update(inside=True, t_in=self.t)
            r["still"] = r["still"] + DT if v < 0.01 else 0.0
            if r["still"] >= 3.0 and not r["stopped"]:
                r.update(stopped=True, stop_x=x)
            if r["stopped"] and x < r["stop_x"] - ROLLBACK:
                pen += self._penalize("경사로 뒤로 밀림")
        elif r["inside"]:
            r["inside"] = False
            if not r["stopped"]:
                pen += self._penalize("경사로 정지 안 함 (3초)")
            if self.t - r["t_in"] > 30:
                pen += self._penalize("경사로 30초 내 통과 못함")

        front = round(0.065 / 0.01)                        # 바퀴축 → 차 앞머리
        for sl, need, k in self.stoplines:                 # ④ 교차로: 신호, 방향지시등, 30초
            lit = self.signal(k)
            if sl - 60 <= idx < sl + 120 and not (v < 0.01 and not can_go(lit, need)):
                self.cross[sl] = self.cross.get(sl, 0.0) + DT     # 신호 기다리는 시간은 빼고 셈
            if prev + front < sl <= idx + front:
                if not can_go(lit, need) and "yellow" not in lit:   # 노란불은 이미 진입 중이면 통과 허용
                    result = "실격: 신호 위반"
                if need and not lamps & need:
                    pen += self._penalize("교차로 방향지시등 미작동")
            if prev < sl + 120 <= idx and self.cross.get(sl, 0.0) > 30:
                pen += self._penalize("교차로 30초 내 통과 못함")

        if self.in_accel():                                # ⑤ 가속구간: ACCEL_MIN 이상 도달
            self.acc_max, self.acc_was = max(self.acc_max, v), True
        elif self.acc_was:
            self.acc_was = False
            if self.acc_max < ACCEL_MIN:
                pen += self._penalize("가속구간 속도 미달")

        dm = self.dum                                      # 돌발: 2초 내 정지, 정지 후 3초 내 비상점멸등, 끄고 출발
        if dm["state"] == "wait" and dm["trigger"] is not None and idx >= dm["trigger"]:
            p = self.track[min(idx + round(DUMMY_AHEAD / 0.01), self.n - 1)]
            self.data.mocap_pos[self.dummy_mocap] = (*p, 0)
            dm.update(state="out", t0=self.t, t_stop=None, still=0.0, hazard=False)
        if dm["state"] == "out":
            if np.hypot(*(self.dummy_pos() - (x, y))) < 0.095:
                result = "실격: 돌발 충돌"
            if v < 0.01 and dm["t_stop"] is None:
                dm["t_stop"] = self.t
            if dm["t_stop"] is None and self.t - dm["t0"] > 2.0:
                pen += self._penalize("돌발 2초 내 정지 안 함")
            dm["hazard"] |= lamps & HAZARD == HAZARD
            if dm["t_stop"] is not None and self.t - dm["t_stop"] > 3.0 and not dm["hazard"]:
                pen += self._penalize("돌발 비상점멸등 미작동 (정지 후 3초)")
            dm["still"] = dm["still"] + DT if v < 0.01 else 0.0
            if dm["still"] >= DUMMY_HOLD or self.t - dm["t0"] > 15:
                self.data.mocap_pos[self.dummy_mocap] = (-3, -3, -1)
                dm["state"] = "cleared"
        elif dm["state"] == "cleared" and v > 0.03:
            dm["state"] = "done"
            if lamps & HAZARD == HAZARD:
                pen += self._penalize("돌발 후 비상점멸등 안 끄고 출발")

        if result is None and self.idx >= self.n - 2:      # 종료: 우측 방향지시등
            if not lamps & RIGHT:
                pen += self._penalize("종료 방향지시등 미작동")
            result = "합격"
        if dist > LANE_HALF + CAR_HALF_W:
            result = "실격: 차로 이탈"
        elif self.score < PASS_SCORE:
            result = "불합격: 점수 미달"
        elif self.t >= TIME_LIMIT and result is None:
            result = "불합격: 시간 초과"

        reward = di * 0.01 * 10 - pen                      # 1m 전진 = +10, 감점은 그대로 빼기
        if result and result != "합격":
            reward -= 50
        info = dict(score=self.score, progress=self.progress, dist=dist, t=self.t, result=result,
                    signal=[self.signal(k) for k in range(len(DIRS))], log=self.log)
        return self._obs(), reward, result is not None, info

    def plot(self, path="run.png"):
        """위에서 본 코스 + 주행 궤적."""
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]  # Windows 한글 폰트
        fig, ax = plt.subplots(figsize=(11, 9))
        ax.set_facecolor("#9fc77e")
        road = dict(color="#606066", zorder=0)
        for y in YS:
            ax.add_patch(plt.Rectangle((XS[0] - RH, y - RH), XS[2] - XS[0] + 2 * RH, 2 * RH, **road))
        for x in XS:
            ax.add_patch(plt.Rectangle((x - RH, YS[0] - RH), 2 * RH, YS[2] - YS[0] + 2 * RH, **road))
        ay0, ay1, alen = PARK_ALLEY
        bx0, bx1, bd = PARK_BAY
        ax.add_patch(plt.Rectangle((XS[0] + RH, ay0), alen, ay1 - ay0, **road))
        ax.add_patch(plt.Rectangle((bx0, ay1), bx1 - bx0, bd, **road))
        for pts, width, color in markings():
            pts = np.asarray(pts)
            ax.plot(pts[:, 0], pts[:, 1], color="gold" if color == YELLOW else "white", lw=width * 50)
        for name, (x, y) in ZONES.items():
            ax.text(x, y + 0.12, name, ha="center", va="center", fontsize=9, bbox=dict(fc="white", alpha=0.8, lw=0))
        tj = np.array(self.traj)
        ax.plot(tj[:, 0], tj[:, 1], "r", lw=1.5)
        ax.set_xlim(0, XS[2] + 0.5)
        ax.set_ylim(0, YS[2] + 0.5)
        ax.set_aspect("equal")
        ax.set_title(f"score={self.score}  progress={self.progress:.2f}/{self.length:.2f}m")
        fig.savefig(path, dpi=100)
        plt.close(fig)


class Expert:
    """규칙을 지키는 교사 제어기 (정답 위치·신호·돌발 상태를 직접 앎, 카메라 안 씀). 반환 ((좌, 우), lamps).
    코스·물리·채점 확인용이자, 뇌 학습 때 정답(교사) 신호."""

    def __init__(self, speed=0.45, look=0.2):
        self.speed, self.look = speed, look
        self.ramp_wait, self.ramp_done, self.hazard, self.park_wait = 0.0, False, False, 0.0

    def __call__(self, env, obs=None):
        x, y, th = env.pose()
        i = env.idx
        k = env.leg[i]
        # 내비게이터 상태 (경로 계획·타이머, 카메라로는 알 수 없는 것): 뇌 학습 때 명령 입력으로 씀
        self.rev, self.rotate, self.hold, self.blink, self.accel = False, 0, False, 0, False
        if i >= env.leg_end[k] - 2 and k + 1 < len(env.kinds):   # 구간 끝 → 다음 구간 (주차 완료면 2초 정지 먼저)
            if env.kinds[k] == "rev" and self.park_wait < 2.0:
                self.park_wait += DT
                self.hold = True
                return (0, 0), 0
            k += 1
        lo, hi = max(i, env.leg_start[k]), env.leg_end[k]
        tx, ty = env.track[min(lo + int(self.look / 0.01), hi)]
        rev = self.rev = env.kinds[k] == "rev"
        d = (tx - x, ty - y) if np.hypot(tx - x, ty - y) > 0.05 else env.track[hi] - env.track[hi - 5]
        err = (np.arctan2(d[1], d[0]) - th - rev * np.pi + np.pi) % (2 * np.pi) - np.pi
        if abs(err) > 0.5:                                  # 방향이 크게 다르면 제자리 회전 (주차 진입·출차)
            s = 0.4 * np.sign(err)
            self.rotate = int(np.sign(err))
            return (-s, s), 0
        turn = np.clip(err * 1.5, -0.35, 0.35)
        speed, lamps = self.speed, 0
        if k + 1 < len(env.kinds):                          # 다음 구간 앞: 끝점에서 멈출 수 있게 감속
            speed = min(speed, 0.25 + (hi - i) * 0.01)   # 0.2 아래는 기어 마찰로 멈춤
        if env.kinds[k] != "road":                          # 주차 후진·전진은 천천히
            speed = min(speed, 0.3)

        if env.progress < 1.0:                              # 출발: 좌측 방향지시등
            lamps |= LEFT
        for sl, need, k in env.stoplines:                   # 교차로: 미리 방향지시등, 내 방향 신호(직진 초록·좌회전 화살표) 아니면 정지선 앞 대기
            if sl - 120 < i < sl + 150:
                lamps |= need
            if sl - 20 <= i <= sl - 9 and not can_go(env.signal(k), need):
                speed = 0
        if env.in_ramp_zone():                              # 경사로: 정지선에서 3.5초 정지 후 힘줘서 출발
            if not self.ramp_done and x >= RAMP[0] + RAMP[1] + RAMP[2] / 2 - 0.02:
                self.ramp_wait += DT
                speed, self.hold = 0, True
                self.ramp_done = self.ramp_wait >= 3.5
            elif self.ramp_done:
                speed = 0.7
        if env.accel is not None and env.accel[0] <= i < env.accel[1] - 30:   # 가속구간
            speed, self.accel = 1.0, True
        if env.n - i < 120:                                 # 종료: 우측 방향지시등
            lamps |= RIGHT
        self.blink = lamps
        if env.dum["state"] == "out" and np.hypot(*(env.dummy_pos() - (x, y))) < 1.0:   # 돌발
            speed, lamps, self.hazard = 0, HAZARD, True
        elif self.hazard:
            self.hazard = False                             # 모형 치워지면 비상점멸등 끄고 출발
        if speed == 0:
            return (0, 0), lamps
        if rev:
            speed = -speed
        return (speed - turn, speed + turn), lamps


def run(controller, env=None):
    """controller(env, obs) → (좌, 우) 또는 ((좌, 우), lamps)."""
    env = env or Env()
    obs, done = env.reset(), False
    total = 0.0
    while not done:
        out = controller(env, obs)
        action, lamps = out if np.ndim(out[0]) else (out, 0)
        obs, r, done, info = env.step(action, lamps)
        total += r
    return env, total, info


def yellow(rgb):
    return (rgb[..., 0] > 150) & (rgb[..., 1] > 110) & (rgb[..., 2] < 90)


def show(controller=None, seed=None):
    """3D 창에서 실시간 주행. 마우스로 시점 조작, 창 닫으면 종료."""
    import mujoco.viewer
    controller = controller or Expert()
    env = Env(seed=seed)
    obs = env.reset()
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.type, v.cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, env.model.body("car").id
        v.cam.distance, v.cam.elevation = 1.0, -30
        done = False
        while v.is_running() and not done:
            t0 = time.time()
            action, lamps = controller(env, obs)
            obs, _, done, info = env.step(action, lamps)
            v.sync()
            time.sleep(max(0, DT - (time.time() - t0)))
        print(info["result"], f"score={info['score']} t={info['t']:.1f}s", info["log"])
        while v.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    import sys
    if "--show" in sys.argv:
        show()
        sys.exit()

    env = Env(seed=0)
    obs = env.reset()
    assert obs.shape == (CAM_H, CAM_W, 3) and yellow(obs).any(), "출발점 카메라에 중앙선이 안 보임"

    _, _, info = run(lambda e, o: (0.5, 0.5), Env(seed=0))     # 직진만: 첫 커브에서 이탈해야 함
    assert info["result"].startswith("실격"), info
    print("직진:", info["result"])

    ex = Expert()
    _, _, info = run(lambda e, o: (ex(e, o)[0], 0), Env(seed=0))   # 램프 안 켜면 방향지시등·비상점멸등 감점
    names = [n for _, n, _ in info["log"]]
    assert {"출발 방향지시등 미작동", "돌발 비상점멸등 미작동 (정지 후 3초)", "교차로 방향지시등 미작동",
            "종료 방향지시등 미작동"} <= set(names), info["log"]
    print("램프 안 켬:", info["result"], info["score"], names)

    for seed in (0, 2):                                    # 규칙 지키는 교사: 초록불(0)·빨간불 대기(2), 돌발 위치 달라도 100점
        t0 = time.perf_counter()
        env, total, info = run(Expert(), Env(seed=seed))
        print(f"교사 seed={seed}:", info["result"], f"score={info['score']} t={info['t']:.1f}s",
              f"돌발@{env.dum['trigger'] / 100:.1f}m", info["log"], f"(계산 {time.perf_counter() - t0:.1f}s)")
        assert info["result"] == "합격" and info["score"] == 100, info
    env.plot()

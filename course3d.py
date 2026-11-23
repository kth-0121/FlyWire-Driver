"""코스 노면 표시를 3D 조립 모형(docs/index.html)의 '코스' 탭 데이터로 내보냄 (sim.py 치수 그대로).

python course3d.py → docs/index.html 의 /*COURSE*/ … /*END*/ 사이를 새 JSON 으로 바꿈.
단위 cm, 기준 = 전체 도안 왼쪽 아래 모서리 (blueprint.py 와 같음).
"""
import json
import re
from pathlib import Path

import numpy as np

import blueprint as bp
import sim
from sim import ACCEL_X, CORNER_R, LANE_HALF, PARK_ALLEY, PARK_BAY, RAMP, RH, XS, XWALK, YS

HTML = Path(__file__).parent / "docs" / "index.html"
O = np.array((bp.X0, bp.Y0))
cm = lambda v: round(float(v) * 100, 2)


def rdp(p, eps):
    """Ramer–Douglas–Peucker: 직선 구간은 양 끝점만 남김."""
    if len(p) < 3:
        return p
    a, b = p[0], p[-1]
    d = b - a
    L = np.linalg.norm(d)
    dist = np.linalg.norm(p - a, axis=1) if L < 1e-9 else np.abs(d[0] * (p[:, 1] - a[1]) - d[1] * (p[:, 0] - a[0])) / L
    i = int(dist.argmax())
    if dist[i] <= eps:
        return np.array([a, b])
    return np.vstack([rdp(p[:i + 1], eps)[:-1], rdp(p[i:], eps)])


def corners(x0, y0, x1, y1, r, c):
    return [[cm(x - O[0]), cm(y - O[1]), cm(r), c] for x, y in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x1 - r, y1 - r), (x0 + r, y1 - r))]


def export():
    lines = [{"c": "w" if color == sim.WHITE else "y", "w": cm(width),
              "p": [[cm(x), cm(y)] for x, y in rdp(np.asarray(pts, float) - O, 0.0005)]}
             for pts, width, color in sim.markings()]
    arcs = corners(XS[0] - RH, YS[0] - RH, XS[2] + RH, YS[2] + RH, CORNER_R + RH, "w") + corners(XS[0], YS[0], XS[2], YS[2], CORNER_R, "y")
    for i in (0, 1):
        for j in (0, 1):
            arcs += corners(XS[i] + RH, YS[j] + RH, XS[i + 1] - RH, YS[j + 1] - RH, CORNER_R - RH, "w")
    c = np.array((XS[1], YS[1]))
    ay0, ay1, alen = PARK_ALLEY
    bx0, bx1, bd = PARK_BAY
    x0, up, flat, hgt = RAMP
    y_lane = YS[0] - LANE_HALF
    pts = [("교차로 중심", *c, "노란 정차금지 44×44 (노랑 1.5cm), 빗금 7cm 간격"),
           ("정지선 가운데 (서쪽으로 진입)", c[0] + RH + XWALK - .02, c[1] + LANE_HALF, "흰 4cm × 32, 세로"),
           ("정지선 가운데 (동쪽으로 진입)", c[0] - RH - XWALK + .02, c[1] - LANE_HALF, "흰 4cm × 32, 세로"),
           ("정지선 가운데 (남쪽으로 진입)", c[0] - LANE_HALF, c[1] + RH + XWALK - .02, "흰 4cm × 32, 가로"),
           ("정지선 가운데 (북쪽으로 진입)", c[0] + LANE_HALF, c[1] - RH - XWALK + .02, "흰 4cm × 32, 가로")]
    for name, a in (("서", (1, 0)), ("동", (-1, 0)), ("남", (0, 1)), ("북", (0, -1))):
        a = np.array(a)
        n = np.array((a[1], -a[0]))
        s = c - a * (RH + .02) + n * (-RH + .035)
        pts.append((f"횡단보도 {name}쪽 첫 줄 (교차로 쪽 끝)", *s, "흰 3.5cm × 15, 7cm 간격 9줄 (교차로 반대쪽으로 15cm)"))
        dc = c - a * (RH + XWALK + .35) + n * LANE_HALF
        pts.append((f"마름모 중심 ({name}쪽)", *dc, "12 × 7.2 마름모, 흰 1.2cm"))
    pts += [*[(f"신호등 기둥 {n}", c[0] + sx * (RH + .1), c[1] + sy * (RH + .1), "높이 33, 받침 10×10")
              for n, sx, sy in (("북동", 1, 1), ("북서", -1, 1), ("남서", -1, -1), ("남동", 1, -1))],
            ("직각주차 통로 왼쪽 아래", XS[0] + RH, ay0, f"통로 {cm(alen):g}×{cm(ay1 - ay0):g}, 노랑 2cm"),
            ("주차칸 왼쪽 아래", bx0, ay1, f"칸 {cm(bx1 - bx0):g}×{cm(bd):g}, 노랑 2cm"),
            ("주차 확인선 왼쪽 끝", bx0, ay1 + bd * .7, "흰 3cm × 30, 가로"),
            ("경사로 판 왼쪽 아래 모서리", x0, YS[0] - RH - .02, f"판 {cm(2 * up + flat):g}×{cm(2 * RH + .04):g}, 높이 {hgt * 100:g}"),
            ("경사로 정지선 아래 끝", x0 + up + flat / 2, YS[0] - RH, "흰 4cm × 32, 세로 (경사로 평지 위)"),
            ("출발선 아래 끝", 1.0, YS[0] - RH, "흰 4cm × 32, 세로"),
            ("직진 화살표 꼬리", 1.25, y_lane, "흰 2.5cm, 길이 30 (오른쪽으로), 머리 8×5"),
            ("도착선 왼쪽 끝", XS[0] - RH, 1.2, "흰 4cm × 32, 가로"),
            ("⑤ 가속 구간 시작 (서쪽으로 달림)", ACCEL_X[1], YS[2] + LANE_HALF, "구간 표시만 (테이프 없음)"),
            ("⑤ 가속 구간 끝", ACCEL_X[0], YS[2] + LANE_HALF, "테이프 없음"),
            ("Arduino 제어함 (남서 기둥 옆)", XS[1] - RH - .15, YS[1] - RH - .13, "8×6")]
    return {"gx": [cm(v - O[0]) for v in bp.GX], "gy": [cm(v - O[1]) for v in bp.GY], "lines": lines, "arcs": arcs,
            "pts": [[n, cm(x - O[0]), cm(y - O[1]), memo] for n, x, y, memo in pts],
            "ramp": [cm(x0 - O[0]), cm(YS[0] - RH - .02 - O[1]), cm(up), cm(flat), cm(hgt), cm(2 * RH + .04)]}


if __name__ == "__main__":
    data = json.dumps(export(), ensure_ascii=False, separators=(",", ":"))
    s = HTML.read_text(encoding="utf-8")
    s, n = re.subn(r"/\*COURSE\*/.*?/\*END\*/", lambda _: f"/*COURSE*/{data}/*END*/", s, flags=re.S)
    assert n == 1, "docs/index.html 에 /*COURSE*/…/*END*/ 표시가 없음"
    HTML.write_text(s, encoding="utf-8", newline="\n")
    print(f"코스 데이터 {len(data) / 1000:.0f} kB → {HTML}")

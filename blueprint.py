"""실물 코스 도안: 코스를 9등분(3×3)한 실측 도안 (sim.py 치수 그대로).

python blueprint.py → blueprint/도안.pdf (전체 + 치수도 + 상세도 + 좌표표 + 조각 4장), 페이지별 PNG.
치수·좌표는 전부 cm, 기준 = 전체 도안 왼쪽 아래 모서리 (조각 좌표는 그 조각 왼쪽 아래 모서리).
각 조각 눈금은 그 조각 왼쪽 아래 모서리 기준 cm. 10cm 가는 격자, 50cm 굵은 격자.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection

import sim
from sim import ACCEL_X, CORNER_R, DIRS, LANE_HALF, PARK_ALLEY, PARK_BAY, RAMP, RH, XS, XWALK, YS

plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]
OUT = Path(__file__).parent / "blueprint"
MARGIN = 0.05                                             # 바깥 테두리선 밖 여유
X0, X1 = XS[0] - RH - MARGIN, XS[2] + RH + MARGIN
Y0, Y1 = YS[0] - RH - MARGIN, YS[2] + RH + MARGIN
NX, NY = 3, 3                                             # 조각 나눔 (가로 × 세로)
GX, GY = np.linspace(X0, X1, NX + 1), np.linspace(Y0, Y1, NY + 1)
ROWS, COLS = ("아래", "가운데", "위"), ("왼쪽", "가운데", "오른쪽")
PIECES = {r * NX + c + 1: (f"{ROWS[r]} {COLS[c]}", GX[c], GX[c + 1], GY[r], GY[r + 1])
          for r in range(NY) for c in range(NX)}             # 번호: 왼쪽 아래 1 → 오른쪽 위 9
GRASS, ROAD, TAPE_Y = "#a8d08d", "#8e8e94", "#f2b800"


def tape(pts, width):
    """선을 실제 폭의 사각형 조각들로."""
    p = np.asarray(pts, float)
    d = np.diff(p, axis=0)
    L = np.linalg.norm(d, axis=1)
    keep = L > 1e-6
    p0, d, L = p[:-1][keep], d[keep], L[keep]
    u = d / L[:, None]
    n = np.c_[-u[:, 1], u[:, 0]] * width / 2
    a, b = p0 - u * width / 4, p0 + d + u * width / 4     # 꺾이는 곳 이음새 겹침
    return np.stack([a - n, b - n, b + n, a + n], axis=1)


def draw_course(ax):
    ax.set_facecolor("#efe8da")                         # 바닥 (흰 테두리선이 보이게)
    ax.add_patch(plt.Polygon(sim.rrect(XS[0] - RH, YS[0] - RH, XS[2] + RH, YS[2] + RH, CORNER_R + RH), color=ROAD, lw=0))
    po = sim.park_outline()
    for i in (0, 1):
        for j in (0, 1):
            ax.add_patch(plt.Polygon(sim.rrect(XS[i] + RH, YS[j] + RH, XS[i + 1] - RH, YS[j + 1] - RH, CORNER_R - RH,
                                               po if (i, j) == (0, 1) else None), color=GRASS, lw=0))
    x0, up, flat, hgt = RAMP                              # 경사로 (따로 만들어 올려놓는 판)
    ax.add_patch(plt.Rectangle((x0, YS[0] - RH - 0.02), 2 * up + flat, 2 * RH + 0.04, fc="none", ec="#7a4b00",
                               hatch="//", lw=1.2, zorder=3))
    for xx in (x0 + up, x0 + up + flat):
        ax.plot([xx, xx], [YS[0] - RH - 0.02, YS[0] + RH + 0.02], color="#7a4b00", lw=1, ls="--", zorder=3)
    ax.text(x0 + up + flat / 2, YS[0] + RH + 0.06, f"경사로 판 {2 * up + flat:.2f}×{2 * RH + 0.04:.2f}m, 높이 {hgt * 100:.2f}cm\n"
            f"(오르막 {up * 100:.0f} / 평지 {flat * 100:.0f} / 내리막 {up * 100:.0f}cm)", ha="center", fontsize=7, zorder=5)

    by_color = {}
    for pts, width, color in sim.markings():
        by_color.setdefault(color, []).append(tape(pts, width))
    for color, polys in by_color.items():
        ax.add_collection(PolyCollection(np.vstack(polys), fc="white" if color == sim.WHITE else TAPE_Y,
                                         ec="none", zorder=4))

    c = np.array((XS[1], YS[1]))                          # 신호등 기둥 + 팔, 도로용 Arduino 제어함
    for d in np.array(DIRS):
        n = np.array((d[1], -d[0]))
        pole, h = c + d * (RH + 0.1) + n * (RH + 0.1), c + d * (RH + 0.1) + n * LANE_HALF
        ax.plot(*np.c_[pole, h], color="k", lw=1.5, zorder=6)
        ax.add_patch(plt.Circle(pole, 0.015, color="k", zorder=6))
        ax.text(*(pole + (pole - c) / np.linalg.norm(pole - c) * 0.07), "신호등\n기둥", ha="center", va="center", fontsize=6, zorder=6)
    sx, sy = XS[1] - RH - 0.1, YS[1] - RH - 0.1
    ax.add_patch(plt.Rectangle((sx - .09, sy - .06), .08, .06, fc="#888", ec="k", lw=.5, zorder=6))
    ax.text(sx - .05, sy - .09, "Arduino 제어함", ha="center", fontsize=6, zorder=6)
    for name, (x, y) in sim.ZONES.items():
        ax.text(x, y + 0.12, name, ha="center", va="center", fontsize=9, zorder=7,
                bbox=dict(fc="white", alpha=0.85, lw=0))


def grid(ax, x0, x1, y0, y1, step_label=0.5):
    for t in ax.texts:
        t.set_clip_on(True)                               # 다른 조각에 걸친 글자는 잘라냄
    for v in np.arange(0, x1 - x0 + 1e-9, 0.1):
        ax.axvline(x0 + v, color="#3a6ea5", lw=0.6 if round(v * 100) % 50 == 0 else 0.2, alpha=0.6, zorder=8)
    for v in np.arange(0, y1 - y0 + 1e-9, 0.1):
        ax.axhline(y0 + v, color="#3a6ea5", lw=0.6 if round(v * 100) % 50 == 0 else 0.2, alpha=0.6, zorder=8)
    xt = np.arange(0, x1 - x0 + 1e-9, step_label)
    yt = np.arange(0, y1 - y0 + 1e-9, step_label)
    ax.set_xticks(x0 + xt, [f"{v * 100:.0f}" for v in xt])
    ax.set_yticks(y0 + yt, [f"{v * 100:.0f}" for v in yt])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")


LEGEND = ("테이프: 흰 2cm 가장자리·테두리선 / 노랑 2cm 중앙선·주차 검지선 / 흰 3.5cm 횡단보도 / 흰 4cm 정지선\n"
          f"차로 폭 {sim.LANE_W * 100:.0f}cm, 도로 폭 {2 * RH * 100:.0f}cm, 모퉁이 반경(중앙선) {CORNER_R * 100:.0f}cm, "
          f"주차 통로 {(PARK_ALLEY[1] - PARK_ALLEY[0]) * 100:.0f}×{PARK_ALLEY[2] * 100:.0f}cm, "
          f"주차칸 {(PARK_BAY[1] - PARK_BAY[0]) * 100:.0f}×{PARK_BAY[2] * 100:.0f}cm")


def cm(v):
    """m → cm 글자 (소수점 필요할 때만)."""
    return f"{round(v * 100, 1):g}"


def dim(ax, a, b, at, text=None, vertical=False, fs=8):
    """치수선: 가로면 x=a~b (높이 at), 세로면 y=a~b (x=at). 글자 기본값 = 길이 cm."""
    p, q = ((at, a), (at, b)) if vertical else ((a, at), (b, at))
    ax.annotate("", q, p, arrowprops=dict(arrowstyle="<->", lw=.8, color="#c00000", shrinkA=0, shrinkB=0), zorder=12)
    for e in (p, q):                                      # 끝 눈금
        tick = ([e[0] - .015, e[0] + .015], [e[1], e[1]]) if vertical else ([e[0], e[0]], [e[1] - .015, e[1] + .015])
        ax.plot(*tick, color="#c00000", lw=.8, zorder=12)
    ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, text or cm(b - a), rotation=90 if vertical else 0, ha="center",
            va="center", fontsize=fs, color="#c00000", zorder=13, bbox=dict(fc="white", ec="none", pad=.5))


def chain(ax, ps, at, vertical=False, fs=8):
    for a, b in zip(ps, ps[1:]):
        dim(ax, a, b, at, vertical=vertical, fs=fs)


def note(ax, x, y, text, fs=7):
    ax.text(x, y, text, fontsize=fs, ha="center", va="center", color="#0b3d91", zorder=13,
            bbox=dict(fc="white", ec="#0b3d91", lw=.5, pad=1.5, alpha=.9))


def page_dims(pdf):
    """전체 치수도: 도로·블록 사슬 치수, 도로 중심선 위치, 모퉁이 반경, 차로 폭."""
    fig, ax = plt.subplots(figsize=(16.5, 11.7))
    draw_course(ax)
    xs = [X0, XS[0] - RH, XS[0] + RH, XS[1] - RH, XS[1] + RH, XS[2] - RH, XS[2] + RH, X1]
    ys = [Y0, YS[0] - RH, YS[0] + RH, YS[1] - RH, YS[1] + RH, YS[2] - RH, YS[2] + RH, Y1]
    chain(ax, xs, Y0 - .12)
    chain(ax, ys, X0 - .12, vertical=True)
    dim(ax, X0, X1, Y0 - .3, f"전체 가로 {cm(X1 - X0)}", fs=10)
    dim(ax, Y0, Y1, X0 - .3, f"전체 세로 {cm(Y1 - Y0)}", vertical=True, fs=10)
    for i, x in enumerate(XS):                            # 도로 중심선 위치 (도안 끝 기준)
        dim(ax, X0, x, Y1 + .06 + .09 * i, f"세로 도로 중심 {cm(x - X0)}")
    for i, y in enumerate(YS):
        dim(ax, Y0, y, X1 + .06 + .09 * i, f"가로 도로 중심 {cm(y - Y0)}", vertical=True)
    cx, cy = XS[2] - CORNER_R, YS[0] + CORNER_R           # 모퉁이 반경 (오른쪽 아래 모퉁이)
    for r, label, ang in ((CORNER_R + RH, "바깥 테두리", -75), (CORNER_R, "중앙선", -50), (CORNER_R - RH, "블록 모서리", -25)):
        a = np.radians(ang)
        ax.annotate("", (cx + r * np.cos(a), cy + r * np.sin(a)), (cx, cy),
                    arrowprops=dict(arrowstyle="->", lw=.8, color="#0b3d91"), zorder=12)
        note(ax, cx + (r + .08) * np.cos(a) + .06, cy + (r + .05) * np.sin(a), f"{label} R{cm(r)}")
    ax.plot(cx, cy, "+", color="#0b3d91", ms=8, zorder=12)
    note(ax, cx - .05, cy + .07, f"원 중심 ({cm(cx - X0)}, {cm(cy - Y0)})")
    dim(ax, YS[0] - RH, YS[0], 2.0, f"차로 {cm(sim.LANE_W)}", vertical=True, fs=7)
    dim(ax, YS[0], YS[0] + RH, 2.0, f"차로 {cm(sim.LANE_W)}", vertical=True, fs=7)
    for i in (0, 1):
        for j in (0, 1):
            w, h = XS[i + 1] - XS[i] - 2 * RH, YS[j + 1] - YS[j] - 2 * RH
            note(ax, (XS[i] + XS[i + 1]) / 2, (YS[j] + YS[j + 1]) / 2 - .25,
                 f"잔디 블록 {cm(w)} × {cm(h)}\n모서리 R{cm(CORNER_R - RH)}", fs=8)
    a0, a1 = ACCEL_X                                      # ⑤ 가속 구간 (상단 도로, 서쪽으로 달림)
    dim(ax, a0, a1, YS[2] + RH - .05, f"⑤ 가속 구간 {cm(a1 - a0)} (x {cm(a0 - X0)}~{cm(a1 - X0)})", fs=7)
    ax.set(xlim=(X0 - .42, X1 + .36), ylim=(Y0 - .4, Y1 + .36), aspect="equal", xticks=[], yticks=[])
    ax.set_title("전체 치수도 (cm) — 기준: 도안 왼쪽 아래 모서리. 빨간 치수 = 길이, 파란 글자 = 반경·크기", fontsize=13)
    fig.text(0.5, 0.02, LEGEND, ha="center", fontsize=10)
    pdf.savefig(fig)
    fig.savefig(OUT / "5_치수도.png", dpi=80)
    plt.close(fig)


def page_detail(pdf):
    """상세도: ④ 교차로, ③ 직각주차, ① 경사로."""
    fig, m = plt.subplot_mosaic([["x", "p"], ["x", "r"]], figsize=(16.5, 11.7), width_ratios=[1.3, 1])
    axs = [m["x"], m["p"], m["r"]]
    c = np.array((XS[1], YS[1]))

    ax = axs[0]                                           # ④ 교차로 (서쪽으로 들어오는 차로 기준 치수)
    draw_course(ax)
    e = RH + XWALK + .5
    s_stop = RH + XWALK - .02                             # 중심 → 정지선 중심
    dim(ax, c[0] + RH, c[0] + s_stop, c[1] + RH + .05, fs=7)
    note(ax, c[0] + s_stop + .05, c[1] + RH + .15, f"정지선: 흰 4cm × {cm(RH)}\n교차로 끝 → 정지선 중심 {cm(s_stop - RH)}")
    dim(ax, c[0] + RH + .02, c[0] + RH + .17, c[1] - RH - .05, "15", fs=7)
    note(ax, c[0] + RH + .12, c[1] - RH - .15, "횡단보도: 흰 3.5cm × 15\n7cm 간격, 교차로 끝 2cm 띄움")
    dim(ax, c[1] - RH, c[1] + RH, c[0] + RH + .32, f"도로 {cm(2 * RH)}", vertical=True, fs=7)
    dim(ax, c[0] - .22, c[0] + .22, c[1] - .3, "44", fs=7)
    note(ax, c[0], c[1] + .3, "정차금지: 노랑 44×44, 빗금 1.5cm 7cm 간격")
    dc = c + np.array((RH + XWALK + .35, 0))
    note(ax, dc[0], dc[1] + LANE_HALF + .12, f"마름모 12×7.2 (흰 1.2cm)\n중심: 교차로 끝에서 {cm(XWALK + .35)}")
    pole = c + np.array((RH + .1, RH + .1))
    note(ax, pole[0] + .12, pole[1] + .14, "신호등 기둥: 교차로 모서리에서\n대각으로 가로 10·세로 10\n높이 33, 팔 → 차로 중심 위\n등 높이 30")
    note(ax, c[0], c[1] - e + .1, f"교차로 중심 ({cm(c[0] - X0)}, {cm(c[1] - Y0)})", fs=8)
    ax.set(xlim=(c[0] - e, c[0] + e), ylim=(c[1] - e, c[1] + e), aspect="equal", title="④ 교차로 (4방향 같은 모양)")

    ax = axs[1]                                           # ③ 직각주차
    draw_course(ax)
    ay0, ay1, alen = PARK_ALLEY
    bx0, bx1, bd = PARK_BAY
    xa = XS[0] + RH
    dim(ax, xa, xa + alen, ay0 - .06, f"통로 길이 {cm(alen)}", fs=7)
    dim(ax, ay0, ay1, xa + alen + .06, f"통로 폭 {cm(ay1 - ay0)}", vertical=True, fs=7)
    dim(ax, bx0, bx1, ay1 + bd + .06, f"칸 폭 {cm(bx1 - bx0)}", fs=7)
    dim(ax, ay1, ay1 + bd, bx1 + .06, f"깊이 {cm(bd)}", vertical=True, fs=7)
    dim(ax, xa, bx0, ay1 + .05, fs=7)
    note(ax, (bx0 + bx1) / 2, ay1 + bd + .17, f"확인선: 흰 3cm, 칸 입구에서 {cm(bd * .7)}")
    note(ax, xa + alen / 2, ay0 + .12, "테두리 전부 노랑 2cm (검지선)")
    note(ax, xa + .3, ay0 - .2, f"통로 왼쪽 아래 모서리 ({cm(xa - X0)}, {cm(ay0 - Y0)})")
    ax.set(xlim=(XS[0] - RH - .05, xa + alen + .25), ylim=(ay0 - .35, ay1 + bd + .3), aspect="equal", title="③ 직각주차")

    ax = axs[2]                                           # ① 경사로
    draw_course(ax)
    x0, up, flat, hgt = RAMP
    yb = YS[0] - RH - .02
    chain(ax, [x0, x0 + up, x0 + up + flat, x0 + 2 * up + flat], yb - .06, fs=7)
    dim(ax, yb, YS[0] + RH + .02, x0 + 2 * up + flat + .06, f"판 폭 {cm(2 * RH + .04)}", vertical=True, fs=7)
    xr = x0 + up + flat / 2
    note(ax, xr, YS[0] + RH + .25, f"경사로 판 시작 x {cm(x0 - X0)}\n높이 {hgt * 100:.2f}cm (경사 15%)\n"
         f"정지선 x {cm(xr - X0)} (평지 가운데)")
    note(ax, xr, yb - .2, "오르막 / 평지 / 내리막")
    ax.set(xlim=(x0 - .25, x0 + 2 * up + flat + .2), ylim=(Y0 - .1, YS[0] + RH + .45), aspect="equal", title="① 경사로")
    for ax in axs:                                        # 확대 창 밖 글자 잘라내기, 눈금은 도안 기준 cm
        for t in ax.texts:
            t.set_clip_on(True)
        ax.xaxis.set_major_formatter(lambda v, _: cm(v - X0))
        ax.yaxis.set_major_formatter(lambda v, _: cm(v - Y0))
        ax.tick_params(labelsize=7)
    fig.text(.6, .04, f"출발선: x {cm(1.0 - X0)}, 아래 도로 바깥 차로 (흰 4cm × {cm(RH)})\n"
             f"도착선: y {cm(1.2 - Y0)}, 왼쪽 도로 바깥 차로 (흰 4cm × {cm(RH)})", fontsize=10)
    fig.suptitle("상세도 (cm) — 좌표 기준: 도안 왼쪽 아래 모서리", fontsize=13)
    pdf.savefig(fig)
    fig.savefig(OUT / "6_상세도.png", dpi=80)
    plt.close(fig)


def piece_of(x, y):
    """점이 속한 조각 번호와 조각 기준 좌표."""
    for k, (name, a0, a1, b0, b1) in PIECES.items():
        if a0 <= x <= a1 and b0 <= y <= b1:
            return f"{k}번 ({cm(x - a0)}, {cm(y - b0)})"


def page_table(pdf):
    """좌표표: 주요 점마다 전체 좌표와 조각 번호·조각 기준 좌표."""
    c = (XS[1], YS[1])
    ay0, ay1, alen = PARK_ALLEY
    bx0, bx1, bd = PARK_BAY
    x0, up, flat, hgt = RAMP
    blocks = [(i + 2 * j + 1, i, j) for j in (0, 1) for i in (0, 1)]
    pts = [("도로 바깥 테두리 왼쪽 아래 (모퉁이 R82의 외접 모서리)", XS[0] - RH, YS[0] - RH, "흰 2cm"),
           ("도로 바깥 테두리 오른쪽 위", XS[2] + RH, YS[2] + RH, "흰 2cm"),
           *[(f"잔디 블록 {k} 왼쪽 아래", XS[i] + RH, YS[j] + RH, "흰 2cm 테두리, 모서리 R18")
             for k, i, j in blocks],
           *[(f"잔디 블록 {k} 오른쪽 위", XS[i + 1] - RH, YS[j + 1] - RH, "") for k, i, j in blocks],
           ("교차로 중심", *c, "노란 정차금지 44×44"),
           ("정지선 가운데 (서쪽으로 진입)", c[0] + RH + XWALK - .02, c[1] + LANE_HALF, "흰 4cm × 32, 세로"),
           ("정지선 가운데 (동쪽으로 진입)", c[0] - RH - XWALK + .02, c[1] - LANE_HALF, "흰 4cm × 32, 세로"),
           ("정지선 가운데 (남쪽으로 진입)", c[0] - LANE_HALF, c[1] + RH + XWALK - .02, "흰 4cm × 32, 가로"),
           ("정지선 가운데 (북쪽으로 진입)", c[0] + LANE_HALF, c[1] - RH - XWALK + .02, "흰 4cm × 32, 가로"),
           *[(f"신호등 기둥 {n}", c[0] + sx * (RH + .1), c[1] + sy * (RH + .1), "높이 33")
             for n, sx, sy in (("북동", 1, 1), ("북서", -1, 1), ("남서", -1, -1), ("남동", 1, -1))],
           ("직각주차 통로 왼쪽 아래", XS[0] + RH, ay0, f"통로 {cm(alen)}×{cm(ay1 - ay0)}, 노랑 2cm"),
           ("주차칸 왼쪽 아래", bx0, ay1, f"칸 {cm(bx1 - bx0)}×{cm(bd)}, 노랑 2cm"),
           ("주차 확인선 왼쪽 끝", bx0, ay1 + bd * .7, "흰 3cm × 30"),
           ("경사로 판 왼쪽 아래 모서리", x0, YS[0] - RH - .02, f"판 {cm(2 * up + flat)}×{cm(2 * RH + .04)}"),
           ("경사로 정지선 아래 끝", x0 + up + flat / 2, YS[0] - RH, "흰 4cm × 32"),
           ("출발선 아래 끝", 1.0, YS[0] - RH, "흰 4cm × 32"),
           ("도착선 왼쪽 끝", XS[0] - RH, 1.2, "흰 4cm × 32"),
           ("⑤ 가속 구간 시작 (서쪽으로 달림)", ACCEL_X[1], YS[2] + LANE_HALF, "구간 표시만 (테이프 없음)"),
           ("⑤ 가속 구간 끝", ACCEL_X[0], YS[2] + LANE_HALF, ""),
           ("Arduino 제어함 (남서 기둥 옆)", XS[1] - RH - .15, YS[1] - RH - .13, "8×6")]
    rows = [[name, f"({cm(x - X0)}, {cm(y - Y0)})", piece_of(x, y), memo]
            for name, x, y, memo in pts]
    fig, ax = plt.subplots(figsize=(16.5, 11.7))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=["항목", "전체 좌표 (x, y)", "조각 (조각 기준 x, y)", "테이프·메모"],
                 colWidths=[.36, .15, .19, .3], loc="upper center", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.45)
    ax.set_title("좌표표 (cm) — 전체 좌표: 도안 왼쪽 아래 모서리 기준 / 조각 좌표: 그 조각 왼쪽 아래 모서리 기준", fontsize=13)
    pdf.savefig(fig)
    fig.savefig(OUT / "7_좌표표.png", dpi=80)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    with PdfPages(OUT / "도안.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(16.5, 11.7))      # A3 가로
        draw_course(ax)
        grid(ax, X0, X1, Y0, Y1)
        for k, (name, x0, x1, y0, y1) in PIECES.items():
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec="red", lw=2.5, ls="--", zorder=9))
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{k}", color="red", fontsize=60, alpha=0.35, ha="center", va="center", zorder=9)
        ax.set_title(f"전체 코스 {(X1 - X0) * 100:.0f} × {(Y1 - Y0) * 100:.0f} cm  →  조각 {len(PIECES)}개 각 "
                     f"{(GX[1] - X0) * 100:.1f} × {(GY[1] - Y0) * 100:.1f} cm (빨간 점선)   눈금: 전체 왼쪽 아래 기준 cm", fontsize=13)
        fig.text(0.5, 0.02, LEGEND, ha="center", fontsize=10)
        pdf.savefig(fig)
        fig.savefig(OUT / "0_전체.png", dpi=80)
        plt.close(fig)
        page_dims(pdf)
        page_detail(pdf)
        page_table(pdf)

        for k, (name, x0, x1, y0, y1) in PIECES.items():
            fig, ax = plt.subplots(figsize=(16.5, 11.7))
            draw_course(ax)
            grid(ax, x0, x1, y0, y1, 0.1)
            ax.tick_params(labelsize=6)
            joins = []                                    # 맞붙는 조각 표시
            for kk, (_, a0, a1, b0, b1) in PIECES.items():
                if kk == k:
                    continue
                if abs(a0 - x1) < 1e-9 and b0 == y0:
                    joins.append(("right", kk))
                elif abs(a1 - x0) < 1e-9 and b0 == y0:
                    joins.append(("left", kk))
                elif abs(b0 - y1) < 1e-9 and a0 == x0:
                    joins.append(("top", kk))
                elif abs(b1 - y0) < 1e-9 and a0 == x0:
                    joins.append(("bottom", kk))
            for side, kk in joins:
                e = 0.04
                pos = {"right": (x1 - e, (y0 + y1) / 2, 90), "left": (x0 + e, (y0 + y1) / 2, 90),
                       "top": ((x0 + x1) / 2, y1 - e, 0), "bottom": ((x0 + x1) / 2, y0 + e, 0)}[side]
                ax.text(pos[0], pos[1], f"  {kk}번 조각과 이어짐  ", rotation=pos[2], color="white", fontsize=11,
                        ha="center", va="center", zorder=10, bbox=dict(fc="red", lw=0))
            ax.set_title(f"조각 {k} / {len(PIECES)} ({name})  {(x1 - x0) * 100:.1f} × {(y1 - y0) * 100:.1f} cm   "
                         f"눈금: 이 조각 왼쪽 아래 모서리 기준 cm (10cm 격자, 50cm 굵은 선)", fontsize=13)
            fig.text(0.5, 0.02, LEGEND, ha="center", fontsize=10)
            pdf.savefig(fig)
            fig.savefig(OUT / f"조각{k}_{name}.png", dpi=80)
            plt.close(fig)
    print(OUT / "도안.pdf")


if __name__ == "__main__":
    main()

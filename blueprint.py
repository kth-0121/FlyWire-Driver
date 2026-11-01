"""실물 코스 도안: 코스를 4등분한 실측 도안 (sim.py 치수 그대로).

python blueprint.py → blueprint/도안.pdf (전체 + 조각 4장), 조각별 PNG.
각 조각 눈금은 그 조각 왼쪽 아래 모서리 기준 cm. 10cm 가는 격자, 50cm 굵은 격자.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection

import sim
from sim import CORNER_R, DIRS, LANE_HALF, PARK_ALLEY, PARK_BAY, RAMP, RH, XS, YS

plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]
OUT = Path(__file__).parent / "blueprint"
MARGIN = 0.05                                             # 바깥 테두리선 밖 여유
X0, X1 = XS[0] - RH - MARGIN, XS[2] + RH + MARGIN
Y0, Y1 = YS[0] - RH - MARGIN, YS[2] + RH + MARGIN
XM, YM = (X0 + X1) / 2, (Y0 + Y1) / 2                     # 4등분 선
PIECES = {1: ("왼쪽 아래", X0, XM, Y0, YM), 2: ("오른쪽 아래", XM, X1, Y0, YM),
          3: ("왼쪽 위", X0, XM, YM, Y1), 4: ("오른쪽 위", XM, X1, YM, Y1)}
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


def main():
    OUT.mkdir(exist_ok=True)
    with PdfPages(OUT / "도안.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(16.5, 11.7))      # A3 가로
        draw_course(ax)
        grid(ax, X0, X1, Y0, Y1)
        for k, (name, x0, x1, y0, y1) in PIECES.items():
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec="red", lw=2.5, ls="--", zorder=9))
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{k}", color="red", fontsize=60, alpha=0.35, ha="center", va="center", zorder=9)
        ax.set_title(f"전체 코스 {(X1 - X0) * 100:.0f} × {(Y1 - Y0) * 100:.0f} cm  →  조각 4개 각 "
                     f"{(XM - X0) * 100:.0f} × {(YM - Y0) * 100:.0f} cm (빨간 점선)   눈금: 전체 왼쪽 아래 기준 cm", fontsize=13)
        fig.text(0.5, 0.02, LEGEND, ha="center", fontsize=10)
        pdf.savefig(fig)
        fig.savefig(OUT / "0_전체.png", dpi=80)
        plt.close(fig)

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
            ax.set_title(f"조각 {k} / 4 ({name})  {(x1 - x0) * 100:.0f} × {(y1 - y0) * 100:.0f} cm   "
                         f"눈금: 이 조각 왼쪽 아래 모서리 기준 cm (10cm 격자, 50cm 굵은 선)", fontsize=13)
            fig.text(0.5, 0.02, LEGEND, ha="center", fontsize=10)
            pdf.savefig(fig)
            fig.savefig(OUT / f"{k}_{name}.png", dpi=80)
            plt.close(fig)
    print(OUT / "도안.pdf")


if __name__ == "__main__":
    main()

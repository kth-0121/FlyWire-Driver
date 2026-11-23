"""코스 9조각을 A4 가로 1:1 로 나눈 출력용 PDF (노면 표시가 있는 종이만).

python course_print.py → blueprint/코스_A4/조각1.pdf … 조각9.pdf (+ docs/course_a4/piece1.pdf … 사이트 링크용 사본)
조각마다 첫 쪽 = 종이 배치도 (어느 칸을 출력했는지), 그 뒤 = 종이 한 장씩 (열 A~G, 행 1~8, 왼쪽 아래가 A1).
종이 한 장이 덮는 넓이 277 × 190 mm (여백 10 mm). 인쇄: 실제 크기 100%, 가로.
"""
import shutil
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

import blueprint as bp
import course3d

OUT = Path(__file__).parent / "blueprint" / "코스_A4"
SITE = Path(__file__).parent / "docs" / "course_a4"
A4, MG, TW, TH = (297, 210), 10, 277, 190
FILL = {"w": "#bdbdbd", "y": "#f2b800"}
ROWS, COLS = ("아래", "가운데", "위"), ("왼쪽", "가운데", "오른쪽")


def load():
    d = course3d.export()
    polys, colors = [], []
    for ln in d["lines"]:
        q = bp.tape(np.array(ln["p"]) / 100, ln["w"] / 100) * 1000          # 테이프 = 사각형 조각들 (mm)
        polys.append(q)
        colors += [FILL[ln["c"]]] * len(q)
    polys = np.vstack(polys)
    rx, ry, up, flat, _, rw = (v * 10 for v in d["ramp"])
    ramp = [(rx, ry, rx + 2 * up + flat, ry), (rx, ry + rw, rx + 2 * up + flat, ry + rw), (rx, ry, rx, ry + rw),
            (rx + up, ry, rx + up, ry + rw), (rx + up + flat, ry, rx + up + flat, ry + rw), (rx + 2 * up + flat, ry, rx + 2 * up + flat, ry + rw)]
    pts = [(n, x * 10, y * 10) for n, x, y, _ in d["pts"]]
    return d, polys, np.array(colors), polys.min(axis=1), polys.max(axis=1), ramp, pts


def content(ax, polys, colors, lo, hi, ramp, pts, x0, y0, x1, y1, ox, oy, fs=5.5, names=True):
    """창 (x0~x1, y0~y1, 전체 좌표 mm) 안의 노면 표시. 눈금 글자는 조각 기준 cm (ox, oy = 조각 왼쪽 아래)."""
    m = (hi[:, 0] > x0) & (lo[:, 0] < x1) & (hi[:, 1] > y0) & (lo[:, 1] < y1)
    ax.add_collection(PolyCollection(polys[m], fc=colors[m], ec="none", zorder=2))
    for a, b, c, e in ramp:
        ax.plot([a, c], [b, e], color="#7a4b00", lw=1, ls="--", zorder=3)
    for n, x, y in pts:
        if x0 <= x <= x1 and y0 <= y <= y1:
            ax.plot(x, y, "+", color="#d62728", ms=8, mew=1, zorder=4)
            if names:
                ax.text(x + 3, y + 3, f"{n} ({(x - ox) / 10:g}, {(y - oy) / 10:g})", fontsize=fs, color="#d62728", zorder=4)
    ax.set(xlim=(x0, x1), ylim=(y0, y1))
    ax.set_xticks([])
    ax.set_yticks([])
    return m.any() or any(x0 <= x <= x1 and y0 <= y <= y1 for _, x, y in pts) or any(
        min(a, c) <= x1 and max(a, c) >= x0 and min(b, e) <= y1 and max(b, e) >= y0 for a, b, c, e in ramp)


def tile_page(k, name, tid, around, x0, y0, w, h, ox, oy, data):
    fig = plt.figure(figsize=(A4[0] / 25.4, A4[1] / 25.4))
    ax = fig.add_axes([MG / A4[0], MG / A4[1], w / A4[0], h / A4[1]])
    content(ax, *data, x0, y0, x0 + w, y0 + h, ox, oy)
    for v in np.arange(ceil((x0 - ox) / 100 - 1e-9) * 100, x0 - ox + w + 1e-6, 100):      # 조각 기준 10cm 격자
        ax.axvline(ox + v, color="#3a6ea5", lw=.3, alpha=.6, zorder=1)
        ax.text(ox + v + 1.5, y0 + 2, f"x {v / 10:g}", fontsize=5.5, color="#3a6ea5", zorder=5)
    for v in np.arange(ceil((y0 - oy) / 100 - 1e-9) * 100, y0 - oy + h + 1e-6, 100):
        ax.axhline(oy + v, color="#3a6ea5", lw=.3, alpha=.6, zorder=1)
        ax.text(x0 + 2, oy + v + 1.5, f"y {v / 10:g}", fontsize=5.5, color="#3a6ea5", zorder=5)
    for s in ax.spines.values():
        s.set(lw=.8, color="k")
    pg = fig.add_axes([0, 0, 1, 1])
    pg.set(xlim=(0, A4[0]), ylim=(0, A4[1]))
    pg.axis("off")
    pg.patch.set_alpha(0)
    pg.text(MG, A4[1] - 5, f"조각 {k} ({name}) · 종이 {tid} · 이 종이 왼쪽 아래 = 조각 기준 ({(x0 - ox) / 10:g}, {(y0 - oy) / 10:g}) cm · "
            f"회색 = 흰 테이프, 노랑 = 노랑 테이프, 굵은 테두리끼리 맞대어 붙임", fontsize=6.5, va="center")
    for (side, label), (x, y, rot) in zip(around, ((MG / 2, MG + h / 2, 90), (MG + w + MG / 2, MG + h / 2, 90),
                                                   (MG + w / 2 + 60, MG / 2, 0), (MG + w / 2, MG + h + 2.2, 0))):
        if not (side == "위" and label is None):
            pg.text(x, y, f"{side}: {label}", fontsize=6, ha="center", va="center", rotation=rot, color="#555")
    pg.plot([MG, MG + 50], [4, 4], "k", lw=1.5)
    for x in (MG, MG + 50):
        pg.plot([x, x], [2.5, 5.5], "k", lw=.8)
    pg.text(MG + 52, 4, "← 50 mm (인쇄: 실제 크기 100%)", fontsize=5.5, va="center")
    return fig


def index_page(k, name, pw, ph, ox, oy, grid, data, n_print):
    fig = plt.figure(figsize=(A4[0] / 25.4, A4[1] / 25.4))
    ax = fig.add_axes([.04, .06, .7, .84])
    ax.set_facecolor("#333")
    content(ax, *data, ox, oy, ox + pw, oy + ph, ox, oy, names=False)
    ax.set_aspect("equal")
    for (c, r), (x0, y0, w, h, on) in grid.items():
        ax.add_patch(Rectangle((x0, y0), w, h, fc="#2ecc71" if on else "none", alpha=.25 if on else 1, ec="none", zorder=5))
        ax.add_patch(Rectangle((x0, y0), w, h, fc="none", ec="white", lw=.6, zorder=6))
        ax.text(x0 + w / 2, y0 + h / 2, f"{'ABCDEFGH'[c]}{r + 1}", color="white" if on else "#888", fontsize=9, ha="center", va="center",
                weight="bold" if on else "normal", zorder=7)
    fig.text(.04, .94, f"코스 조각 {k} ({name}) {pw / 10:g} × {ph / 10:g} cm — A4 종이 배치도", fontsize=13, weight="bold")
    fig.text(.76, .88, f"초록 칸 = 출력한 종이 {n_print}장\n회색 글자 칸 = 표시가 없어 생략\n\n"
             "1. 실제 크기 100%, 가로로 인쇄\n   (각 장 아래 50mm 눈금 확인)\n2. 굵은 테두리 밖 여백을 잘라 냄\n"
             "3. 조각 왼쪽 아래 모서리에서\n   각 장 맨 위에 적힌 좌표 자리에\n   마스킹테이프로 살짝 붙임\n   (하늘색 선 = 조각 기준 10cm)\n"
             "4. 회색·노랑 띠의 가장자리를\n   송곳으로 찍거나 칼로 살짝\n   그어 판에 자국을 냄\n5. 종이를 떼고 자국을 따라\n   테이프를 붙임\n\n"
             "※ 종이는 반드시 뗄 것 —\n   흰 종이가 남으면 차 카메라가\n   차선으로 봄 (바닥은 검정)\n\n칸 이름: 열 A→ 오른쪽, 행 1→ 위\n(A1 = 왼쪽 아래)", fontsize=8.5, va="top", linespacing=1.5)
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SITE.mkdir(parents=True, exist_ok=True)
    d, *data = load()
    gx, gy, total = [v * 10 for v in d["gx"]], [v * 10 for v in d["gy"]], 0
    for r in range(3):
        for c in range(3):
            k, name, ox, oy, pw, ph = r * 3 + c + 1, f"{ROWS[r]} {COLS[c]}", gx[c], gy[r], gx[c + 1] - gx[c], gy[r + 1] - gy[r]
            nc, nr, grid = ceil(pw / TW - 1e-6), ceil(ph / TH - 1e-6), {}
            probe = plt.figure().add_subplot()
            for j in range(nr):
                for i in range(nc):
                    x0, y0 = ox + i * TW, oy + j * TH
                    w, h = min(TW, ox + pw - x0), min(TH, oy + ph - y0)
                    grid[i, j] = (x0, y0, w, h, content(probe, *data, x0, y0, x0 + w, y0 + h, ox, oy))
            plt.close(probe.figure)
            tid = lambda i, j: (f"{'ABCDEFGH'[i]}{j + 1}" + ("" if grid[i, j][4] else " (빈 칸, 생략)")) if (i, j) in grid else "조각 끝"
            n = sum(g[4] for g in grid.values())
            with PdfPages(OUT / f"조각{k}.pdf") as pdf:
                pdf.savefig(fig := index_page(k, name, pw, ph, ox, oy, grid, data, n))
                plt.close(fig)
                for fig in (
                        tile_page(k, name, tid(i, j), (("왼쪽", tid(i - 1, j)), ("오른쪽", tid(i + 1, j)), ("아래", tid(i, j - 1)), ("위", tid(i, j + 1))),
                                  x0, y0, w, h, ox, oy, data) for (i, j), (x0, y0, w, h, on) in sorted(grid.items(), key=lambda t: (t[0][1], t[0][0])) if on):
                    pdf.savefig(fig)
                    plt.close(fig)
            shutil.copy(OUT / f"조각{k}.pdf", SITE / f"piece{k}.pdf")
            total += n
            print(f"조각 {k} ({name}): 종이 {n}장 / {nc * nr}칸")
    print(f"모두 {total}장 (+ 배치도 9장) → {OUT}")


if __name__ == "__main__":
    main()

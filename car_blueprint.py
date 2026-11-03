"""자동차 설계도: 윗면·옆면 배치 (sim.py 치수 그대로, mm). 배선·판 가공은 assembly.py.

python car_blueprint.py → blueprint/자동차.pdf, blueprint/5_자동차.png
좌표: 바퀴축 중심 = 0, 앞(+x), 왼쪽(+y). 실물 키트 측정값이 다르면 sim.py 값을 고치고 다시 실행.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

import car3d
import sim

plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]
OUT = Path(__file__).parent / "blueprint"
mm = 1000
F, R = sim.CAR_FRONT * mm, sim.CAR_REAR * mm                  # 바퀴축 → 앞머리 65 / 뒤끝 152.5
PLATE_W = 135
WR, WHW, WY = sim.WHEEL_R * mm, sim.WHEEL_HW * mm, sim.WHEEL_Y * mm
CAM_X, CAM_Z = sim.CAM_X * mm, sim.CAM_Z * mm
# 부품 위치 (sim.py CAR_XML 의 geom 위치와 같음): 이름, 중심 x, 중심 y, 길이, 폭, 색
PARTS = [("L298N", 12, 0, 43, 43, "#cc2222"), ("18650×2", -95, 0, 77, 41, "#222222"),
         ("Uno (윗판)", -75, 0, 69, 53, "#00748c")]
LEDS = [("좌 지시등", 58, 58, "orange"), ("우 지시등", 58, -58, "orange"), ("좌 지시등", -145, 58, "orange"),
        ("우 지시등", -145, -58, "orange")]


def dim(ax, p, q, text, off=(0, 0), **kw):
    """치수선."""
    (x0, y0), (x1, y1) = p, q
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="<->", lw=.8, color="#444"))
    ax.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], text, ha="center", va="center", fontsize=8,
            color="#222", bbox=dict(fc="white", ec="none", pad=1), **kw)


def top(ax):
    ax.add_patch(Rectangle((-R, -PLATE_W / 2), F + R, PLATE_W, fc="#dfeffa", ec="#5b8db8", lw=1.2))
    for s in (1, -1):                                         # 바퀴 (판 바깥)
        ax.add_patch(Rectangle((-WR, s * WY - WHW), 2 * WR, 2 * WHW, fc="#333", ec="k"))
        ax.add_patch(Rectangle((-59, s * 56 - 11.25), 70, 22.5, fc="#f2c200", ec="#a07d00", alpha=.6))   # TT 모터 (판 밑)
    ax.add_patch(Circle((-125, 0), 12, fc="#bbb", ec="k"))
    ax.text(-125, -33, "캐스터", ha="center", fontsize=7)
    for name, x, y, l, w, c in PARTS:
        ax.add_patch(Rectangle((x - l / 2, y - w / 2), l, w, fc=c, ec="k", alpha=.35 if "Uno" in name else .8))
        ax.text(x, y + (w / 2 + 7 if "Uno" in name else 0), name, ha="center", va="center", fontsize=7,
                color="k" if "Uno" in name else "white")
    for _, x, y, c in LEDS:
        ax.add_patch(Circle((x, y), 4, fc=c, ec="k", lw=.6))
    ax.add_patch(Rectangle((CAM_X - 8, -13.5), 10, 27, fc="#553322", ec="k"))
    ax.text(CAM_X + 8, 0, "ESP32-CAM\n(기둥 위)", fontsize=7, va="center")
    ax.arrow(F + 15, 0, 25, 0, width=2, color="#2a7")
    ax.text(F + 28, 8, "앞", color="#2a7", fontsize=9, ha="center")
    dim(ax, (-R, -PLATE_W / 2 - 25), (F, -PLATE_W / 2 - 25), f"판 길이 {F + R:.0f}")
    dim(ax, (0, -PLATE_W / 2 - 45), (F, -PLATE_W / 2 - 45), f"축→앞 {F:.0f}")
    dim(ax, (-R, -PLATE_W / 2 - 45), (0, -PLATE_W / 2 - 45), f"축→뒤 {R:.1f}")
    dim(ax, (-R - 20, -PLATE_W / 2), (-R - 20, PLATE_W / 2), f"판 폭 {PLATE_W}", rotation=90)
    dim(ax, (F + 60, -WY - WHW), (F + 60, WY + WHW), f"바퀴 바깥 폭 {2 * (WY + WHW):.0f}", rotation=90)
    dim(ax, (-40, -WY), (-40, WY), f"좌우 바퀴 간격 {2 * WY:.0f}", rotation=90)
    ax.plot([0, 0], [-WY - 30, WY + 30], "k-.", lw=.6)
    ax.text(2, WY + 32, "바퀴축 (기준 0)", fontsize=7)
    ax.set(xlim=(-R - 45, F + 90), ylim=(-PLATE_W / 2 - 60, WY + 50), aspect="equal", title="윗면 (mm)")


def side(ax):
    z0, z2 = car3d.Z1, car3d.Z2                               # 아래판·윗판 밑면 높이 (car3d.py 와 같음)
    ax.axhline(0, color="#777", lw=1)
    ax.add_patch(Rectangle((-R, z0), F + R, 3, fc="#5b8db8"))                     # 아래판
    ax.add_patch(Rectangle((-145, z2), 150, 3, fc="#9cc3e0"))                     # 윗판 (서포트 30mm 위)
    for x in (-140, -5):
        ax.plot([x, x], [z0 + 3, z2], color="#c9a33a", lw=2)
    ax.add_patch(Rectangle((-59, WR - 9.4), 70, 18.8, fc="#f2c200", ec="#a07d00", alpha=.7))   # TT 모터 (판 밑)
    ax.add_patch(Circle((0, WR), WR, fc="#333", ec="k", alpha=.85))
    ax.add_patch(Circle((-125, 6), 6, fc="#bbb", ec="k"))                          # 볼캐스터 + 서포트
    ax.plot([-125, -125], [12, z0], color="#c9a33a", lw=3)
    ax.add_patch(Rectangle((-133.5, z0 + 3), 77, 21, fc="#222", alpha=.8))
    ax.text(-95, z0 + 13, "18650×2", color="white", fontsize=7, ha="center", va="center")
    ax.add_patch(Rectangle((-9.5, z0 + 3), 43, 27, fc="#cc2222", alpha=.8))
    ax.text(12, z0 + 16, "L298N", color="white", fontsize=7, ha="center", va="center")
    ax.add_patch(Rectangle((-109.3, z2 + 7), 68.6, 6, fc="#00748c"))
    ax.text(-75, z2 + 18, "Uno", fontsize=7, ha="center")
    ax.add_patch(Rectangle((CAM_X - 12, z0 + 3), 6, CAM_Z - 15 - z0, fc="#262626"))   # 카메라 기둥
    s, c = sim.CAM_PITCH, 30
    cam = Polygon([(CAM_X - 7, CAM_Z + 12), (CAM_X + 7, CAM_Z + 12), (CAM_X + 7, CAM_Z - 12), (CAM_X - 7, CAM_Z - 12)],
                  fc="#553322", ec="k")
    ax.add_patch(cam)
    import numpy as np
    th = np.radians(sim.CAM_PITCH)
    half = np.radians(sim.CAM_FOVY / 2)
    for a, ls in ((th, "-"), (th - half, ":"), (th + half, ":")):                  # 광축, 화각 위/아래
        L = 230
        ax.plot([CAM_X, CAM_X + L * np.cos(a)], [CAM_Z, CAM_Z - L * np.sin(a)], color="#2a7", ls=ls, lw=1)
    ax.text(CAM_X + 120, CAM_Z - 40, f"아래로 {sim.CAM_PITCH}°\n세로 화각 {sim.CAM_FOVY:.0f}°\n(렌즈 대각 {sim.CAM_DIAG_FOV}°)",
            fontsize=7, color="#2a7")
    dim(ax, (CAM_X + 25, 0), (CAM_X + 25, CAM_Z), f"렌즈 높이 {CAM_Z:.0f}", rotation=90)
    dim(ax, (0, -12), (CAM_X, -12), f"{CAM_X:.0f}")
    dim(ax, (-R - 15, 0), (-R - 15, 2 * WR), f"바퀴 Ø{2 * WR:.0f}", rotation=90)
    ax.set(xlim=(-R - 35, F + 250), ylim=(-25, CAM_Z + 120), aspect="equal", title="옆면 (mm, 왼쪽에서 봄)")


if __name__ == "__main__":
    from matplotlib.backends.backend_pdf import PdfPages
    OUT.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(16.5, 11.7))                    # A3 가로 (배선은 assembly.py → 조립설계도.pdf)
    top(fig.add_axes([.03, .1, .45, .8]))
    side(fig.add_axes([.52, .1, .45, .8]))
    fig.suptitle("FlyDrive 자동차 설계도 (자체 제작 차체, 2WD + ESP32-CAM, sim.py 치수)", fontsize=14)
    with PdfPages(OUT / "자동차.pdf") as pdf:
        pdf.savefig(fig)
    fig.savefig(OUT / "8_자동차.png", dpi=80)
    print("저장:", OUT / "자동차.pdf", OUT / "8_자동차.png")

"""자동차 3D 도안: 실제 부품 치수로 배치 (mm). 자체 제작 2단 차체.

python car3d.py → blueprint/자동차_3D.pdf (조립도·분해도·앞·옆·위), blueprint/9_자동차_3D.png
좌표: 바퀴축 중심 바닥 = (0, 0, 0), 앞 +x, 왼쪽 +y, 위 +z.
주행에 중요한 값(판 외곽, 바퀴·축 위치, 카메라 높이·각도)은 sim.py 그대로. 나머지(판 높이, 층 간격)는 실물 부품에 맞춘 값.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import sim

plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]
OUT = Path(__file__).parent / "blueprint"
F, R = sim.CAR_FRONT * 1000, sim.CAR_REAR * 1000          # 바퀴축 → 판 앞 65 / 뒤 152.5
W = 135                                                   # 판 폭
WR, WY = sim.WHEEL_R * 1000, sim.WHEEL_Y * 1000           # 바퀴 반지름 32.5, 바퀴 중심 y ±82.5
T = 3                                                     # 판 두께 (MDF 3mm)
Z1 = 42                                                   # 아래판 밑면: TT 모터(높이 18.8, 축 중심)가 판 밑에 붙는 높이
Z2 = Z1 + T + 30                                          # 윗판 밑면: M3 서포트 30mm
CAM = (sim.CAM_X * 1000, 0, sim.CAM_Z * 1000)             # 렌즈 중심 (60, 0, 150)

# (이름, 중심 x, y, z, 크기 x, y, z, 색, 분해도에서 올리는 층)
PARTS = [
    ("아래판 (MDF 3mm)", (F - R) / 2, 0, Z1 + T / 2, F + R, W, T, "#9ecae1", 0),
    ("윗판 (MDF 3mm)", -72.5, 0, Z2 + T / 2, 145, 100, T, "#c6dbef", 3),
    # TT 모터 70×22.5×18.8: 축이 기어박스 끝에서 11mm → 몸체 중심 x = -24, 바퀴 안쪽 면에 2mm 띄워 붙임
    ("TT 모터 L", -24, 56, WR, 70, 22.5, 18.8, "#f2c200", -1),
    ("TT 모터 R", -24, -56, WR, 70, 22.5, 18.8, "#f2c200", -1),
    ("18650×2 홀더", -95, 0, Z1 + T + 10.5, 77, 41, 21, "#333333", 1),
    ("L298N", -3, 0, Z1 + T + 13.5, 43, 43, 27, "#d62728", 1),
    ("MP1584 5V", -80, 45, Z1 + T + 3, 22, 17, 6, "#2ca02c", 1),
    ("Arduino Uno", -75, 0, Z2 + T + 7.5, 68.6, 53.4, 15, "#00748c", 4),
    ("카메라 기둥", CAM[0] - 9, 0, (Z1 + T + CAM[2] - 12) / 2, 6, 20, CAM[2] - 12 - Z1 - T, "#444444", 2),
]
STANDOFFS = [(x, y) for x in (-140, -5) for y in (-42, 42)]
LEDS = [(58, 58, "orange"), (58, -58, "orange"), (-145, 58, "orange"), (-145, -58, "orange")]   # 방향지시등 (assembly.py 구멍)


def box(ax, c, s, color, alpha=.95):
    c, s = np.array(c, float), np.array(s, float) / 2
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s + c
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    ax.add_collection3d(Poly3DCollection([v[f] for f in faces], fc=color, ec="k", lw=.3, alpha=alpha))


def cyl(ax, c, axis, r, h, color, n=24):
    """axis: 0=x, 1=y, 2=z 방향 원기둥."""
    t = np.linspace(0, 2 * np.pi, n)
    ring = np.c_[r * np.cos(t), r * np.sin(t)]
    ends = []
    for d in (-h / 2, h / 2):
        p = np.zeros((n, 3))
        others = [i for i in range(3) if i != axis]
        p[:, others[0]], p[:, others[1]], p[:, axis] = ring[:, 0], ring[:, 1], d
        ends.append(p + c)
    sides = [[ends[0][i], ends[0][i + 1], ends[1][i + 1], ends[1][i]] for i in range(n - 1)]
    ax.add_collection3d(Poly3DCollection(sides + ends, fc=color, ec="none", alpha=.95))


def car(ax, explode=0.0, labels=True):
    up = lambda layer: explode * layer                    # 분해도: 층마다 위로 띄움
    for name, x, y, z, sx, sy, sz, color, layer in PARTS:
        box(ax, (x, y, z + up(layer)), (sx, sy, sz), color, .45 if "판" in name else .95)
        if labels:
            ax.text(x, y + (sy / 2 + 8) * np.sign(y or 1), z + up(layer) + sz / 2 + 6, name, fontsize=7, zorder=20)
    for s in (1, -1):                                     # 바퀴 65×26 + 축
        cyl(ax, (0, s * WY, WR), 1, WR, 26, "#222222")
        cyl(ax, (0, s * (WY - 17), WR), 1, 2.7, 10, "#bbbbbb")
    for x, y in STANDOFFS:                                # M3 서포트 30mm
        cyl(ax, (x, y, Z1 + T + 15 + up(2)), 2, 2.5, 30, "#c9a33a")
    cyl(ax, (-125, 0, Z1 / 2 + 6), 2, 10, Z1 - 12, "#999999")       # 볼 캐스터 + 스페이서 (판 밑면까지 42)
    cyl(ax, (-125, 0, 6), 1, 6, 12, "#dddddd")
    # ESP32-CAM: 보드 27×40.5 를 30° 아래로 기울여 렌즈가 (60, 0, 150)
    th = np.radians(sim.CAM_PITCH)
    u = np.array([np.cos(th), 0, -np.sin(th)])            # 광축
    v = np.array([np.sin(th), 0, np.cos(th)])             # 보드 위쪽
    c = np.array(CAM) + np.array([0, 0, up(2)]) - u * 5
    w = np.array([0, 1, 0])
    pts = [c + a * v * 20 + b * w * 13.5 for a, b in ((-1, -1), (-1, 1), (1, 1), (1, -1))]
    ax.add_collection3d(Poly3DCollection([pts], fc="#5a3a22", ec="k", lw=.4))
    lens = np.array(CAM) + np.array([0, 0, up(2)])
    ax.plot(*np.c_[lens, lens + u * 60], color="#2a7", lw=1.5)
    if labels:
        ax.text(*(lens + [0, 0, 14]), "ESP32-CAM (120° 렌즈, 30° 아래)", fontsize=7, color="#1a6", zorder=20)
    for x, y, color in LEDS:
        ax.scatter([x], [y], [Z1 + T + 4], color=color, s=25, edgecolor="k", depthshade=False)
    ax.set(xlim=(-170, 140), ylim=(-110, 110), zlim=(0, 200 + explode * 4))
    ax.set_box_aspect((310, 220, 200 + explode * 4))
    ax.set_xlabel("x 앞 (mm)", fontsize=7), ax.set_ylabel("y 왼쪽", fontsize=7), ax.set_zlabel("z 위", fontsize=7)
    ax.tick_params(labelsize=6)


if __name__ == "__main__":
    from matplotlib.backends.backend_pdf import PdfPages
    OUT.mkdir(exist_ok=True)
    views = [("조립도", 25, -55, 0, True), ("분해도 (층별)", 22, -55, 35, True),
             ("앞에서", 0, 0, 0, False), ("왼쪽에서", 0, -90, 0, False), ("위에서", 90, -90, 0, False)]
    with PdfPages(OUT / "자동차_3D.pdf") as pdf:
        fig = plt.figure(figsize=(16.5, 11.7))
        for i, (title, el, az, ex, lab) in enumerate(views[:2]):
            ax = fig.add_axes([.02 + .49 * i, .1, .48, .82], projection="3d")
            car(ax, ex, lab)
            ax.view_init(el, az)
            ax.set_title(title, fontsize=13)
        fig.suptitle(f"FlyDrive 자동차 3D 도안 (mm) — 아래판 {F + R:.0f}×{W}, 바퀴 Ø{2 * WR:.0f} 축 간격 {2 * WY:.0f}, "
                     f"렌즈 높이 {CAM[2]:.0f}·{sim.CAM_PITCH}° 아래", fontsize=13)
        fig.text(.5, .04, "층 구성: 바닥 ─ TT 모터·바퀴·볼캐스터(판 밑면 42mm) ─ 아래판(배터리·L298N·MP1584·카메라 기둥) "
                 "─ M3 서포트 30mm ×4 ─ 윗판(Uno)\nLED: 노랑 4 (앞뒤 좌우 모서리, 방향지시등·비상등). 배선·가공은 조립설계도.pdf", ha="center", fontsize=10)
        pdf.savefig(fig)
        fig.savefig(OUT / "9_자동차_3D.png", dpi=80)
        plt.close(fig)
        fig = plt.figure(figsize=(16.5, 11.7))
        for i, (title, el, az, ex, lab) in enumerate(views[2:]):
            ax = fig.add_axes([.01 + .33 * i, .05, .32, .88], projection="3d")
            car(ax, ex, lab)
            ax.view_init(el, az)
            if el == 0 and az == 0:
                ax.set_proj_type("ortho")
            ax.set_title(title, fontsize=13)
        pdf.savefig(fig)
        fig.savefig(OUT / "10_자동차_3면.png", dpi=80)
        plt.close(fig)
    print(OUT / "자동차_3D.pdf")

"""조립 설계도: 주문서 부품을 실제로 어떻게 연결·조립하는지.

python assembly.py → blueprint/조립설계도.pdf
  1 차량 배선도          2 배선표 (선 하나씩)
  3·4 차체판 가공도 1:1 (A4 가로, 100% 로 출력해 MDF 판에 붙이고 자름)
  5 신호등 (Nano 배선 + 등 치수 + 교차로 배치)      6 조립 순서·재료 배분
좌표: 바퀴축 중심 = (0, 0), 앞 +x, 왼쪽 +y (mm). 부품 위치는 car3d.py 와 같음.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, Polygon, Rectangle

import car3d
import sim

plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
OUT = Path(__file__).parent / "blueprint"
F, R, W = car3d.F, car3d.R, car3d.W                     # 아래판: 바퀴축 앞 65, 뒤 152.5, 폭 135
RED, BLK = "#d62728", "#222222"

# ---------------------------------------------------------------- 1. 차량 배선도
# 부품: 이름 → (x, y, w, h, 제목, 색), 핀: (부품, 핀) → 좌표
BOXES = {
    "bat": (12, 120, 42, 50, "18650×2 홀더\n(스위치, 7.4V)\n주문 7·8", "#444444"),
    "ml": (12, 60, 42, 30, "TT 모터 왼쪽\n주문 1", "#f2c200"),
    "mr": (12, 200, 42, 30, "TT 모터 오른쪽\n주문 1", "#f2c200"),
    "drv": (92, 60, 60, 170, "L298N\n주문 4", "#f3c9c9"),
    "uno": (200, 50, 70, 190, "Arduino Uno\n(키트)", "#cfe8ee"),
    "esp": (345, 60, 60, 70, "ESP32-CAM\n주문 5·6", "#e2d6cc"),
    "buck": (345, 160, 60, 30, "MP1584 (5.0V)\n주문 10", "#cfe9cf"),
}
PINS = {
    ("bat", "+"): (24, 170), ("bat", "−"): (42, 120),
    ("ml", "a"): (54, 80), ("ml", "b"): (54, 70), ("mr", "a"): (54, 220), ("mr", "b"): (54, 210),
    ("drv", "OUT1"): (92, 80), ("drv", "OUT2"): (92, 70), ("drv", "OUT4"): (92, 220), ("drv", "OUT3"): (92, 210),
    ("drv", "12V"): (107, 230), ("drv", "GND"): (107, 60), ("drv", "5V (안 씀)"): (132, 60),
    ("drv", "ENA"): (152, 190), ("drv", "IN1"): (152, 175), ("drv", "IN2"): (152, 160),
    ("drv", "IN3"): (152, 130), ("drv", "IN4"): (152, 115), ("drv", "ENB"): (152, 100),
    ("uno", "D5"): (200, 190), ("uno", "D7"): (200, 175), ("uno", "D8"): (200, 160),
    ("uno", "D9"): (200, 130), ("uno", "D10"): (200, 115), ("uno", "D6"): (200, 100),
    ("uno", "VIN"): (215, 240), ("uno", "GND"): (215, 50),
    ("uno", "D2"): (270, 210), ("uno", "D3"): (270, 175), ("uno", "D12"): (270, 120), ("uno", "D11"): (270, 90),
    ("esp", "GPIO14"): (345, 120), ("esp", "GPIO15"): (345, 90), ("esp", "5V"): (360, 130), ("esp", "GND"): (390, 60),
    ("buck", "OUT+"): (360, 160), ("buck", "IN+"): (360, 190), ("buck", "IN−"): (390, 190),
}
VBUS, GBUS = 262, 22                                     # +7.4V 레일, 공통 GND 레일 (y)
# (시작 핀, 끝 핀 또는 레일, 색, 꺾는 점들)
WIRES = [
    (("bat", "+"), "V", RED, []), (("bat", "−"), "G", BLK, []),
    (("drv", "12V"), "V", RED, []), (("drv", "GND"), "G", BLK, []),
    (("uno", "VIN"), "V", RED, []), (("uno", "GND"), "G", BLK, []),
    (("buck", "IN+"), "V", RED, []), (("buck", "IN−"), "G", BLK, [(390, 200), (412, 200)]),
    (("esp", "GND"), "G", BLK, []), (("buck", "OUT+"), ("esp", "5V"), "#ff7f0e", []),
    (("drv", "OUT1"), ("ml", "a"), "#1f77b4", []), (("drv", "OUT2"), ("ml", "b"), "#17becf", []),
    (("drv", "OUT4"), ("mr", "a"), "#1f77b4", []), (("drv", "OUT3"), ("mr", "b"), "#17becf", []),
    (("uno", "D5"), ("drv", "ENA"), "#9467bd", []), (("uno", "D7"), ("drv", "IN1"), "#2ca02c", []),
    (("uno", "D8"), ("drv", "IN2"), "#8c564b", []), (("uno", "D9"), ("drv", "IN3"), "#2ca02c", []),
    (("uno", "D10"), ("drv", "IN4"), "#8c564b", []), (("uno", "D6"), ("drv", "ENB"), "#9467bd", []),
    (("esp", "GPIO14"), ("uno", "D12"), "#e377c2", []),
]


def resistor(ax, x, y, text, vertical=False, w=14, h=5):
    """저항 기호 (가운데 x, y)."""
    if vertical:
        w, h = h, w
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc="white", ec="k", lw=1, zorder=3))
    ax.text(x + (5 if vertical else 0), y + (0 if vertical else 5), text, fontsize=6.5,
            ha="left" if vertical else "center", va="center" if vertical else "bottom")


def led(ax, x, y, color, text):
    """LED 기호: 왼쪽 + (애노드) → 오른쪽 − (캐소드)."""
    ax.add_patch(Polygon([(x - 4, y + 4), (x - 4, y - 4), (x + 3, y)], fc=color, ec="k", lw=.8, zorder=3))
    ax.plot([x + 3, x + 3], [y - 4, y + 4], "k", lw=1.2, zorder=3)
    ax.text(x, y + 6, text, fontsize=6.5, ha="center")


def page_wiring():
    fig = plt.figure(figsize=(16.5, 11.7))
    ax = fig.add_axes([.01, .03, .98, .9])
    for x, y, w, h, t, c in BOXES.values():
        ax.add_patch(Rectangle((x, y), w, h, fc=c, ec="k", lw=1.2, alpha=.35))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=10, weight="bold")
    for (box, pin), (x, y) in PINS.items():
        bx, by, bw, bh = BOXES[box][:4]
        ax.add_patch(Circle((x, y), 1.4, fc="white", ec="k", zorder=4))
        dx = 3 if x <= bx else -3 if x >= bx + bw else 0
        dy = 3 if y <= by else -3 if y >= by + bh else 0
        ax.text(x + dx, y + dy, pin, fontsize=7.5, zorder=5, ha="left" if dx > 0 else "right" if dx < 0 else "center",
                va="bottom" if dy > 0 else "top" if dy < 0 else "center", family=["Consolas", "Malgun Gothic"])
    ax.plot([20, 412], [VBUS, VBUS], color=RED, lw=3)
    ax.plot([20, 412], [GBUS, GBUS], color=BLK, lw=3)
    ax.text(20, VBUS + 3, "+7.4V (배터리 + → 홀더 스위치 뒤)", color=RED, fontsize=9)
    ax.text(20, GBUS - 7, "공통 GND (모든 부품의 GND 를 하나로)", color=BLK, fontsize=9)
    for a, b, color, via in WIRES:
        p = PINS[a]
        if b in ("V", "G"):
            q = (via[-1][0] if via else p[0], VBUS if b == "V" else GBUS)
        else:
            q = PINS[b]
            if not via and p[1] != q[1]:
                via = [((p[0] + q[0]) / 2, p[1]), ((p[0] + q[0]) / 2, q[1])]
        pts = np.array([p, *via, q])
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2, solid_capstyle="round", zorder=2)
        if b in ("V", "G"):
            ax.add_patch(Circle(q, 1.6, fc=color, ec=color, zorder=4))
    # 방향지시등: 핀마다 LED 2개, LED 마다 220Ω 따로
    for pin, ys, names in (("D2", (226, 212), ("왼쪽 앞", "왼쪽 뒤")), ("D3", (191, 177), ("오른쪽 앞", "오른쪽 뒤"))):
        x0, y0 = PINS[("uno", pin)]
        ax.plot([x0, 280, 280], [y0, ys[0], ys[1]], color="#bcbd22", lw=2)
        ax.plot([280, 280], [y0, ys[0]], color="#bcbd22", lw=2)
        for y, n in zip(ys, names):
            ax.plot([280, 330], [y, y], color="#bcbd22", lw=2, zorder=2)
            resistor(ax, 292, y, "220Ω")
            led(ax, 314, y, "#ffd21f", f"노랑 LED {n}")
            ax.add_patch(Circle((330, y), 1.4, fc=BLK, zorder=4))
    ax.plot([330, 330], [226, GBUS], color=BLK, lw=2, zorder=1)
    ax.add_patch(Circle((330, GBUS), 1.6, fc=BLK, zorder=4))
    ax.text(283, 238, "방향지시등 (키트 노랑 LED 4 + 220Ω 4)\n비상등 = D2·D3 동시 깜빡임", fontsize=8)
    # 분압: D11(5V) → 10k → GPIO15 (3.3V), GPIO15 → 10k+10k → GND
    x0, y0 = PINS[("uno", "D11")]
    ax.plot([x0, 345], [y0, y0], color="#7f7f7f", lw=2, zorder=1)
    resistor(ax, 290, y0, "10kΩ (키트)")
    ax.add_patch(Circle((308, y0), 1.6, fc="#7f7f7f", zorder=4))
    ax.plot([308, 308], [y0, GBUS], color="#7f7f7f", lw=2, zorder=1)
    resistor(ax, 308, 72, "10kΩ", vertical=True)
    resistor(ax, 308, 50, "10kΩ", vertical=True)
    ax.text(312, 100, "5V→3.3V 분압\n(10k : 20k)", fontsize=7.5)
    ax.text(275, 135, "ESP32 → Uno (3.3V 그대로 읽힘)", fontsize=7, color="#e377c2")
    notes = ("① L298N 의 ENA·ENB 점퍼 캡을 빼고 그 핀에 D5·D6 연결 (속도 PWM).  12V 점퍼(5V 레귤레이터)는 그대로 두되 5V 단자는 쓰지 않음.\n"
             "② MP1584 는 ESP32-CAM 을 꽂기 전에 IN 에 배터리를 물리고 OUT 을 5.0V 로 맞춤 (가변저항 돌림). 처음엔 20V 가까이 나올 수 있음.\n"
             "③ ESP32-CAM 은 업로드할 때만 MB 보드에 꽂고, 차에는 CAM 보드만 올림 (MB 는 노트북 연결용).  ④ Uno 는 D11·D12 를 SoftwareSerial 로 ESP32 와 통신.\n"
             "⑤ 모터가 반대로 돌면 그 모터의 OUT 선 두 가닥을 맞바꿈.  ⑥ 선 굵기: 전원·모터 = 22AWG 전선 키트(주문 16), 신호 = 키트 점퍼선.")
    ax.text(5, -2, notes, fontsize=8.5, va="top")
    ax.set(xlim=(0, 420), ylim=(-40, 280), aspect="equal")
    ax.axis("off")
    fig.suptitle("① 차량 배선도 — 빨강 = +7.4V, 검정 = GND, 주황 = 5V. 선이 교차해도 점이 없으면 연결 아님", fontsize=14)
    return fig


def page_table():
    rows = []
    for a, b, color, _ in WIRES:
        name = lambda p: f"{BOXES[p[0]][4].splitlines()[0]} {p[1]}"
        rows.append((name(a), {"V": "+7.4V 레일", "G": "GND 레일"}.get(b) or name(b),
                     {RED: "빨강", BLK: "검정"}.get(color, "아무 색")))
    rows += [("Uno D2", "220Ω → 노랑 LED 왼쪽 앞 / 왼쪽 뒤 (각각)", "점퍼"), ("Uno D3", "220Ω → 노랑 LED 오른쪽 앞 / 뒤 (각각)", "점퍼"),
             ("LED 4개 짧은 다리(−)", "GND 레일", "검정"), ("Uno D11", "10kΩ → ESP32-CAM GPIO15", "점퍼"),
             ("ESP32-CAM GPIO15", "10kΩ → 10kΩ → GND", "점퍼"),
             ("+7.4V 레일·GND 레일", "브레드보드 위 + / − 줄 (아래판, 미니 브레드보드 또는 전선 묶음)", "")]
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([.03, .03, .94, .88])
    ax.axis("off")
    t = ax.table(cellText=[(i + 1, *r) for i, r in enumerate(rows)], colLabels=["#", "부터", "까지", "전선"],
                 loc="upper center", cellLoc="left", colWidths=[.04, .3, .5, .1])
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.32)
    fig.suptitle("② 배선표 — 위에서부터 차례로 연결하고 ✓ 표시", fontsize=14)
    return fig


# ---------------------------------------------------------------- 3·4. 차체판 1:1
A4 = (297, 210)
STANDOFFS = car3d.STANDOFFS                                          # (-140|-5, ±42)
UNO_HOLES = [(14.0, 2.5), (15.3, 50.7), (66.1, 7.6), (66.1, 35.5)]   # Uno R3, USB 쪽 모서리 기준
UNO_X0, UNO_Y0 = -75 - 68.6 / 2, -53.4 / 2                            # USB 가 뒤(−x)를 봄
LED_XY = [(58, 58), (58, -58), (-145, 58), (-145, -58)]
LX = -3                                                              # L298N 중심 x (조립모형_3D.html 과 같음)
# 카메라: 렌즈 (60, 0, 150), 30° 아래. 카메라판 뒤 아래 모서리가 기둥 앞면에 닿음 (조립모형_3D.html 과 같은 계산)
_t = np.radians(sim.CAM_PITCH)
_U, _V, _W = np.array([np.cos(_t), 0, -np.sin(_t)]), np.array([np.sin(_t), 0, np.cos(_t)]), np.array([0, 1, 0])
_C = np.array(car3d.CAM, float) - (10 * _V + 9 * _U)
_cw = lambda a, b, c: a * _W + b * _V + c * _U + _C
_BB, _TB = _cw(0, -20, -14.5), _cw(0, 6, -14.5)
MAST_X, MAST_H = _BB[0], _TB[2] - (car3d.Z1 + 3)                        # 기둥 앞면 x, 기둥 높이
WEDGE = (_TB[0] - _BB[0], _TB[2] - _BB[2])                               # 쐐기 가로·세로 (30°)


def sheet():
    """A4 가로, 1 단위 = 1 mm."""
    fig = plt.figure(figsize=(A4[0] / 25.4, A4[1] / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, A4[0]), ylim=(0, A4[1]), aspect="equal")
    ax.axis("off")
    ax.plot([10, 60], [8, 8], "k", lw=2)                             # 출력 크기 확인용 50mm
    for x in (10, 60):
        ax.plot([x, x], [6, 10], "k", lw=1)
    ax.text(62, 8, "← 자로 재서 정확히 50mm 이어야 함 (인쇄: 실제 크기 100%, 페이지 맞춤 끔)", fontsize=6, va="center")
    return fig, ax


def hole(ax, x, y, d, text=None, dy=3):
    ax.add_patch(Circle((x, y), d / 2, fc="white", ec="k", lw=.8))
    ax.plot([x - d, x + d], [y, y], "k", lw=.3)
    ax.plot([x, x], [y - d, y + d], "k", lw=.3)
    if text:
        ax.text(x, y + d / 2 + dy, text, fontsize=5, ha="center")


def zone(ax, x, y, w, h, text, color):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=color, ec=color, lw=.8, ls="--", alpha=.18))
    ax.text(x, y, text, fontsize=5.5, ha="center", va="center")


def page_bottom():
    fig, ax = sheet()
    ox, oy = 40 + R, 30 + W / 2                                       # 차 좌표 → 종이 좌표
    P = lambda x, y: (ox + x, oy + y)
    ax.add_patch(Rectangle(P(-R, -W / 2), F + R, W, fc="none", ec="k", lw=1.2))
    zone(ax, *P(-95, 0), 77, 41, "18650 홀더\n(양면테이프)", "#444444")
    zone(ax, *P(LX, 0), 43, 43, "L298N\n방열판 앞", RED)
    zone(ax, *P(-80, 45), 22, 17, "MP1584", "#2ca02c")
    zone(ax, *P(MAST_X - 3, 0), 6, 20, "기둥", "#444444")
    for s in (1, -1):                                                 # TT 모터 + 키트 고정판은 판 밑면
        zone(ax, *P(-24, s * 56), 70, 22.5, "TT 모터 (밑면)", "#f2c200")
        zone(ax, *P(-23, s * 36.75), 30, 16, "모터 고정판 (밑면)\n실물 대고 표시", "#1f77b4")
        for x in (-34, -12):
            hole(ax, *P(x, s * 36.75), 3.2)
        hole(ax, *P(-62, s * 40), 8, "모터선 Ø8")
    for x, y in STANDOFFS:
        hole(ax, *P(x, y), 3.2, "서포트 Ø3.2")
    for dx in (-18.5, 18.5):
        for dy in (-18.5, 18.5):
            hole(ax, *P(LX + dx, dy), 3.2)
    ax.text(*P(LX, 25), "L298N 구멍 37×37 (실물 대고 확인)", fontsize=5, ha="center")
    for x, y in LED_XY:
        hole(ax, *P(x, y), 5, "LED Ø5")
    for y in (8, -8):
        hole(ax, *P(-125, y), 3.2)
    ax.text(*P(-125, 13), "캐스터 2구멍 (간격은 실물 대고 확인)", fontsize=5, ha="center")
    ax.plot(*zip(P(0, -W / 2 - 6), P(0, W / 2 + 6)), "k-.", lw=.6)
    ax.text(*P(1, W / 2 + 7), "바퀴축 (바퀴 중심선)", fontsize=6)
    ax.annotate("", P(F + 18, 0), P(F + 3, 0), arrowprops=dict(arrowstyle="-|>", color="#2a7"))
    ax.text(*P(F + 20, 0), "앞", color="#2a7", fontsize=9, va="center")
    for (a, b, t) in ((P(-R, -W / 2 - 8), P(F, -W / 2 - 8), f"{F + R:.1f}"), (P(-R, -W / 2 - 14), P(0, -W / 2 - 14), f"{R}"),
                      (P(0, -W / 2 - 14), P(F, -W / 2 - 14), f"{F:.0f}")):
        ax.annotate("", b, a, arrowprops=dict(arrowstyle="<->", lw=.6))
        ax.text((a[0] + b[0]) / 2, a[1] + 1, t, fontsize=6, ha="center")
    ax.annotate("", P(-R - 8, W / 2), P(-R - 8, -W / 2), arrowprops=dict(arrowstyle="<->", lw=.6))
    ax.text(*P(-R - 10, 0), f"{W}", fontsize=6, rotation=90, ha="right", va="center")
    ax.text(10, 200, "③ 아래판 1:1 — MDF 3mm (주문 16). 위에서 본 모습. 굵은 선을 칼로 여러 번 그어 자르고, 구멍은 송곳 → Ø3 드릴.",
            fontsize=8, weight="bold")
    ax.text(10, 194, "점선 칸은 부품 자리 (구멍 없음). TT 모터·고정판·캐스터는 판 밑면에 붙음. 캐스터는 서포트 30mm 로 띄워 판 밑면 높이 42mm.",
            fontsize=6.5)
    return fig


def page_upper():
    fig, ax = sheet()
    ox, oy = 30 + 145, 30 + 50                                           # 윗판 x -145~0
    P = lambda x, y: (ox + x, oy + y)
    ax.add_patch(Rectangle(P(-145, -50), 145, 100, fc="none", ec="k", lw=1.2))
    zone(ax, *P(-75, 0), 68.6, 53.4, "Arduino Uno\nUSB → 뒤", "#00748c")
    for x, y in STANDOFFS:
        hole(ax, *P(x, y), 3.2, "서포트 Ø3.2")
    for bx, by in UNO_HOLES:
        hole(ax, *P(UNO_X0 + bx, UNO_Y0 + by), 3.2)
    ax.text(*P(-75, -33), "Uno 구멍 4개 Ø3.2 (Uno 를 대보고 맞으면 뚫기)", fontsize=5.5, ha="center")
    hole(ax, *P(-20, 9), 12, "배선 Ø12")
    ax.annotate("", P(15, 0), P(3, 0), arrowprops=dict(arrowstyle="-|>", color="#2a7"))
    ax.text(*P(17, 0), "앞", color="#2a7", fontsize=9, va="center")
    ax.annotate("", P(0, -57), P(-145, -57), arrowprops=dict(arrowstyle="<->", lw=.6))
    ax.text(*P(-72.5, -56), "145", fontsize=6, ha="center")
    ax.annotate("", P(-152, 50), P(-152, -50), arrowprops=dict(arrowstyle="<->", lw=.6))
    ax.text(*P(-154, 0), "100", fontsize=6, rotation=90, ha="right", va="center")
    # 카메라 기둥 부품 (오른쪽)
    x0, y0 = 205, 30
    for i in range(2):                                                 # 기둥 2장 겹쳐 붙임 (6mm)
        ax.add_patch(Rectangle((x0 + i * 26, y0), 20, MAST_H, fc="none", ec="k", lw=1.2))
        ax.text(x0 + i * 26 + 10, y0 + MAST_H / 2, f"기둥 {i + 1}\n20×{MAST_H:.0f}", fontsize=6, ha="center", va="center", rotation=90)
    ax.add_patch(Rectangle((x0 + 54, y0 + 120), 27, 26, fc="none", ec="k", lw=1.2))
    ax.text(x0 + 67.5, y0 + 133, "카메라판\n27×26", fontsize=6, ha="center", va="center")
    t = np.tan(np.radians(sim.CAM_PITCH))
    for i in range(2):                                                 # 30° 쐐기
        bx = x0 + 54 + i * 18
        ax.add_patch(Polygon([(bx, y0 + 85), (bx + WEDGE[0], y0 + 85), (bx, y0 + 85 + WEDGE[1])], fc="none", ec="k", lw=1.2))
        ax.text(bx + 1, y0 + 88, "30°", fontsize=6)
    ax.text(x0 + 54, y0 + 112, "쐐기 ×2", fontsize=6)
    for i in range(2):                                                 # 기둥 받침 삼각
        by = y0 + i * 32
        ax.add_patch(Polygon([(x0 + 54, by), (x0 + 74, by), (x0 + 54, by + 25)], fc="none", ec="k", lw=1.2))
    ax.text(x0 + 54, y0 + 62, "기둥 뒤\n받침 ×2", fontsize=6)
    ax.text(10, 200, "④ 윗판·카메라 기둥 1:1 — MDF 3mm. 기둥 2장 겹쳐 순간접착제 → 아래판 '기둥' 자리에 세우고 뒤에 받침 2개.",
            fontsize=8, weight="bold")
    ax.text(10, 194, f"기둥 맨 위 앞면에 쐐기 2개(30° 가 아래로) → 카메라판 → ESP32-CAM. 렌즈 중심이 바닥에서 {car3d.CAM[2]:.0f}mm, "
            f"바퀴축 앞 {car3d.CAM[0]:.0f}mm 인지 자로 확인.", fontsize=6.5)
    return fig


# ---------------------------------------------------------------- 5. 신호등
EW, NS = (2, 3, 4, 5), (6, 7, 8, 9)                               # road_signal.ino 와 같은 핀
LAMPS = (("빨강", "#e31a1c"), ("노랑", "#ffd21f"), ("화살표(초록)", "#33a02c"), ("초록", "#33a02c"))


def page_signal():
    fig = plt.figure(figsize=(16.5, 11.7))
    # (a) 배선
    ax = fig.add_axes([.02, .08, .47, .82])
    ax.add_patch(Rectangle((0, 10), 40, 170, fc="#cfe0f5", ec="k", lw=1.2))
    ax.text(20, 95, "Arduino Nano\n주문 11\n\nUSB ← 휴대폰\n충전기 5V", ha="center", va="center", fontsize=10, weight="bold")
    y = 170
    for group, pins in (("동서 방향 등 2개 (마주 봄)", EW), ("남북 방향 등 2개 (마주 봄)", NS)):
        ax.text(48, y + 8, group + "  (저항 = 220Ω)", fontsize=9, weight="bold")
        for pin, (name, c) in zip(pins, LAMPS):
            ax.add_patch(Circle((40, y), 1.2, fc="white", ec="k", zorder=4))
            ax.text(38, y, f"D{pin}", fontsize=8, ha="right", va="center")
            ax.plot([40, 52, 52], [y, y + 3, y - 3], color=c, lw=1.6)
            ax.plot([40, 52], [y, y - 3], color=c, lw=1.6)
            for k, yy in enumerate((y + 3, y - 3)):
                ax.plot([52, 118], [yy, yy], color=c, lw=1.6, zorder=1)
                resistor(ax, 66, yy, "", w=10, h=3.5)
                led(ax, 84, yy, c, "")
                ax.text(121, yy, f"{name} — 등 {'①' if k == 0 else '②'}", fontsize=7, va="center")
            y -= 18
        y -= 6
    ax.plot([118, 118], [173, 12], color=BLK, lw=2)
    ax.plot([40, 118], [12, 12], color=BLK, lw=2)
    ax.text(42, 14, "Nano GND", fontsize=8)
    ax.text(0, 0, "LED 16개 (빨강 4·노랑 4·초록 8, 주문 12~14), 저항 16개 (주문 15). 핀 하나에 LED 2개 = 약 20mA.\n"
                  "기둥에서 Nano 까지: 신호선 4 + GND 1 = 5가닥 (전선 키트 6색) — 코스 판 밑으로.", fontsize=8, va="top")
    ax.set(xlim=(-5, 150), ylim=(-12, 185), aspect="equal")
    ax.axis("off")
    ax.set_title("(a) 배선 — arduino/road_signal/road_signal.ino 그대로 업로드", fontsize=11)
    # (b) 등 머리 치수
    ax = fig.add_axes([.52, .55, .46, .36])
    sp, hw, hh = 28, 136, 32
    ax.add_patch(Rectangle((-hw / 2, -hh / 2), hw, hh, fc="#151515"))
    for j, (name, c) in enumerate(LAMPS):
        x = (j - 1.5) * sp
        ax.add_patch(Circle((x, 0), 5, fc=c))
        ax.text(x, -hh / 2 - 6, name, fontsize=8, ha="center", va="top")
    ax.text(.5 * sp, 0, "←", color="k", fontsize=13, ha="center", va="center", weight="bold")
    for a, b, t, yy in ((-hw / 2, hw / 2, f"{hw}", hh / 2 + 10), (-1.5 * sp, -.5 * sp, f"{sp}", hh / 2 + 4)):
        ax.annotate("", (b, yy), (a, yy), arrowprops=dict(arrowstyle="<->", lw=.7))
        ax.text((a + b) / 2, yy + 1.5, t, fontsize=8, ha="center")
    ax.annotate("", (hw / 2 + 6, hh / 2), (hw / 2 + 6, -hh / 2), arrowprops=dict(arrowstyle="<->", lw=.7))
    ax.text(hw / 2 + 8, 0, f"{hh}", fontsize=8, va="center")
    ax.text(-hw / 2, -hh / 2 - 22, "검정 폼보드 상자 136×32, 깊이 20 (mm). 앞판에 Ø10 구멍 4개 (28 간격)에 10mm LED 를 끼움.\n"
            "화살표 칸: 초록 LED 앞에 ← 모양만 뚫은 검정 테이프. 보는 사람 기준 왼쪽부터 빨강·노랑·화살표·초록.", fontsize=8, va="top")
    ax.set(xlim=(-80, 85), ylim=(-62, 32), aspect="equal")
    ax.axis("off")
    ax.set_title("(b) 등 머리 4개 (앞에서 본 모습, mm)", fontsize=11)
    # (c) 교차로 배치 (위에서)
    ax = fig.add_axes([.52, .05, .46, .45])
    c, RH, lh = np.array((sim.XS[1], sim.YS[1])), sim.RH, sim.LANE_HALF
    for d in ((1, 0), (0, 1)):
        ax.add_patch(Rectangle((-.9 if d[0] else -RH, -RH if d[0] else -.9), 1.8 if d[0] else 2 * RH,
                               2 * RH if d[0] else 1.8, fc="#555", ec="none"))
    ax.plot([-.9, .9], [0, 0], color="#fc0", lw=1)
    ax.plot([0, 0], [-.9, .9], color="#fc0", lw=1)
    for k, d in enumerate(np.array(sim.DIRS, float)):
        n = np.array((d[1], -d[0]))
        pole, head = d * (RH + .1) + n * (RH + .1), d * (RH + .1) + n * lh
        ew = abs(d[0]) > 0
        col = "#1f77b4" if ew else "#d62728"
        ax.plot(*zip(pole, head), color=col, lw=3)
        ax.add_patch(Circle(pole, .02, fc=col))
        hd = Rectangle(head - abs(d) * .01 - abs(n) * .068, *(2 * (abs(d) * .01 + abs(n) * .068)), fc="k")
        ax.add_patch(hd)
        ax.annotate("", -d * .55 + n * lh, -d * .8 + n * lh, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5))
        ax.text(*(pole + n * .07), f"{'동서' if ew else '남북'}\nD{'2-5' if ew else '6-9'}", color=col, fontsize=8,
                ha="center", va="center")
    ax.text(-.88, -.88, f"기둥 = 교차로 중심에서 진행 방향으로 {100 * (RH + .1):.0f}cm 지나서 오른쪽 {100 * (RH + .1):.0f}cm.\n"
            f"기둥 높이 {100 * (.3 + .03):.0f}cm, 팔 길이 {100 * (RH + .1 - lh):.0f}cm, 등 중심 높이 30cm (팔 끝에 매닮).\n"
            "기둥·팔: 검정 폼보드 6cm 띠를 3번 접어 사각 기둥. 화살표 = 그 등을 보며 달려오는 차.", fontsize=8,
            bbox=dict(fc="white", ec="none", alpha=.9))
    ax.set(xlim=(-.9, .9), ylim=(-.9, .9), aspect="equal")
    ax.axis("off")
    ax.set_title("(c) ④교차로 신호등 배치 (위에서, m) — 화살표 방향으로 오는 차가 그 등을 봄", fontsize=11)
    fig.suptitle("⑤ 신호등: 도로용 Arduino Nano + 10mm LED 16개", fontsize=14)
    return fig


# ---------------------------------------------------------------- 6. 조립 순서
STEPS = """[자동차]  (주문 번호 = 주문서 no)
1. 차체판: ③④ 페이지를 실제 크기로 출력 → 50mm 눈금 확인 → MDF 클립보드(16, 집게 떼어냄)에 붙이고 칼·실톱으로 자름. 구멍은 송곳으로 표시 후 드릴.
2. 모터: 키트 고정판에 TT 모터(1)를 긴 나사 2개로 고정 → 아래판 밑면 '모터 고정판' 자리에 대고 구멍 표시 → 나사 또는 순간접착제. 키트 바퀴(1) 끼움.
3. 캐스터(2): 서포트(14·15) 로 띄워 아래판 밑면 뒤쪽에. 평평한 책상에 올려 아래판이 수평(판 밑면 높이 42mm)이 되게 너트로 높이 조절.
4. 아래판 위: 18650 홀더(6) 양면테이프, L298N(3) 나사 4개, MP1584(9) 양면테이프. 모터선은 Ø8 구멍으로 올려 L298N OUT 단자에.
5. ★ MP1584 를 5.0V 로 맞춤: 배터리(7)만 연결하고 OUT 을 테스터(21)로 재며 가변저항을 돌림. 맞춘 뒤에 ESP32-CAM 연결.
6. 서포트 30mm 4개 + 윗판 + Uno (M3 나사·너트로 살짝 띄움). USB 가 뒤를 보게.
7. 카메라: 기둥 2장 붙임 → 아래판 앞쪽 기둥 자리 + 뒤 받침 → 쐐기 30° → 카메라판 → ESP32-CAM(4) 에 광각 렌즈(5) 끼워 고정. 렌즈 높이 150mm.
8. 방향지시등: 키트 노랑 LED 4개를 모서리 Ø5 구멍에 꽂고 220Ω 연결.
9. 배선: ②배선표 차례대로. 마지막에 GND 가 전부 하나로 이어졌는지 확인. L298N ENA·ENB 점퍼 캡 제거.
10. 시험: 바퀴를 들고 앞으로 명령 → 두 바퀴가 앞으로 도는지 (반대면 OUT 두 선 맞바꿈). 깜빡이 좌/우/비상 확인.

[코스]  (5.24 × 4.14 m, 조각 9개 = blueprint 조각1~9)
11. 조각 1개 = 검정 폼보드 60×90 5장: 세로로 3장(60×90) + 그 위에 가로로 2장(48cm 로 잘라). 뒷면 이음매는 박스테이프.
    앞면 이음매에 흰 속살이 보이면 차선으로 오인 → 검정 절연테이프로 덮음. 조각끼리도 같은 방법.
12. 도안(5_치수도·7_좌표표)으로 선 자리 표시 → 흰·노랑 절연테이프. 3.5cm 횡단보도·4cm 정지선은 19mm 테이프 2줄.
13. 경사로: 길이 45.5 + 30 + 45.5cm, 폭 68cm, 높이 6.75cm. 갈빗대(사다리꼴) 5개를 폼보드 2겹으로 → 상판 3장 → 위에 차선 테이프.
14. 신호등: ⑤ 페이지. 등 머리 4개, 기둥·팔 4개. Nano 에 road_signal.ino 업로드, 휴대폰 충전기로 전원.
15. 보행자 모형: 폼보드를 지름 5cm × 14cm 원통으로 말고 파랑 절연테이프로 감쌈 + 머리. 차 카메라는 '파랑'으로 사람을 인식.
※ 바닥이 흰색이면 뇌가 전부 차선으로 봄 → 바닥은 반드시 검정. 초록 잔디는 필요 없음 (밝은 초록은 신호 초록불로 오인).
※ 바닥 조명이 반사돼 하얗게 빛나지 않는지 차 카메라 화면으로 확인."""


def page_steps():
    fig = plt.figure(figsize=(16.5, 11.7))
    fig.text(.03, .95, "⑥ 조립 순서", fontsize=16, weight="bold", va="top")
    fig.text(.03, .91, STEPS, fontsize=10.5, va="top", linespacing=1.55)
    # MDF 클립보드 228×315 두 장 재단 배치 (위 40mm 는 집게 자리)
    ax = fig.add_axes([.70, .05, .29, .5])
    for ox in (0, 240):
        ax.add_patch(Rectangle((ox, 0), 228, 315, fc="#e8d5b9", ec="k"))
        ax.add_patch(Rectangle((ox, 275), 228, 40, fc="#bbbbbb", ec="k", hatch="//"))
        ax.text(ox + 114, 295, "집게 자리", ha="center", va="center", fontsize=7)
    for x, y, w, h, t in ((5, 5, F + R, W, "아래판"), (5, 145, 145, 100, "윗판"),
                          (245, 5, 20, MAST_H, "기둥1"), (270, 5, 20, MAST_H, "기둥2"),
                          (295, 5, 27, 26, "카메라판"), (295, 40, 60, 40, "쐐기·받침")):
        ax.add_patch(Rectangle((x, y), w, h, fc="#c29a6b", ec="k"))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8, rotation=90 if h > 2 * w else 0)
    ax.set(xlim=(-5, 473), ylim=(-5, 320), aspect="equal")
    ax.set_title("MDF 클립보드 228×315 (주문 16) 2장에서 자르기 (둘째 장 남는 곳은 실수 대비)", fontsize=10)
    return fig


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    pages = [("11_차량배선도", page_wiring), ("12_배선표", page_table), ("13_아래판_1대1", page_bottom),
             ("14_윗판_1대1", page_upper), ("15_신호등", page_signal), ("16_조립순서", page_steps)]
    with PdfPages(OUT / "조립설계도.pdf") as pdf:
        for name, page in pages:
            fig = page()
            pdf.savefig(fig)
            fig.savefig(OUT / f"{name}.png", dpi=80 if fig.get_figwidth() > 12 else 110)
            plt.close(fig)
    print(OUT / "조립설계도.pdf")

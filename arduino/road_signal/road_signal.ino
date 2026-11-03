// 도로용 Arduino (남는 Uno, Nano 도 그대로 됨): ④교차로 4색 신호등 4개 (sim.py SIGNAL_PHASES 와 같은 순서·시간)
// 등 하나 = 빨강·노랑·초록 좌회전화살표·초록 LED 4개. 마주 보는 두 등(동서끼리, 남북끼리)은 같은 핀에 병렬.
// LED마다 220옴 저항 따로. 핀 하나에 LED 2개 = 약 20mA (핀 한도 40mA 안).
// 화살표: 초록 LED 위에 ← 모양으로 구멍 낸 검은 종이/테이프를 붙임.

//                 빨강 노랑 화살표 초록
const int EW[4] = {2, 3, 4, 5};    // 동서 방향 진행 차로를 보는 등 2개
const int NS[4] = {6, 7, 8, 9};    // 남북 방향 진행 차로를 보는 등 2개
enum { R = 1, Y = 2, A = 4, G = 8 };

struct Phase { unsigned long ms; byte ew, ns; };
const Phase PHASES[] = {
  {6000, G, R},          // 동서 직진
  {2000, Y, R},
  {5000, R | A, R},      // 동서 좌회전
  {2000, Y, R},
  {6000, R, G},          // 남북 직진
  {2000, R, Y},
  {5000, R, R | A},      // 남북 좌회전
  {2000, R, Y},
};

void show(const int pins[4], byte lamps) {
  for (int i = 0; i < 4; i++) digitalWrite(pins[i], lamps >> i & 1);
}

void setup() {
  for (int i = 0; i < 4; i++) { pinMode(EW[i], OUTPUT); pinMode(NS[i], OUTPUT); }
}

void loop() {
  for (const Phase &p : PHASES) {
    show(EW, p.ew);
    show(NS, p.ns);
    delay(p.ms);
  }
}

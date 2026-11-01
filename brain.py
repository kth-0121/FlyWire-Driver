"""FlyWire 전체 뇌 LIF 모델 (Shiu et al. 2024, Nature 파라미터).

뉴런 138,639개, 시냅스 연결 1,509만 개, 가중치 = 시냅스 수 x 흥분(+1)/억제(-1) x W_SYN.
brain = Brain(load_weights()); spikes = brain.step(ext_input)
"""
import time
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA = Path(__file__).parent / "data"
SRC = "https://github.com/philshiu/Drosophila_brain_model/raw/main/"
FILES = ("Completeness_783.csv", "Connectivity_783.parquet")

# Shiu et al. 2024 파라미터 (mV, ms)
V_0, V_RESET, V_TH = -52.0, -52.0, -45.0
T_MBR, TAU, T_REF, DELAY = 20.0, 5.0, 2.2, 1.8
W_SYN = 0.275
W_POI = 250 * W_SYN      # 외부 포아송 입력 1회의 세기
DT = 0.1


def load_annotations():
    """뉴런 인덱스 순서에 맞춘 FlyWire 분류표 (Schlegel et al. 2024): super_class, side, pos_x/y/z 등."""
    f = DATA / "neuron_annotations.tsv"
    if not f.exists():
        urllib.request.urlretrieve("https://github.com/flyconnectome/flywire_annotations/raw/main/"
                                   "supplemental_files/Supplemental_file1_neuron_annotations.tsv", f)
    ids = pd.read_csv(DATA / FILES[0], index_col=0).index
    a = pd.read_csv(f, sep="\t", low_memory=False,
                    usecols=["root_id", "super_class", "cell_type", "side", "pos_x", "pos_y", "pos_z"])
    return a.drop_duplicates("root_id").set_index("root_id").reindex(ids).reset_index()


def load_weights():
    """(pre, post) CSR 가중치 행렬 [mV]. 데이터 없으면 내려받음."""
    DATA.mkdir(exist_ok=True)
    for f in FILES:
        if not (DATA / f).exists():
            urllib.request.urlretrieve(SRC + f, DATA / f)
    n = len(pd.read_csv(DATA / FILES[0]))
    df = pd.read_parquet(DATA / FILES[1], columns=["Presynaptic_Index", "Postsynaptic_Index",
                                                   "Excitatory x Connectivity"])
    w = (df["Excitatory x Connectivity"].to_numpy() * W_SYN).astype(np.float32)
    return sp.csr_matrix((w, (df["Presynaptic_Index"], df["Postsynaptic_Index"])), shape=(n, n))


class Brain:
    # 전부 제자리(in-place) 연산: Windows에선 13만 원소 배열을 매 스텝 새로 할당하는 게 연산보다 10배 느림
    def __init__(self, W, dt=DT):
        self.W, self.dt, self.n = W, dt, W.shape[0]
        f32 = dict(dtype=np.float32)
        self.v = np.full(self.n, V_0, **f32)
        self.g = np.zeros(self.n, **f32)
        self.ref = np.zeros(self.n, **f32)
        self.tmp = np.empty(self.n, **f32)
        self.free = np.empty(self.n, dtype=bool)
        empty = np.array([], dtype=np.int64)
        self.queue = deque([empty] * round(DELAY / dt))   # 시냅스 지연 버퍼

    def step(self, ext=None):
        """dt 한 스텝. ext: (뉴런 인덱스 배열) 이번 스텝 외부 입력 받는 뉴런. 반환: 발화한 뉴런 인덱스."""
        v, g, tmp = self.v, self.g, self.tmp
        arrived = self.queue.popleft()
        if len(arrived):
            # 발화 뉴런 행만 모아 더함: 활동이 희소할수록 빠름
            rows = self.W[arrived]
            np.add.at(g, rows.indices, rows.data)
        if ext is not None:
            g[ext] += W_POI
        np.less_equal(self.ref, 0, out=self.free)
        np.subtract(v, V_0, out=tmp)                      # dv = dt/T_MBR * (g - (v - V_0)), 불응기면 0
        np.subtract(g, tmp, out=tmp)
        tmp *= self.dt / T_MBR
        tmp *= self.free
        v += tmp
        g *= 1 - self.dt / TAU
        self.ref -= self.dt
        spk = np.flatnonzero(v >= V_TH)
        self.v[spk] = V_RESET
        self.ref[spk] = T_REF
        self.queue.append(spk)
        return spk


if __name__ == "__main__":
    t0 = time.perf_counter()
    W = load_weights()
    print(f"로드: {time.perf_counter() - t0:.1f}s, 뉴런 {W.shape[0]:,}, 연결 {W.nnz:,}")

    rng = np.random.default_rng(0)
    for n_in, rate in [(0, 0), (100, 150), (2000, 150)]:     # 입력 뉴런 수, 포아송 Hz
        brain = Brain(W)
        inputs = rng.choice(brain.n, n_in, replace=False)
        p = rate * DT / 1000
        steps, total_spk = 2000, 0                            # 200ms 생체 시간
        t0 = time.perf_counter()
        for _ in range(steps):
            total_spk += len(brain.step(inputs[rng.random(n_in) < p]))
        el = time.perf_counter() - t0
        print(f"입력 {n_in:>4}개 @{rate}Hz: {el / steps * 1e6:6.0f} us/step, "
              f"실시간 대비 x{steps * DT / 1000 / el:.2f}, 스텝당 발화 {total_spk / steps:.1f}개")
        if n_in:
            assert total_spk > 0, "입력이 있는데 아무도 발화 안 함"

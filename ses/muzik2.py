"""Müzik, sürüm 2: canlı, akılda kalan, modern. 100 BPM, Fa majör, çift zamanlı ritim hissi.
Yapı (ölçü = 2,4 sn): 0 kanca + şıklatma · 1 kick girer · 3 (7.20) tam ritim · 5 (12.00) yükseliş ·
6–7 (14.40) etkinlik: ıslık ezgisi · 8 (19.20) söz: sadeleşir · NEFES sessizlik · LOGO_VURUS dolu akor."""
import numpy as np
from scipy import signal
import zaman as Z
from sentez import SR, N, hz, tt, zarf, suz, gurultu, koy, yanki, tus, pad, bas, kick, rim, shaker, can, supurge

rng = np.random.default_rng(21)
V, O = Z.VURUS, Z.OLCU
A16 = V / 4                                   # on altılık = 0,15 sn


def cimbir(m, vel=0.6, dur=0.5):
    """Marimba/kalimba arası tok, parlak çimdik: kanca ezgisi."""
    t = tt(dur)
    f = hz(m)
    y = np.sin(2 * np.pi * f * t + 2.4 * vel * np.exp(-t * 30) * np.sin(2 * np.pi * f * 3.98 * t))
    y += 0.25 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 16)
    return y * np.exp(-t * 8.5) * zarf(len(t), 0.002, 0.05) * vel


def alkis(vel=0.6):
    t = tt(0.16)
    y = np.zeros(len(t))
    for d in (0.0, 0.011, 0.023):
        i = int(d * SR)
        y[i:] += gurultu(0.16)[: len(t) - i] * np.exp(-t[: len(t) - i] * 60)
    return suz(y, [900, 4200], "band") * vel


def siklatma(vel=0.5):
    t = tt(0.07)
    return (suz(gurultu(0.07), [1800, 5200], "band") * np.exp(-t * 110) + 0.3 * np.sin(2 * np.pi * 2300 * t) * np.exp(-t * 160)) * vel


def acik_zil(vel=0.3):
    t = tt(0.14)
    return suz(gurultu(0.14), 7200, "high") * np.exp(-t * 26) * zarf(len(t), 0.003, 0.03) * vel


def islik(m, dur, vel=0.4):
    """Islık benzeri yumuşak ezgi sesi (etkinlik sahnesi)."""
    t = tt(dur)
    f = hz(m) * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(1, t * 6))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.12 * np.sin(4 * np.pi * np.cumsum(f) / SR)
    y += 0.05 * suz(gurultu(dur), [hz(m) * 0.9, hz(m) * 1.3], "band")[: len(t)]
    return y * zarf(len(t), 0.03, 0.08) * vel


# ölçü: (bas notası, pad, kanca için ölçek notaları)
AKOR = {
    0: (41, [53, 57, 60, 67]), 1: (38, [50, 53, 57, 60]), 2: (34, [46, 50, 53, 57, 60]), 3: (36, [48, 52, 55, 62]),
    4: (45, [52, 57, 60, 64]), 5: (34, [46, 53, 57, 62, 65]), 6: (38, [50, 57, 60, 65]), 7: (36, [48, 55, 60, 65]),
}
# Kanca: 16 adımlık desen (None = sus). İki ölçülük soru-cevap; akılda kalsın diye hep aynı ritim.
K1 = [69, None, 72, None, 74, 72, None, 69, None, 72, None, 77, None, 74, 72, None]
K2 = [69, None, 72, None, 74, 72, None, 77, None, 76, None, 72, None, 74, None, 72]
K3 = [70, None, 74, None, 77, 74, None, 70, None, 74, None, 79, None, 77, 74, None]
K4 = [67, None, 72, None, 76, 72, None, 79, None, 76, None, 72, None, 74, 76, None]
KANCA = {0: K1, 1: K2, 2: K3, 3: K4, 4: K1, 5: K3, 6: K2, 7: K4}
ISLIK = [(0, 81, 1.5), (1.5, 84, 0.5), (2, 86, 1), (3, 84, 1), (4, 81, 1.5), (5.5, 79, 0.5), (6, 77, 1.5),     # ölçü 6–7
         (8, 79, 1), (9, 81, 1), (10, 84, 1.5), (11.5, 86, 0.5), (12, 88, 1), (13, 86, 1), (14, 84, 2)]


def muzik():
    kuru = np.zeros((N, 2)); davul = np.zeros((N, 2)); alt = np.zeros((N, 2)); yastik = np.zeros((N, 2))
    kickler = []
    for b in range(8):
        t0 = b * O
        bm, pd = AKOR[b]
        # yastık (pad): alçak, sıcak
        p = pad(pd, O, 0.8 if b < 5 else 1.0) * (0.19 if b < 3 else 0.19)
        i = int(t0 * SR); yastik[i:i + len(p)] += p[: N - i]
        # kanca
        guc = 0.56 + (0.04 if b >= 3 else 0) + (0.05 if b >= 5 else 0)
        for k, m in enumerate(KANCA[b]):
            if m is None:
                continue
            sal = 0.012 if k % 2 else 0.0                                   # hafif sallantı (swing)
            koy(kuru, cimbir(m, guc * (1.0 if k % 4 == 0 else 0.8) * rng.uniform(0.92, 1.05)), t0 + k * A16 + sal, 0.5, pan=0.3 * np.sin(k * 0.9))
            if b >= 5:                                                      # yükselişte bir oktav üstten çiftlenir
                koy(kuru, cimbir(m + 12, guc * 0.42), t0 + k * A16 + sal, 0.5, pan=-0.3 * np.sin(k * 0.9))
        # akor vuruşları: tersten (ve'lerde) parlak tuş
        if b >= 0:
            for o in ([1.5, 3.5] if b < 3 else [0.5, 1.5, 2.5, 3.5]):
                for j, m in enumerate(pd[-3:]):
                    koy(kuru, tus(m + 12, 0.16, 0.30 if b < 3 else 0.36), t0 + o * V + j * 0.004, 0.42, pan=(j - 1) * 0.35)
        # bas: zıplayan, senkoplu
        if b >= 0:
            desen = [(0, bm, 0.9)] if b < 3 else [(0, bm, 0.55), (0.75, bm, 0.2), (1.5, bm + 12, 0.3), (2, bm, 0.55), (2.75, bm + 7, 0.2), (3.5, bm + 12, 0.3)]
            for o, m, sv in desen:
                koy(alt, bas(m, sv * V * 1.6, 0.62), t0 + o * V, 0.6)
        # ritim
        for k in range(16):
            tk = t0 + k * A16
            sal = 0.012 if k % 2 else 0.0
            if k in (4, 12):                                                # 2 ve 4: baştan itibaren şıklatma, sonra alkış
                koy(davul, siklatma(0.5), tk, 0.5, pan=0.2)
                if b >= 3:
                    koy(davul, alkis(0.42 if b < 5 else 0.5), tk, 0.5, pan=-0.1)
            if (k in (0, 8) or (b >= 3 and k in (7, 10)) or (b >= 5 and k == 14)):
                koy(davul, kick(0.85 if k in (0, 8) else 0.6), tk, 0.5); kickler.append(tk)
            if b >= 3:
                koy(davul, shaker(0.16 if k % 2 else 0.26, 0.04), tk + sal, 0.5, pan=-0.35)
                if k % 4 == 2:
                    koy(davul, acik_zil(0.22 if b < 5 else 0.28), tk, 0.5, pan=0.35)
            elif k % 2 == 0:
                koy(davul, shaker(0.18, 0.05), tk, 0.5, pan=-0.3)
            if b >= 5 and k % 2 == 1:
                koy(davul, shaker(0.12, 0.03), tk + sal, 0.5, pan=0.4)       # tef hissi
        if b in (2, 4, 7):                                                  # ölçü sonu küçük geçişler
            for j, k in enumerate((13, 14, 15)):
                koy(davul, rim(0.35 + 0.08 * j), t0 + k * A16, 0.5, pan=-0.2 + 0.2 * j)
    # etkinlik: ıslık ezgisi (ölçü 6–7)
    for vb, m, sv in ISLIK:
        koy(kuru, islik(m, sv * V * 0.95, 0.20), 6 * O + vb * V, 1, pan=0.1)
    # yükseliş ve geçiş süpürmeleri
    koy(kuru, supurge(1.0, 500, 8000, 0.10), 3 * O - 1.0, 1, 0)
    koy(kuru, supurge(1.1, 500, 9000, 0.12), 5 * O - 1.1, 1, 0); koy(kuru, can(89, 0.16), 5 * O, 1, 0.3)
    koy(kuru, supurge(0.8, 700, 9000, 0.09), 6 * O - 0.8, 1, 0); koy(kuru, can(93, 0.12), 6 * O, 1, -0.3)
    # ölçü 8: söz — sadeleşir (yarı zamanlı), sıcak
    t8 = 8 * O
    for o, bm, pd in [(0, 34, [46, 53, 57, 62, 65]), (2, 36, [48, 55, 60, 64, 67]), (4, 38, [50, 57, 60, 65, 69])]:
        dur = min(2 * V, Z.NEFES - (t8 + o * V))
        if dur <= 0.1:
            continue
        p = pad(pd, max(0.2, dur - 0.4), 1.0) * 0.26
        i = int((t8 + o * V) * SR); yastik[i:i + len(p)] += p[: N - i]
        koy(alt, bas(bm, dur * 0.95, 0.6), t8 + o * V, 0.6)
        for j, m in enumerate(pd):
            koy(kuru, tus(m + 12, dur * 0.8, 0.40), t8 + o * V + j * 0.03, 0.42, pan=(j - 2) * 0.2)
        koy(davul, kick(0.7), t8 + o * V, 0.5); kickler.append(t8 + o * V)
        koy(davul, siklatma(0.45), t8 + (o + 1) * V, 0.5, pan=0.2)
    for k, m in enumerate([77, None, 81, None, 84, None, 81, None, 79, None, 84, None, 88, None, 86, None, 84, None, 81, None]):
        if m is not None and t8 + k * A16 * 2 < Z.NEFES - 0.2:
            koy(kuru, cimbir(m, 0.34), t8 + k * A16 * 2, 0.5, pan=0.25 * np.sin(k))
    koy(kuru, supurge(0.9, 900, 9000, 0.10), Z.NEFES - 0.9, 1, 0)
    # kick'e bağlı hafif pompalama: yastık ve bas canlı nefes alır
    pompa = np.ones(N)
    for tk in kickler:
        i = int(tk * SR); n = int(0.22 * SR)
        if i + n < N:
            pompa[i:i + n] = np.minimum(pompa[i:i + n], 0.55 + 0.45 * np.linspace(0, 1, n) ** 0.7)
    yastik *= pompa[:, None]
    kes = np.ones(N); i = int(Z.NEFES * SR); f = int(0.04 * SR)
    kes[i - f:i] = np.linspace(1, 0, f); kes[i:] = 0
    ana = (kuru * 0.95 + yastik + alt + davul * 0.95) * kes[:, None]
    # LOGO_VURUS: dolu sıcak akor, kapanışın altında çınlar
    son = np.zeros((N, 2)); tv = Z.LOGO_VURUS
    for j, m in enumerate([41, 53, 60, 65, 69, 72, 79]):
        koy(son, tus(m, 4.2, 0.62), tv + j * 0.012, 0.5, pan=(j - 3) * 0.14)
    for j, m in enumerate([77, 81, 84, 89]):
        koy(son, cimbir(m, 0.5, 1.2), tv + j * 0.06, 0.5, pan=(j - 1.5) * 0.3)
    p = pad([41, 53, 57, 60, 67, 72], 2.9, 1.0) * 0.50
    i = int(tv * SR); son[i:i + len(p)] += p[: N - i]
    koy(son, bas(29, 3.0, 0.8), tv, 0.6); koy(son, kick(0.8), tv, 0.5); koy(son, alkis(0.3), tv, 0.5)
    koy(son, suz(gurultu(1.6), 6000, "high") * np.exp(-tt(1.6) * 3.2) * 0.10, tv, 1, 0)       # yumuşak zil
    koy(son, can(84, 0.22, 3.0), tv, 1, 0.25); koy(son, can(89, 0.13, 3.0), tv + 0.3, 1, -0.25)
    for j, m in enumerate([72, 77, 81, 84]):                    # yazı açılırken küçük yükselen süs
        koy(son, cimbir(m, 0.36, 0.8), Z.YAZI_ACILIS + 0.5 + j * A16, 0.5, pan=(j - 1.5) * 0.3)
    for j, m in enumerate([53, 60, 67, 72, 77]):                # kapanış kartında akorun yumuşak yankısı
        koy(son, tus(m, 1.8, 0.36), Z.KAPANIS_SLOGAN + j * 0.03, 0.5, pan=(j - 2) * 0.2)
    m = yanki(ana, 1.6, 0.16) + yanki(son, 3.0, 0.30)
    return signal.sosfilt(signal.butter(2, 28, "high", fs=SR, output="sos"), m, axis=0)

"""Özgün müzik ve ses efektleri: tamamı kodla sentezlenir (örnek dosyası yok)."""
import numpy as np
from scipy import signal
import zaman as Z

SR = 48000
N = int(Z.SURE * SR)
rng = np.random.default_rng(7)


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tt(dur):
    return np.arange(int(dur * SR)) / SR


def zarf(n, a=0.005, r=0.05):
    e = np.ones(n)
    na, nr = max(1, int(a * SR)), max(1, int(r * SR))
    e[:na] = np.linspace(0, 1, na) ** 0.7
    e[-nr:] *= np.linspace(1, 0, nr) ** 1.5
    return e


def suz(x, f, tur="low", q=2):
    if np.ndim(f) == 0:
        f = min(f, SR * 0.45)
    sos = signal.butter(q, f, tur, fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def gurultu(dur):
    return rng.standard_normal(int(dur * SR))


def koy(buf, x, t, kazanc=1.0, pan=0.0):
    """Mono x'i stereo buf'a t anında, eşit güçlü pan ile ekler."""
    i = int(round(t * SR))
    if i >= len(buf) or i < 0:
        return
    x = x[: len(buf) - i]
    a = (pan + 1) * np.pi / 4
    buf[i:i + len(x), 0] += x * np.cos(a) * kazanc
    buf[i:i + len(x), 1] += x * np.sin(a) * kazanc


def yanki(buf, sure=2.2, islak=0.22, parlak=5200):
    """Sentez oda: üstel sönümlü gürültü ile evrişim."""
    n = int(sure * SR)
    out = buf.copy()
    for c in range(2):
        ir = np.random.default_rng(11 + c).standard_normal(n) * np.exp(-np.arange(n) / (sure * SR / 6.9))
        ir[: int(0.012 * SR)] = 0
        ir = suz(suz(ir, parlak), 180, "high")
        ir /= np.sqrt(np.sum(ir ** 2))
        out[:, c] += islak * signal.fftconvolve(buf[:, c], ir)[: len(buf)]
    return out


# ---------------------------------------------------------------- çalgılar
def tus(m, dur, vel=0.6):
    """Yumuşak elektrik piyano: FM, hızlı sönen parlaklık."""
    t = tt(dur + 0.6)
    f = hz(m)
    ind = (0.9 + 1.6 * vel) * np.exp(-t * 7)
    y = np.sin(2 * np.pi * f * t + ind * np.sin(2 * np.pi * f * t))
    y += 0.18 * vel * np.sin(2 * np.pi * f * 4 * t) * np.exp(-t * 14)
    e = np.exp(-t / (0.9 + 60 / f)) * zarf(len(t), 0.004, 0.25)
    return y * e * vel


def pad(notalar, dur, parlak=1.0):
    t = tt(dur + 1.2)
    L = np.zeros(len(t)); R = np.zeros(len(t))
    for m in notalar:
        f = hz(m)
        for h in range(1, 7):
            g = parlak ** (h - 1) / h ** 1.9
            ph = rng.uniform(0, 6.28, 2)
            L += g * np.sin(2 * np.pi * f * h * 2 ** (-4 / 1200) * t + ph[0])
            R += g * np.sin(2 * np.pi * f * h * 2 ** (4 / 1200) * t + ph[1])
    e = zarf(len(t), 0.7, 1.2) * (1 + 0.06 * np.sin(2 * np.pi * 0.21 * t))
    return np.stack([L * e, R * e], 1) / len(notalar)


def bas(m, dur, vel=0.7):
    t = tt(dur)
    f = hz(m)
    y = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) * np.exp(-t * 5)
    return y * zarf(len(t), 0.012, 0.09) * np.exp(-t * 0.9) * vel


def kick(vel=1.0):
    t = tt(0.32)
    f = 46 + 70 * np.exp(-t * 32)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 13) * vel


def rim(vel=1.0):
    t = tt(0.12)
    y = suz(gurultu(0.12), [900, 3800], "band") * np.exp(-t * 55)
    y += 0.5 * np.sin(2 * np.pi * 1750 * t) * np.exp(-t * 90)
    return y * vel


def shaker(vel=1.0, uzun=0.07):
    t = tt(uzun)
    return suz(gurultu(uzun), 6500, "high") * np.exp(-t / (uzun / 4)) * zarf(len(t), 0.004, 0.02) * vel


def can(m, vel=0.5, dur=2.5):
    t = tt(dur)
    f = hz(m)
    y = sum(g * np.sin(2 * np.pi * f * k * t) * np.exp(-t * d) for k, g, d in [(1, 1, 2.2), (2.76, 0.35, 5), (5.4, 0.15, 9)])
    return y * zarf(len(t), 0.002, 0.3) * vel


def supurge(dur, f0, f1, vel=1.0, q=1.2):
    """Hışırtı: merkez frekansı kayan bant geçiren gürültü."""
    n = int(dur * SR)
    x = gurultu(dur)
    f = np.geomspace(f0, f1, n)
    out = np.zeros(n); blok = 256
    zi = None
    for i in range(0, n, blok):
        fc = f[i]
        sos = signal.butter(1, [fc / (1 + 0.5 / q), min(fc * (1 + 0.5 / q), SR * 0.45)], "band", fs=SR, output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        out[i:i + blok], zi = signal.sosfilt(sos, x[i:i + blok], zi=zi)
    return out * np.sin(np.linspace(0, np.pi, n)) ** 1.5 * vel


# ---------------------------------------------------------------- müzik
AKOR = [  # (ölçü, bas, pad, arpej)
    (0, 41, [53, 57, 60, 67], [53, 60, 65, 69, 72]),          # Fa(9)
    (1, 38, [50, 53, 57, 60], [50, 57, 62, 65, 69]),          # Re m7
    (2, 34, [46, 50, 53, 57, 60], [58, 62, 65, 69, 74]),      # Si♭ maj7
    (3, 36, [48, 52, 55, 62], [55, 60, 64, 67, 72]),          # Do
    (4, 38, [50, 53, 57, 60, 64], [57, 62, 65, 69, 74]),      # Re m9
    (5, 34, [46, 53, 57, 62, 65], [58, 65, 69, 74, 77]),      # Si♭ maj7 (yükseliş)
]
EZGI = [  # (zaman vuruş cinsinden, nota, süre vuruş, şiddet)
    (0, 72, 2, .5), (2, 69, 2, .42), (4, 74, 2, .5), (6, 77, 2, .45),
    (8, 74, 1.5, .5), (9.5, 70, 2.5, .42), (12, 72, 1, .5), (13, 76, 1, .52), (14, 79, 2, .55),
    (16, 81, 2, .55), (18, 77, 2, .5),
    (20, 82, 1, .62), (21, 81, 1, .6), (22, 77, 1, .58), (23, 74, 1, .56),
    (24, 79, 1, .6), (25, 82, 1, .62), (26, 84, 1.4, .66),
]


def muzik():
    kuru = np.zeros((N, 2)); davul = np.zeros((N, 2)); alt = np.zeros((N, 2))
    V, O = Z.VURUS, Z.OLCU
    for b, bm, pd, arp in AKOR:
        t0 = b * O
        guc = 1.0 + (0.25 if b >= 5 else 0)
        p = pad(pd, O, 0.75 if b < 5 else 0.95) * 0.20 * guc
        i = int(t0 * SR); kuru[i:i + len(p)] += p[: N - i]
        desen = [0, 2, 1, 2, 3, 2, 4, 2] if b >= 5 else [0, 2, 1, 2, 3, 2, 1, 2]
        for k, d in enumerate(desen):
            vel = (0.34 if k % 2 else 0.44) * guc * rng.uniform(0.9, 1.05)
            koy(kuru, tus(arp[d], V * 0.9, vel), t0 + k * V / 2 + rng.uniform(0, 0.006), 0.5, pan=0.35 * np.sin(k * 1.3))
        if b >= 3:
            for k, (o, m) in enumerate([(0, bm), (1.5, bm), (2, bm + 12 if b == 5 else bm), (3.5, bm + 7)]):
                koy(alt, bas(m, V * (1.3 if o in (0, 2) else 0.45), 0.6), t0 + o * V, 0.55)
    # ölçü 6: Sol m7 -> Do7sus, 16.50'de susar
    t6 = 6 * O
    for o, bm, pd, arp in [(0, 43, [55, 58, 62, 65], [55, 62, 65, 70, 74]), (2, 36, [48, 53, 58, 62], [60, 65, 70, 72, 77])]:
        dur = 2 * V if o == 0 else Z.NEFES - (t6 + 2 * V)
        p = pad(pd, dur - 0.5, 0.95) * 0.25
        i = int((t6 + o * V) * SR); kuru[i:i + len(p)] += p[: N - i]
        koy(alt, bas(bm, dur * 0.9, 0.6), t6 + o * V, 0.55)
        for k, d in enumerate([0, 2, 1, 3] if o == 0 else [0, 2, 3]):
            koy(kuru, tus(arp[d], V * 0.9, 0.46), t6 + o * V + k * V / 2, 0.5, pan=0.3 * np.sin(k * 1.3))
    for vb, m, sv, vel in EZGI:
        koy(kuru, tus(m, sv * V, vel), vb * V, 0.62, pan=-0.12)
    # ritim: 7.20'den nefese kadar
    t = Z.SAHNE["S4"]; k = 0
    while t < Z.NEFES - 0.05:
        vurus = k % 8
        yuk = t >= Z.EL_DEGISTIRME
        if vurus in (0, 4) or (yuk and vurus == 3):
            koy(davul, kick(0.9 if vurus == 0 else 0.75), t, 0.5)
        if vurus in (2, 6):
            koy(davul, rim(0.5), t, 0.5, pan=0.15)
        koy(davul, shaker(0.20 if vurus % 2 else 0.32), t + 0.004, 0.5, pan=-0.3)
        if yuk:
            koy(davul, shaker(0.13, 0.05), t + Z.VURUS / 4, 0.5, pan=0.3)
        t += Z.VURUS / 2; k += 1
    # yükselişe hazırlık ve nefese giriş
    koy(kuru, supurge(1.1, 500, 7000, 0.10), Z.EL_DEGISTIRME - 1.1, 1, 0)
    koy(kuru, can(89, 0.16), Z.EL_DEGISTIRME, 1, 0.3)
    koy(kuru, supurge(0.9, 900, 9000, 0.09), Z.NEFES - 0.9, 1, 0)
    # müzik NEFES'te susar: kuru sinyal kısa sönümle kesilir, yankı kuyruğu kalır
    kes = np.ones(N); i = int(Z.NEFES * SR); f = int(0.05 * SR)
    kes[i - f:i] = np.linspace(1, 0, f); kes[i:] = 0
    ana = (kuru + alt + davul * 0.8) * kes[:, None]
    # 16.80: dolu sıcak akor, kapanışın altında çınlar
    son = np.zeros((N, 2)); tv = Z.LOGO_VURUS
    for j, m in enumerate([41, 53, 60, 65, 69, 72, 79]):
        koy(son, tus(m, 4.2, 0.62), tv + j * 0.012, 0.5, pan=(j - 3) * 0.14)
    p = pad([41, 53, 57, 60, 67, 72], 3.6, 1.0) * 0.50
    i = int(tv * SR); son[i:i + len(p)] += p[: N - i]
    koy(son, bas(29, 3.6, 0.8), tv, 0.6); koy(son, kick(0.7), tv, 0.5)
    koy(son, can(84, 0.22, 3.5), tv, 1, 0.25); koy(son, can(89, 0.13, 3.5), tv + 0.3, 1, -0.25)
    koy(son, tus(81, 2, 0.4), Z.YAZI_ACILIS + 0.9, 0.5, 0.2); koy(son, tus(84, 2.5, 0.38), Z.KAPANIS_SLOGAN, 0.5, -0.2)
    for j, m in enumerate([53, 60, 67, 72, 77]):      # kapanış kartında akorun yumuşak yankısı
        koy(son, tus(m, 2.4, 0.36), Z.KAPANIS_SLOGAN + j * 0.03, 0.5, pan=(j - 2) * 0.2)
    m = yanki(ana, 2.4, 0.26) + yanki(son, 3.2, 0.30)
    return signal.sosfilt(signal.butter(2, 28, "high", fs=SR, output="sos"), m, axis=0)


# ---------------------------------------------------------------- efektler
def tik(f=2400, vel=1.0):
    t = tt(0.03)
    return (suz(gurultu(0.03), [f * 0.6, f * 1.6], "band") * np.exp(-t * 260) + 0.4 * np.sin(2 * np.pi * f * 0.5 * t) * np.exp(-t * 300)) * vel


def pin_sesi(vel=1.0, ton=52):
    t = tt(0.5)
    y = np.sin(2 * np.pi * np.cumsum(hz(ton) * (1 + 1.2 * np.exp(-t * 40))) / SR) * np.exp(-t * 11)
    y += 0.5 * tus(ton + 24, 0.3, 0.5)[: len(t)] + 0.25 * np.pad(tik(1800), (0, len(t) - int(0.03 * SR)))
    return y * vel


def dokunus():
    t = tt(0.09)
    return np.sin(2 * np.pi * np.cumsum(190 + 260 * np.exp(-t * 70)) / SR) * np.exp(-t * 45) + 0.5 * np.pad(tik(3000), (0, len(t) - int(0.03 * SR)))


def kagit(dur, vel=1.0):
    """Kâğıt çanta: rastgele kısa çıtırtı taneleri."""
    n = int(dur * SR); y = np.zeros(n)
    for _ in range(int(dur * 70)):
        i = rng.integers(0, n - 600); L = rng.integers(120, 600)
        y[i:i + L] += rng.standard_normal(L) * np.exp(-np.arange(L) / (L / 5)) * rng.uniform(0.2, 1)
    return suz(y, [1500, 9000], "band") * np.sin(np.linspace(0, np.pi, n)) ** 0.6 * vel


def efektler():
    b = np.zeros((N, 2))
    # S1: süzülüş, pin vuruşu, halka
    koy(b, supurge(0.30, 5000, 700, 0.32), 0.0, 1, -0.1)
    koy(b, pin_sesi(0.9), Z.PIN_VURUS, 1, 0)
    koy(b, supurge(0.6, 400, 2200, 0.10), Z.PIN_VURUS + 0.02, 1, 0)
    koy(b, supurge(0.55, 700, 3600, 0.20), 1.25, 1, 0.1)
    # S2: telefon yükselir, odak çizgisi, harfler, kartlar
    koy(b, supurge(0.7, 260, 1500, 0.24), 1.70, 1, 0)
    koy(b, can(77, 0.10, 0.8), Z.ODAK_CIZGI, 1, 0)
    ns = len(Z.YAZI)
    for i, ch in enumerate(Z.YAZI):
        if ch != " ":
            koy(b, tik(rng.uniform(2100, 3100), rng.uniform(0.28, 0.42)), Z.YAZI_BAS + i * (Z.YAZI_SON - Z.YAZI_BAS) / ns, 1, rng.uniform(-0.15, 0.15))
    for i, tk in enumerate(Z.KARTLAR):
        koy(b, supurge(0.30, 900 + 250 * i, 4200 + 600 * i, 0.26), tk, 1, (-0.5, 0.0, 0.5)[i])
        koy(b, tus(72 + (0, 2, 5)[i], 0.25, 0.22), tk + 0.26, 1, (-0.5, 0.0, 0.5)[i])
    koy(b, supurge(0.38, 3000, 800, 0.18), Z.GRUP_GECIS - 0.1, 1, 0)
    # S3: dolgu, dokunuş, başarı, fırlama, kamera geçişi
    d = Z.DOLGU_SON - Z.DOLGU_BAS; t = tt(d)
    dol = np.sin(2 * np.pi * np.cumsum(np.geomspace(330, 700, len(t))) / SR) * zarf(len(t), 0.05, 0.06) * 0.10
    koy(b, dol + supurge(d, 800, 5200, 0.12)[: len(t)], Z.DOLGU_BAS, 1, 0)
    koy(b, dokunus() * 0.8, Z.DOKUNUS, 1, 0)
    koy(b, can(84, 0.22, 1.4), Z.DOKUNUS + 0.07, 1, -0.1)
    koy(b, can(89, 0.22, 1.8), Z.DOKUNUS + 0.19, 1, 0.1)
    koy(b, supurge(0.45, 600, 6000, 0.28), Z.FIRLAMA - 0.05, 1, 0)
    koy(b, supurge(0.62, 5000, 220, 0.42, q=0.8), Z.KAMERA_GECIS - 0.08, 1, 0)
    # S4: evler, rota çizimi, yeşil pin, cam kart
    for i, te in enumerate(Z.EVLER):
        koy(b, tus([65, 69, 72, 74, 77, 81][i], 0.2, 0.2), te, 1, -0.6 + 0.24 * i)
        koy(b, supurge(0.14, 500, 1600, 0.10), te - 0.05, 1, -0.6 + 0.24 * i)
    r = Z.ROTA_SON - Z.ROTA_BAS; n = int(r * SR)
    ciz = suz(gurultu(r), [2400, 5200], "band") * (0.6 + 0.4 * np.sin(2 * np.pi * 9 * tt(r))) * zarf(n, 0.15, 0.2) * 0.045
    pan = np.linspace(-0.6, 0.6, n) * np.pi / 4 + np.pi / 4
    i = int(Z.ROTA_BAS * SR); b[i:i + n, 0] += ciz * np.cos(pan); b[i:i + n, 1] += ciz * np.sin(pan)
    koy(b, pin_sesi(0.7, 57), Z.YESIL_PIN, 1, 0.4)
    koy(b, supurge(0.26, 1200, 4800, 0.16), Z.CAM_KART, 1, 0.3)
    # S5: çanta, kurdele, el değiştirme, bildirim
    for i, tk in enumerate(Z.KATLAMA):                         # tulumun katlanışı: yumuşak kumaş hışırtısı
        koy(b, supurge(0.16, 700 + 150 * i, 2600, 0.13, q=0.9), tk, 1, (-0.2, 0.2, 0.0)[i])
    koy(b, kagit(0.30, 0.24), Z.KATLAMA[-1] + 0.08, 1, 0)
    for i, ta in enumerate(Z.ADIMLAR):                         # yürüyüş: yumuşak adımlar + kâğıt çıtırtısı
        koy(b, suz(kick(0.30), 400), ta, 1, (-0.25, 0.25, -0.1)[i])
        koy(b, kagit(0.12, 0.10), ta + 0.03, 1, 0)
    koy(b, supurge(Z.KURDELE_SON - Z.KURDELE_BAS, 1800, 7500, 0.20, q=2), Z.KURDELE_BAS, 1, 0)
    koy(b, tus(81, 0.3, 0.25), Z.KURDELE_SON, 1, 0)
    koy(b, kagit(0.42, 0.30), Z.EL_DEGISTIRME - 0.10, 1, 0)
    koy(b, supurge(0.3, 500, 2000, 0.14), Z.EL_DEGISTIRME - 0.12, 1, 0)
    koy(b, can(88, 0.20, 1.0), Z.BILDIRIM, 1, 0.3)
    koy(b, can(93, 0.20, 1.4), Z.BILDIRIM + 0.12, 1, 0.3)
    koy(b, supurge(0.5, 2600, 500, 0.16), Z.S6_GECIS, 1, 0)
    # S7-S8: yeşil pinin yaslanışı, yazı açılışı, kapanış
    koy(b, pin_sesi(0.45, 60), Z.YESIL_YASLAN, 1, -0.3)
    koy(b, supurge(0.5, 900, 3800, 0.10), Z.YAZI_ACILIS - 0.05, 1, 0.2)
    koy(b, supurge(0.4, 700, 2600, 0.08), Z.KAPANIS_SLOGAN, 1, 0)
    koy(b, tik(2600, 0.35), Z.KAPANIS_BUTON, 1, 0)
    return yanki(b, 1.2, 0.12, 7000)


def soz_tiklari(buf, anlar):
    """S6: her kelime otururken yumuşak tık (anlar seslendirmeden ölçülür)."""
    for i, t in enumerate(anlar):
        koy(buf, tik(1500, 0.22), t, 1, (-0.2, 0.2)[i % 2])

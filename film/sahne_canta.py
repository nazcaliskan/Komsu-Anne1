"""S5 · Elden ele (10.00–13.30). Kapı önü; kâğıt çanta, içinde katlanmış bebek tulumu.
Mercan çizgi çantayı bir kez sarıp fiyonklu kurdele olur; çanta 12.00'de el değiştirir;
bildirim iner; 12.90–13.30 arası sahne solar, kurdele çözülüp yeniden tek çizgi olur.

Devirler: 9.85–10.00 çizgi K.s5_giris'te (S4 ile aynı nokta); t=13.30 çizginin ucu K.s6_bas'ta."""
import functools
import numpy as np, cv2
from ortak import *  # noqa: F401,F403
from ortak import _yd_noktalar

ARALIK = (9.85, 13.3)

DUVAR = (243, 224, 205)
KAPI = (140, 167, 148)
KAPI_KOYU = (118, 147, 127)
PERVAZ = (252, 244, 234)
TAS = (234, 215, 197)
IP = (186, 146, 104)
IP_KOYU = (164, 126, 88)
KRAFT_KOYU = (196, 157, 113)
KRAFT_ACIK = (226, 191, 149)
GOZ = (58, 42, 34)

FIGUR = {
    "sol": dict(ten=(249, 220, 200), sac=(107, 62, 38), ust=(170, 154, 214), kol=(152, 135, 200), tip="uzun"),
    "sag": dict(ten=(227, 170, 130), sac=(196, 81, 58), ust=(127, 168, 142), kol=(110, 152, 126), tip="ortu"),
}


def _duzen(K):
    if K.dikey:
        return dict(s=1.0, cxL=235, cxR=845, sy=900, gL=(478, 1130), gR=(602, 1130),
                    kapi=(290, 360, 790, 1580), yer=1580, saksi=(945, 1580), lamba=(140, 540), plaka=(880, 700),
                    kart=(70, 170, 1010, 330))
    return dict(s=0.85, cxL=662, cxR=1258, sy=505, gL=(905, 700), gR=(1015, 700),
                kapi=(720, 70, 1200, 1400), yer=1400, saksi=(1640, 1060), lamba=(330, 300), plaka=(1300, 400),
                kart=(990, 64, 1850, 212))


def _elips(dst, cx, cy, rx, ry, renk, alfa=1.0, bulanik=0, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cokgen(dst, np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1), renk, alfa, bulanik=bulanik)


# ------------------------------------------------------------------ arka plan (değişmez, önbellekte)
PAY = 140


@functools.lru_cache(maxsize=2)
def _arka(dikey):
    K = DIKEY if dikey else YATAY
    D = _duzen(K)
    W, H = K.W + 2 * PAY, K.H + 2 * PAY
    im = np.empty((H, W, 3), np.uint8); im[:] = DUVAR
    o = PAY
    # duvarda hafif yatay kaplama çizgileri
    for y in range(-PAY, K.H + PAY, 118):
        cizgi_ince(im, np.array([(0, y + o), (W, y + o)]), (236, 215, 195), 3, 0.7)
    x0, y0, x1, y1 = D["kapi"]
    yer = D["yer"]
    cokgen(im, np.array([(0, yer + o), (W, yer + o), (W, H), (0, H)]), TAS)
    cokgen(im, np.array([(0, yer + o), (W, yer + o), (W, yer + o + 16), (0, yer + o + 16)]), (222, 201, 182))
    # kapı: pervaz, kanat, üst pencere, paneller, tokmak
    kutu(im, x0 - 34 + o, y0 - 34 + o, x1 + 34 + o, y1 + o, 26, PERVAZ)
    golge(im, x0 + o, y0 + o, x1 + o, y1 + o, 12, bulanik=10, alfa=0.10, dy=0)
    kutu(im, x0 + o, y0 + o, x1 + o, y1 + o + 40, 12, KAPI)
    w = x1 - x0
    kutu(im, x0 + 0.14 * w + o, y0 + 60 + o, x1 - 0.14 * w + o, y0 + 60 + 0.38 * w + o, 18, (255, 241, 208))
    cizgi_ince(im, np.array([((x0 + x1) / 2 + o, y0 + 60 + o), ((x0 + x1) / 2 + o, y0 + 60 + 0.38 * w + o)]), KAPI, 12)
    for py0, py1 in [(y0 + 60 + 0.46 * w, y0 + 60 + 1.15 * w), (y0 + 60 + 1.23 * w, min(y1 - 50, y0 + 60 + 2.1 * w))]:
        if py1 > py0 + 40:
            kutu(im, x0 + 0.14 * w + o, py0 + o, x1 - 0.14 * w + o, py1 + o, 16, None, cerceve=KAPI_KOYU, cerceve_kalinlik=7)
    _elips(im, x1 - 0.10 * w + o, y0 + 60 + 0.80 * w + o, 17, 17, BAL)
    # paspas
    if K.dikey:
        kutu(im, x0 + 30 + o, yer + 30 + o, x1 - 30 + o, yer + 130 + o, 22, (214, 150, 128))
    # numara plakası
    px, py = D["plaka"]
    kutu(im, px - 62 + o, py - 46 + o, px + 62 + o, py + 46 + o, 18, PERVAZ)
    yaz(im, "12", px + o, py + 22 + o, "serif", 64, METIN2, hiza="orta")
    # duvar lambası
    lx, ly = D["lamba"]
    _elips(im, lx + o, ly + o, 150, 150, (255, 236, 190), 0.55, bulanik=50)
    cizgi_ince(im, np.array([(lx + o, ly - 110 + o), (lx + o, ly - 56 + o)]), METIN2, 10)
    kutu(im, lx - 40 + o, ly - 60 + o, lx + 40 + o, ly + 56 + o, 16, (255, 244, 214), cerceve=METIN2, cerceve_kalinlik=8)
    # saksı ve yapraklar
    sx, syy = D["saksi"]
    r = np.random.default_rng(4)
    for i in range(11):
        ang = np.radians(-90 + (i - 5) * 17 + r.uniform(-5, 5))
        L = r.uniform(230, 380)
        ucx, ucy = sx + np.cos(ang) * L, syy - 150 + np.sin(ang) * L
        orta = np.array([(sx + ucx) / 2, (syy - 150 + ucy) / 2])
        d = np.array([ucx - sx, ucy - (syy - 150)]); d /= np.linalg.norm(d); n = np.array([-d[1], d[0]])
        yaprak = egri([(sx, syy - 150), orta + n * 46, (ucx, ucy), orta - n * 46, (sx, syy - 150)], 10)
        cokgen(im, yaprak + o, [(124, 158, 136), (104, 140, 117), (148, 178, 156)][i % 3])
    cokgen(im, np.array([(sx - 96, syy - 170), (sx + 96, syy - 170), (sx + 74, syy), (sx - 74, syy)]) + o, (201, 128, 104))
    kutu(im, sx - 108 + o, syy - 196 + o, sx + 108 + o, syy - 160 + o, 12, (214, 142, 118))
    return cv2.GaussianBlur(im, (0, 0), 4.5)


# ------------------------------------------------------------------ baş hareleri (önbellekte)
BAS_OLCEK = 1.15


@functools.lru_cache(maxsize=8)
def _bas(kim, s, yon):
    """Baş haresi (merkezi yüz ortası). Birim: 1 = tasarım pikseli * s * BAS_OLCEK."""
    v = FIGUR[kim]
    u = s * BAS_OLCEK
    k = bos_katman((int(320 * u), int(380 * u)))
    c = lambda x, y: (x * u, y * u)
    def el(cx, cy, rx, ry, renk, alfa=1.0):
        _elips(k, cx * u, cy * u, rx * u, ry * u, renk, alfa)
    fx = 160 + yon * 6                                   # yüz hafifçe karşıdakine dönük
    if v["tip"] == "uzun":
        el(160, 180, 90, 102, v["ten"])
        a = np.linspace(np.pi, 2 * np.pi, 30)
        ust = np.stack([160 + 96 * np.cos(a), 150 + 106 * np.sin(a)], 1)
        alt = np.stack([160 - yon * 14 + 100 * np.cos(a[::-1]), 152 + 40 * np.sin(a[::-1])], 1)
        cokgen(k, np.concatenate([ust, alt]) * u, v["sac"])
    else:
        el(160, 184, 118, 132, v["sac"])                # başörtüsü
        el(fx - yon * 2, 186, 78, 92, v["ten"])
    for gx in (fx - 34, fx + 34):                        # kapalı, gülümseyen gözler
        a = np.linspace(np.pi * 1.12, np.pi * 1.88, 12)
        cizgi_ince(k, np.stack([(gx + 15 * np.cos(a)) * u, (192 + 11 * np.sin(a)) * u], 1), GOZ, 5.5 * u)
        el(gx + (-14 if gx < fx else 14), 222, 17, 14, (242, 150, 140), 0.55)
    a = np.linspace(np.pi * 0.14, np.pi * 0.86, 14)
    cizgi_ince(k, np.stack([(fx + 24 * np.cos(a)) * u, (214 + 20 * np.sin(a)) * u], 1), (156, 74, 58), 5.5 * u)
    return k


def _koyu(renk, k):
    return tuple(float(np.clip(c * k, 0, 255)) for c in renk)


# ------------------------------------------------------------------ zaman / poz
def _kamera(t, K):
    """Tasarım -> ekran: yavaş itiş + hafif kayma. Dönüş: T(noktalar), zf."""
    zf = 0.93 + 0.07 * cik(ara(t, 9.85, 10.8)) + 0.035 * ara(t, 10.6, 13.3)
    kay = np.array([7 * np.sin(0.9 * t + 1), 4 * np.cos(0.7 * t)], np.float32)
    c = np.array([K.cx, K.cy], np.float32)
    def T(p):
        return c + (np.asarray(p, np.float32) - c) * zf + kay
    return T, zf, kay


def _gecis(t):
    return gir_cik(ara(t, Z.EL_DEGISTIRME - 0.25, Z.EL_DEGISTIRME + 0.25))


def _canta_poz(t, D):
    """Tutma noktası (tasarım), açı (derece)."""
    x = _gecis(t)
    g = np.array(D["gL"], np.float32) * (1 - x) + np.array(D["gR"], np.float32) * x
    g = g + (0, -46 * D["s"] * np.sin(np.pi * x) + 3.5 * np.sin(1.6 * t))
    aci = 2.0 * np.sin(1.9 * t) - 8.0 * np.sin(2 * np.pi * x)
    return g, aci


def _canta_T(t, K, D):
    """Çanta-yerel (birim: tasarım pikseli / s) -> ekran dönüşümü."""
    T, zf, _ = _kamera(t, K)
    g, aci = _canta_poz(t, D)
    c, s_ = np.cos(np.radians(aci)), np.sin(np.radians(aci))
    R = np.array([[c, -s_], [s_, c]], np.float32) * D["s"]
    def B(p):
        return T(np.asarray(p, np.float32).reshape(-1, 2) @ R.T + g)
    return B, zf


# çanta-yerel kurdele yolları
_C, _L, _R = (0, 282), (-116, 282), (116, 282)
ON1 = np.linspace(_C, _L, 10).astype(np.float32)
ARKA = egri([_L, (-64, 266), (0, 260), (64, 266), _R], 8)
ON2 = np.linspace(_R, _C, 10).astype(np.float32)
SARGI = [ON1, ARKA, ON2]
_ilmek = egri([_C, (-42, 244), (-96, 232), (-112, 276), (-84, 310), (-36, 298), _C], 9)
_ilmek2 = _ilmek * (-1, 1) + (0, 0)
_kuyruk1 = egri([_C, (-22, 322), (-50, 372)], 8)
FIYONK = np.concatenate([_ilmek, _ilmek2, _kuyruk1])
KUYRUK2 = egri([_C, (24, 324), (54, 368)], 8)


def _parca_uzunluk(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


_SU = np.cumsum([_parca_uzunluk(p) for p in SARGI]); _SU = _SU / _SU[-1]


def _cizgi_durumu(t):
    """Çizginin evreleri: yaklaşma (baş, kuyruk), sargı, fiyonk, ikinci kuyruk, çıkış (baş, kuyruk)."""
    if t < 10.0:
        yb, yk = 0.004, 0.0
    else:
        yb = gir_cik(ara(t, 10.0, Z.KURDELE_BAS))
        yk = max(0.0, yb - 0.45)
        if t > Z.KURDELE_BAS:
            yk = kar(0.55, 1.0, yumusak(ara(t, Z.KURDELE_BAS, Z.KURDELE_BAS + 0.32)))
    sargi = ara(t, Z.KURDELE_BAS, Z.KURDELE_BAS + 0.26)
    fiyonk = cik(ara(t, Z.KURDELE_BAS + 0.26, Z.KURDELE_SON), 2)
    k2 = cik(ara(t, Z.KURDELE_SON - 0.09, Z.KURDELE_SON + 0.03))
    r = gir_cik(ara(t, Z.S6_GECIS, Z.S6_GECIS + 0.24))
    if r > 0:
        fiyonk = 1 - ara(r, 0, 0.5); sargi = 1 - ara(r, 0.5, 1.0); k2 = 1 - ara(r, 0, 0.3)
    cb = gir_cik(ara(t, Z.S6_GECIS + 0.04, 13.27))
    ck = 0.82 * yumusak(ara(t, Z.S6_GECIS + 0.22, 13.27))
    return yb, yk, sargi, fiyonk, k2, cb, min(ck, cb)


def _yaklasma(t, K, D):
    B, _ = _canta_T(t, K, D)
    c = B([_C])[0]
    s5 = np.array(K.s5_giris, np.float32)
    k = D["s"]
    return bezier(s5, s5 + (190 * k, 110 * k), c + (210 * k, -30 * k), c, 40)


def _cikis(t, K, D):
    B, _ = _canta_T(t, K, D)
    c = B([_C])[0]
    s6 = np.array(K.s6_bas, np.float32)
    return bezier(c, c + (-40, -260), s6 + (-120, -170), s6, 48)


# ------------------------------------------------------------------ çizim parçaları
def _govde(dst, T, zf, D, kim, cx, yon, t):
    v = FIGUR[kim]
    s, sy = D["s"], D["sy"] + 3 * np.sin(1.5 * t + (0 if yon > 0 else 1.3))
    alt = 2400
    sol = [(cx - 36 * s, sy - 14 * s), (cx - 112 * s, sy + 2 * s), (cx - 174 * s, sy + 50 * s),
           (cx - 192 * s, sy + 180 * s), (cx - 200 * s, sy + 430 * s), (cx - 214 * s, alt)]
    L = egri(sol, 10)
    govde = np.concatenate([L, (L * (-1, 1) + (2 * cx, 0))[::-1]])
    if v["tip"] == "uzun":                               # arkaya dökülen saç: gövdenin ve boynun arkasında
        hx, hy = cx + yon * 8 * s, sy - 152 * s
        sac = egri([(hx - 100 * s, hy - 40 * s), (hx - 118 * s, hy + 90 * s), (hx - 96 * s, hy + 230 * s), (hx, hy + 262 * s),
                    (hx + 96 * s, hy + 230 * s), (hx + 118 * s, hy + 90 * s), (hx + 100 * s, hy - 40 * s)], 8)
        cokgen(dst, T(sac), v["sac"])
        cokgen(dst, T(np.array([(cx - 30 * s, sy - 90 * s), (cx + 30 * s, sy - 90 * s), (cx + 34 * s, sy + 10 * s), (cx - 34 * s, sy + 10 * s)])), _koyu(v["ten"], 0.94))
    cokgen(dst, T(govde), v["ust"])
    if v["tip"] == "uzun":
        a = np.linspace(0, np.pi, 20)
        cokgen(dst, T(np.stack([cx + 44 * s * np.cos(a), sy - 16 * s + 40 * s * np.sin(a)], 1)), v["ten"])
        for i in range(3):
            bx, by = T([(cx, sy + (120 + i * 105) * s)])[0]
            daire(dst, bx, by, 9 * s * zf, _koyu(v["ust"], 0.86))
    else:                                                # başörtüsünün omuza inen kısmı
        dok = egri([(cx - 96 * s, sy - 150 * s), (cx - 104 * s, sy - 30 * s), (cx - 60 * s, sy + 44 * s), (cx, sy + 62 * s),
                    (cx + 60 * s, sy + 44 * s), (cx + 104 * s, sy - 30 * s), (cx + 96 * s, sy - 150 * s)], 8)
        cokgen(dst, T(dok), v["sac"])
    return sy


def _bas_ciz(dst, T, zf, D, kim, cx, yon, sy, t):
    s = D["s"]
    x = _gecis(t)
    egim = yon * (2.5 + 3.5 * np.sin(np.pi * x)) + 1.2 * np.sin(1.1 * t + yon)
    hx, hy = T([(cx + yon * 8 * s, sy - 152 * s)])[0]
    yapistir(dst, _bas(kim, round(s, 3), yon), hx, hy, olcek=zf / BAS_OLCEK, aci=egim, capa=(0.5, 180 / 380))


def _kol(dst, T, zf, D, v, omuz, el, tercih):
    """İki eklemli kol: omuz -> dirsek -> el (tasarım koordinatı)."""
    s = D["s"]
    L = 215 * s
    o, e = np.array(omuz, np.float32), np.array(el, np.float32)
    d = e - o
    uz = np.linalg.norm(d)
    if uz > 2 * L * 0.98:
        e = o + d / uz * 2 * L * 0.98; d = e - o; uz = np.linalg.norm(d)
    n = np.array([-d[1], d[0]]) / max(uz, 1e-6)
    if n @ np.array(tercih, np.float32) < 0:
        n = -n
    dirsek = (o + e) / 2 + n * np.sqrt(max(L * L - (uz / 2) ** 2, 0)) * 0.8
    yol = T(egri([o, dirsek, e], 8))
    cizgi_ince(dst, yol, v["kol"], 64 * s * zf)
    ex, ey = T([e])[0]
    daire(dst, ex, ey, 34 * s * zf, v["ten"])


def _canta_ciz(dst, B, zf, s):
    k = s * zf
    # arka sap
    cizgi_ince(dst, B(egri([(-44, 124), (-36, 40), (8, -2), (50, 42), (60, 124)], 8)), IP_KOYU, 8 * k)
    # katlanmış tulum (çantanın ağzından görünür)
    a = np.radians(-8); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    tl = lambda P: np.asarray(P, np.float32) @ R.T + (14, 52)
    tulum = [(-56, 0), (-28, -14), (-14, -4), (0, 0), (14, -4), (28, -14), (56, 0), (90, 30), (72, 58), (54, 42),
             (54, 120), (-54, 120), (-54, 42), (-72, 58), (-90, 30)]
    cokgen(dst, B(tl(tulum)), (253, 248, 240))
    for y in (16, 36, 56):
        cizgi_ince(dst, B(tl([(-52, y), (52, y)])), (137, 172, 150), 7 * k)
    cizgi_ince(dst, B(tl(egri([(-28, -13), (-14, 0), (0, 5), (14, 0), (28, -13)], 6))), (137, 172, 150), 5.5 * k)
    # çanta gövdesi
    on = np.array([(-113, 122), (113, 122), (119, 398), (113, 406), (-113, 406), (-119, 398)], np.float32)
    cokgen(dst, B(on), KRAFT)
    cokgen(dst, B([(66, 122), (113, 122), (119, 398), (113, 406), (86, 406)]), KRAFT_KOYU, 0.55)
    cokgen(dst, B([(-113, 122), (113, 122), (114, 150), (-114, 150)]), KRAFT_ACIK)
    cizgi_ince(dst, B([(-114, 150), (114, 150)]), KRAFT_KOYU, 3 * k, 0.7)
    cizgi_ince(dst, B([(-40, 152), (-44, 404)]), KRAFT_KOYU, 3 * k, 0.35)
    # ön sap
    cizgi_ince(dst, B(egri([(-52, 126), (-44, 42), (0, 3), (44, 42), (52, 126)], 8)), IP, 9 * k)


def _bildirim(dst, t, K, D):
    a = ara(t, Z.BILDIRIM, Z.BILDIRIM + 0.36)
    if a <= 0:
        return
    x0, y0, x1, y1 = D["kart"]
    dy = -(1 - geri(a, 1.1)) * (y1 + 60) + 3 * np.sin(1.3 * t)
    alfa = min(1.0, a * 3)
    y0, y1 = y0 + dy, y1 + dy
    cam_kart(dst, x0, y0, x1, y1, r=40, alfa=alfa, opak=0.88)
    h = y1 - y0
    ik = h * 0.56
    yapistir(dst, marka("ikon", int(ik)), x0 + 34 + ik / 2, (y0 + y1) / 2 - 2, alfa=alfa)
    tx = x0 + 34 + ik + 26
    yaz(dst, "Annea · şimdi", tx, y0 + h * 0.36, "nunito", 28, METIN2, 700, alfa=alfa)
    boyut = 42
    while True:
        w1 = yazi_olcu("Aslı:", "nunito", boyut, 900)[0]
        w2 = yazi_olcu("Çok yakıştı, teşekkürler!", "nunito", boyut, 600)[0]
        if tx + w1 + 12 + w2 <= x1 - 30 or boyut <= 30:
            break
        boyut -= 1
    yaz(dst, "Aslı:", tx, y0 + h * 0.74, "nunito", boyut, METIN, 900, alfa=alfa)
    yaz(dst, "Çok yakıştı, teşekkürler!", tx + w1 + 12, y0 + h * 0.74, "nunito", boyut, METIN, 600, alfa=alfa)


# ------------------------------------------------------------------ sahne
def ciz(t, K, tuval):
    D = _duzen(K)
    s = D["s"]
    T, zf, kay = _kamera(t, K)
    sonme = yumusak(ara(t, Z.S6_GECIS, 13.22))
    cipla = tuval.copy() if sonme > 0 else None
    yb, yk, sargi, fiyonk, k2, cb, ck = _cizgi_durumu(t)
    B, _ = _canta_T(t, K, D)
    kal = 13 * zf

    if sonme < 1:
        # arka plan: daha az ölçeklenir (derinlik), tuvalin nefes alan ışığıyla çarpılır
        zb = 1.05 + (zf - 1) * 0.55
        Mb = afin(zb, 0, (K.cx + PAY, K.cy + PAY), (K.cx + kay[0] * 0.5, K.cy + kay[1] * 0.5))
        arka = cv2.warpAffine(_arka(K.dikey), Mb, (K.W, K.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        isik = tuval.astype(np.float32) / np.array(KREM, np.float32)
        tuval[:] = np.clip(arka.astype(np.float32) * isik, 0, 255).astype(np.uint8)

        # figürler
        syL = _govde(tuval, T, zf, D, "sol", D["cxL"], +1, t)
        syR = _govde(tuval, T, zf, D, "sag", D["cxR"], -1, t)
        _bas_ciz(tuval, T, zf, D, "sol", D["cxL"], +1, syL, t)
        _bas_ciz(tuval, T, zf, D, "sag", D["cxR"], -1, syR, t)

        # kurdelenin arkadan geçen yarısı (çantanın ARKASINDA)
        if sargi > _SU[0]:
            u = (min(sargi, _SU[1]) - _SU[0]) / (_SU[1] - _SU[0])
            cizgi(tuval, B(yol_parca(ARKA, 0, u)), kal * 1.1, parlama=0.6)
        g, _ = _canta_poz(t, D)
        gs = T([g + (10 * s, 432 * s)])[0]
        _elips(tuval, gs[0], gs[1], 150 * s * zf, 30 * s * zf, (110, 70, 45), 0.10, bulanik=12)
        _canta_ciz(tuval, B, zf, s)

        # kollar: veren 12.00'den sonra bırakır, alan 11.6'dan itibaren uzanır
        tutL = 1 - yumusak(ara(t, Z.EL_DEGISTIRME, Z.EL_DEGISTIRME + 0.42))
        tutR = yumusak(ara(t, Z.EL_DEGISTIRME - 0.45, Z.EL_DEGISTIRME - 0.02))
        for kim, cx, yon, sy, tut in (("sol", D["cxL"], +1, syL, tutL), ("sag", D["cxR"], -1, syR, tutR)):
            v = FIGUR[kim]
            ic_omuz, dis_omuz = (cx + yon * 152 * s, sy + 74 * s), (cx - yon * 156 * s, sy + 74 * s)
            ic_dur = np.array((cx + yon * 225 * s, sy + 330 * s + 5 * np.sin(1.4 * t + yon)), np.float32)
            dis_el = np.array((cx - yon * 172 * s, sy + 478 * s + 3 * np.sin(1.2 * t)), np.float32)
            ic_el = ic_dur * (1 - tut) + (g + (-yon * 17 * s, 6 * s)) * tut
            _kol(tuval, T, zf, D, v, dis_omuz, dis_el, (-yon, 0.2))
            _kol(tuval, T, zf, D, v, ic_omuz, ic_el, (-yon, 0.35))

        _bildirim(tuval, t, K, D)

        if sonme > 0:                                    # geriye bulanıklaşıp çıplak zemine döner
            tuval[:] = karistir(bulanik(tuval, 16 * sonme), cipla, sonme)
    else:
        tuval[:] = cipla

    # ---- mercan çizgi: öndeki parçalar (hepsi ortak cizgi() ile)
    if sargi > 0:
        cizgi(tuval, B(yol_parca(ON1, 0, min(1, sargi / _SU[0]))), kal * 1.1, parlama=0.6)
    if sargi > _SU[1]:
        cizgi(tuval, B(yol_parca(ON2, 0, (sargi - _SU[1]) / (1 - _SU[1]))), kal * 1.1, parlama=0.6)
    if fiyonk > 0:
        cizgi(tuval, B(yol_parca(FIYONK, 0, fiyonk)), kal, parlama=0.6)
    if k2 > 0:
        cizgi(tuval, B(yol_parca(KUYRUK2, 0, k2)), kal, parlama=0.6)
    if yk < 1 and t < Z.S6_GECIS:                        # yaklaşan kuyruklu çizgi
        def yak(u):
            b_, k_ = _cizgi_durumu(u)[:2]
            return yol_parca(_yaklasma(u, K, D), k_, max(b_, k_ + 0.004))
        cizgi(tuval, alt_zaman(yak, t, 4) if t > 10.0 else yak(t), kal)
    if cb > 0:                                           # çözülüp S6'ya giden çizgi
        def cik_(u):
            d = _cizgi_durumu(u)
            return yol_parca(_cikis(u, K, D), d[6], max(d[5], d[6] + 0.004))
        cizgi(tuval, alt_zaman(cik_, t, 4) if t < 13.27 else cik_(t), kal)
    return tuval

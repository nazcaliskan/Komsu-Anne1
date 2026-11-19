"""S5 · Elden ele (9.85–14.40) — yüz ve figür yok: nesneler, iki el ve hareket.

Tulum katlanıp kâğıt çantaya girer; çizgi çantayı bir kez sarıp kurdele olur;
tek el çantayı alır ve kamera onunla birlikte "bir sokak öteye" kayar (A kapısı,
sokak, B kapısı paralaksla akar); 12.00'de çanta ikinci ele geçer; bildirim iner;
"sarsın" derken tulum çantadan çıkıp kollarını açar; sahne söner, kurdele çözülüp
tek çizgi olarak etkinlik sahnesine akar.

Düz palet korunur; gerçeklik fiziksel ipuçlarından gelir: sol üstten yumuşak ışık,
kâğıt lifi ve kat izleri, kumaş dokusu, hacim gölgeleri, temas gölgesi, sığ alan derinliği.
"""
import functools
import numpy as np, cv2
import ortak as O
from ortak import Z

ARALIK = (9.85, Z.ETK_BAS)
SON_BAS, SON_BIT = Z.S5_GECIS, Z.S5_GECIS + 0.32       # sahne söner
UC_VARIS = Z.S5_GECIS + 0.35                            # çizginin ucu K.etk_giris'e varır

TEN1, TEN2 = (249, 220, 200), (227, 170, 130)
KOL1, KOL2 = (250, 166, 148), O.NIL                 # veren el yavruağzı, alan el nil yeşili
KRAFT_KOYU, KRAFT_ACIK, SAP = (170, 132, 94), (228, 196, 158), (150, 112, 78)
YURU_BAS, YURU_SON = 10.90, 11.98
DUSUS = 10.36                                        # tulum çantaya düşer
CIK_BAS, CIK_TEPE = 13.02, 13.50                     # tulum çantadan çıkar ("sarsın" 13.46)


# ------------------------------------------------------------------ yerleşim
@functools.lru_cache(maxsize=None)
def _yer(dikey):
    K = O.DIKEY if dikey else O.YATAY
    s = 1.0 if dikey else 0.80                      # yatayda daha yakın çekim
    bw, bh = 490 * s, 560 * s
    ust = 930 if dikey else 446                      # çanta ön yüzünün üst kenarı (dinlenme)
    return dict(K=K, s=s, bw=bw, bh=bh, bx=K.cx, ust=ust, pivot_y=ust - 120 * s, gy=ust + bh - 70 * s,
                D=(2200 if dikey else 1750))


def _elips(cx, cy, rx, ry, aci=0.0, n=40):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.stack([rx * np.cos(a), ry * np.sin(a)], 1)
    c, s_ = np.cos(aci), np.sin(aci)
    return p @ np.array([[c, s_], [-s_, c]], np.float32) + (cx, cy)


@functools.lru_cache(maxsize=None)
def _doku(h, w, sx, sy, tohum):
    """Önbellekli doku gürültüsü (std 1): sx, sy bulanıklık yarıçapı."""
    g = np.random.default_rng(tohum).standard_normal((h, w)).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), sigmaX=max(sx, 0.3), sigmaY=max(sy, 0.3))
    return g / (g.std() + 1e-6)


def _carp(kat, m):
    """Ön çarpılmış katmanın rengini m (h, w) ile çarpar (alfa değişmez)."""
    kat[..., :3] = np.clip(kat[..., :3].astype(np.float32) * m[..., None], 0, kat[..., 3:].astype(np.float32)).astype(np.uint8)


def _hacim(kat, d, sig, guc=0.5, sinir=0.16):
    """Sol üstten ışık: şeklin sol-üst kenarı aydınlanır, sağ-alt kenarı gölgelenir (yumuşak hacim)."""
    a = kat[..., 3].astype(np.float32) / 255
    ab = cv2.GaussianBlur(a, (0, 0), sig)
    d = max(1, int(round(d)))
    e = np.roll(ab, (-d, -d), (0, 1)) - np.roll(ab, (d, d), (0, 1))
    _carp(kat, 1 + np.clip(guc * e, -sinir, sinir * 0.7))


# ------------------------------------------------------------------ arka plan (yarı çözünürlük, bulanık, paralaks)
@functools.lru_cache(maxsize=None)
def _isik(dikey):
    """Sol üstten yönlü ışık + ortam kararması (yarı çözünürlük çarpanı)."""
    K = O.DIKEY if dikey else O.YATAY
    h, w = K.H // 2, K.W // 2
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x /= w; y /= h
    isik = 1.07 - 0.11 * (0.55 * x + 0.45 * y)
    kose = np.clip(np.hypot(x - 0.42, y - 0.45) * 1.5 - 0.5, 0, 1) ** 2
    return (isik - 0.10 * kose).astype(np.float32)


def _arka(t, Y, kay, hiz):
    K, s, gy, D = Y["K"], Y["s"], Y["gy"], Y["D"]
    h, w = K.H // 2, K.W // 2
    g = np.empty((h, w, 3), np.uint8)
    q = 0.5

    def X(wx, f=1.0):
        return (K.cx + (wx * s - kay) * f) * q

    def kutu(x0, y0, x1, y1, renk, r=0, f=1.0, alfa=1.0):
        O.kutu(g, X(x0, f), y0 * q, X(x1, f), y1 * q, r * q, renk, alfa)

    Dw = D / s
    g[:] = (250, 240, 228)                                           # sokak aralığı: açık gök-duvar
    # uzak katman: ağaçlar, lamba, çit — daha bulanık (sığ alan derinliği)
    for wx, boy, ton in [(700, 520, (150, 214, 190)), (980, 640, (110, 198, 168)), (1280, 560, (170, 224, 202)),
                         (1560, 660, (120, 202, 174)), (1840, 540, (156, 216, 192))]:
        wx = wx * Dw / 2200
        O.kutu(g, X(wx - 12, .62), (gy - boy * s * .55) * q, X(wx + 12, .62), gy * q, 4, (176, 146, 120))
        for dx, dy, r in [(0, 0, 150), (-90, 60, 110), (95, 50, 115)]:
            O.daire(g, X(wx + dx, .62), (gy - boy * s + dy * s) * q, r * s * q, ton)
            O.daire(g, X(wx + dx - r * .25, .62), (gy - boy * s + dy * s - r * s * .25) * q, r * s * q * .55, (255, 255, 240), 0.14)
    lx = 1120 * Dw / 2200
    O.kutu(g, X(lx - 7, .8), (gy - 760 * s) * q, X(lx + 7, .8), gy * q, 3, (120, 104, 96))
    O.daire(g, X(lx, .8), (gy - 780 * s) * q, 34 * s * q, (255, 226, 170))
    for i in range(-2, 30):                                           # çit
        cx = 640 + i * 62
        if 600 < cx < Dw - 740:
            kutu(cx, gy - 250 * s, cx + 40, gy, (255, 251, 246), r=14)
            kutu(cx + 30, gy - 240 * s, cx + 40, gy, (214, 204, 192), alfa=0.5)
    kutu(600, gy - 150 * s, Dw - 740, gy - 120 * s, (255, 251, 246))
    g = O.bulanik(g, 3.2 if K.dikey else 2.4)
    # orta katman: A evi, B evi
    kutu(-2600, 0, 620, gy, (255, 228, 216))                         # A evi: yavruağzı açık
    kutu(Dw - 760, 0, Dw + 2600, gy, (212, 240, 228))                # B evi: nil açık
    for wx in (620, Dw - 760):                                        # duvar köşelerinde ince gölge
        kutu(wx - 14, 0, wx + 14, gy, (150, 120, 100), alfa=0.16)
    for i in range(1, 9):                                             # duvar derzleri
        y = gy - i * 230 * s
        O.kutu(g, 0, y * q, w, (y + 3) * q, 0, (255, 255, 255), 0.16)
        O.kutu(g, 0, (y + 3) * q, w, (y + 6) * q, 0, (150, 120, 100), 0.07)
    kutu(-760, gy - 900 * s, -440, gy - 500 * s, (255, 246, 232), r=16)          # A penceresi
    kutu(-735, gy - 875 * s, -465, gy - 525 * s, (255, 232, 190), r=8)
    kutu(-735, gy - 875 * s, -465, gy - 820 * s, (150, 120, 90), r=8, alfa=0.18)  # pervaz gölgesi
    for wx, kapi, pano in [(0, O.SOMON, (253, 160, 146)), (Dw, O.NIL, (128, 208, 180))]:
        koyu = tuple(int(c * 0.80) for c in kapi)
        kutu(wx - 240, gy - 990 * s, wx + 240, gy, (255, 246, 236), r=22)        # söve
        kutu(wx - 200, gy - 950 * s, wx + 200, gy, kapi, r=12)
        kutu(wx - 200, gy - 950 * s, wx + 200, gy - 905 * s, koyu, r=12, alfa=0.45)   # söve gölgesi (ışık üstten)
        kutu(wx - 200, gy - 950 * s, wx - 172, gy, koyu, alfa=0.25)
        for dx in (-100, 0, 100):                                                 # ahşap tahta araları
            kutu(wx + dx - 3, gy - 940 * s, wx + dx + 3, gy, koyu, alfa=0.40)
        kutu(wx - 150, gy - 890 * s, wx + 150, gy - 560 * s, pano, r=12)
        kutu(wx - 150, gy - 500 * s, wx + 150, gy - 70 * s, pano, r=12)
        for y0, y1 in ((890, 560), (500, 70)):                                    # pano kabartması
            kutu(wx - 150, gy - y0 * s, wx + 150, gy - (y0 - 22) * s, koyu, r=10, alfa=0.35)
            kutu(wx - 150, gy - (y1 + 16) * s, wx + 150, gy - y1 * s, (255, 250, 240), r=10, alfa=0.30)
        O.daire(g, X(wx + 158), (gy - 470 * s) * q, 15 * s * q, (255, 226, 170))
        O.daire(g, X(wx + 164), (gy - 462 * s) * q, 15 * s * q, (90, 70, 50), 0.25)
    kutu(Dw + 330, gy - 880 * s, Dw + 640, gy - 500 * s, (255, 250, 240), r=16)  # B penceresi
    kutu(Dw + 355, gy - 855 * s, Dw + 615, gy - 525 * s, (255, 236, 198), r=8)
    kutu(Dw + 355, gy - 855 * s, Dw + 615, gy - 800 * s, (150, 120, 90), r=8, alfa=0.18)
    O.daire(g, X(Dw - 392), (gy - 712 * s) * q, 80 * s * q, (120, 110, 100), 0.20)  # levha gölgesi
    O.daire(g, X(Dw - 400), (gy - 720 * s) * q, 78 * s * q, (255, 250, 244))     # B kapısının yanında yeşil pin (tulum ve bildirim örtmez)
    O.pin(g, X(Dw - 400), (gy - 665 * s) * q, 116 * s * q, O.NIL, delik=(255, 250, 244))
    # zemin ve paspaslar; yakın katman derzleri daha hızlı akar
    O.kutu(g, 0, gy * q, w, h, 0, (238, 217, 196))
    for wx, renk in [(0, (240, 150, 130)), (Dw, (70, 170, 138))]:
        kutu(wx - 250, gy + 26 * s, wx + 250, gy + 120 * s, renk, r=18)
        kutu(wx - 250, gy + 100 * s, wx + 250, gy + 128 * s, (120, 90, 70), r=14, alfa=0.22)
    for i in range(-6, 40):
        x = X(i * 300, 1.3)
        if -4 < x < w + 4:
            O.cizgi_ince(g, [(x, (gy + 12 * s) * q), (x - 40 * q, h)], (206, 180, 156), 2, 0.7)
    g = O.bulanik(g, (4.6 if K.dikey else 3.4))
    # sıva / zemin dokusu (kamerayla birlikte kayar), duvar-zemin birleşiminde ortam gölgesi, yönlü ışık
    f = g.astype(np.float32)
    dk = _doku(h, w, 7.0, 7.0, 21) * 0.011 + _doku(h, w, 1.6, 1.6, 22) * 0.005
    f *= (1 + np.roll(dk, -int(kay * q) % w, 1))[..., None]
    yy = np.arange(h, dtype=np.float32)[:, None]
    ao = 0.20 * np.exp(-np.abs(yy - gy * q) / (34 * s * q)) * (yy <= gy * q) + 0.12 * np.exp(-np.abs(yy - gy * q) / (60 * s * q)) * (yy > gy * q)
    f *= (_isik(K.dikey) * (1 - ao))[..., None]
    g = np.clip(f, 0, 255).astype(np.uint8)
    k = int(abs(hiz) * 0.28) | 1                                      # yatay hareket bulanıklığı (hıza göre)
    if k > 2:
        g = cv2.blur(g, (min(k, 61), 1))
    z = 1 + 0.035 * O.ara(t, ARALIK[0], ARALIK[1])                    # kamera çok yavaş iter
    M = np.array([[2 * z, 0, K.cx * (1 - z)], [0, 2 * z, Y["gy"] * (1 - z)]], np.float32)
    return cv2.warpAffine(g, M, (K.W, K.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ------------------------------------------------------------------ çanta hareleri (yerel başlangıç: sapların tepesi)
@functools.lru_cache(maxsize=None)
def _canta(dikey):
    Y = _yer(dikey)
    s, bw, bh = Y["s"], Y["bw"], Y["bh"]
    ox, oy = bw / 2 + 60 * s, 40 * s
    boyut = (int(bw + 120 * s), int(bh + 200 * s))
    arka, on = O.bos_katman(boyut), O.bos_katman(boyut)
    ust = 120 * s

    def sap(k, dy, renk):
        P = O.egri([(-0.20 * bw, ust + 10 * s + dy), (-0.15 * bw, 30 * s + dy), (0, 4 * s + dy), (0.15 * bw, 30 * s + dy), (0.20 * bw, ust + 10 * s + dy)], 16) + (ox, oy)
        O.cizgi_ince(k, P + (3 * s, 3 * s), (90, 64, 44), 10 * s, 0.35)                   # ip gölgesi
        O.cizgi_ince(k, P, renk, 10 * s)
        O.cizgi_ince(k, P - (2 * s, 2 * s), (214, 176, 136), 3 * s, 0.55)                  # ip üstünde ışık

    sap(arka, -18 * s, (136, 100, 70))
    O.cokgen(arka, np.array([(-bw / 2 + 8 * s, ust - 24 * s), (bw / 2 - 8 * s, ust - 24 * s), (bw / 2, ust + 40 * s), (-bw / 2, ust + 40 * s)]) + (ox, oy), KRAFT_KOYU)
    O.cokgen(arka, np.array([(-bw / 2 + 8 * s, ust - 24 * s), (bw / 2 - 8 * s, ust - 24 * s), (bw / 2 - 6 * s, ust - 6 * s), (-bw / 2 + 6 * s, ust - 6 * s)]) + (ox, oy), (120, 90, 62), 0.45)  # iç gölge
    X0, X1, Y0, Y1 = ox - bw / 2, ox + bw / 2, oy + ust, oy + ust + bh
    O.kutu(on, X0, Y0, X1, Y1, 10 * s, O.KRAFT)
    yan = X1 - bw * 0.20                                                                    # gölgeli yan yüz
    O.kutu(on, yan, Y0, X1, Y1, 8 * s, (140, 100, 62), 0.20)
    # kâğıt lifi dokusu + sol üstten ışık + dipte kararma
    h, w = on.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = (xx - X0) / bw, (yy - Y0) / bh
    m = 1 + _doku(h, w, 0.7, 7 * s, 31) * 0.020 + _doku(h, w, 16 * s, 16 * s, 32) * 0.018 + _doku(h, w, 0.6, 0.6, 33) * 0.010
    m *= 1.05 - 0.09 * u - 0.07 * np.clip(v, 0, 1) ** 2
    _carp(on, m)
    koyu, acik = (120, 86, 54), (255, 236, 206)
    def kat(P, a=0.22):                                                                     # kat izi: koyu çizgi + yanında ışık
        P = np.asarray(P, np.float32)
        O.cizgi_ince(on, P, koyu, 2.4 * s, a)
        O.cizgi_ince(on, P - (2.2 * s, 2.2 * s), acik, 2.0 * s, a * 1.3)
    kat([(yan, Y0 + 4 * s), (yan, Y1 - 4 * s)], 0.30)
    kat([(X0 + 6 * s, Y1 - bh * 0.20), (yan, Y1 - bh * 0.20)], 0.20)
    kat([(X0 + bw * 0.22, Y0 + 44 * s), (X0 + bw * 0.22, Y1 - 6 * s)], 0.14)
    kat([(yan, Y1 - bh * 0.20), (X1 - bw * 0.10, Y1 - bh * 0.30), (X1 - 4 * s, Y1 - bh * 0.20)], 0.20)
    kat([(X1 - bw * 0.10, Y0 + 44 * s), (X1 - bw * 0.10, Y1 - bh * 0.30)], 0.16)
    O.kutu(on, X0, Y0, X1, Y0 + 40 * s, 8 * s, KRAFT_ACIK, 0.85)                           # ağız kıvrımı
    O.kutu(on, X0, Y0 + 40 * s, X1, Y0 + 52 * s, 4 * s, koyu, 0.16)                        # kıvrımın gölgesi
    O.cizgi_ince(on, [(X0 + 8 * s, Y0 + 2 * s), (X1 - 8 * s, Y0 + 2 * s)], (255, 244, 224), 3 * s, 0.7)   # kenar ışığı
    O.cizgi_ince(on, [(X0 + 2 * s, Y0 + 10 * s), (X0 + 2 * s, Y1 - 10 * s)], (255, 244, 224), 3 * s, 0.45)
    sap(on, 4 * s, SAP)
    _hacim(on, 3 * s, 4 * s, 0.6, 0.10)
    return arka, on, (ox, oy)


def _tulum(Y, f1, f2):
    """Çizgili tulum haresi. f1: kollar içe (0..1), f2: alt yarı yukarı (0..1). Çapa: yaka (üst orta)."""
    s = Y["s"]
    W, Hg = 250 * s, O.kar(310, 176, f2) * s
    ox, oy = 230 * s, 30 * s
    k = O.bos_katman((int(460 * s), int(480 * s)))
    krem = (255, 250, 240)
    L = 150 * (1 - f1)
    if L > 6:
        for yn in (-1, 1):
            O.cokgen(k, np.array([(yn * (W / 2 - 30 * s), 6 * s), (yn * (W / 2 + L * s * 0.82), 6 * s + L * s * 0.50),
                                  (yn * (W / 2 + L * s * 0.60), 6 * s + L * s * 0.95), (yn * (W / 2 - 30 * s), 110 * s)]) + (ox, oy), krem)
    bacak = 112 * (1 - f2)
    if bacak > 5:
        for yn in (-1, 1):
            O.kutu(k, ox + yn * W / 4 - W / 4 + 5 * s, oy + Hg - 40 * s, ox + yn * W / 4 + W / 4 - 5 * s, oy + Hg + bacak * s, 34 * s, krem)
    O.kutu(k, ox - W / 2, oy, ox + W / 2, oy + Hg, 36 * s, krem)
    h, w = k.shape[:2]
    a = k[..., 3:].astype(np.float32) / 255                           # şekli çizgili kumaşla doldur
    y = (np.arange(h, dtype=np.float32) - oy) / (27 * s)
    serit = ((np.floor(y) % 2) == 1)[:, None, None]
    renk = np.where(serit, np.array((104, 196, 166), np.float32), np.array(krem, np.float32)) * np.ones((1, w, 1), np.float32)
    xx = (np.arange(w, dtype=np.float32) - ox) / (W / 2)              # kumaş dokusu + gövde hacmi (kenarlara doğru koyulaşır)
    m = (1 + _doku(h, w, 0.5, 0.5, 41) * 0.022 + _doku(h, w, 9 * s, 3 * s, 42) * 0.020) * (1.03 - 0.07 * np.clip(np.abs(xx[None, :] + 0.15), 0, 1.6) ** 2)
    k[..., :3] = np.clip(renk * m[..., None] * a + 0.5, 0, 255).astype(np.uint8)
    O.cokgen(k, _elips(ox, oy + 2 * s, 46 * s, 24 * s), (226, 214, 198))                 # yaka içi
    O.cizgi_ince(k, _elips(ox, oy + 2 * s, 46 * s, 24 * s)[:21], O.ADACAYI, 5 * s)
    for i in range(3):
        if 58 + i * 38 < Hg / s - 20:
            O.daire(k, ox + 1.5 * s, oy + (59.5 + i * 38) * s, 5.5 * s, (70, 60, 50), 0.3)
            O.daire(k, ox, oy + (58 + i * 38) * s, 5.5 * s, O.ADACAYI)
    golge = (96, 82, 70)
    if f1 > 0.3:                                                      # katlanan kolların kıvrım gölgesi
        g = min(1.0, (f1 - 0.3) / 0.5)
        for yn in (-1, 1):
            x = ox + yn * (W / 2 - 36 * s)
            O.cokgen(k, np.array([(x - 9 * s, oy + 22 * s), (x + 9 * s, oy + 22 * s), (x + 9 * s, oy + Hg - 18 * s), (x - 9 * s, oy + Hg - 18 * s)]), golge, 0.16 * g, bulanik=4 * s)
            O.cizgi_ince(k, [(x, oy + 26 * s), (x, oy + Hg - 22 * s)], golge, 2.4 * s, 0.26 * g)
    if f2 > 0.3:
        g = min(1.0, (f2 - 0.3) / 0.5)
        O.cokgen(k, np.array([(ox - W / 2 + 20 * s, oy + Hg - 28 * s), (ox + W / 2 - 20 * s, oy + Hg - 28 * s), (ox + W / 2 - 20 * s, oy + Hg - 8 * s), (ox - W / 2 + 20 * s, oy + Hg - 8 * s)]), golge, 0.18 * g, bulanik=4 * s)
        O.cizgi_ince(k, [(ox - W / 2 + 22 * s, oy + Hg - 16 * s), (ox + W / 2 - 22 * s, oy + Hg - 16 * s)], golge, 2.4 * s, 0.26 * g)
    k[..., :3] = np.minimum(k[..., :3], k[..., 3:])
    _hacim(k, 3 * s, 5 * s, 0.7, 0.12)
    return k, (ox, oy)


# ------------------------------------------------------------------ kol ve el
def _kol(tuval, K, tut, s, taraf, kol, ten, g):
    """Kadrajın yan kenarından giren, dirsekten hafif bükülü kol; sapları kavramış kapalı el.
    tut: elin tuttuğu nokta; taraf: -1 sol, +1 sağ; g: 0..1 giriş."""
    if K.dikey:                                       # yan kenardan girer
        tut = np.asarray(tut, np.float32) + np.array([taraf * 1150 * s, -150 * s], np.float32) * (1 - g)
        dirsek = tut + np.array([taraf * 300, -150], np.float32) * s
        cikis = np.array([(-200 if taraf < 0 else K.W + 200) + taraf * 1150 * s * (1 - g), tut[1] - 520 * s], np.float32)
    else:                                             # yatayda kısa kol: üst kenardan, yakından girer
        tut = np.asarray(tut, np.float32) + np.array([taraf * 420 * s, -620 * s], np.float32) * (1 - g)
        dirsek = tut + np.array([taraf * 210, -190], np.float32) * s
        cikis = np.array([tut[0] + taraf * 470 * s, min(-160.0, tut[1] - 700 * s)], np.float32)
    yon = dirsek - tut; yon /= np.linalg.norm(yon)
    dik = np.array([-yon[1], yon[0]], np.float32)
    kat = O.bos_katman(K)
    yol = O.egri([tut + yon * 92 * s, dirsek, cikis], 14)
    O.cizgi_ince(kat, yol, kol, 100 * s)                                                   # kol (dirsekte yumuşak kıvrım)
    O.cizgi_ince(kat, [tut + yon * 76 * s, tut + yon * 122 * s], tuple(min(255, int(c * 1.10)) for c in kol), 112 * s)   # kol ağzı
    koyu = tuple(int(c * 0.84) for c in ten)
    O.cokgen(kat, np.array([tut + yon * 8 * s + dik * 37 * s, tut + yon * 80 * s + dik * 37 * s, tut + yon * 80 * s - dik * 37 * s, tut + yon * 8 * s - dik * 37 * s]), ten)   # bilek
    aci = float(np.arctan2(yon[1], yon[0]))
    O.cokgen(kat, _elips(tut[0], tut[1], 66 * s, 56 * s, -aci), ten)                       # yumruk
    for i in (-1, 0, 1):                                                                   # parmak araları
        c = tut - yon * 30 * s + dik * i * 24 * s
        O.cizgi_ince(kat, [c - yon * 22 * s, c + yon * 16 * s], koyu, 3.2 * s, 0.8)
    b = tut + yon * 6 * s - dik * taraf * 34 * s + (0, 30 * s)                             # başparmak
    O.cokgen(kat, _elips(b[0], b[1], 34 * s, 20 * s, -aci - 0.5 * taraf), ten)
    O.cizgi_ince(kat, _elips(b[0], b[1], 34 * s, 20 * s, -aci - 0.5 * taraf)[8:26], koyu, 2.6 * s, 0.6)
    x0, x1 = max(0, int(min(tut[0], cikis[0]) - 160 * s)), min(K.W, int(max(tut[0], cikis[0]) + 160 * s))
    y0, y1 = max(0, int(min(cikis[1], dirsek[1]) - 140 * s)), min(K.H, int(tut[1] + 140 * s))
    if x1 <= x0 or y1 <= y0:
        return
    kes = np.ascontiguousarray(kat[y0:y1, x0:x1])
    _hacim(kes, 9 * s, 12 * s, 0.9, 0.17)                                                  # yumuşak hacim gölgesi
    # elin altına düşen yumuşak gölge (çantaya / tulumun üstüne)
    gl = cv2.GaussianBlur(kes[..., 3], (0, 0), 14 * s).astype(np.float32) / 255 * 0.22
    bolge = tuval[y0:y1, x0:x1].astype(np.float32)
    dy, dx = int(16 * s), int(10 * s)
    bolge[dy:, dx:] *= (1 - gl[:-dy, :-dx])[..., None]
    tuval[y0:y1, x0:x1] = bolge.astype(np.uint8)
    O.katman_bas(tuval, kes, 1.0, x0, y0)


# ------------------------------------------------------------------ hareket
def _kay(t, Y):
    """Kameranın yana kayışı (dünya pikseli); yürüyüş dışında da çok yavaş sürer."""
    return Y["D"] * O.gir_cik(O.ara(t, YURU_BAS, YURU_SON)) + 22 * Y["s"] * (t - ARALIK[0]) - 22 * Y["s"] * (YURU_SON - ARALIK[0]) * O.ara(t, YURU_BAS, YURU_SON)


def _durum(t, Y):
    """Çantanın sap tepesi (ekran) ve açısı."""
    s = Y["s"]
    yuru = O.yumusak(O.ara(t, 11.00, 11.15)) * (1 - O.yumusak(O.ara(t, 11.85, 12.02)))
    kalk = 46 * s * O.yumusak(O.ara(t, 10.98, 11.16))
    cek = O.yumusak(O.ara(t, 12.22, 13.0))                            # alan el çantayı kendine çeker
    x = Y["bx"] + 9 * s * np.sin(t * 0.8) + 12 * s * np.sin(2 * np.pi * (t - 11.1) / 0.6) * yuru + 64 * s * cek
    y = Y["pivot_y"] - kalk + 9 * s * np.cos(2 * np.pi * (t - 11.1) / 0.3) * yuru + 4 * s * np.sin(t * 1.3) - 14 * s * cek
    aci = 3.6 * np.sin(2 * np.pi * (t - 11.1) / 0.6 + 0.6) * yuru - 1.6 * cek + 0.7 * np.sin(t * 1.1) * cek
    if t > DUSUS:                                                     # tulum düşünce küçük esneme
        y += 7 * s * np.exp(-(t - DUSUS) * 16) * np.cos((t - DUSUS) * 40)
    if t > Z.EL_DEGISTIRME:
        d = t - Z.EL_DEGISTIRME
        aci += 3.2 * np.exp(-d * 4.5) * np.sin(2 * np.pi * 1.8 * d)
        y += 10 * s * np.exp(-d * 6) * np.sin(2 * np.pi * 2.2 * d)
    return float(x), float(y), float(aci)


def _kurdele_yollari(Y):
    """Yerel koordinatta bant: ön yay (soldan sağa), arka yay (sağdan sola), fiyonk merkezi."""
    s, bw, bh = Y["s"], Y["bw"], Y["bh"]
    yb = 120 * s + bh * 0.54
    a, b = bw / 2 + 9 * s, 15 * s
    th = np.linspace(np.pi, 0, 30)
    on = np.stack([a * np.cos(th), yb + b * np.sin(th)], 1).astype(np.float32)
    th = np.linspace(0, -np.pi, 30)
    arka = np.stack([a * np.cos(th), yb + b * np.sin(th)], 1).astype(np.float32)
    return on, arka, np.array([0, yb + b], np.float32)


def _fiyonk(c, s, g, sal=0.0):
    """Fiyonk ilmekleri ve uçları (yerel), g: 0..1 büyüme; sal: uçların gecikmeli salınımı (piksel)."""
    out = []
    for yn in (-1, 1):
        out.append(O.bezier(c, c + np.array([yn * 150 + sal * 0.3, -112]) * s * g, c + np.array([yn * 168 + sal * 0.4, 66]) * s * g, c, 26))
        out.append(O.bezier(c, c + np.array([yn * 26, 44]) * s * g, c + np.array([yn * 66 + sal * 0.7, 70]) * s * g, c + np.array([yn * 76 + sal, 122]) * s * g, 12))
    return out


def _bildirim(tuval, t, Y):
    K = Y["K"]
    p = O.ara(t, Z.BILDIRIM, Z.BILDIRIM + 0.40)
    if p <= 0:
        return
    alfa = O.ara(t, Z.BILDIRIM, Z.BILDIRIM + 0.12)
    if K.dikey:                                                       # iki satır: üstte ikon + ad + saat, altta büyük mesaj
        x0, x1, yuk, pt, ik = 54, 1026, 252, 62, 70
        y0 = O.kar(-yuk - 40, 132, O.geri(p, 1.3)) + 3 * np.sin(t * 1.7)
        O.cam_kart(tuval, x0, y0, x1, y0 + yuk, r=44, alfa=alfa, opak=0.88)
        O.yapistir(tuval, O.marka("ikon", ik), x0 + 40 + ik / 2, y0 + 70, alfa=alfa)
        w = O.yaz(tuval, "Annea", x0 + 40 + ik + 18, y0 + 84, "nunito", 36, O.METIN, 800, alfa)
        O.yaz(tuval, "şimdi", x1 - 42, y0 + 84, "nunito", 32, O.METIN2, 600, alfa, hiza="sag")
        tx, ty = x0 + 42, y0 + 196
    else:
        x0, x1, yuk, pt, ik = 1920 - 50 - 770, 1920 - 50, 150, 38, 80
        y0 = O.kar(-yuk - 40, 60, O.geri(p, 1.3)) + 3 * np.sin(t * 1.7)
        O.cam_kart(tuval, x0, y0, x1, y0 + yuk, r=30, alfa=alfa, opak=0.88)
        O.yapistir(tuval, O.marka("ikon", ik), x0 + 30 + ik / 2, y0 + yuk / 2, alfa=alfa)
        tx, ty = x0 + 30 + ik + 24, y0 + yuk * 0.74
        O.yaz(tuval, "Annea · şimdi", tx, y0 + yuk * 0.36, "nunito", 23, O.METIN2, 700, alfa)
    w = O.yaz(tuval, "Aslı:", tx, ty, "nunito", pt, O.METIN, 900, alfa)
    O.yaz(tuval, " Çok yakıştı, teşekkürler!", tx + w, ty, "nunito", pt, O.METIN, 600, alfa)


# ------------------------------------------------------------------ çizgi (yalnız ortak.cizgi)
PARLAK = (255, 190, 176)


def _kurdele_ciz(tuval, yol, kal, parlama=0.5):
    """Kurdele: filmin çizgisi + üstünde ince açık bir geçişle saten parlaklığı (ikisi de ortak.cizgi)."""
    O.cizgi(tuval, yol, kal, parlama=parlama)
    son = yol[-1] if isinstance(yol, list) else yol
    O.cizgi(tuval, np.asarray(son, np.float32) - (0, kal * 0.20), max(2.0, kal * 0.22), alfa=0.55, renk=PARLAK, parlama=0)


def _serbest(tuval, t, Y, c0):
    """Sönmeyle birlikte kurdele çözülür, tek çizgi K.etk_giris'e akar."""
    K, s = Y["K"], Y["s"]
    hedef = np.array(K.etk_giris, np.float32)
    R = O.bezier(c0, c0 + np.array([-150, -330]) * s, hedef + (40, 260), hedef, 60)

    def yol(u):
        h = O.gir_cik(O.ara(u, SON_BAS, UC_VARIS))
        kuy = h - O.kar(0.42, 0.075, O.ara(u, SON_BAS + 0.05, UC_VARIS))
        return O.yol_parca(R, max(0.0, kuy), max(h, 0.002))
    O.cizgi(tuval, O.alt_zaman(yol, t), O.kar(16 * s, 10, O.ara(t, SON_BAS, UC_VARIS - 0.05)))


def ciz(t, K, tuval):
    Y = _yer(K.dikey)
    s, bw, bh = Y["s"], Y["bw"], Y["bh"]
    ciplak = tuval.copy()
    son = O.yumusak(O.ara(t, SON_BAS, SON_BIT))
    px, py, aci = _durum(t, Y)
    M = O.afin(1, aci, (0, 0), (px, py))
    on_y, arka_y, fc = _kurdele_yollari(Y)

    if son < 1:
        kay = _kay(t, Y)
        hiz = kay - _kay(t - 1 / Z.FPS, Y)
        arka_plan = _arka(t, Y, kay, hiz).astype(np.float32) * (ciplak.astype(np.float32) / np.array(O.KREM, np.float32))
        tuval[:] = np.clip(arka_plan, 0, 255).astype(np.uint8)
        kalk = O.yumusak(O.ara(t, 10.98, 11.16))
        gy_golge = Y["ust"] + bh + 4 * s                               # zeminde: geniş yumuşak gölge + dar temas gölgesi
        O.cokgen(tuval, _elips(px + 26 * s, gy_golge + 6 * s, bw * (0.62 - 0.10 * kalk), 30 * s), (90, 60, 40), 0.24 - 0.10 * kalk, bulanik=16 * s)
        if kalk < 1:
            O.cokgen(tuval, _elips(px + 6 * s, gy_golge - 2 * s, bw * 0.50, 10 * s), (70, 46, 30), 0.38 * (1 - kalk), bulanik=5 * s)

        c_arka, c_on, (ox, oy) = _canta(K.dikey)
        Ms = O.afin(1, aci, (ox, oy), (px, py))
        # kurdele: sarılma (10.40–10.74) ve fiyonk (10.70–10.83)
        bas_ = np.array(K.s5_giris, np.float32)
        sol = O.nokta(M, on_y[:1])[0]
        yaklas = O.bezier(bas_, bas_ + np.array([-190, -60]) * s, sol + np.array([-150, -130]) * s, sol, 30)
        kb = Z.KURDELE_BAS

        def uc(u):
            return O.ara(u, kb, kb + 0.11) + O.ara(u, kb + 0.11, kb + 0.23) + O.ara(u, kb + 0.23, kb + 0.34)

        def kuyruk(u):
            return O.yumusak(O.ara(u, kb + 0.10, kb + 0.31))
        kal = O.kar(13, 20 * s, O.yumusak(O.ara(t, kb + 0.05, kb + 0.34)))

        def parca(i, P):
            def f(u):
                a, b = max(kuyruk(u), i) - i, min(uc(u), i + 1) - i
                return O.yol_parca(P, a, b) if b > a + 1e-4 else None
            if t > kb + 0.36:
                return [P] if i else []                # sarım bitti: yaklaşma parçası çekildi, bant kaldı
            L = [p for p in O.alt_zaman(f, t) if p is not None]
            return L if len(L) == 4 else ([f(t)] if f(t) is not None else [])
        arka_ekran, on_ekran = O.nokta(M, arka_y), O.nokta(M, on_y)

        # tulum: katlanır, çantaya düşer; "sarsın" derken çıkıp kollarını açar
        k1, k2, k3 = Z.KATLAMA
        cik = O.gir_cik(O.ara(t, CIK_BAS, CIK_TEPE))
        f1 = O.gir_cik(O.ara(t, k1 - 0.03, k2 - 0.02)) - O.geri(O.ara(t, CIK_BAS + 0.26, CIK_TEPE + 0.06), 2.0)
        f2 = O.gir_cik(O.ara(t, k2 - 0.02, k3 - 0.03)) - O.gir_cik(O.ara(t, CIK_BAS + 0.10, CIK_TEPE - 0.10))
        dus = O.gir(O.ara(t, k3 - 0.04, DUSUS), 2)
        if cik > 0.02:                                                 # arkasında sıcak ışık
            c = O.nokta(M, [(-104 * s * cik, (36 - 110 * cik) * s)])[0]
            O.cokgen(tuval, _elips(c[0], c[1], 300 * s, 300 * s), (255, 240, 206), 0.42 * cik * (0.9 + 0.1 * np.sin(t * 5)), bulanik=60 * s)

        L = parca(2, arka_ekran)                                       # arka yarı: çantanın ARKASINDA
        if L:
            O.cizgi(tuval, L, kal, parlama=0.5)
        O.donustur(tuval, c_arka, Ms)
        hr, capa = _tulum(Y, round(float(np.clip(f1, -0.12, 1)), 3), round(float(np.clip(f2, 0, 1)), 3))
        if dus < 1:
            hx = Y["bx"] + 14 * s * np.sin(t * 2.1)
            hy = O.kar(Y["ust"] - 500 * s + 14 * s * np.sin(t * 3.0), Y["ust"] - 84 * s, dus)
            ta = 5 * np.sin(t * 2.4) * (1 - dus) + 2.0 * yay_sal(f1) - 2.0 * yay_sal(f2)
            O.donustur(tuval, hr, O.afin(1 + 0.05 * np.sin(np.pi * f1) + 0.05 * np.sin(np.pi * f2), ta, capa, (hx, hy)))
        else:
            yer = O.nokta(M, [(-104 * s * cik, (36 - 250 * cik) * s + 5 * s * np.sin(t * 3.1) * cik)])[0]   # sola ve alçakta: el önünü kapatmaz
            O.donustur(tuval, hr, O.afin(1 + 0.04 * cik, aci + (-5 + 2.5 * np.sin(t * 2.3)) * cik, capa, yer))
        O.donustur(tuval, c_on, Ms)
        if t < kb:                                                     # devir: K.s5_giris'te nokta, sonra kısa kıpırtı
            g = O.yumusak(O.ara(t, 10.0, 10.25))
            x = np.linspace(-22, 22, 12) * g
            O.cizgi(tuval, np.stack([bas_[0] + x, bas_[1] + 6 * g * np.sin(x / 9 + t * 9)], 1), 13)
        else:
            L = parca(0, yaklas)                                       # yaklaşma ve ön yarı: çantanın ÖNÜNDE
            if L:
                O.cizgi(tuval, L, kal, parlama=0.6)
            L = parca(1, on_ekran)
            if L:
                _kurdele_ciz(tuval, L, kal, 0.6)
            g = O.geri(O.ara(t, kb + 0.30, Z.KURDELE_SON), 2.2)
            if g > 0.01:
                sal = (_durum(t - 0.09, Y)[2] - aci) * 14 + (_durum(t - 0.09, Y)[0] - px) * 0.8      # uçlar gecikerek izler
                for P in _fiyonk(fc, s, g * (1 + 0.03 * np.sin(2 * np.pi * (t - 11.1) / 0.3) * kalk), sal):
                    _kurdele_ciz(tuval, O.nokta(M, P), 12 * s, 0.45)
                O.cizgi(tuval, O.nokta(M, [fc, fc + (0.1, 0)]), 30 * s * min(g, 1.0), parlama=0.3)
        # eller: yan kenarlardan giren, dirsekten bükülü kollar
        g1 = O.cik(O.ara(t, 10.84, 11.02)) - O.gir(O.ara(t, 12.03, 12.34), 2)
        if g1 > 0:
            _kol(tuval, K, O.nokta(M, [(-56 * s, 24 * s)])[0], s, -1, KOL1, TEN1, g1)
        g2 = O.cik(O.ara(t, 11.80, 11.99))
        if g2 > 0:
            _kol(tuval, K, O.nokta(M, [(56 * s, 24 * s)])[0], s, 1, KOL2, TEN2, g2)
        for gec in (0.0, 0.14):                                        # 12.00: sıcak ışık nabzı
            p = O.ara(t, Z.EL_DEGISTIRME + gec, Z.EL_DEGISTIRME + gec + 0.6)
            if 0 < p < 1:
                O.halka(tuval, px, py + 22 * s, (70 + 230 * O.cik(p)) * s, 0.42 * (1 - p), (255, 222, 180), 6 * s)
        _bildirim(tuval, t, Y)
        if son > 0:                                                    # geriye bulanıklaşıp çıplak zemine söner
            tuval = O.karistir(O.bulanik(tuval, 16 * son), ciplak, son)
    else:
        tuval = ciplak
    if t >= SON_BAS:
        px0, py0, aci0 = _durum(SON_BAS, Y)
        _serbest(tuval, t, Y, O.nokta(O.afin(1, aci0, (0, 0), (px0, py0)), [fc])[0])
    return tuval


def yay_sal(f):
    """Katlama adımının ortasında küçük savrulma (0 -> 0)."""
    return float(np.sin(np.pi * np.clip(f, 0, 1)))

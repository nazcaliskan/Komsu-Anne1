"""S5 · Elden ele (9.85–13.30) — yüz ve figür yok: nesneler, iki el ve hareket.

Tulum katlanıp kâğıt çantaya girer; çizgi çantayı bir kez sarıp kurdele olur;
tek el çantayı alır ve kamera onunla birlikte "bir sokak öteye" kayar (A kapısı,
sokak, B kapısı paralaksla akar); 12.00'de çanta ikinci ele geçer; bildirim iner;
sahne söner, kurdele çözülüp tek çizgi olarak S6'ya akar.
"""
import functools
import numpy as np, cv2
import ortak as O
from ortak import Z

ARALIK = (9.85, 13.3)

TEN1, TEN2 = (249, 220, 200), (227, 170, 130)
KOL1, KOL2 = (168, 150, 214), (122, 158, 136)       # lavanta, adaçayı
KRAFT_KOYU, KRAFT_ACIK, SAP = (178, 140, 100), (226, 192, 152), (156, 118, 82)
YURU_BAS, YURU_SON = 10.90, 11.98
DUSUS = 10.36                                        # tulum çantaya düşer


# ------------------------------------------------------------------ yerleşim
@functools.lru_cache(maxsize=None)
def _yer(dikey):
    K = O.DIKEY if dikey else O.YATAY
    s = 1.0 if dikey else 0.68
    bw, bh = 490 * s, 560 * s
    ust = 930 if dikey else 492                      # çanta ön yüzünün üst kenarı (dinlenme)
    return dict(K=K, s=s, bw=bw, bh=bh, bx=K.cx, ust=ust, pivot_y=ust - 120 * s, gy=ust + bh - 70 * s,
                D=(2200 if dikey else 1750), yon1=np.array((-0.78, -0.62) if dikey else (-0.90, -0.44), np.float32))


def _elips(cx, cy, rx, ry, aci=0.0, n=40):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.stack([rx * np.cos(a), ry * np.sin(a)], 1)
    c, s_ = np.cos(aci), np.sin(aci)
    return p @ np.array([[c, s_], [-s_, c]], np.float32) + (cx, cy)


# ------------------------------------------------------------------ arka plan (yarı çözünürlük, bulanık, paralaks)
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
    # uzak katman: ağaçlar, lamba (yalnız evler arasındaki aralıkta görünür)
    for wx, boy, ton in [(700, 520, (176, 206, 186)), (980, 640, (154, 190, 168)), (1280, 560, (186, 212, 194)),
                         (1560, 660, (160, 196, 174)), (1840, 540, (178, 208, 188))]:
        wx = wx * Dw / 2200
        O.kutu(g, X(wx - 12, .62), (gy - boy * s * .55) * q, X(wx + 12, .62), gy * q, 4, (176, 146, 120))
        for dx, dy, r in [(0, 0, 150), (-90, 60, 110), (95, 50, 115)]:
            O.daire(g, X(wx + dx, .62), (gy - boy * s + dy * s) * q, r * s * q, ton)
    lx = 1120 * Dw / 2200
    O.kutu(g, X(lx - 7, .8), (gy - 760 * s) * q, X(lx + 7, .8), gy * q, 3, (120, 104, 96))
    O.daire(g, X(lx, .8), (gy - 780 * s) * q, 34 * s * q, (255, 226, 170))
    for i in range(-2, 30):                                           # çit
        cx = 640 + i * 62
        if 600 < cx < Dw - 740:
            kutu(cx, gy - 250 * s, cx + 40, gy, (255, 251, 246), r=14)
    kutu(600, gy - 150 * s, Dw - 740, gy - 120 * s, (255, 251, 246))
    # orta katman: A evi, B evi
    kutu(-2600, 0, 620, gy, (246, 222, 200))
    kutu(Dw - 760, 0, Dw + 2600, gy, (226, 234, 222))
    for i in range(1, 9):                                             # duvar derzleri
        y = gy - i * 230 * s
        O.kutu(g, 0, y * q, w, (y + 3) * q, 0, (255, 255, 255), 0.16)
    kutu(-760, gy - 900 * s, -440, gy - 500 * s, (255, 246, 232), r=16)          # A penceresi
    kutu(-735, gy - 875 * s, -465, gy - 525 * s, (255, 232, 190), r=8)
    for wx, kapi, pano in [(0, (233, 146, 120), (241, 172, 148)), (Dw, (122, 158, 136), (148, 180, 160))]:
        kutu(wx - 240, gy - 990 * s, wx + 240, gy, (255, 246, 236), r=22)        # söve
        kutu(wx - 200, gy - 950 * s, wx + 200, gy, kapi, r=12)
        kutu(wx - 150, gy - 890 * s, wx + 150, gy - 560 * s, pano, r=12)
        kutu(wx - 150, gy - 500 * s, wx + 150, gy - 70 * s, pano, r=12)
        O.daire(g, X(wx + 158), (gy - 470 * s) * q, 15 * s * q, (255, 226, 170))
    kutu(Dw + 330, gy - 880 * s, Dw + 640, gy - 500 * s, (255, 250, 240), r=16)  # B penceresi
    kutu(Dw + 355, gy - 855 * s, Dw + 615, gy - 525 * s, (255, 236, 198), r=8)
    O.daire(g, X(Dw), (gy - 1085 * s) * q, 78 * s * q, (255, 250, 244))           # B kapısının üstünde yeşil pin
    O.pin(g, X(Dw), (gy - 1030 * s) * q, 116 * s * q, O.ADACAYI, delik=(255, 250, 244))
    # zemin ve paspaslar; yakın katman derzleri daha hızlı akar
    O.kutu(g, 0, gy * q, w, h, 0, (238, 217, 196))
    O.kutu(g, 0, gy * q, w, (gy + 10 * s) * q, 0, (224, 198, 174))
    for wx, renk in [(0, (214, 160, 124)), (Dw, (150, 178, 158))]:
        kutu(wx - 250, gy + 26 * s, wx + 250, gy + 120 * s, renk, r=18)
    for i in range(-6, 40):
        x = X(i * 300, 1.3)
        if -4 < x < w + 4:
            O.cizgi_ince(g, [(x, (gy + 12 * s) * q), (x - 40 * q, h)], (214, 190, 166), 2, 0.6)
    g = O.bulanik(g, (5.5 if K.dikey else 4.0))
    k = int(abs(hiz) * 0.28) | 1                                      # yatay hareket bulanıklığı (hıza göre)
    if k > 2:
        g = cv2.blur(g, (min(k, 61), 1))
    z = 1 + 0.03 * O.ara(t, ARALIK[0], ARALIK[1])                     # kamera çok yavaş iter
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
        P = O.egri([(-0.20 * bw, ust + 10 * s + dy), (-0.15 * bw, 30 * s + dy), (0, 4 * s + dy), (0.15 * bw, 30 * s + dy), (0.20 * bw, ust + 10 * s + dy)], 16)
        O.cizgi_ince(k, P + (ox, oy), renk, 10 * s)

    sap(arka, -18 * s, (140, 104, 72))
    O.cokgen(arka, np.array([(-bw / 2 + 8 * s, ust - 24 * s), (bw / 2 - 8 * s, ust - 24 * s), (bw / 2, ust + 40 * s), (-bw / 2, ust + 40 * s)]) + (ox, oy), KRAFT_KOYU)
    O.kutu(on, ox - bw / 2, oy + ust, ox + bw / 2, oy + ust + bh, 10 * s, O.KRAFT)
    O.kutu(on, ox - bw / 2, oy + ust, ox + bw / 2, oy + ust + 40 * s, 8 * s, KRAFT_ACIK)
    O.kutu(on, ox + bw * 0.30, oy + ust + 40 * s, ox + bw / 2, oy + ust + bh, 8 * s, (150, 110, 70), 0.14)   # yan gölge
    O.cizgi_ince(on, [(ox - bw * 0.30, oy + ust + 44 * s), (ox - bw * 0.30, oy + ust + bh - 6 * s)], (255, 240, 214), 3 * s, 0.35)
    O.kutu(on, ox - bw / 2, oy + ust + bh - 46 * s, ox + bw / 2, oy + ust + bh, 10 * s, (150, 110, 70), 0.10)
    sap(on, 4 * s, SAP)
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
    a = k[..., 3:].astype(np.float32) / 255                           # şekli çizgili kumaşla doldur
    y = (np.arange(k.shape[0], dtype=np.float32) - oy) / (27 * s)
    serit = ((np.floor(y) % 2) == 1)[:, None, None]
    k[..., :3] = (np.where(serit, np.array(O.ADACAYI, np.float32) * 0.5 + 127 * 0.5 + 30, np.array(krem, np.float32)) * a + 0.5).astype(np.uint8)
    O.cokgen(k, _elips(ox, oy + 2 * s, 46 * s, 24 * s), (236, 226, 210))                 # yaka
    O.cizgi_ince(k, _elips(ox, oy + 2 * s, 46 * s, 24 * s)[:21], O.ADACAYI, 5 * s)
    for i in range(3):
        O.daire(k, ox, oy + (58 + i * 38) * s, 5.5 * s, O.ADACAYI, 0.9 * (1 if 58 + i * 38 < Hg / s - 20 else 0))
    if f1 > 0.5:                                                      # katlanan kolların kıvrım gölgesi
        for yn in (-1, 1):
            O.cizgi_ince(k, [(ox + yn * (W / 2 - 34 * s), oy + 26 * s), (ox + yn * (W / 2 - 34 * s), oy + Hg - 22 * s)], (120, 100, 84), 3 * s, 0.22 * (f1 - 0.5) * 2)
    if f2 > 0.5:
        O.cizgi_ince(k, [(ox - W / 2 + 22 * s, oy + Hg - 16 * s), (ox + W / 2 - 22 * s, oy + Hg - 16 * s)], (120, 100, 84), 3 * s, 0.25 * (f2 - 0.5) * 2)
    return k, (ox, oy)


# ------------------------------------------------------------------ el
def _el(dst, x, y, yon, kol, ten, s):
    """Sapları kavramış kapalı el; kol `yon` doğrultusunda kadrajdan çıkar."""
    yon = np.asarray(yon, np.float32); yon /= np.linalg.norm(yon)
    dik = np.array([-yon[1], yon[0]], np.float32)
    p = np.array([x, y], np.float32)

    def bant(a, b, gen, renk):
        O.cokgen(dst, np.array([p + yon * a + dik * gen / 2, p + yon * b + dik * gen / 2, p + yon * b - dik * gen / 2, p + yon * a - dik * gen / 2]), renk)

    koyu = tuple(int(c * 0.86) for c in ten)
    bant(84 * s, 1700 * s, 98 * s, kol)
    bant(70 * s, 124 * s, 112 * s, tuple(min(255, int(c * 1.12)) for c in kol))          # kol ağzı
    bant(10 * s, 76 * s, 74 * s, ten)                                                     # bilek
    aci = float(np.arctan2(yon[1], yon[0]))
    O.cokgen(dst, _elips(p[0], p[1], 66 * s, 56 * s, -aci), ten)                          # yumruk
    for i in (-1, 0, 1):                                                                  # parmak araları
        c = p - yon * 30 * s + dik * i * 24 * s
        O.cizgi_ince(dst, [c - yon * 22 * s, c + yon * 16 * s], koyu, 3.2 * s, 0.75)
    b = p + yon * 6 * s + dik * (34 * s if yon[0] < 0 else -34 * s) + (0, 30 * s)         # başparmak
    O.cokgen(dst, _elips(b[0], b[1], 34 * s, 20 * s, -aci + 0.5 * np.sign(yon[0])), ten)
    O.cizgi_ince(dst, _elips(b[0], b[1], 34 * s, 20 * s, -aci + 0.5 * np.sign(yon[0]))[8:26], koyu, 2.6 * s, 0.6)


# ------------------------------------------------------------------ hareket
def _kay(t, Y):
    """Kameranın yana kayışı (dünya pikseli)."""
    return Y["D"] * O.gir_cik(O.ara(t, YURU_BAS, YURU_SON)) + 26 * Y["s"] * (t - ARALIK[0]) - 26 * Y["s"] * (YURU_SON - ARALIK[0]) * O.ara(t, YURU_BAS, YURU_SON)


def _durum(t, Y):
    """Çantanın sap tepesi (ekran), açısı."""
    s = Y["s"]
    yuru = O.yumusak(O.ara(t, 11.00, 11.15)) * (1 - O.yumusak(O.ara(t, 11.85, 12.02)))
    kalk = 46 * s * O.yumusak(O.ara(t, 10.98, 11.16))
    x = Y["bx"] + 9 * s * np.sin(t * 0.8) + 12 * s * np.sin(2 * np.pi * (t - 11.1) / 0.6) * yuru
    y = Y["pivot_y"] - kalk + 9 * s * np.cos(2 * np.pi * (t - 11.1) / 0.3) * yuru + 4 * s * np.sin(t * 1.3)
    aci = 3.6 * np.sin(2 * np.pi * (t - 11.1) / 0.6 + 0.6) * yuru
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


def _fiyonk(c, s, g):
    """Fiyonk ilmekleri ve uçları (yerel), g: 0..1 büyüme."""
    out = []
    for yn in (-1, 1):
        out.append(O.bezier(c, c + np.array([yn * 150, -112]) * s * g, c + np.array([yn * 168, 66]) * s * g, c, 26))
        out.append(O.bezier(c, c + np.array([yn * 26, 44]) * s * g, c + np.array([yn * 66, 70]) * s * g, c + np.array([yn * 76, 122]) * s * g, 12))
    return out


def _bildirim(tuval, t, Y):
    K, s = Y["K"], Y["s"]
    p = O.ara(t, Z.BILDIRIM, Z.BILDIRIM + 0.40)
    if p <= 0:
        return
    if K.dikey:
        x0, x1, y0, yuk, pt, ik = 50, 1030, 150, 196, 47, 104
    else:
        x0, x1, y0, yuk, pt, ik = 1920 - 70 - 800, 1920 - 70, 64, 150, 38, 80
    y0 = O.kar(-yuk - 40, y0, O.geri(p, 1.3)) + 3 * np.sin(t * 1.7)
    alfa = O.ara(t, Z.BILDIRIM, Z.BILDIRIM + 0.12)
    O.cam_kart(tuval, x0, y0, x1, y0 + yuk, r=38 if K.dikey else 30, alfa=alfa, opak=0.86)
    O.yapistir(tuval, O.marka("ikon", ik), x0 + 30 + ik / 2, y0 + yuk / 2, alfa=alfa)
    tx = x0 + 30 + ik + 24
    O.yaz(tuval, "Annea · şimdi", tx, y0 + yuk * 0.36, "nunito", pt * 0.60, O.METIN2, 700, alfa)
    w = O.yaz(tuval, "Aslı:", tx, y0 + yuk * 0.74, "nunito", pt, O.METIN, 900, alfa)
    O.yaz(tuval, " Çok yakıştı, teşekkürler!", tx + w, y0 + yuk * 0.74, "nunito", pt, O.METIN, 600, alfa)


# ------------------------------------------------------------------ çizgi (yalnız ortak.cizgi)
def _serbest(t, Y, c0):
    """12.90'dan sonra: kurdele çözülür, tek çizgi K.s6_bas'a akar."""
    K, s = Y["K"], Y["s"]
    hedef = np.array(K.s6_bas, np.float32)
    R = O.bezier(c0, c0 + np.array([-150, -330]) * s, hedef + (-230, -190), hedef, 60)
    def yol(u):
        h = O.gir_cik(O.ara(u, Z.S6_GECIS, 13.25))
        kuy = h - O.kar(0.42, 0.085, O.ara(u, Z.S6_GECIS + 0.05, 13.25))
        return O.yol_parca(R, max(0.0, kuy), max(h, 0.002))
    O.cizgi(_serbest.tuval, O.alt_zaman(yol, t), O.kar(16 * s, 10, O.ara(t, Z.S6_GECIS, 13.2)))


def ciz(t, K, tuval):
    Y = _yer(K.dikey)
    s, bw, bh = Y["s"], Y["bw"], Y["bh"]
    ciplak = tuval.copy()
    son = O.yumusak(O.ara(t, Z.S6_GECIS, 13.22))
    px, py, aci = _durum(t, Y)
    M = O.afin(1, aci, (0, 0), (px, py))
    on_y, arka_y, fc = _kurdele_yollari(Y)

    if son < 1:
        kay = _kay(t, Y)
        hiz = (_kay(t, Y) - _kay(t - 1 / Z.FPS, Y))
        arka_plan = _arka(t, Y, kay, hiz).astype(np.float32) * (ciplak.astype(np.float32) / np.array(O.KREM, np.float32))
        tuval[:] = np.clip(arka_plan, 0, 255).astype(np.uint8)
        kalk = O.yumusak(O.ara(t, 10.98, 11.16))
        gy_golge = Y["ust"] + bh + 6 * s                                # çanta gölgesi
        O.cokgen(tuval, _elips(px, gy_golge, bw * (0.56 - 0.08 * kalk), 26 * s), (110, 70, 45), 0.26 - 0.10 * kalk, bulanik=12 * s)

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
            L = [p for p in O.alt_zaman(f, t) if p is not None]
            return L if len(L) == 4 or (L and t > kb + 0.34) else (L[-1:] if f(t) is not None else [])
        arka_ekran, on_ekran = O.nokta(M, arka_y), O.nokta(M, on_y)

        L = parca(2, arka_ekran)                                       # arka yarı: çantanın ARKASINDA
        if L:
            O.cizgi(tuval, L, kal, parlama=0.5)
        O.donustur(tuval, c_arka, Ms)
        # tulum: katlanır, çantaya düşer
        k1, k2, k3 = getattr(Z, 'KATLAMA', [9.95, 10.12, 10.28])        # ses efektleriyle aynı anlar
        f1 = O.gir_cik(O.ara(t, k1 - 0.03, k2 - 0.02))
        f2 = O.gir_cik(O.ara(t, k2 - 0.02, k3 - 0.03))
        dus = O.gir(O.ara(t, k3 - 0.04, DUSUS), 2)
        hr, capa = _tulum(Y, round(f1, 3), round(f2, 3))
        if dus < 1:
            hx = Y["bx"] + 14 * s * np.sin(t * 2.1)
            hy = O.kar(Y["ust"] - 500 * s + 14 * s * np.sin(t * 3.0), Y["ust"] - 84 * s, dus)
            ta = 5 * np.sin(t * 2.4) * (1 - dus) + 2.0 * yay_sal(f1) - 2.0 * yay_sal(f2)
            O.donustur(tuval, hr, O.afin(1 + 0.05 * np.sin(np.pi * f1) + 0.05 * np.sin(np.pi * f2), ta, capa, (hx, hy)))
        else:
            O.donustur(tuval, hr, O.afin(1, aci, capa, O.nokta(M, [(0, 120 * s - 84 * s)])[0]))
        O.donustur(tuval, c_on, Ms)
        if t < kb:                                                     # devir: K.s5_giris'te nokta, sonra kısa kıpırtı
            g = O.yumusak(O.ara(t, 10.0, 10.25))
            x = np.linspace(-22, 22, 12) * g
            O.cizgi(tuval, np.stack([bas_[0] + x, bas_[1] + 6 * g * np.sin(x / 9 + t * 9)], 1), 13)
        else:
            for i, P in ((0, yaklas), (1, on_ekran)):                  # yaklaşma ve ön yarı: çantanın ÖNÜNDE
                L = parca(i, P)
                if L:
                    O.cizgi(tuval, L, kal, parlama=0.6)
            g = O.geri(O.ara(t, kb + 0.30, Z.KURDELE_SON), 2.2)
            if g > 0.01:
                for P in _fiyonk(fc, s, g):
                    O.cizgi(tuval, O.nokta(M, P), 12 * s, parlama=0.45)
                O.cizgi(tuval, O.nokta(M, [fc, fc + (0.1, 0)]), 30 * s * min(g, 1.0), parlama=0.3)
        # eller
        yon1 = Y["yon1"]; yon2 = yon1 * (-1, 1)
        g1 = O.cik(O.ara(t, 10.84, 11.02)) - O.gir(O.ara(t, 12.03, 12.32), 2)
        if 0 < g1:
            p1 = O.nokta(M, [(-56 * s, 24 * s)])[0] + yon1 * 1100 * s * (1 - g1)
            _el(tuval, p1[0], p1[1], yon1, KOL1, TEN1, s)
        g2 = O.cik(O.ara(t, 11.80, 11.99))
        if g2 > 0:
            p2 = O.nokta(M, [(56 * s, 24 * s)])[0] + yon2 * 1100 * s * (1 - g2)
            _el(tuval, p2[0], p2[1], yon2, KOL2, TEN2, s)
        for gec in (0.0, 0.14):                                        # 12.00: sıcak ışık nabzı
            p = O.ara(t, Z.EL_DEGISTIRME + gec, Z.EL_DEGISTIRME + gec + 0.6)
            if 0 < p < 1:
                O.halka(tuval, px, py + 22 * s, (70 + 230 * O.cik(p)) * s, 0.42 * (1 - p), (255, 222, 180), 6 * s)
        _bildirim(tuval, t, Y)
        if son > 0:                                                    # 12.90–13.22: geriye bulanıklaşıp çıplak zemine söner
            tuval = O.karistir(O.bulanik(tuval, 16 * son), ciplak, son)
    else:
        tuval = ciplak
    if t >= Z.S6_GECIS:
        px0, py0, aci0 = _durum(Z.S6_GECIS, Y)
        _serbest.tuval = tuval
        _serbest(t, Y, O.nokta(O.afin(1, aci0, (0, 0), (px0, py0)), [fc])[0])
    return tuval


def yay_sal(f):
    """Katlama adımının ortasında küçük savrulma (0 -> 0)."""
    return float(np.sin(np.pi * f))

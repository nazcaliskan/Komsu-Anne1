"""S2 (Sor) + S3 (Tek dokunuş): telefon, soru sayfası, cevap kartları, grup sayfası, buton dolgusu.

Çizgi sahipliği: t=2.00'de K.s2_giris'te devralınır, t=Z.FIRLAMA (6.60) anında K.dugme'de bırakılır.
Arayüz düz bir ekran haresine (780x1688) çizilir, telefon gövdesiyle birlikte hafif perspektifle basılır.
"""
import functools
import numpy as np, cv2
from ortak import *  # noqa: F401,F403  (Z dahil)
import ortak as O

ARALIK = (1.6, 7.1)

UW, UH = 780, 1688            # ekran haresi (uygulamanın 390x844'ünün 2 katı)
CER, PAY = 22, 44             # gövde çerçevesi, hare kenar payı (bulanıklık için)
GW, GH = UW + 2 * (CER + PAY), UH + 2 * (CER + PAY)
EKRAN_R = 86
ALAN = (60, 310, 720, 430)    # soru alanı (arayüz pikseli)
ALAN_ALT = 428                # odak çizgisinin oturduğu alt kenar
CIZ_X0, CIZ_X1 = 78, 702
DUGME = (60, 1135, 720, 1225)
DUGME_CY, DUGME_H = 1180, 90
KAPAK0, KAPAK1 = DUGME[0] + DUGME_H / 2, DUGME[2] - DUGME_H / 2     # hap uçlarının merkezleri
YUVA = [(30, 770, 750, 960), (30, 980, 750, 1170), (30, 1190, 750, 1380)]
CEVAP = [("Elif", "Bizde beyaz gürültü işe yaradı.", "a"),
         ("Zeynep", "Akşam banyosu mucize oldu.", "b"),
         ("Merve", "Geçecek, yalnız değilsin.", "d")]
ETIKET = "2 sokak ötede"
KART_SURE = 0.33
GOVDE_RENK = (50, 42, 40)


@functools.lru_cache(maxsize=None)
def _yer(dikey):
    K = DIKEY if dikey else YATAY
    cx, cy, en, boy = K.dugme
    sf = en / (DUGME[2] - DUGME[0])
    son = (cx, cy - (DUGME_CY - UH / 2) * sf, sf)
    if dikey:
        return dict(c0=(540, 1010), s0=0.80, son=son, kart=(900, 236),
                    kc=[(522, 884), (548, 1152), (528, 1420)], ka=[-1.6, 1.2, -0.8], yon=[-1, 1, -1],
                    hos=(540, 992, 760, 176), grup=None)
    return dict(c0=(600, 545), s0=0.536, son=son, kart=(860, 226),
                kc=[(1360, 268), (1405, 540), (1368, 812)], ka=[-1.5, 1.0, -1.2], yon=[1, 1, 1],
                hos=(1408, 716, 800, 186), grup=(1408, 384, 880, 320))


# ------------------------------------------------------------------ telefonun duruşu
def _poz(t, K):
    """(cx, cy, ölçek, yaw, pitch, roll). Hiç tam durmaz: yavaş itme + hafif salınım."""
    Y = _yer(K.dikey)
    sal = lambda f, ph: np.sin(2 * np.pi * f * t + ph)
    s = Y["s0"] * (1 + 0.012 * max(0.0, t - 2.4))
    yaw, pitch, roll = -5 + 1.6 * sal(0.21, 0), 4 + 1.0 * sal(0.17, 1.3), 0.8 + 0.5 * sal(0.13, 2.1)
    cx, cy = Y["c0"][0] + 5 * sal(0.11, 0.4), Y["c0"][1] + 6 * sal(0.14, 1.9)
    y = cik(ara(t, 1.6, Z.TELEFON_OTUR), 3)                       # yükseliş
    cy = kar(K.H + GH * Y["s0"] / 2 + 30, cy, y)
    yaw, pitch, roll = yaw - 7 * (1 - y), pitch + 10 * (1 - y), roll + 3 * (1 - y)
    g = gir_cik(ara(t, 4.55, 6.5))                                # butona yaklaşma
    fx, fy, fs = Y["son"]
    if t > 6.5:                                                   # devirden sonra çok yavaş itme (buton merkezi sabit)
        d = 0.004 * (t - 6.6)
        bx, by = K.dugme[0], K.dugme[1]
        fx, fy, fs = bx + (fx - bx) * (1 + d), by + (fy - by) * (1 + d), fs * (1 + d)
    s = s * (fs / s) ** g
    cx, cy = kar(cx, fx, g), kar(cy, fy, g)
    return cx, cy, s, yaw * (1 - g), pitch * (1 - g), roll * (1 - g)


def _kose(poz, w=GW, h=GH):
    cx, cy, s, yaw, pitch, roll = poz
    P = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]], np.float32)
    a, b, c = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    Q = P @ (Ry @ Rx @ Rz).T
    k = 3200 / (3200 + Q[:, 2])
    return np.stack([cx + Q[:, 0] * k * s, cy + Q[:, 1] * k * s], 1).astype(np.float32)


def _matris(t, K):
    """(gövde haresi -> ekran, arayüz pikseli -> ekran) perspektif matrisleri."""
    Mg = dortgen((GW, GH), _kose(_poz(t, K)))
    T = np.array([[1, 0, PAY + CER], [0, 1, PAY + CER], [0, 0, 1]], np.float32)
    return Mg, (Mg @ T).astype(np.float32)


def _oteleme(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], np.float32)


# ------------------------------------------------------------------ arayüz sayfaları
def _hap(im, x, y, metin, boyut, yazi_renk, zemin_renk, yuk, dolgu=22, agirlik=800, cerceve=None, sagdan=False):
    w, asc, desc = yazi_olcu(metin, "nunito", boyut, agirlik)
    en = w + 2 * dolgu
    if sagdan:
        x -= en
    kutu(im, x, y, x + en, y + yuk, yuk / 2, zemin_renk, cerceve=cerceve, cerceve_kalinlik=2)
    yaz(im, metin, x + dolgu, y + yuk / 2 + boyut * 0.345, "nunito", boyut, yazi_renk, agirlik)
    return en


def _ust(im, baslik):
    kutu(im, UW / 2 - 66, 28, UW / 2 + 66, 68, 20, (20, 17, 16))          # kamera adası (markasız)
    daire(im, UW / 2 + 40, 48, 8, (38, 36, 50))
    daire(im, UW / 2 + 42, 46, 2.5, (90, 96, 130))
    yaz(im, "09:41", 58, 64, "nunito", 28, METIN, 800)
    kutu(im, 664, 40, 714, 64, 7, None, cerceve=METIN, cerceve_kalinlik=2.5)
    kutu(im, 669, 45, 700, 59, 3, METIN)
    for i in range(4):
        kutu(im, 596 + i * 12, 60 - 6 - i * 4, 604 + i * 12, 64, 2, METIN)
    cizgi_ince(im, [(66, 120), (46, 138), (66, 156)], METIN, 4.5)
    yaz(im, baslik, UW / 2, 152, "nunito", 40, METIN, 900, hiza="orta")
    yapistir(im, marka("ikon", 50), 716, 136)


def _alt_cubuk(im, etkin):
    kutu(im, 0, 1556, UW, UH, 0, BEYAZ)
    cizgi_ince(im, [(0, 1556), (UW, 1556)], CIZGI_GRI, 2)
    for i in range(5):
        x = 78 + 156 * i
        r = MERCAN if i == etkin else METIN3
        kutu(im, x - 18, 1592, x + 18, 1628, 11, None, cerceve=r, cerceve_kalinlik=4)
        if i == etkin:
            daire(im, x, 1610, 6, r)
    kutu(im, 290, 1660, 490, 1668, 4, METIN, 0.55)


def _kart_zemin(im, x0, y0, x1, y1, r=44):
    golge(im, x0, y0, x1, y1, r, bulanik=14, alfa=0.07, dy=6)
    kutu(im, x0, y0, x1, y1, r, BEYAZ)


@functools.lru_cache(maxsize=None)
def _soru_taban():
    im = np.empty((UH, UW, 3), np.uint8)
    im[:] = KREM
    _ust(im, "Soru sor")
    _kart_zemin(im, 30, 220, 750, 650)
    yaz(im, "Sorun", 60, 290, "nunito", 32, METIN, 800)
    kutu(im, *ALAN, 28, (255, 253, 251), cerceve=CIZGI_GRI, cerceve_kalinlik=3)
    yaz(im, "Durumu biraz anlatırsan yanıtlar daha isabetli olur.", 60, 484, "nunito", 25, METIN3, 600)
    x = 60
    for ad, sec in [("Uyku", True), ("Beslenme", False), ("Sağlık", False), ("Gelişim", False)]:
        x += 14 + (_hap(im, x, 530, ad, 28, (91, 74, 153), (241, 236, 250), 60, cerceve=LAVANTA) if sec
                   else _hap(im, x, 530, ad, 28, METIN2, KREM2, 60))
    yaz(im, "Yakınındaki annelerden", 40, 730, "nunito", 36, METIN, 800)
    for x0, y0, x1, y1 in YUVA:                                   # yanıt bekleyen iskelet kartlar
        _kart_zemin(im, x0, y0, x1, y1, 36)
        daire(im, x0 + 86, (y0 + y1) / 2, 46, KREM2)
        kutu(im, x0 + 160, y0 + 48, x0 + 380, y0 + 76, 14, KREM2)
        kutu(im, x0 + 160, y0 + 108, x0 + 600, y0 + 136, 14, KREM2)
    _alt_cubuk(im, 2)
    return im


@functools.lru_cache(maxsize=None)
def _grup_taban():
    im = np.empty((UH, UW, 3), np.uint8)
    im[:] = KREM
    _ust(im, "Grup")
    _kart_zemin(im, 30, 220, 750, 560)
    yaz(im, "Beşiktaş · 0–6 ay anneleri", 60, 312, "nunito", 47, METIN, 900)
    yaz(im, "Akran grubu · 128 anne", 60, 366, "nunito", 31, METIN2, 700)
    for i, tp in enumerate("abcdef"):
        x = 104 + 66 * i
        daire(im, x, 462, 47, BEYAZ)
        yapistir(im, avatar(84, tp), x, 462)
    _hap(im, 104 + 66 * 6 - 30, 432, "+122", 28, (75, 119, 92), ADACAYI_ACIK, 60)
    _kart_zemin(im, 30, 590, 750, 1296)
    yaz(im, "Grubun paylaşımlarını görmek ve", 60, 668, "nunito", 35, METIN, 600)
    yaz(im, "sohbete katılmak için gruba katıl.", 60, 716, "nunito", 35, METIN, 600)
    cizgi_ince(im, [(60, 770), (720, 770)], CIZGI_GRI, 2)
    for i, (ad, zr, nr) in enumerate([("Paylaşımlar", MERCAN_ACIK, MERCAN), ("Grup sohbeti", ADACAYI_ACIK, ADACAYI),
                                       ("Buluşmalar", (253, 243, 220), BAL)]):
        y = 842 + 92 * i
        daire(im, 94, y, 32, zr)
        daire(im, 94, y, 11, nr)
        yaz(im, ad, 148, y + 12, "nunito", 33, METIN, 700)
        cizgi_ince(im, [(690, y - 12), (702, y), (690, y + 12)], METIN3, 3.5)
    cizgi_ince(im, [(60, 1090), (720, 1090)], CIZGI_GRI, 2)
    kutu(im, *DUGME, DUGME_H / 2, BEYAZ, cerceve=MERCAN, cerceve_kalinlik=3.5)
    yaz(im, "Gruba katıl", UW / 2, DUGME_CY + 14, "nunito", 40, MERCAN, 800, hiza="orta")
    yaz(im, "Kimse otomatik eklenmez; istediğin zaman ayrılırsın.", UW / 2, 1266, "nunito", 24, METIN3, 600, hiza="orta")
    _alt_cubuk(im, 3)
    return im


def _soru(t):
    im = _soru_taban().copy()
    n = len(Z.YAZI)
    k = int(np.floor(ara(t, Z.YAZI_BAS, Z.YAZI_SON) * n + 1e-6))
    if t >= Z.YAZI_SON:
        k = n
    x = 86.0
    if k == 0:
        yaz(im, "Aklındaki soruyu yaz…", x, 386, "nunito", 44, METIN3, 600, alfa=0.9)
    else:
        x += yaz(im, Z.YAZI[:k], x, 386, "nunito", 44, METIN, 700)
    yaziyor = Z.YAZI_BAS <= t <= Z.YAZI_SON + 0.1
    if t >= 2.44 and (yaziyor or (t * 1.8) % 1 < 0.55):            # imleç
        kutu(im, x + 4, 344, x + 8, 398, 2, MERCAN)
    for i, tk in enumerate(Z.KARTLAR):                             # kalkan kartın yuvası boşalır
        a = ara(t, tk, tk + 0.10)
        if a > 0:
            x0, y0, x1, y1 = YUVA[i]
            kutu(im, x0 - 2, y0 - 2, x1 + 2, y1 + 14, 36, KREM, a)
            kutu(im, x0 + 3, y0 + 3, x1 - 3, y1 - 3, 34, None, a * 0.9, cerceve=CIZGI_GRI, cerceve_kalinlik=3)
    return im


def _ekran(t):
    kay = gir_cik(ara(t, Z.GRUP_GECIS, 4.8))
    if kay >= 1:
        return _grup_taban()
    soru = _soru(t)
    dx = int(round(kay * UW))
    if dx <= 0:
        return soru
    out = np.empty_like(soru)
    out[:, :UW - dx] = soru[:, dx:]
    out[:, UW - dx:] = _grup_taban()[:, :dx]
    return out


# ------------------------------------------------------------------ telefon gövdesi
@functools.lru_cache(maxsize=None)
def _govde():
    """Markasız cihaz: mat metal çerçeve, ince pah, sol üstten gelen kenar ışığı, yan tuşlar."""
    k = np.zeros((GH, GW, 4), np.uint8)
    for x0, y0, x1, y1 in [(GW - PAY - 3, PAY + 430, GW - PAY + 7, PAY + 590),          # güç
                           (PAY - 7, PAY + 330, PAY + 3, PAY + 410), (PAY - 7, PAY + 450, PAY + 3, PAY + 590)]:  # ses
        kutu(k, x0, y0, x1, y1, 4, (74, 64, 60))
    R = EKRAN_R + CER
    kutu(k, PAY, PAY, GW - PAY, GH - PAY, R, (58, 49, 46))
    y, x = np.mgrid[0:GH, 0:GW].astype(np.float32)
    isik = 1.42 - 0.62 * np.clip((x / GW * 0.55 + y / GH * 0.45), 0, 1)                 # sol üst aydınlık, sağ alt koyu
    k[..., :3] = np.clip(k[..., :3] * isik[..., None], 0, 255).astype(np.uint8)
    kutu(k, PAY + 1.5, PAY + 1.5, GW - PAY - 1.5, GH - PAY - 1.5, R - 1.5, None, cerceve=(176, 160, 150), cerceve_kalinlik=2.5, alfa=0.85)  # dış pah
    kutu(k, PAY + 7, PAY + 7, GW - PAY - 7, GH - PAY - 7, R - 7, None, cerceve=(34, 28, 26), cerceve_kalinlik=3)                          # pah gölgesi
    kutu(k, PAY + CER - 3, PAY + CER - 3, GW - PAY - CER + 3, GH - PAY - CER + 3, EKRAN_R + 3, None, cerceve=(16, 13, 12), cerceve_kalinlik=6)  # cam kenarı
    hl = np.zeros((GH, GW, 4), np.uint8)                                                # sol üst kenarda ince ışık çizgisi
    kutu(hl, PAY + 3, PAY + 3, GW - PAY - 3, GH - PAY - 3, R - 3, None, cerceve=(255, 244, 232), cerceve_kalinlik=2)
    yog = np.clip(1.15 - (x / GW + y / GH) * 1.9, 0, 1) ** 1.5 * 0.75
    katman_bas(k, (hl * yog[..., None]).astype(np.uint8))
    return k


@functools.lru_cache(maxsize=None)
def _yansima():
    """Ekran camındaki çok hafif çapraz yansıma (alfa haritası, iki ekran genişliğinde; telefon dönerken kayar)."""
    y, x = np.mgrid[0:UH, 0:2 * UW].astype(np.float32)
    d = x - 0.42 * y
    a = 0.060 * np.exp(-((d - 560) / 190) ** 2) + 0.034 * np.exp(-((d - 980) / 70) ** 2)
    return a.astype(np.float32)


@functools.lru_cache(maxsize=None)
def _kose_maske():
    """Ekranın yuvarlak köşeleri için alfa (yalnız dört köşe karesi kullanılır)."""
    m = np.zeros((UH, UW, 3), np.uint8)
    kutu(m, 0, 0, UW, UH, EKRAN_R, BEYAZ)
    return m[..., :1].astype(np.float32) / 255


def _telefon_hare(t, yaw=0.0):
    g = _govde().copy()
    e = _ekran(t)
    o, R = PAY + CER, EKRAN_R + 2
    roi = g[o:o + UH, o:o + UW]
    eski = [roi[ys, xs, :3].copy() for ys in (slice(0, R), slice(UH - R, UH)) for xs in (slice(0, R), slice(UW - R, UW))]
    kay = int(np.clip(UW / 2 - 46 * yaw + 26 * np.sin(0.45 * t), 0, UW))
    a = _yansima()[:, kay:kay + UW, None]
    roi[..., :3] = (e + (255.0 - e) * a + 0.5).astype(np.uint8)
    m, i = _kose_maske(), 0
    for ys in (slice(0, R), slice(UH - R, UH)):
        for xs in (slice(0, R), slice(UW - R, UW)):
            a = m[ys, xs]
            roi[ys, xs, :3] = (roi[ys, xs, :3] * a + eski[i] * (1 - a) + 0.5).astype(np.uint8)
            i += 1
    return g


# ------------------------------------------------------------------ öne kalkan kartlar
@functools.lru_cache(maxsize=None)
def _kart(i, w, h):
    ad, cevap, tip = CEVAP[i]
    k = np.zeros((h, w, 4), np.uint8)
    kutu(k, 0, 0, w, h, 34, BEYAZ)
    kutu(k, 1, 1, w - 1, h - 1, 33, None, cerceve=CIZGI_GRI, cerceve_kalinlik=2)
    yapistir(k, avatar(128, tip), 96, h / 2)
    yaz(k, ad, 186, h / 2 - 22, "nunito", 43, METIN, 800)
    _hap(k, w - 30, h / 2 - 64, ETIKET, 27, TUGLA, MERCAN_ACIK, 50, dolgu=20, sagdan=True)
    gen = yaz(k, cevap, 186, h / 2 + 52, "nunito", 40, METIN, 600)
    assert gen < w - 186 - 24, (cevap, gen)
    return k


@functools.lru_cache(maxsize=None)
def _hos_kart(w, h):
    k = np.zeros((h, w, 4), np.uint8)
    kutu(k, 0, 0, w, h, 38, BEYAZ)
    kutu(k, 1, 1, w - 1, h - 1, 37, None, cerceve=CIZGI_GRI, cerceve_kalinlik=2)
    r = h * 0.29
    cx = 44 + r
    daire(k, cx, h / 2, r, ADACAYI)
    cizgi_ince(k, [(cx - r * 0.42, h / 2 + r * 0.02), (cx - r * 0.10, h / 2 + r * 0.34), (cx + r * 0.46, h / 2 - r * 0.30)], BEYAZ, r * 0.17)
    x = cx + r + 34
    yaz(k, "Hoş geldin!", x, h / 2 + 4, "nunito", 64, METIN, 900)
    yaz(k, "Artık gruptasın.", x, h / 2 + 50, "nunito", 31, METIN2, 600)
    return k


def _dikdortgen_kose(cx, cy, w, h, aci=0.0, olcek=1.0):
    P = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], np.float32) * olcek
    c, s = np.cos(np.radians(aci)), np.sin(np.radians(aci))
    return P @ np.array([[c, s], [-s, c]], np.float32) + (cx, cy)


def _golge(tuval, P, renk, alfa, sigma):
    """Çokgenin yumuşak gölgesi/ışığı; büyük bulanıklıkta küçültülmüş maskeyle hızlı."""
    if alfa <= 0.004:
        return
    P = np.asarray(P, np.float32)
    H, W = tuval.shape[:2]
    pay = int(sigma * 3) + 2
    x0, y0 = max(0, int(P[:, 0].min()) - pay), max(0, int(P[:, 1].min()) - pay)
    x1, y1 = min(W, int(P[:, 0].max()) + pay), min(H, int(P[:, 1].max()) + pay)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    k = max(1, int(sigma / 3))
    w, h = max(2, (x1 - x0) // k), max(2, (y1 - y0) // k)
    mk = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mk, [np.round((P - (x0, y0)) / k * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    mk = cv2.GaussianBlur(mk, (0, 0), max(0.6, sigma / k))
    mk = cv2.resize(mk, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    O._birlestir(tuval, renk, mk * alfa, x0, y0)


_KENAR = {}


def _kenar(hare):
    """Kartın kalınlığı: aynı biçimin koyu krem kopyası (altından ince kenar olarak görünür)."""
    if id(hare) not in _KENAR:
        k = np.zeros_like(hare)
        a = hare[..., 3:].astype(np.float32) / 255
        k[..., :3] = (np.array((216, 201, 190), np.float32) * a + 0.5).astype(np.uint8)
        k[..., 3] = hare[..., 3]
        _KENAR[id(hare)] = k
    return _KENAR[id(hare)]


def _kalkan(tuval, hare, q, alfa, yuk):
    """Dört köşeye oturan hareyi kâğıt kalınlığı ve katmanlı gölgeyle basar (ışık sol üstten). yuk: 0..1 yükseklik."""
    if alfa <= 0.01:
        return
    w = np.linalg.norm(q[1] - q[0])
    ic = q * 0.98 + q.mean(0) * 0.02
    _golge(tuval, ic + (w * 0.030 * yuk, w * (0.014 + 0.062 * yuk)), (104, 66, 42), (0.07 + 0.11 * yuk) * alfa, w * (0.014 + 0.036 * yuk))  # uzun, yumuşak
    _golge(tuval, ic + (2 + 3 * yuk, 5 + 6 * yuk), (84, 52, 34), 0.20 * alfa, 4 + 5 * yuk)                                               # temas
    M = dortgen((hare.shape[1], hare.shape[0]), q)
    kal = 2.0 + 3.5 * yuk
    donustur(tuval, _kenar(hare), _oteleme(0.6, kal) @ M, alfa)
    donustur(tuval, hare, M, alfa)


@functools.lru_cache(maxsize=None)
def _grup_kart(w, h):
    """Yatay kurulum: grup sayfasının başlığını büyüten, öne kalkmış kart."""
    k = np.zeros((h, w, 4), np.uint8)
    kutu(k, 0, 0, w, h, 40, BEYAZ)
    kutu(k, 1, 1, w - 1, h - 1, 39, None, cerceve=CIZGI_GRI, cerceve_kalinlik=2)
    gen = yaz(k, "Beşiktaş · 0–6 ay anneleri", 46, 104, "nunito", 61, METIN, 900)
    assert gen < w - 80, gen
    yaz(k, "Akran grubu · 128 anne", 46, 160, "nunito", 38, METIN2, 700)
    for i, tp in enumerate("abcdef"):
        x = 46 + 54 + 84 * i
        daire(k, x, 242, 58, BEYAZ)
        yapistir(k, avatar(106, tp), x, 242)
    _hap(k, 46 + 54 + 84 * 6 - 34, 206, "+122", 34, (75, 119, 92), ADACAYI_ACIK, 72, dolgu=26)
    return k


# ------------------------------------------------------------------ çizgi yolları
def _yay_orani(P, i):
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return float(d[i] / d[-1])


def _yol_giris(u, K):
    """S1'den gelen çizginin telefonun solundan dolanıp soru alanının alt kenarına yattığı yol."""
    _, Mu = _matris(u, K)
    FL, FR = nokta(Mu, [(CIZ_X0, ALAN_ALT), (CIZ_X1, ALAN_ALT)])
    sol = nokta(Mu, [(-CER - 4, ALAN_ALT)])[0]
    G = np.array(K.s2_giris, np.float32)
    k = _poz(u, K)[2] / 0.8
    A0 = (G[0], G[1] - 270 * (1 if K.dikey else 0.6))
    c1 = G + (-46 * k, 96 * k)
    c2 = (sol[0] - 56 * k, kar(G[1], FL[1], 0.42))
    c3 = (sol[0] - 34 * k, FL[1] + 92 * k)
    n = 22
    P = egri([A0, G, c1, c2, c3, FL, FR], n)
    return P, _yay_orani(P, n), _yay_orani(P, 5 * n)


def _yol_butona(u, K):
    """Odak çizgisinden kalkıp butonun sol ucuna akan yol."""
    _, Mu = _matris(u, K)
    FL, FR, BL = nokta(Mu, [(CIZ_X0, ALAN_ALT), (CIZ_X1, ALAN_ALT), (KAPAK0, DUGME_CY)])
    k = _poz(u, K)[2]
    m1 = FR + (70 * k, 150 * k)
    m2 = ((FR[0] + BL[0]) / 2 + 60 * k, kar(FR[1], BL[1], 0.55))
    m3 = BL + (-150 * k, -150 * k)
    m4 = BL + (-120 * k, -22 * k)
    n = 22
    P = egri([FL, FR, m1, m2, m3, m4, BL], n)
    return P, _yay_orani(P, n)


def _cizgi_ciz(t, K, tuval, Mu):
    olc = _poz(t, K)[2]
    if t < 2.0 or t >= Z.FIRLAMA:
        return
    if t < 2.5:                                                    # devralma ve odak çizgisine yatış
        def f(u):
            P, sG, sFL = _yol_giris(u, K)
            x = ara(u, 2.0, 2.47)
            bas = kar(sG, 1.0, 1 - (1 - x) ** 1.7)
            kuy = sFL * gir_cik(ara(u, 2.02, 2.5))
            return yol_parca(P, min(kuy, bas - 0.01), bas)
        cizgi(tuval, alt_zaman(f, t), kar(10, 7, yumusak(ara(t, 2.1, 2.5))))
    elif t < Z.GRUP_GECIS:                                         # odak çizgisi
        P = nokta(Mu, [(CIZ_X0, ALAN_ALT), (CIZ_X1, ALAN_ALT)])
        cizgi(tuval, P, 7, parlama=1 + 0.18 * np.sin(2 * np.pi * 0.9 * (t - 2.5)))
    elif t < Z.DOLGU_BAS:                                          # butona akış
        def f(u):
            P, sFR = _yol_butona(u, K)
            bas = kar(sFR, 1.0, gir_cik(ara(u, Z.GRUP_GECIS, 5.42)))
            kuy = gir_cik(ara(u, Z.GRUP_GECIS + 0.10, Z.DOLGU_BAS))
            return yol_parca(P, min(kuy, bas - 0.004), bas)
        cizgi(tuval, alt_zaman(f, t), kar(7, 9, ara(t, 5.2, Z.DOLGU_BAS)))
    else:                                                          # buton dolgusu: çizgi hap olur
        x = ara(t, Z.DOLGU_BAS, Z.DOLGU_SON)
        uc = kar(KAPAK0, KAPAK1, gir_cik(x))
        yuk = kar(9 / olc, DUGME_H, cik(ara(t, Z.DOLGU_BAS, Z.DOLGU_BAS + 0.17), 2))
        bas = 0.055 * np.sin(np.pi * ara(t, Z.DOKUNUS - 0.03, Z.DOKUNUS + 0.20)) ** 2   # dokunuşta hafif basılma
        yuk *= 1 - bas
        P = nokta(Mu, [(KAPAK0 + bas * 60, DUGME_CY), (uc - bas * 60, DUGME_CY)])
        boy = np.linalg.norm(nokta(Mu, [(UW / 2, DUGME_CY - 50)])[0] - nokta(Mu, [(UW / 2, DUGME_CY + 50)])[0]) / 100
        cizgi(tuval, P, yuk * boy, parlama=kar(1.0, 0.45, x))


def _beyaz_yazi(tuval, Mu, uc_x):
    """Dolgunun geçtiği yerde buton yazısı beyaza döner (arayüz düzleminde, aynı perspektifle)."""
    hr = yazi("Gruba katıl", "nunito", 40, BEYAZ, 800)
    w, asc, _ = yazi_olcu("Gruba katıl", "nunito", 40, 800)
    sol, ust = UW / 2 - w / 2 - 4, DUGME_CY + 14 - asc - 4
    n = int(np.clip(uc_x - sol, 0, hr.shape[1]))
    if n >= 2:
        donustur(tuval, np.ascontiguousarray(hr[:, :n]), Mu @ _oteleme(sol, ust))


# ------------------------------------------------------------------ sahne
def ciz(t, K, tuval):
    Y = _yer(K.dikey)
    poz = _poz(t, K)
    Mg, Mu = _matris(t, K)
    s = poz[2]

    # telefon: gölge + gövde (yükselirken dikey hareket bulanıklığı, kartlar kalkınca sığ alan derinliği)
    hare = _telefon_hare(t, poz[3])
    hiz = abs(_poz(t, K)[1] - _poz(t - 1 / Z.FPS, K)[1])
    kb = min(48, int(hiz * 0.5 / s))
    if kb >= 3:
        hare = cv2.blur(hare, (1, kb | 1))
    kalkis = max(yumusak(ara(t, tk, tk + KART_SURE)) for tk in Z.KARTLAR) * (1 - ara(t, Z.GRUP_GECIS, 4.78))
    if kalkis > 0.02:
        hare = cv2.GaussianBlur(hare, (0, 0), 2.2 * kalkis / s)
    kose = nokta(Mg, [(PAY, PAY), (GW - PAY, PAY), (GW - PAY, GH - PAY), (PAY, GH - PAY)])
    orta = kose.mean(0)
    yerde = cik(ara(t, 1.6, Z.TELEFON_OTUR), 2)
    _golge(tuval, (kose - orta) * 1.05 + orta, (255, 241, 224), 0.34 * yerde, 64 * s)                 # ekran ışığının çevreye yansıması
    _golge(tuval, (kose - orta) * 0.95 + orta + (64 * s, 92 * s), (98, 62, 40), 0.20, 58 * s)         # uzun yumuşak gölge (ışık sol üstten)
    _golge(tuval, (kose - orta) * 0.985 + orta + (9 * s, 15 * s), (70, 44, 30), 0.34, 11 * s)         # temas gölgesi
    donustur(tuval, hare, Mg)

    # çizgi (t >= FIRLAMA'da birleştirici devralır; buton düz mercan kalır)
    if t >= Z.FIRLAMA:
        B = nokta(Mu, [(DUGME[0], DUGME[1]), (DUGME[2], DUGME[3])])
        kutu(tuval, B[0][0], B[0][1], B[1][0], B[1][1], (B[1][1] - B[0][1]) / 2, MERCAN)
    else:
        _cizgi_ciz(t, K, tuval, Mu)
    bas_h = np.sin(np.pi * ara(t, Z.DOKUNUS - 0.04, Z.DOKUNUS + 0.24)) ** 2
    if bas_h > 0.01:                                               # dokunuşta ekran hafifçe içe basılır
        bx, by = nokta(Mu, [(UW / 2 + 150, DUGME_CY + 6)])[0]
        a = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        _golge(tuval, np.stack([bx + 78 * s * np.cos(a), by + 62 * s * np.sin(a)], 1), (120, 44, 30), 0.26 * bas_h, 16 * s)
    if t >= Z.DOLGU_BAS:
        _beyaz_yazi(tuval, Mu, kar(KAPAK0, KAPAK1, gir_cik(ara(t, Z.DOLGU_BAS, Z.DOLGU_SON))) + DUGME_H * 0.42)

    # cevap kartları: ekrandaki yuvadan kameraya doğru kalkar
    kw, kh = Y["kart"]
    cikis = gir(ara(t, Z.GRUP_GECIS, 4.8), 2)
    for i in (0, 1, 2):
        tk = Z.KARTLAR[i]
        if t < tk or cikis >= 1:
            continue
        p = geri(ara(t, tk, tk + KART_SURE), 1.3)
        x0, y0, x1, y1 = YUVA[i]
        yuva = nokta(Mu, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        hx, hy = Y["kc"][i]
        hx += 5 * np.sin(1.1 * t + i * 2.0) + Y["yon"][i] * K.W * 0.62 * cikis
        hy += 7 * np.sin(0.9 * t + i * 1.7)
        hedef = _dikdortgen_kose(hx, hy, kw, kh, Y["ka"][i] * (1 + 2.5 * cikis), 1 + 0.012 * (t - tk))
        q = yuva + (hedef - yuva) * p
        _kalkan(tuval, _kart(i, kw, kh), q, ara(t, tk, tk + 0.07) * (1 - cikis), min(1.0, max(0.0, p)))

    # dokunuş: parmak izi + dalga halkaları
    dx, dy = nokta(Mu, [(UW / 2 + 150, DUGME_CY + 6)])[0]
    pa = ara(t, Z.DOKUNUS - 0.20, Z.DOKUNUS - 0.04) * (1 - ara(t, Z.DOKUNUS + 0.06, Z.DOKUNUS + 0.26))
    if pa > 0.01:
        pr = 44 * s * (1.35 - 0.35 * yumusak(ara(t, Z.DOKUNUS - 0.20, Z.DOKUNUS)) - 0.12 * ara(t, Z.DOKUNUS, Z.DOKUNUS + 0.05))
        daire(tuval, dx, dy, pr, BEYAZ, 0.42 * pa)
        daire(tuval, dx, dy, pr, METIN, 0.30 * pa, kalinlik=3)
    for j in range(3):
        h = ara(t, Z.DOKUNUS + 0.07 * j, Z.DOKUNUS + 0.07 * j + 0.62)
        if 0 < h < 1:
            halka(tuval, dx, dy, s * (40 + 300 * cik(h, 2)), (1 - h) ** 1.5 * 0.85, MERCAN if j != 1 else BEYAZ, 5 * s)

    # yatay: grup başlığı sağdaki boşluğa, öne kalkmış kart olarak büyür
    if Y["grup"] is not None and t >= 4.86:
        gx, gy, gw, gh = Y["grup"]
        tg = 4.86
        p = geri(ara(t, tg, tg + 0.44), 1.15)
        A = nokta(Mu, [(30, 220), (750, 220), (750, 560), (30, 560)])
        hedef = _dikdortgen_kose(gx + 5 * np.sin(1.0 * t), gy + 6 * np.sin(0.8 * t + 1), gw, gh, 1.1, 1 + 0.008 * (t - tg))
        _kalkan(tuval, _grup_kart(gw, gh), A + (hedef - A) * p, ara(t, tg, tg + 0.09), min(1.0, max(0.0, p)))

    # "Hoş geldin!" kartı butondan kalkar: dikeyde butonun altındaki boşluğa, yatayda grup kartının altına
    t0 = Z.DOKUNUS + 0.05
    if t >= t0:
        hx, hy, hw, hh = Y["hos"]
        p = geri(ara(t, t0, t0 + 0.36), 1.5)
        B = nokta(Mu, [(DUGME[0] + 60, DUGME[1]), (DUGME[2] - 60, DUGME[1]), (DUGME[2] - 60, DUGME[3]), (DUGME[0] + 60, DUGME[3])])
        if K.dikey:
            c = nokta(Mu, [(UW / 2, 1428)])[0]
            hedef = _dikdortgen_kose(c[0], c[1], hw, hh, -1.0, s / Y["son"][2] * (1 + 0.012 * (t - t0)))
        else:
            hedef = _dikdortgen_kose(hx + 4 * np.sin(1.3 * t), hy - 10 * (t - t0), hw, hh, -1.2, 1 + 0.02 * (t - t0))
        _kalkan(tuval, _hos_kart(hw, hh), B + (hedef - B) * p, ara(t, t0, t0 + 0.08), min(1.0, max(0.0, p)))
    return tuval


def dugme_kutusu(t, K):
    """Butonun ekrandaki (cx, cy, en, boy) kutusu — devir denetimi için."""
    _, Mu = _matris(t, K)
    B = nokta(Mu, [(DUGME[0], DUGME[1]), (DUGME[2], DUGME[3])])
    return ((B[0][0] + B[1][0]) / 2, (B[0][1] + B[1][1]) / 2, B[1][0] - B[0][0], B[1][1] - B[0][1])

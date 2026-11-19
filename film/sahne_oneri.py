"""Öneri + güven sahnesi (20.40–25.60): annelerin birbirine önerdiği üç kart
(kreş, çocuk doktoru, pişik kremi) kelimeyle aynı anda öne kalkar; çizgi kartları
tek hatla bağlar, sonra "Doğrulanmış anneler" kalkanını çizer. Yüz ve insan yok."""
import functools, json
import numpy as np, cv2
import ortak as O
from ortak import Z

ARALIK = (Z.ONERI_BAS, Z.SLOGAN_BAS)


def _kelime_anlari():
    v = dict(kres=20.60, doktor=21.06, pisik=22.09, dogrulanmis=23.00, anneler=23.54, onersin=24.13, guvenle=24.81)
    try:
        d = json.load(open(f"{O.KOK}/cikti/ses/kilit_lilyv3_2.json"))["satirlar"]
        a = {k["k"].lower(): k["bas"] for s in d[9:11] for k in s["kelimeler"]}
        for anahtar, kelime in [("kres", "kreş"), ("doktor", "doktor"), ("pisik", "pişik"), ("dogrulanmis", "doğrulanmış"),
                                ("anneler", "anneler"), ("onersin", "önersin"), ("guvenle", "güvenle")]:
            v[anahtar] = a.get(kelime, v[anahtar])
    except Exception:
        pass
    return v


W_ = _kelime_anlari()
T_KART = [W_["kres"], W_["doktor"], W_["pisik"]]
KARTLAR = [("Kreş", "Minik Adımlar Kreşi", "3 anne önerdi"),
           ("Çocuk doktoru", "Mahallende, 4 dk", "5 anne önerdi"),
           ("Pişik kremi", "Kokusuz, işe yarıyor", "7 anne önerdi")]
GOLGE = (96, 70, 58)


# ------------------------------------------------------------------ yerleşim
@functools.lru_cache(maxsize=2)
def _yer(dikey):
    if dikey:
        return dict(kw=900, kh=300, merkez=[(540, 880), (540, 1230), (540, 1580)], egim=[-1.0, 0.8, -0.6],
                    kalkan=(540, 312), kb=272, etiket=(540, 548, "orta", 62), alt=(540, 612, "orta", 38))
    return dict(kw=540, kh=430, merkez=[(330, 735), (960, 735), (1590, 735)], egim=[-0.8, 0.5, -0.6],
                kalkan=(700, 250), kb=244, etiket=(850, 246, "sol", 66), alt=(852, 306, "sol", 38))


def _elips(cx, cy, rx, ry, n=40):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


def _don(P, aci, merkez):
    a = np.radians(aci)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    return (np.asarray(P, np.float32) - merkez) @ R.T + merkez


# ------------------------------------------------------------------ nesneler (karo içine)
def _nesne_kres(k, cx, cy, s):
    O.cokgen(k, _elips(cx + 6 * s, cy + 66 * s, 88 * s, 13 * s), GOLGE, 0.22, bulanik=6 * s)
    x0, x1, y0, y1 = cx - 62 * s, cx + 30 * s, cy - 8 * s, cy + 62 * s
    O.kutu(k, x0, y0, x1, y1, 4 * s, (255, 250, 243))
    O.cokgen(k, np.array([(x1 - 22 * s, y0), (x1, y0), (x1, y1), (x1 - 22 * s, y1)]), GOLGE, 0.07)          # gölgede kalan yan
    cati = np.array([(x0 - 14 * s, y0 + 2 * s), ((x0 + x1) / 2, cy - 66 * s), (x1 + 14 * s, y0 + 2 * s)])
    O.cokgen(k, cati, O.SOMON)
    O.cokgen(k, np.array([((x0 + x1) / 2, cy - 66 * s), (x1 + 14 * s, y0 + 2 * s), ((x0 + x1) / 2 + 6 * s, y0 + 2 * s)]), O.SOMON_KOYU, 0.35)
    O.cokgen(k, np.array([(x0 - 2 * s, y0 + 2 * s), (x1 + 2 * s, y0 + 2 * s), (x1, y0 + 9 * s), (x0, y0 + 9 * s)]), GOLGE, 0.14)  # saçak gölgesi
    O.kutu(k, cx - 30 * s, cy + 22 * s, cx - 6 * s, y1, 5 * s, O.NIL)                                        # kapı
    O.daire(k, cx - 12 * s, cy + 44 * s, 2.2 * s, (255, 250, 243))
    O.kutu(k, cx + 2 * s, cy + 12 * s, cx + 22 * s, cy + 34 * s, 3 * s, O.YAVRUAGZI)                          # pencere
    O.cizgi_ince(k, np.array([(cx + 12 * s, cy + 12 * s), (cx + 12 * s, cy + 34 * s)]), (255, 250, 243), 2 * s)
    O.cizgi_ince(k, np.array([(cx + 2 * s, cy + 23 * s), (cx + 22 * s, cy + 23 * s)]), (255, 250, 243), 2 * s)
    for (bx, by, bw, renk, aci) in [(cx + 60 * s, cy + 45 * s, 34 * s, O.NIL, 0), (cx + 58 * s, cy + 14 * s, 28 * s, O.YAVRUAGZI, 12)]:   # ahşap küpler
        P = _don(O._yd_noktalar(bx - bw / 2, by - bw / 2, bx + bw / 2, by + bw / 2, 5 * s), aci, (bx, by))
        O.cokgen(k, P, renk)
        O.cokgen(k, _don(np.array([(bx + bw * 0.18, by - bw / 2), (bx + bw / 2, by - bw / 2), (bx + bw / 2, by + bw / 2), (bx + bw * 0.18, by + bw / 2)]), aci, (bx, by)), GOLGE, 0.10)
        O.daire(k, bx - bw * 0.14, by - bw * 0.1, bw * 0.13, (255, 255, 255), 0.55)


def _nesne_stetoskop(k, cx, cy, s):
    koyu, acik = (58, 88, 92), (126, 160, 162)
    O.cokgen(k, _elips(cx + 46 * s, cy + 66 * s, 44 * s, 10 * s), GOLGE, 0.24, bulanik=6 * s)
    sol = O.egri([(cx - 58 * s, cy - 62 * s), (cx - 60 * s, cy - 30 * s), (cx - 40 * s, cy - 2 * s), (cx - 16 * s, cy + 8 * s)], 14)
    sag = O.egri([(cx + 26 * s, cy - 62 * s), (cx + 28 * s, cy - 30 * s), (cx + 8 * s, cy - 2 * s), (cx - 16 * s, cy + 8 * s)], 14)
    govde = O.egri([(cx - 16 * s, cy + 8 * s), (cx - 22 * s, cy + 40 * s), (cx + 2 * s, cy + 64 * s), (cx + 30 * s, cy + 56 * s), (cx + 46 * s, cy + 36 * s)], 14)
    for P in (sol, sag, govde):
        O.cizgi_ince(k, P + (2 * s, 4 * s), GOLGE, 9 * s, 0.14)
    for P in (sol, sag):                                           # metal kulaklık boruları
        O.cizgi_ince(k, P, (176, 184, 190), 6 * s)
        O.cizgi_ince(k, P - (1.4 * s, 0), (236, 240, 243), 2 * s, 0.9)
    O.cizgi_ince(k, sol[-9:], koyu, 9 * s); O.cizgi_ince(k, sag[-9:], koyu, 9 * s)
    O.cizgi_ince(k, govde, koyu, 9 * s)
    O.cizgi_ince(k, govde[2:-3] - (1.6 * s, 1.6 * s), acik, 2.4 * s, 0.8)                                    # hortumda ışık
    for ex in (cx - 58 * s, cx + 26 * s):
        O.daire(k, ex, cy - 66 * s, 7 * s, koyu); O.daire(k, ex - 2 * s, cy - 68 * s, 2.2 * s, acik, 0.8)
    gx, gy = cx + 52 * s, cy + 30 * s                              # göğüs parçası
    O.daire(k, gx, gy, 27 * s, (150, 160, 168))
    O.daire(k, gx, gy, 23 * s, (214, 219, 224))
    O.daire(k, gx, gy, 15 * s, (240, 243, 246))
    O.cokgen(k, _don(_elips(gx - 9 * s, gy - 11 * s, 9 * s, 4 * s), -35, (gx - 9 * s, gy - 11 * s)), (255, 255, 255), 0.9)
    O.daire(k, gx, gy, 24 * s, (255, 255, 255), 0.5, kalinlik=1.6 * s)


def _nesne_krem(k, cx, cy, s):
    m = np.array((cx, cy), np.float32)
    aci = -20
    D = lambda P: _don(np.asarray(P, np.float32) * s + m, aci, m)
    O.cokgen(k, _don(_elips(cx + 8 * s, cy + 70 * s, 62 * s, 12 * s), 0, m), GOLGE, 0.22, bulanik=6 * s)
    O.cokgen(k, D([(-38, -78), (38, -78), (31, 42), (-31, 42)]), (253, 251, 248))                             # gövde
    O.cokgen(k, D([(-36.6, -26), (36.6, -26), (33.6, 20), (-33.6, 20)]), O.YAVRUAGZI)                          # renk bandı
    O.cokgen(k, D([(-38, -78), (38, -78), (37.4, -64), (-37.4, -64)]), (226, 222, 216))                        # kıvrılmış uç
    for x in range(-30, 31, 10):
        O.cizgi_ince(k, D([(x, -76), (x, -66)]), (198, 192, 186), 1.6 * s)
    O.cokgen(k, D([(12, -64), (37.4, -64), (31, 42), (12, 42)]), GOLGE, 0.08)                                 # gölgeli yan
    O.cokgen(k, D([(-30, -62), (-22, -62), (-20, 40), (-27, 40)]), (255, 255, 255), 0.75)                      # parlak şerit
    a = np.linspace(0, 2 * np.pi, 30)                                                                         # damla simgesi
    damla = np.stack([9 * np.sin(a) * (1 - 0.5 * np.cos(a) * (np.cos(a) > 0)), -3 - 11 * np.cos(a) - 5 * (np.cos(a) > 0) * np.cos(a) ** 3], 1)
    O.cokgen(k, D(damla), O.SOMON)
    O.cokgen(k, D([(-31, 42), (31, 42), (18, 56), (-18, 56)]), (232, 228, 222))                                # omuz
    O.cokgen(k, D(O._yd_noktalar(-21, 55, 21, 86, 6) / 1.0), O.NIL)                                            # kapak
    O.cokgen(k, D([(6, 56), (21, 56), (21, 85), (6, 85)]), O.NIL_KOYU, 0.30)
    for x in (-13, -5, 3):
        O.cizgi_ince(k, D([(x, 60), (x, 82)]), (255, 255, 255), 1.6 * s, 0.35)


NESNE = [_nesne_kres, _nesne_stetoskop, _nesne_krem]


# ------------------------------------------------------------------ hareler
@functools.lru_cache(maxsize=8)
def _kart(i, dikey):
    """Kartın değişmeyen kısmı. Dönüş: (hare, yıldız merkezleri, yıldız yarıçapı)."""
    Y = _yer(dikey)
    w, h = Y["kw"], Y["kh"]
    baslik, alt, hap = KARTLAR[i]
    k = O.bos_katman((w, h + 8))
    O.kutu(k, 0, 7, w, h + 7, 30, (228, 214, 202))                 # kâğıt kalınlığı
    O.kutu(k, 0, 0, w, h, 30, O.BEYAZ)
    karo = (O.NIL_ACIK, O.YAVRUAGZI_ACIK, O.NIL_ACIK)[i]
    if dikey:
        O.kutu(k, 28, 28, 272, 272, 26, karo)
        NESNE[i](k, 150, 146, 1.22)
        O.yaz(k, baslik, 304, 104, "nunito", 58, O.METIN, 800)
        O.yaz(k, alt, 304, 158, "nunito", 38, O.METIN2, 600)
        r, ys, xs, adim, hb = 21, 226, 326, 48, 36
        hx1, hy0, hy1 = w - 28, 196, 256
    else:
        O.kutu(k, 26, 26, w - 26, 216, 24, karo)
        NESNE[i](k, w / 2, 120, 0.98)
        O.yaz(k, baslik, 32, 280, "nunito", 46, O.METIN, 800)
        O.yaz(k, alt, 32, 322, "nunito", 30, O.METIN2, 600)
        r, ys, xs, adim, hb = 16, 376, 50, 37, 28
        hx1, hy0, hy1 = w - 26, 352, 400
    gen = O.yazi_olcu(hap, "nunito", hb, 800)[0]
    O.kutu(k, hx1 - gen - 2 * hb * 0.62, hy0, hx1, hy1, (hy1 - hy0) / 2, O.KREM2)
    O.yaz(k, hap, hx1 - hb * 0.62, (hy0 + hy1) / 2 + hb * 0.34, "nunito", hb, O.METIN2, 800, hiza="sag")
    yildiz = [(xs + j * adim, ys) for j in range(5)]
    for (x, y) in yildiz:
        O.cokgen(k, _yildiz_n(x, y, r), O.CIZGI_GRI)
    # sol üstten gelen ışık: sağ alta doğru çok hafif koyulaşma
    yy, xx = np.mgrid[0:h + 8, 0:w].astype(np.float32)
    kararma = np.clip((xx / w * 0.5 + yy / h * 0.5) - 0.35, 0, 1) * 0.045
    k[..., :3] = (k[..., :3].astype(np.float32) * (1 - kararma[..., None])).astype(np.uint8)
    return k, yildiz, r


def _yildiz_n(x, y, r):
    a = np.radians(-90 + np.arange(10) * 36)
    rr = np.where(np.arange(10) % 2 == 0, r, r * 0.46)
    return np.stack([x + rr * np.cos(a), y + rr * np.sin(a)], 1)


@functools.lru_cache(maxsize=4)
def _yildiz(r):
    k = O.bos_katman((int(r * 2 + 8), int(r * 2 + 8)))
    c = r + 4
    O.cokgen(k, _yildiz_n(c, c, r), O.SOMON)
    O.cokgen(k, _yildiz_n(c - r * 0.12, c - r * 0.14, r * 0.42), (255, 255, 255), 0.22)
    return k


@functools.lru_cache(maxsize=2)
def _rozet(dikey):
    b = 36 if dikey else 28
    gen = O.yazi_olcu("Doğrulanmış anne", "nunito", b, 800)[0]
    h = int(b * 1.72)
    w = int(gen + b * 2.5)
    k = O.bos_katman((w + 12, h + 16))
    O.cokgen(k, O._yd_noktalar(6, 9, w + 6, h + 9, h / 2), GOLGE, 0.20, bulanik=4)
    O.kutu(k, 6, 4, w + 6, h + 4, h / 2, O.NIL_ACIK, cerceve=O.BEYAZ, cerceve_kalinlik=3)
    O.yaz(k, "Doğrulanmış anne", 6 + b * 0.62, 4 + h / 2 + b * 0.34, "nunito", b, O.NIL_KOYU, 800)
    cx, cy, u = 6 + w - b * 0.95, 4 + h / 2, b / 36
    O.daire(k, cx, cy, 15 * u, O.NIL)
    O.cizgi_ince(k, np.array([(cx - 7 * u, cy + 0.5 * u), (cx - 2 * u, cy + 5.5 * u), (cx + 7.5 * u, cy - 5 * u)]), O.BEYAZ, 4 * u)
    return k


@functools.lru_cache(maxsize=2)
def _lekeler(dikey):
    """Arka plan ışık lekeleri: 1/8 çözünürlükte bir kez kurulur."""
    K = O.DIKEY if dikey else O.YATAY
    w, h = K.W // 8, K.H // 8
    k = O.bos_katman((w, h))
    for (fx, fy, fr, renk, al) in [(0.22, 0.20, 0.42, O.NIL_ACIK, 0.85), (0.86, 0.62, 0.46, O.YAVRUAGZI_ACIK, 0.8), (0.18, 0.92, 0.36, O.NIL_ACIK, 0.6)]:
        r = fr * min(w, h)
        O.cokgen(k, _elips(fx * w, fy * h, r, r * 0.9, 48), renk, al, bulanik=7.5)
    return k


# ------------------------------------------------------------------ çizginin yolu
def _kalkan_yolu(cx, cy, boy):
    """Kalkanın dış hattı: üst ortadan başlar, saat yönünde döner, üst ortada kapanır."""
    h, w = boy / 2, boy * 0.41
    ust = np.array([(0, -h), (w * 0.72, -h)], np.float32)
    kose = O.bezier((w * 0.72, -h), (w, -h), (w, -h), (w, -h * 0.72), 8)
    yan = O.bezier((w, -h * 0.72), (w, h * 0.25), (w * 0.62, h * 0.72), (0, h), 22)
    sag = np.concatenate([ust, kose[1:], yan[1:]])
    sol = sag[::-1] * (-1, 1)
    return np.concatenate([sag, sol[1:]]) + (cx, cy)


@functools.lru_cache(maxsize=2)
def _yol(dikey):
    """Tek hat: giriş → kartların altı → kalkan → çıkış. Dönüş: (noktalar, birikimli uzunluk, anahtar uzunluklar)."""
    K = O.DIKEY if dikey else O.YATAY
    Y = _yer(dikey)
    g, c = np.array(K.oneri_giris, np.float32), np.array(K.slogan_giris, np.float32)
    kx, ky = Y["kalkan"]
    kal = _kalkan_yolu(kx, ky, Y["kb"])
    tepe = kal[0]
    D = lambda a, b: np.array([a, b], np.float32)
    if dikey:
        y1, y2, y3 = 1052, 1402, 1752
        parca = [O.bezier(g, (60, 640), (30, 980), (110, y1), 30), D((110, y1), (970, y1)),
                 O.bezier((970, y1), (1062, y1), (1062, y2), (970, y2), 30), D((970, y2), (110, y2)),
                 O.bezier((110, y2), (18, y2), (18, y3), (110, y3), 30), D((110, y3), (970, y3)),
                 O.bezier((970, y3), (1075, y3), (1066, 1300), (1058, 760), 40),
                 O.bezier((1058, 760), (1050, 300), (880, 120), tepe, 40), kal,
                 O.bezier(tepe, (330, tepe[1] - 20), (60, 330), c, 40)]
        anahtar_i = (0, 1, 3, 5, 7, 8, 9)
    else:
        y = 978
        parca = [O.bezier(g, (120, 330), (10, 940), (80, y), 34), D((80, y), (606, y)), D((606, y), (1236, y)), D((1236, y), (1840, y)),
                 O.bezier((1840, y), (1925, y), (1900, 700), (1896, 470), 30),
                 O.bezier((1896, 470), (1890, 60), (1100, 40), tepe, 44), kal,
                 O.bezier(tepe, (560, tepe[1] - 10), (430, 190), c, 34)]
        anahtar_i = (0, 1, 2, 3, 5, 6, 7)
    P = [parca[0]]
    sonlar = []
    for p in parca[1:]:
        sonlar.append(sum(len(q) for q in P) - 1)
        P.append(np.asarray(p, np.float32)[1:])
    sonlar.append(sum(len(q) for q in P) - 1)
    P = np.concatenate(P).astype(np.float32)
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    L = [float(d[sonlar[i]]) for i in anahtar_i]          # [alt çizgi başı, kart1 sonu, kart2 sonu, kart3 sonu, kalkan başı, kalkan sonu, çıkış sonu]
    return P, d, L


def _anahtar(t, kareler):
    """[(t, değer), ...] arasında yumuşak geçişli parçalı eğri; aralarda bekler."""
    if t <= kareler[0][0]:
        return kareler[0][1]
    for (t0, v0), (t1, v1) in zip(kareler, kareler[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * O.gir_cik(O.ara(t, t0, t1)) if v1 != v0 else v0
    return kareler[-1][1]


def _cizgi_yolu(t, dikey):
    P, d, L = _yol(dikey)
    tk, tg = T_KART, Z.ONERI_GUVEN
    bas = _anahtar(t, [(Z.ONERI_BAS, 0), (tk[0], L[0]), (tk[0] + 0.02, L[0]), (tk[0] + 0.42, L[1]),
                       (tk[1], L[1]), (tk[1] + 0.48, L[2]), (tk[2], L[2]), (tk[2] + 0.50, L[3]),
                       (tg, L[3]), (tg + 0.32, L[4]), (tg + 0.80, L[5]),
                       (Z.ONERI_GECIS + 0.02, L[5]), (Z.SLOGAN_BAS - 0.03, L[6])])
    kuyruk = _anahtar(t, [(tg, 0), (tg + 0.90, L[4]), (Z.ONERI_GECIS + 0.04, L[4]), (Z.SLOGAN_BAS - 0.03, L[6] - 120)])
    if t < Z.ONERI_BAS + 0.03:
        kuyruk = 0
    bas = max(bas, kuyruk + 2)
    return O.yol_parca(P, kuyruk / d[-1], bas / d[-1])


# ------------------------------------------------------------------ kamera
def _kamera(t, K):
    u = O.ara(t, *ARALIK)
    s = 1 + 0.028 * np.sin(np.pi * u)                    # uçlarda birim (devir noktaları sabit), arada sürekli hareket
    dx, dy = 9 * np.sin(np.pi * u) * np.sin(2.2 * u + 0.4), -12 * np.sin(np.pi * u)
    return O.afin(s, 0, (K.cx, K.cy), (K.cx + dx, K.cy + dy)), s


# ------------------------------------------------------------------ çizim
def ciz(t, K, tuval):
    dk = K.dikey
    Y = _yer(dk)
    M, ks = _kamera(t, K)
    zemin = tuval.copy()
    sol = O.yumusak(O.ara(t, Z.ONERI_GECIS, Z.SLOGAN_BAS - 0.05))

    # arkada yumuşak nil / yavruağzı ışık lekeleri (palet + derinlik); önbellekten, kamerayla hafifçe kayar
    gir = O.yumusak(O.ara(t, Z.ONERI_BAS, Z.ONERI_BAS + 0.5))
    if gir > 0:
        p = O.nokta(M, [(K.cx, K.cy)])[0]
        O.yapistir(tuval, _lekeler(dk), p[0], p[1] + 8 * np.sin(t * 0.8), alfa=gir, olcek=8 * ks)

    # kartlar
    w, h = Y["kw"], Y["kh"]
    for i in range(3):
        p = O.ara(t, T_KART[i], T_KART[i] + 0.55)
        if p <= 0:
            continue
        kart, yildizlar, yr = _kart(i, dk)
        e = O.geri(p, 1.5)
        olcek = (0.70 + 0.30 * e) * ks * (1 + 0.006 * np.sin(1.7 * t + i * 2.1))
        alfa = min(1.0, p * 3.2)
        cx, cy = O.nokta(M, [Y["merkez"][i]])[0]
        cy += (1 - O.cik(p)) * 90 + 3.5 * np.sin(1.3 * t + i * 1.9)
        aci = Y["egim"][i] + (1 - O.cik(p)) * (7 if i % 2 else -7) + 0.25 * np.sin(0.9 * t + i)
        hw, hh = w * olcek / 2, h * olcek / 2
        kalk = 0.55 + 0.45 * O.cik(p)                                # yükseldikçe gölge uzar ve yumuşar
        O.golge(tuval, cx - hw + 14, cy - hh + 10, cx + hw + 22, cy + hh + 10, 30, bulanik=34 * kalk, alfa=0.15 * alfa, dy=30 * kalk, renk=GOLGE)
        O.golge(tuval, cx - hw + 4, cy - hh + 4, cx + hw + 4, cy + hh + 4, 30, bulanik=7, alfa=0.16 * alfa, dy=6, renk=GOLGE)
        Mk = O.afin(olcek, aci, (w / 2, h / 2), (cx, cy))
        O.donustur(tuval, kart, Mk, alfa)
        for j, (sx, sy) in enumerate(yildizlar):                     # yıldızlar sırayla dolar
            q = O.ara(t, T_KART[i] + 0.34 + j * 0.075, T_KART[i] + 0.34 + j * 0.075 + 0.24)
            if q > 0:
                x, y = O.nokta(Mk, [(sx, sy)])[0]
                O.yapistir(tuval, _yildiz(yr), x, y, alfa=min(1, q * 3), olcek=olcek * (0.4 + 0.6 * O.geri(q, 2.4)), aci=aci)
        tb = W_["onersin"] + i * 0.09                                 # "Doğrulanmış anne" rozeti
        q = O.ara(t, tb, tb + 0.6)
        if q > 0:
            rz = _rozet(dk)
            x, y = O.nokta(Mk, [(w - rz.shape[1] / 2 - (6 if dk else 0), 4 if dk else 2)])[0]
            O.yapistir(tuval, rz, x, y, alfa=min(1, q * 4), olcek=olcek * max(0.05, float(O.yay(q, 5.5, 1.6))), aci=aci - 2 + 3 * (1 - O.cik(q)))

    # kalkan: dolgu, onay, etiket
    tg = Z.ONERI_GUVEN
    kx, ky = Y["kalkan"]
    kal = O.nokta(M, _kalkan_yolu(kx, ky, Y["kb"] - 16))
    dolu = O.yumusak(O.ara(t, tg + 0.62, tg + 0.95))
    kmer = O.nokta(M, [(kx, ky)])[0]
    if dolu > 0:
        O.cokgen(tuval, kal + (5, 14), GOLGE, 0.20 * dolu, bulanik=12)
        O.cokgen(tuval, kal, O.NIL, dolu)
        O.cokgen(tuval, O.nokta(M, _kalkan_yolu(kx - Y["kb"] * 0.05, ky - Y["kb"] * 0.07, Y["kb"] * 0.72)), (255, 255, 255), 0.10 * dolu)
    par = O.ara(t, W_["guvenle"], W_["guvenle"] + 0.75)               # "güvenle": kalkan hafifçe parlar
    if 0 < par < 1:
        for gec in (0.0, 0.22):
            q = O.ara(par, gec, 1)
            O.halka(tuval, kmer[0], kmer[1], Y["kb"] * (0.55 + 0.75 * O.cik(q)) * ks, 0.55 * (1 - q) ** 1.5, O.NIL, 5)
    for anahtar, (x, y, hiza, boy), metin, renk, ag in [("dogrulanmis", Y["etiket"], "Doğrulanmış anneler", O.METIN, 800),
                                                         ("anneler", Y["alt"], "Güvenli, denetlenen topluluk", O.NIL_KOYU, 700)]:
        q = O.ara(t, W_[anahtar] + 0.05, W_[anahtar] + 0.45)
        if q > 0:
            p = O.nokta(M, [(x, y)])[0]
            O.yaz(tuval, metin, p[0], p[1] + (1 - O.cik(q)) * 22, "nunito", boy, renk, ag, alfa=O.yumusak(q), hiza=hiza, olcek=ks)

    # sönme: sahne geriye bulanıklaşıp çıplak zemine döner
    if sol > 0:
        tuval = O.karistir(O.bulanik(tuval, 16 * sol), zemin, sol)

    # çizgi (tek hat) — her şeyin üstünde, sönmeden etkilenmez
    yollar = [O.nokta(M, p) for p in O.alt_zaman(lambda u: _cizgi_yolu(u, dk), t, n=4)]
    hareketli = float(np.abs(yollar[0][-1] - yollar[-1][-1]).max()) > 0.5 or float(np.abs(yollar[0][0] - yollar[-1][0]).max()) > 0.5
    O.cizgi(tuval, yollar if hareketli else yollar[-1], 12 if dk else 11)
    onay = O.ara(t, tg + 0.78, tg + 1.10)                             # onay işareti: iki parça, o da cizgi() ile
    if onay > 0 and sol < 1:
        u = Y["kb"] / 272
        isaret = np.array([(kx - 44 * u, ky + 2 * u), (kx - 12 * u, ky + 36 * u), (kx + 48 * u, ky - 38 * u)], np.float32)
        O.cizgi(tuval, O.nokta(M, O.yol_parca(isaret, 0, O.cik(onay))), 15 * u, alfa=(1 - sol) ** 2, renk=O.BEYAZ, parlama=0)
    return tuval

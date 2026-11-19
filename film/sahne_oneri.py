"""Öneri + güven sahnesi: annelerin birbirine önerdiği üç kart (kreş, çocuk doktoru,
pişik kremi) kelimeyle aynı anda öne kalkar; çizgi kartları bir ipe dizer gibi tek sakin
hatla bağlar, sonra "Doğrulanmış anneler" kalkanını çizer. Yüz ve insan yok."""
import functools, json
import numpy as np, cv2
import ortak as O
from ortak import Z

ARALIK = (Z.ONERI_BAS, Z.SLOGAN_BAS)


def _kelime_anlari():
    v = dict(kres=19.60, doktor=20.25, deneyen=20.63, soylesin=21.03, dogrulanmis=21.90, anneler=22.63, icin=23.31, rahat=23.77, etsin=24.23)
    try:
        d = json.load(open(f"{O.KOK}/cikti/ses/kilit_lilyv3_1.json"))["satirlar"]
        a = {}
        for s in d:
            if s["metin"].startswith("Kreşi") or s["metin"].lower().startswith("doğrulanmış"):
                a.update({k["k"].lower(): k["bas"] for k in s["kelimeler"]})
        for anahtar, kelime in [("kres", "kreşi"), ("doktor", "doktoru"), ("deneyen", "deneyen"), ("soylesin", "söylesin"), ("dogrulanmis", "doğrulanmış"),
                                ("anneler", "anneler"), ("icin", "için"), ("rahat", "rahat"), ("etsin", "etsin")]:
            v[anahtar] = a.get(kelime, v[anahtar])
    except Exception:
        pass
    return v


W_ = _kelime_anlari()
T_KART = [W_["kres"], W_["doktor"], W_["deneyen"]]
KARTLAR = [("Kreş", "3 anne önerdi"), ("Çocuk doktoru", "5 anne önerdi"), ("Pişik kremi", "7 anne önerdi")]
GOLGE = (96, 70, 58)


# ------------------------------------------------------------------ yerleşim
@functools.lru_cache(maxsize=2)
def _yer(dikey):
    if dikey:      # kartlar alt alta, ip sol kenarda aşağı iner
        return dict(kw=820, kh=260, merkez=[(592, 868), (592, 1212), (592, 1556)], egim=[-0.8, 0.6, -0.5],
                    dugum=[(96, 868), (106, 1212), (96, 1556)],
                    kalkan=(540, 312), kb=272, etiket=(540, 548, "orta", 62), alt=(540, 612, "orta", 38))
    # yatay: kartlar yan yana, ip üstlerinden geçer (ipe asılı gibi)
    return dict(kw=500, kh=400, merkez=[(360, 762), (960, 762), (1560, 762)], egim=[-0.7, 0.4, -0.5],
                dugum=[(360, 506), (960, 514), (1560, 506)],
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
    baslik, hap = KARTLAR[i]
    k = O.bos_katman((w, h + 8))
    O.kutu(k, 0, 7, w, h + 7, 30, (228, 214, 202))                 # kâğıt kalınlığı
    O.kutu(k, 0, 0, w, h, 30, O.BEYAZ)
    karo = (O.NIL_ACIK, O.YAVRUAGZI_ACIK, O.NIL_ACIK)[i]
    if dikey:
        O.kutu(k, 28, 28, 232, 232, 26, karo)
        NESNE[i](k, 130, 126, 1.04)
        O.yaz(k, baslik, 264, 114, "nunito", 62, O.METIN, 800)
        r, ys, xs, adim, hb = 20, 190, 286, 46, 36
        hx1, hy0, hy1 = w - 28, 160, 220
    else:
        O.kutu(k, 26, 26, w - 26, 206, 24, karo)
        NESNE[i](k, w / 2, 114, 0.94)
        O.yaz(k, baslik, 32, 274, "nunito", 48, O.METIN, 800)
        r, ys, xs, adim, hb = 16, 336, 50, 37, 28
        hx1, hy0, hy1 = w - 26, 312, 360
    gen = O.yazi_olcu(hap, "nunito", hb, 800)[0]
    O.kutu(k, hx1 - gen - 2 * hb * 0.62, hy0, hx1, hy1, (hy1 - hy0) / 2, O.KREM2)
    O.yaz(k, hap, hx1 - hb * 0.62, (hy0 + hy1) / 2 + hb * 0.34, "nunito", hb, O.METIN2, 800, hiza="sag")
    yildiz = [(xs + j * adim, ys) for j in range(5)]
    for (x, y) in yildiz:
        O.cokgen(k, _yildiz_n(x, y, r), O.CIZGI_GRI)
    yy, xx = np.mgrid[0:h + 8, 0:w].astype(np.float32)             # sol üstten ışık: sağ alta doğru çok hafif koyulaşma
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
    """Arka plan ışık lekeleri (hafif): 1/8 çözünürlükte bir kez kurulur."""
    K = O.DIKEY if dikey else O.YATAY
    w, h = K.W // 8, K.H // 8
    k = O.bos_katman((w, h))
    for (fx, fy, fr, renk, al) in [(0.22, 0.20, 0.42, O.NIL_ACIK, 0.45), (0.86, 0.62, 0.46, O.YAVRUAGZI_ACIK, 0.40), (0.18, 0.92, 0.36, O.NIL_ACIK, 0.30)]:
        r = fr * min(w, h)
        O.cokgen(k, _elips(fx * w, fy * h, r, r * 0.9, 48), renk, al, bulanik=7.5)
    return k


# ------------------------------------------------------------------ çizginin yolu
def _kalkan_yolu(cx, cy, boy):
    """Kalkanın dış hattı (kapalı): üst ortadan başlar, saat yönünde döner."""
    h, w = boy / 2, boy * 0.41
    ust = np.array([(0, -h), (w * 0.72, -h)], np.float32)
    kose = O.bezier((w * 0.72, -h), (w, -h), (w, -h), (w, -h * 0.72), 8)
    yan = O.bezier((w, -h * 0.72), (w, h * 0.25), (w * 0.62, h * 0.72), (0, h), 22)
    sag = np.concatenate([ust, kose[1:], yan[1:]])
    sol = sag[::-1] * (-1, 1)
    return np.concatenate([sag, sol[1:]]) + (cx, cy)


def _kalkan_soldan(cx, cy, boy):
    """Aynı hat, sol kenarın ortasından başlayıp saatin tersine (aşağı doğru) dönen kapalı tur."""
    P = _kalkan_yolu(cx, cy, boy)[:-1][::-1]
    i = int(np.argmin(np.hypot(P[:, 0] - (cx - boy * 0.41), P[:, 1] - (cy - boy * 0.12))))
    P = np.roll(P, -i, axis=0)
    return np.concatenate([P, P[:1]])


@functools.lru_cache(maxsize=2)
def _yol(dikey):
    """Tek hat, sırayla: ipin ucu → (ip) → giriş → kalkan turu → çıkış.
    Dönüş: (noktalar, birikimli uzunluk, [düğüm3, düğüm2, düğüm1, giriş, kalkan başı, kalkan sonu, çıkış] uzunlukları)."""
    K = O.DIKEY if dikey else O.YATAY
    Y = _yer(dikey)
    g, c = np.array(K.oneri_giris, np.float32), np.array(K.slogan_giris, np.float32)
    kal = _kalkan_soldan(*Y["kalkan"], Y["kb"])
    k0 = kal[0]
    d1, d2, d3 = (np.array(p, np.float32) for p in Y["dugum"])
    if dikey:
        ip = [O.bezier(d3 + (4, 96), d3 + (2, 60), d3 + (0, 30), d3, 10), O.bezier(d3, d3 - (0, 120), d2 + (0, 120), d2, 24),
              O.bezier(d2, d2 - (0, 120), d1 + (0, 120), d1, 24), O.bezier(d1, d1 - (0, 130), g + (-40, 150), g, 24)]
        giris_kalkan = O.bezier(g, g + (40, -150), k0 + (-150, -190), k0, 36)
        cikis = O.bezier(k0, k0 + (0, 150), (40, 470), (46, 610), 30), O.bezier((46, 610), (50, 680), (90, 700), c, 14)
    else:
        ip = [O.bezier(d3 + (250, -10), d3 + (160, 0), d3 + (80, 0), d3, 10), O.bezier(d3, d3 - (200, 0), d2 + (200, 0), d2, 24),
              O.bezier(d2, d2 - (200, 0), d1 + (200, 0), d1, 24), O.bezier(d1, d1 - (190, 0), g + (-150, 170), g, 26)]
        giris_kalkan = O.bezier(g, g + (120, -140), k0 + (-120, -170), k0, 34)
        cikis = O.bezier(k0, k0 + (0, 120), (470, 430), (380, 400), 26), O.bezier((380, 400), (330, 380), (350, 332), c, 14)
    parca = ip + [giris_kalkan, kal, *cikis]
    P, sonlar = [parca[0]], []
    for p in parca[1:]:
        sonlar.append(sum(len(q) for q in P) - 1)
        P.append(np.asarray(p, np.float32)[1:])
    sonlar.append(sum(len(q) for q in P) - 1)
    P = np.concatenate(P).astype(np.float32)
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    L = [float(d[sonlar[i]]) for i in (0, 1, 2, 3, 4, 5, 7)]
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
    # ipin ucu: girişten düğümlere iner (uzunluk azalır)
    uc = _anahtar(t, [(Z.ONERI_BAS, L[3]), (tk[0], L[2]), (tk[0] + 0.05, L[2]), (tk[1], L[1]), (tk[1] + 0.03, L[1]), (tk[2], L[0]), (tk[2] + 0.30, 0.0),
                      (tg + 0.10, 0.0), (tg + 0.95, L[4]), (Z.ONERI_GECIS + 0.03, L[4]), (Z.SLOGAN_BAS - 0.03, L[6] - 120)])
    # öbür uç: ONERI_GUVEN'de girişten kalkana uzanır ve turu çizer; sonda çıkışa akar
    bas = _anahtar(t, [(tg, L[3]), (tg + 0.28, L[4]), (tg + 0.85, L[5]), (Z.ONERI_GECIS, L[5]), (Z.SLOGAN_BAS - 0.03, L[6])])
    bas = max(bas, uc + 2)
    return O.yol_parca(P, uc / d[-1], bas / d[-1])


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
    sol = O.yumusak(O.ara(t, Z.ONERI_GECIS, Z.SLOGAN_BAS - 0.04))
    tg = Z.ONERI_GUVEN

    gir = O.yumusak(O.ara(t, Z.ONERI_BAS, Z.ONERI_BAS + 0.5))        # arkada hafif ışık lekeleri
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
        cy += (1 - O.cik(p)) * 70 + 3.5 * np.sin(1.3 * t + i * 1.9)
        aci = Y["egim"][i] + (1 - O.cik(p)) * (6 if i % 2 else -6) + 0.25 * np.sin(0.9 * t + i)
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
        tb = W_["anneler"] + i * 0.10                                 # "Doğrulanmış anne" rozeti
        q = O.ara(t, tb, tb + 0.6)
        if q > 0:
            rz = _rozet(dk)
            x, y = O.nokta(Mk, [(w - rz.shape[1] / 2 - (6 if dk else 0), 4 if dk else 2)])[0]
            O.yapistir(tuval, rz, x, y, alfa=min(1, q * 4), olcek=olcek * max(0.05, float(O.yay(q, 5.5, 1.6))), aci=aci - 2 + 3 * (1 - O.cik(q)))

    # kalkan: dolgu, parlama, etiket
    kx, ky = Y["kalkan"]
    kal = O.nokta(M, _kalkan_yolu(kx, ky, Y["kb"] - 16))
    dolu = O.yumusak(O.ara(t, tg + 0.70, tg + 1.05))
    kmer = O.nokta(M, [(kx, ky)])[0]
    if dolu > 0:
        O.cokgen(tuval, kal + (5, 14), GOLGE, 0.20 * dolu, bulanik=12)
        O.cokgen(tuval, kal, O.NIL, dolu)
        O.cokgen(tuval, O.nokta(M, _kalkan_yolu(kx - Y["kb"] * 0.05, ky - Y["kb"] * 0.07, Y["kb"] * 0.72)), (255, 255, 255), 0.10 * dolu)
    for tp in (W_["icin"], W_["rahat"] + 0.08):                      # "için rahat etsin": kalkan yumuşakça parlar
        q = O.ara(t, tp, tp + 0.9)
        if 0 < q < 1:
            O.halka(tuval, kmer[0], kmer[1], Y["kb"] * (0.56 + 0.62 * O.cik(q)) * ks, 0.42 * (1 - q) ** 1.6, O.NIL, 5)
    for anahtar, (x, y, hiza, boy), metin, renk, ag in [("dogrulanmis", Y["etiket"], "Doğrulanmış anneler", O.METIN, 800),
                                                         ("anneler", Y["alt"], "Güvenli, denetlenen topluluk", O.NIL_KOYU, 700)]:
        q = O.ara(t, W_[anahtar] + 0.05, W_[anahtar] + 0.45)
        if q > 0:
            p = O.nokta(M, [(x, y)])[0]
            O.yaz(tuval, metin, p[0], p[1] + (1 - O.cik(q)) * 22, "nunito", boy, renk, ag, alfa=O.yumusak(q), hiza=hiza, olcek=ks)

    if sol > 0:                                                      # sönme: sahne geriye bulanıklaşıp çıplak zemine döner
        tuval = O.karistir(O.bulanik(tuval, 16 * sol), zemin, sol)

    # çizgi (tek hat) — her şeyin üstünde, sönmeden etkilenmez
    kal_c = 12 if dk else 11
    yollar = [O.nokta(M, p) for p in O.alt_zaman(lambda u: _cizgi_yolu(u, dk), t, n=4)]
    hareketli = float(np.abs(yollar[0][-1] - yollar[-1][-1]).max()) > 0.5 or float(np.abs(yollar[0][0] - yollar[-1][0]).max()) > 0.5
    O.cizgi(tuval, yollar if hareketli else yollar[-1], kal_c)
    for i in range(3):                                               # ipteki düğümler: kart belirirken parlar (ip toplanınca söner)
        q = O.ara(t, T_KART[i] - 0.02, T_KART[i] + 0.5)
        kalan = 1 - O.ara(t, tg + 0.10 + (2 - i) * 0.22, tg + 0.30 + (2 - i) * 0.22)
        if q > 0 and kalan > 0:
            dp = O.nokta(M, [Y["dugum"][i]])
            O.cizgi(tuval, np.concatenate([dp, dp]), kal_c * (1 + 1.5 * O.geri(q, 2.2) - 0.55 * O.cik(q)) * kalan, alfa=min(1, q * 5))
            if q < 1:
                O.halka(tuval, dp[0, 0], dp[0, 1], 14 + 46 * O.cik(q), 0.6 * (1 - q) ** 1.5, O.SOMON, 4)
    onay = O.ara(t, tg + 0.85, tg + 1.15)                             # onay işareti: o da cizgi() ile
    if onay > 0 and sol < 1:
        u = Y["kb"] / 272
        isaret = np.array([(kx - 44 * u, ky + 2 * u), (kx - 12 * u, ky + 36 * u), (kx + 48 * u, ky - 38 * u)], np.float32)
        O.cizgi(tuval, O.nokta(M, O.yol_parca(isaret, 0, O.cik(onay))), 15 * u, alfa=(1 - sol) ** 2, renk=O.BEYAZ, parlama=0)
    return tuval

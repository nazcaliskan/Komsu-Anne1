"""S4 · Sokak (7.20–10.00). Yukarıdan, hafif eğik görülen mahalle; evler yerden yükselir,
mercan çizgi A evinden B evine rotayı çizer, yeşil pin bekler, cam kart açılır.

Devirler: 7.05–7.60 A evinin kapı önü = K.evA (birleştirici deliği buradan açar);
t >= 10.00 çizginin ucu = K.s5_giris (S5 ile uzun geçiş)."""
import functools
import numpy as np, cv2
from ortak import *  # noqa: F401,F403
from ortak import _yd_noktalar, _maske, _birlestir

ARALIK = (7.05, 10.25)

# ------------------------------------------------------------------ dünya (harita düzlemi, piksel)
SOKAKLAR = [300 + 700 * k for k in range(-2, 6)]        # sokak eksenleri (x ve y için aynı ızgara)
YARI = 60                                               # sokak yarı genişliği
OX = OY = 1700                                          # doku kayması: dünya (-1700..) -> doku 0
DOKU = 5900
YUVA = [(40, 40), (330, 40), (40, 330), (330, 330)]     # ada içindeki dört ev yuvası (210x210)

SOKAK_RENK = (236, 229, 221)        # açık, sıcak asfalt
SERIT = (250, 246, 240)             # şerit ve yaya geçidi çizgileri
KALDIRIM = (252, 247, 241)
KALDIRIM_GOLGE = (212, 201, 190)
ADA_RENK = (244, 234, 223)
BAHCE = (229, 234, 214)
TABAN = (231, 217, 203)
GUNES = (1.0, 0.72)                 # güneş sol üstte: gölgeler sağ alta düşer (yükseklik başına)
GOLGE_RENK = (92, 70, 58)
CATILAR = [(247, 201, 168), (226, 150, 128), (250, 240, 228), (201, 220, 206), (240, 214, 170), (232, 176, 150)]
DUVAR = (247, 233, 219)


def _duzen(K):
    """Biçime göre A/B evleri, kapı önleri, rota ve pinin yeri (dünya koordinatı)."""
    if K.dikey:
        A = (690, 1800, 900, 2010); kapiA = (940, 1905)
        B = (1390, 690, 1600, 900); kapiB = (1495, 940)
        rota = [kapiA, (1000, 1905), (1000, 1000), (1495, 1000), kapiB]
        pin_kay = (46, 2)
        sB = (620, 800)
    else:
        A = (400, 690, 610, 900); kapiA = (505, 940)
        B = (1800, 400, 2010, 610); kapiB = (1760, 505)
        rota = [kapiA, (505, 1000), (1700, 1000), (1700, 505), kapiB]
        pin_kay = (2, -46)
        sB = (1150, 600)
    return dict(A=A, B=B, kapiA=np.array(kapiA, np.float32), kapiB=np.array(kapiB, np.float32),
                rota=_yuvarla(rota), pin=np.array(kapiB, np.float32) + pin_kay, sB=sB)


def _yuvarla(pts, r=58, n=12):
    """Köşeleri yay ile yumuşatılmış çoklu çizgi."""
    P = [np.array(p, np.float32) for p in pts]
    out = [P[0]]
    for i in range(1, len(P) - 1):
        a, b, c = P[i - 1], P[i], P[i + 1]
        ra = min(r, np.linalg.norm(b - a) / 2); rc = min(r, np.linalg.norm(c - b) / 2)
        p0 = b + (a - b) / np.linalg.norm(a - b) * ra
        p1 = b + (c - b) / np.linalg.norm(c - b) * rc
        u = np.linspace(0, 1, n)[:, None]
        out.extend((1 - u) ** 2 * p0 + 2 * (1 - u) * u * b + u ** 2 * p1)
    out.append(P[-1])
    Q = np.array(out, np.float32)
    # eşit aralıklı yeniden örnekle (yol_parca düzgün ilerlesin)
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    s = np.linspace(0, d[-1], int(d[-1] / 8))
    return np.stack([np.interp(s, d, Q[:, 0]), np.interp(s, d, Q[:, 1])], 1).astype(np.float32)


@functools.lru_cache(maxsize=1)
def _evler():
    """Tüm evler: (x0, y0, x1, y1, yükseklik, çatı rengi, ozel). İki biçimin A/B yuvaları sabit ölçüde."""
    r = np.random.default_rng(21)
    ozel = {(690, 1800): "A", (1390, 690): "B", (400, 690): "A", (1800, 400): "B"}
    L = []
    for sx in SOKAKLAR[:-1]:
        for sy in SOKAKLAR[:-1]:
            bx, by = sx + YARI, sy + YARI
            for ox, oy in YUVA:
                x0, y0 = bx + ox, by + oy
                oz = ozel.get((x0, y0))
                if oz is None and r.random() < 0.14:
                    continue                                   # boş yuva: bahçe/ağaç
                if oz:
                    w = h = 210; yuk = 104
                    cati = CATILAR[0] if oz == "A" else CATILAR[3]
                else:
                    w, h = r.integers(165, 211, 2); yuk = int(r.integers(64, 118))
                    x0 += int(r.integers(0, 211 - w)); y0 += int(r.integers(0, 211 - h))
                    cati = CATILAR[int(r.integers(0, len(CATILAR)))]
                L.append((x0, y0, x0 + int(w), y0 + int(h), yuk, cati, oz))
    return L


@functools.lru_cache(maxsize=1)
def _doku():
    """Değişmeyen zemin dokusu: asfalt, şeritler, yaya geçitleri, kaldırımlar, bahçeler, ağaçlar ve gölgeleri."""
    T = np.empty((DOKU, DOKU, 3), np.uint8)
    T[:] = SOKAK_RENK
    def P(pts):
        return np.round((np.asarray(pts, np.float32) + (OX, OY)) * 16).astype(np.int32)
    def N(x, y):
        return (int(round((x + OX) * 16)), int(round((y + OY) * 16)))
    def dolu(x0, y0, x1, y1, renk, r=0):
        cv2.fillPoly(T, [P(_yd_noktalar(x0, y0, x1, y1, r))], renk, cv2.LINE_AA, 4)
    lo, hi = SOKAKLAR[0], SOKAKLAR[-1]
    for sk in SOKAKLAR:                                   # kesik orta şerit (kavşaklarda yok)
        for u in range(lo, hi, 64):
            if min(abs(u + 17 - k) for k in SOKAKLAR) > YARI + 70:
                dolu(sk - 2.5, u, sk + 2.5, u + 34, SERIT, 2)
                dolu(u, sk - 2.5, u + 34, sk + 2.5, SERIT, 2)
    for sx in SOKAKLAR:                                   # yaya geçitleri: her kavşağın dört kolunda
        for sy in SOKAKLAR:
            for k in range(-2, 3):
                o = k * 22
                dolu(sx + o - 6, sy - YARI - 52, sx + o + 6, sy - YARI - 12, SERIT, 2)
                dolu(sx + o - 6, sy + YARI + 12, sx + o + 6, sy + YARI + 52, SERIT, 2)
                dolu(sx - YARI - 52, sy + o - 6, sx - YARI - 12, sy + o + 6, SERIT, 2)
                dolu(sx + YARI + 12, sy + o - 6, sx + YARI + 52, sy + o + 6, SERIT, 2)
    for sx in SOKAKLAR[:-1]:                              # adalar: kaldırım (hafif yüksek, kenar gölgeli) + iç zemin
        for sy in SOKAKLAR[:-1]:
            bx, by = sx + YARI, sy + YARI
            dolu(bx + 5, by + 6, bx + 585, by + 586, KALDIRIM_GOLGE, 46)
            dolu(bx, by, bx + 580, by + 580, KALDIRIM, 46)
            dolu(bx + 24, by + 24, bx + 556, by + 556, ADA_RENK, 30)
    for x0, y0, x1, y1, yuk, cati, oz in _evler():
        dolu(x0 - 22, y0 - 22, x1 + 22, y1 + 22, BAHCE, 26)
        dolu(x0, y0, x1, y1, TABAN, 6)
    for x0, y0, x1, y1 in [(770, 1880, 940, 1930), (1470, 900, 1520, 940), (480, 900, 530, 940), (1760, 480, 1800, 530)]:
        dolu(x0, y0, x1, y1, KALDIRIM, 8)                 # kapı önü patikaları (iki biçimin A/B evleri)
    # yüzey dokusu: ince tane + iri lekeler (asfalt/taş hissi)
    r = np.random.default_rng(9)
    n = 1024
    def sarmal(sigma):                                     # döşenebilir (kenarları saran) bulanık gürültü
        p = int(sigma * 4) + 2
        g = np.pad(r.standard_normal((n, n)).astype(np.float32), p, mode="wrap")
        return cv2.GaussianBlur(g, (0, 0), sigma)[p:-p, p:-p]
    tane, leke = sarmal(0.8), sarmal(26)
    doku = (tane / tane.std() * 2.2 + leke / leke.std() * 2.6)[..., None]
    for y in range(0, DOKU, n):
        for x in range(0, DOKU, n):
            blok = T[y:y + n, x:x + n]
            blok[:] = np.clip(blok.astype(np.float32) + doku[:blok.shape[0], :blok.shape[1]], 0, 255).astype(np.uint8)
    # ağaçlar: uzun yumuşak gölge + hacimli tepe (açık üst-sol, koyu alt-sağ)
    r = np.random.default_rng(5)
    agac = []
    for sx in SOKAKLAR[:-1]:
        for sy in SOKAKLAR[:-1]:
            bx, by = sx + YARI, sy + YARI
            for _ in range(4):
                if r.random() < 0.5:
                    agac.append((bx + 290 + r.normal(0, 5), by + r.uniform(50, 530), r.uniform(24, 36)))
                else:
                    agac.append((bx + r.uniform(50, 530), by + 290 + r.normal(0, 5), r.uniform(24, 36)))
    G = np.zeros((DOKU, DOKU), np.uint8)
    for x, y, rr in agac:
        cv2.line(G, N(x + 8, y + 6), N(x + rr * 1.9 * GUNES[0], y + rr * 1.9 * GUNES[1]), 255, int(rr * 1.7), cv2.LINE_AA, 4)
    G = cv2.GaussianBlur(G, (0, 0), 7)
    for y in range(0, DOKU, n):
        blok = T[y:y + n]
        g = G[y:y + n, :, None].astype(np.float32) / 255 * 0.24
        blok[:] = (blok * (1 - g) + np.array(GOLGE_RENK, np.float32) * g).astype(np.uint8)
    for x, y, rr in agac:
        cv2.circle(T, N(x + rr * 0.06, y + rr * 0.06), int(rr * 16), (104, 138, 116), -1, cv2.LINE_AA, 4)
        cv2.circle(T, N(x - rr * 0.10, y - rr * 0.10), int(rr * 0.86 * 16), (132, 164, 141), -1, cv2.LINE_AA, 4)
        for dx, dy, k in [(-0.34, -0.30, 0.46), (0.10, -0.42, 0.34), (-0.42, 0.12, 0.30)]:
            cv2.circle(T, N(x + rr * dx, y + rr * dy), int(rr * k * 16), (158, 187, 163), -1, cv2.LINE_AA, 4)
        cv2.circle(T, N(x - rr * 0.40, y - rr * 0.36), int(rr * 0.20 * 16), (184, 207, 186), -1, cv2.LINE_AA, 4)
    return cv2.GaussianBlur(T, (0, 0), 0.7)


# ------------------------------------------------------------------ zaman eğrileri
def _ilerleme(t):
    x = ara(t, Z.ROTA_BAS, Z.ROTA_SON)
    return 0.3 * x + 0.7 * yumusak(x)


def _uc_dunya(t, D):
    return yol_nokta(D["rota"], _ilerleme(t))


def _kamera(t, K, D):
    """Dünya -> ekran afin dönüşümü ve yakınlık. Kamera çizginin ucunu yumuşakça izler, hiç durmaz."""
    capa = np.mean([_uc_dunya(u, D) for u in np.linspace(t - 0.35, t + 0.3, 9)], axis=0)
    a = gir_cik(ara(t, 7.7, 9.6)); b = gir_cik(ara(t, 9.6, 10.0))
    s = np.array(K.evA, np.float32) * (1 - a) + np.array(D["sB"], np.float32) * a
    s = s * (1 - b) + np.array(K.s5_giris, np.float32) * b
    z = 1.10 - 0.10 * yumusak(ara(t, 7.05, 8.2)) + 0.05 * ara(t, 8.2, 9.7)
    z *= 1 + 1.3 * gir(ara(t, 9.7, 10.25), 2)
    aci = -2.0 + 3.5 * ara(t, 7.05, 10.25)
    return afin(z, aci, capa, s), z, s


def _yukselme(t, t0):
    return float(yay((t - t0) / 0.75, 5.0, 1.3)) if t > t0 else 0.0


def _ton(renk, k):
    return tuple(float(np.clip(c * k, 0, 255)) for c in renk)


def _ev_ciz(dst, M, z, ev, f):
    x0, y0, x1, y1, yuk, cati, oz = ev
    taban = nokta(M, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    h = yuk * z * f
    if h < 1.5:
        return
    ust = taban - (0, h)
    orta = taban.mean(0)
    kenarlar = []
    for i in range(4):
        a, b = taban[i], taban[(i + 1) % 4]
        n = (a + b) / 2 - orta
        n = n / (np.linalg.norm(n) + 1e-6)
        if n[1] > -0.15:
            kenarlar.append(((a[1] + b[1]) / 2, i, n))
    for _, i, n in sorted(kenarlar, key=lambda e: e[0]):
        a, b = taban[i], taban[(i + 1) % 4]
        ua = ust[i]
        p = lambda u, v: a + (b - a) * u + (ua - a) * v
        def dort(u0, u1, v0, v1, renk, alfa=1.0):
            cokgen(dst, np.array([p(u0, v0), p(u1, v0), p(u1, v1), p(u0, v1)]), renk, alfa)
        isik = 0.90 + 0.05 * n[1] - 0.10 * max(n[0], 0) + 0.06 * max(-n[0], 0)     # güneş solda: sola bakan yüz aydınlık
        dort(0, 1, 0, 1, _ton(DUVAR, isik))
        dort(0, 1, 0.84, 1, GOLGE_RENK, 0.16)                    # saçak gölgesi
        dort(0, 1, 0, 0.16, GOLGE_RENK, 0.10)                    # duvar dibinde ortam kararması
        if n[1] > 0.8 and h > 14:                                # güney duvarı: kapı ve pencereler
            dort(0.42, 0.58, 0.0, 0.58, (142, 166, 149) if oz == "B" else (186, 112, 92))
            dort(0.42, 0.58, 0.52, 0.58, GOLGE_RENK, 0.22)
            for u0 in (0.12, 0.70):
                dort(u0 - 0.012, u0 + 0.192, 0.25, 0.30, _ton(DUVAR, 0.78))          # denizlik
                dort(u0, u0 + 0.18, 0.30, 0.70, (255, 236, 190))                     # sıcak iç ışık
                cokgen(dst, np.array([p(u0, 0.70), p(u0 + 0.13, 0.70), p(u0, 0.44)]), (255, 250, 232), 0.85)   # cam yansıması
    tepe = orta - (0, h)
    sacak = tepe + (ust - tepe) * 1.08
    m01, m32 = (sacak[0] + sacak[3]) / 2, (sacak[1] + sacak[2]) / 2
    aydin, golgeli = _ton(cati, 1.04), _ton(cati, 0.87)
    cokgen(dst, np.array([sacak[0], sacak[1], m32, m01]), aydin)             # ışık alan yüz
    cokgen(dst, np.array([m01, m32, sacak[2], sacak[3]]), golgeli)           # gölgede kalan yüz
    kal = max(1.2, 1.7 * z)
    if h > 10:                                                               # kiremit sıraları (yüz başına tek maske)
        for p0, p1, q0, q1, renk in ((sacak[0], m01, sacak[1], m32, _ton(aydin, 0.93)), (m01, sacak[3], m32, sacak[2], _ton(golgeli, 0.92))):
            mk, mx, my = _maske([np.array([p0 + (p1 - p0) * k, q0 + (q1 - q0) * k]) for k in (0.25, 0.5, 0.75)], kapali=False, kalinlik=kal)
            _birlestir(dst, renk, mk * 0.75, mx, my)
    cizgi_ince(dst, np.array([m01, m32]), _ton(cati, 1.12), max(1.5, 2.6 * z), 0.9)   # mahya
    if (x0 * 3 + y0) % 3 == 0 and h > 20:                                    # baca
        c0 = sacak[0] + (sacak[1] - sacak[0]) * 0.70 + (sacak[3] - sacak[0]) * 0.22
        e1, e2 = (sacak[1] - sacak[0]) * 0.11, (sacak[3] - sacak[0]) * 0.11
        yk = np.array([0, -h * 0.16])
        cokgen(dst, np.array([c0 + e2, c0 + e1 + e2, c0 + e1 + e2 + (h * 0.20, h * 0.14), c0 + e2 + (h * 0.20, h * 0.14)]), GOLGE_RENK, 0.22)
        cokgen(dst, np.array([c0 + e2, c0 + e1 + e2, c0 + e1 + e2 + yk, c0 + e2 + yk]), _ton(DUVAR, 0.82))
        cokgen(dst, np.array([c0 + yk, c0 + e1 + yk, c0 + e1 + e2 + yk, c0 + e2 + yk]), _ton(DUVAR, 0.98))


@functools.lru_cache(maxsize=4)
def _izgara(W, H):
    y, x = np.mgrid[0:H // 8, 0:W // 8].astype(np.float32)
    return x * 8 + 4, y * 8 + 4


def _alan_derinligi(tuval, merkez, guc):
    """Sığ alan derinliği: odak merkezinden uzaklaştıkça kenarlar yumuşar."""
    H, W = tuval.shape[:2]
    x, y = _izgara(W, H)
    d = np.hypot(x - merkez[0], y - merkez[1]) / (0.5 * np.hypot(W, H))
    m = np.clip((d - 0.30) / 0.62, 0, 1) ** 1.4 * guc
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    bl = bulanik(tuval, 3.0 + 5.0 * guc)
    tuval[:] = (tuval * (1 - m) + bl * m + 0.5).astype(np.uint8)


# ------------------------------------------------------------------ sahne
def ciz(t, K, tuval):
    D = _duzen(K)
    M, z, s = _kamera(t, K, D)
    # zemin dokusu (nefes alan ışığı korumak için tuvalin tonuyla çarpılır)
    Md = M.copy(); Md[:, 2] -= M[:, :2] @ np.array([OX, OY], np.float32)
    harita = cv2.warpAffine(_doku(), Md, (K.W, K.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    isik = tuval.astype(np.float32) / np.array(KREM, np.float32)
    tuval[:] = np.clip(harita.astype(np.float32) * isik, 0, 255).astype(np.uint8)

    # evler: A'dan dışa doğru dalga hâlinde yükselir
    gorunen = []
    for ev in _evler():
        c = nokta(M, [[(ev[0] + ev[2]) / 2, (ev[1] + ev[3]) / 2]])[0]
        if -260 < c[0] < K.W + 260 and -260 < c[1] < K.H + 420:
            uzak = np.hypot((ev[0] + ev[2]) / 2 - D["kapiA"][0], (ev[1] + ev[3]) / 2 - D["kapiA"][1])
            t0 = Z.EVLER[min(len(Z.EVLER) - 1, int(uzak / 330))] + (ev[0] * 7 + ev[1] * 3) % 9 / 100
            if ev[6] == "A":
                t0 = Z.EVLER[0]
            gorunen.append((nokta(M, [[ev[0], ev[3]], [ev[2], ev[3]]])[:, 1].max(), ev, _yukselme(t, t0)))
    for _, ev, f in gorunen:                                     # tek yönlü güneş: uzun yumuşak gölge + dip kararması
        if f > 0.02:
            x0, y0, x1, y1, yuk = ev[:5]
            k = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
            g = np.concatenate([k, k + (yuk * f * GUNES[0], yuk * f * GUNES[1])])
            hull = cv2.convexHull(nokta(M, g)).reshape(-1, 2)
            cokgen(tuval, hull, GOLGE_RENK, 0.21, bulanik=6 * z)
            cokgen(tuval, nokta(M, [[x0 - 9, y0 - 9], [x1 + 9, y0 - 9], [x1 + 9, y1 + 9], [x0 - 9, y1 + 9]]), GOLGE_RENK, 0.14 * min(1, f), bulanik=7 * z)
    for _, ev, f in sorted(gorunen, key=lambda e: e[0]):
        _ev_ciz(tuval, M, z, ev, f)

    # yeşil pin: ~8.4'te düşer, 9.60'ta çizgi ulaşınca zıplar
    px, py = nokta(M, [D["pin"]])[0]
    boy = 122 * z
    dus = ara(t, 8.40, 8.78)
    if dus > 0:
        dy = -(1 - geri(dus, 1.4)) * 300 - 28 * z * np.sin(np.pi * ara(t, Z.YESIL_PIN, Z.YESIL_PIN + 0.32))
        ez = 1 - 0.13 * np.sin(np.pi * ara(t, 8.72, 8.98)) + 0.08 * np.sin(np.pi * ara(t, Z.YESIL_PIN, Z.YESIL_PIN + 0.32))
        a_pin = min(1.0, dus * 4)
        yer = max(0.0, 1 - (-dy) / 260)
        cokgen(tuval, np.array([(px - 9 * z, py), (px + 9 * z, py + 4 * z), (px + boy * 0.80, py + boy * 0.50), (px + boy * 0.62, py + boy * 0.22)]) + (0, 0),
               GOLGE_RENK, 0.26 * a_pin * yer, bulanik=4 * z)
        for i in range(3):
            x = ara(t, Z.YESIL_PIN + 0.09 * i, Z.YESIL_PIN + 0.09 * i + 0.62)
            if 0 < x < 1:
                halka(tuval, px, py - 4 * z, (18 + 120 * cik(x, 2)) * z, (1 - x) * 0.75, ADACAYI, kalinlik=max(2, 5 * z * (1 - x * 0.5)))
        pin(tuval, px, py + dy, boy, ADACAYI, a_pin, olcek_x=2 - ez, olcek_y=ez, delik=(250, 244, 236))

    # sığ alan derinliği: hep hafif, kapıya itişte belirgin
    _alan_derinligi(tuval, s, 0.38 + 0.62 * yumusak(ara(t, 9.55, 10.1)))

    # kapıya doğru itiş: yakınlaşma bulanıklığı (çizgi ve kart net kalır)
    guc = 0.11 * gir(ara(t, 9.72, 10.25), 1.6)
    if guc > 0.002:
        tuval[:] = radyal_bulanik(tuval, s, guc, adim=6 + int(guc * 60))

    # mercan çizgi: t >= 7.60 (öncesinde noktayı birleştirici çizer)
    if t >= 7.60:
        def yol(u):
            Mu = _kamera(u, K, D)[0]
            return nokta(Mu, yol_parca(D["rota"], 0.0, max(_ilerleme(u), 0.0015)))
        kal = 13 * z
        if _ilerleme(t) < 1 or t < 10.0:
            cizgi(tuval, alt_zaman(yol, t, 4), kal)
        else:
            cizgi(tuval, yol(t), kal)

    # cam kart: "Defne · bir sokak ötede" (9.20'de açılır, geçişe kadar okunur)
    a = cik(ara(t, Z.CAM_KART, Z.CAM_KART + 0.26))
    if a > 0.01:
        metin, boyut = "Defne · bir sokak ötede", 46
        mw, asc, desc = yazi_olcu(metin, "nunito", boyut, 800)
        av = 80
        w, h = 28 + av + 22 + mw + 38, 120
        cx = float(np.clip(px, 44 + w / 2, K.W - 44 - w / 2))
        y1 = py - boy - 30 + (1 - a) * 26
        y1 = max(y1, (150 if K.dikey else 60) + h)
        x0, y0 = cx - w / 2, y1 - h
        cam_kart(tuval, x0, y0, x0 + w, y1, r=36, alfa=a, opak=0.84)
        yapistir(tuval, avatar(av, "b"), x0 + 28 + av / 2, y0 + h / 2, alfa=a)
        yaz(tuval, metin, x0 + 28 + av + 22, y0 + h / 2 + asc * 0.36, "nunito", boyut, METIN, 800, alfa=a)
    return tuval

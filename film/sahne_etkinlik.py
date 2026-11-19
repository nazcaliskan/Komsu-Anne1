"""Etkinlik sahnesi (14.40–19.15): yukarıdan park köşesi, piknik örtüsü, kelimeyle beliren nesneler,
çizginin süs ipi olması ve etkinlik kartında "Katılıyorum" dokunuşu. Yüz ve insan figürü yok."""
import functools, json
import numpy as np, cv2
import ortak as O
from ortak import Z

ARALIK = (Z.ETK_BAS, Z.SOZ_BAS)
T0, T_SON = Z.ETK_BAS, Z.SOZ_BAS
BITIS = T_SON - 0.03                     # 19.12: bundan sonra ekranda yalnız çizgi

try:
    _k = json.load(open(f"{O.KOK}/cikti/ses/kilit_lilyv3_5.json"))["satirlar"]
    _w = {k["k"].lower(): k["bas"] for s in (_k[6], _k[7]) for k in s["kelimeler"]}
except Exception:
    _w = {}
T_KAHVE, T_OYUN, T_ATOLYE = _w.get("kahve", 14.73), _w.get("oyun", 15.68), _w.get("atölye", 16.26)
T_MAHALLE = _w.get("mahalle", 17.93)

M = 70                                   # arka plan payı (kamera kayması için)
ISIK = np.array([10.0, 14.0])            # gölge yönü: ışık sol üstten
AHSAP, KAHVE, KOYU = (222, 186, 140), (116, 76, 52), (70, 58, 52)
BAYRAK = [O.SEFTALI, O.ADACAYI_ACIK, O.BAL, O.LAVANTA, (255, 250, 240), O.MERCAN_ACIK]


def _yer(K):
    if K.dikey:
        return dict(ortu=(540, 800), s=1.0, A=(96, 330), B=(984, 330), sark=95, n=9, by=(78, 98),
                    kart=(86, 1236), ks=1.0, agac=[(-20, 560, 1.0), (1100, 610, 0.9), (10, 1190, 1.1), (1090, 1160, 0.8)],
                    patika=[(-80, 500), (260, 450), (640, 250), (1160, 170)])
    return dict(ortu=(590, 610), s=0.92, A=(120, 150), B=(1800, 150), sark=78, n=15, by=(74, 92),
                kart=(1120, 338), ks=0.8, agac=[(-10, 330, 1.0), (1090, 1000, 1.0), (1900, 560, 1.1), (60, 1010, 0.8)],
                patika=[(-80, 880), (500, 990), (1200, 1010), (2000, 900)])


def _rng(n):
    return np.random.default_rng(n)


def _gurultu(h, w, olcek, tohum):
    k = _rng(tohum).standard_normal((max(2, h // olcek + 2), max(2, w // olcek + 2))).astype(np.float32)
    return cv2.resize(k, (w, h), interpolation=cv2.INTER_CUBIC)


def _el(dst, cx, cy, rx, ry, renk, alfa=1.0, aci=0.0, n=40):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(np.radians(aci)), np.sin(np.radians(aci))
    x, y = rx * np.cos(a), ry * np.sin(a)
    O.cokgen(dst, np.stack([cx + x * c - y * s, cy + x * s + y * c], 1), renk, alfa)


# ------------------------------------------------------------------ arka plan (önbellek)
@functools.lru_cache(maxsize=2)
def _arka(K):
    Y = _yer(K)
    W, H = K.W + 2 * M, K.H + 2 * M
    img = np.empty((H, W, 3), np.float32)
    img[:] = (183, 206, 172)
    g = _gurultu(H, W, 90, 1) * 7 + _gurultu(H, W, 14, 2) * 3.5 + _rng(3).standard_normal((H, W)).astype(np.float32) * 2.2
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g += np.sin((xx * 0.6 + yy) / 62.0) * 2.2                         # biçme izleri
    img += g[..., None] * np.array([0.8, 1.0, 0.7], np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # patika: açık, yumuşak kenarlı
    kat = O.bos_katman((W, H))
    yol = O.egri([(x + M, y + M) for x, y in Y["patika"]], 30)
    O.cizgi_ince(kat, yol + ISIK * 0.35, (150, 170, 140), 168, 0.5)
    O.cizgi_ince(kat, yol, (240, 229, 210), 160)
    kat = O.bulanik(kat, 2.2)
    O.katman_bas(img, kat)
    b = img.astype(np.float32)
    mk = kat[..., 3:].astype(np.float32) / 255
    b += mk * (_gurultu(H, W, 5, 4) * 3.0)[..., None]
    img = np.clip(b, 0, 255).astype(np.uint8)
    r = _rng(5)                                                       # küçük kır çiçekleri
    for _ in range(int(W * H / 26000)):
        x, y = r.uniform(0, W), r.uniform(0, H)
        _el(img, x, y, 3.2, 3.2, [(255, 252, 244), (250, 226, 150), (247, 214, 196)][r.integers(3)], 0.85, n=10)
    for ax, ay, s in Y["agac"]:                                        # ağaç gölgeleri (çimde)
        for dx, dy, rr in _AGAC:
            _golge_el(img, ax + M + dx * s + 30, ay + M + dy * s + 42, rr * s, 20, 0.20)
    ox, oy = Y["ortu"][0] + M, Y["ortu"][1] + M                        # örtü + gölgesi
    ort = _ortu(Y["s"])
    O.yapistir(img, _golge(ort, 14, 0.30), ox + 12, oy + 16, aci=-6)
    O.yapistir(img, ort, ox, oy, aci=-6)
    return img


_AGAC = [(0, 0, 120), (-90, 50, 80), (80, 70, 90), (40, -90, 75), (-70, -70, 70), (110, -20, 66)]


def _golge_el(img, cx, cy, r, blur, alfa):
    a = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    O.cokgen(img, np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1), (52, 70, 56), alfa, bulanik=blur)


@functools.lru_cache(maxsize=2)
def _taclar(K):
    """Ağaç tepeleri: ön planda, hafif bulanık (sığ alan derinliği)."""
    Y = _yer(K)
    kat = O.bos_katman((K.W + 2 * M, K.H + 2 * M))
    r = _rng(8)
    for ax, ay, s in Y["agac"]:
        for i, (dx, dy, rr) in enumerate(_AGAC):
            cx, cy, R = ax + M + dx * s, ay + M + dy * s, rr * s
            ton = [(92, 128, 104), (104, 142, 114), (118, 154, 124)][i % 3]
            _el(kat, cx, cy, R, R, ton)
            _el(kat, cx - R * 0.22, cy - R * 0.26, R * 0.66, R * 0.62, tuple(min(255, c + 16) for c in ton), 0.8)
            for _ in range(5):
                a, d = r.uniform(0, 6.28), r.uniform(0.2, 0.75) * R
                _el(kat, cx + d * np.cos(a) - 6, cy + d * np.sin(a) - 8, R * 0.16, R * 0.14, (150, 182, 150), 0.5, n=14)
    return O.bulanik(kat, 2.6)


@functools.lru_cache(maxsize=2)
def _benek(K):
    """Yaprak arası ışık benekleri: yavaşça kayan hafif koyuluk haritası."""
    W, H = K.W + 2 * M, K.H + 2 * M
    g = cv2.GaussianBlur(_gurultu(H, W, 60, 21), (0, 0), 14)
    g = np.clip((g - 0.15) * 2.2, 0, 1)
    out = (255 - g * 12).astype(np.uint8)
    return cv2.merge([out, out, np.clip(out.astype(np.int16) + 3, 0, 255).astype(np.uint8)])


@functools.lru_cache(maxsize=2)
def _ortu(s):
    w, h = int(660 * s), int(560 * s)
    kat = O.bos_katman((w, h))
    O.kutu(kat, 0, 0, w, h, 10, (251, 244, 233))
    kare = 70 * s
    for i in range(int(w / kare) + 1):
        if i % 2 == 0:
            O.kutu(kat, i * kare, 0, min(w, (i + 1) * kare), h, 0, (236, 150, 128), 0.42)
    for j in range(int(h / kare) + 1):
        if j % 2 == 0:
            O.kutu(kat, 0, j * kare, w, min(h, (j + 1) * kare), 0, (236, 150, 128), 0.42)
    f = kat.astype(np.float32)
    doku = _rng(11).standard_normal((h, w)).astype(np.float32) * 3.0 + _gurultu(h, w, 80, 12) * 7 + _gurultu(h, w, 3, 13) * 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    doku += -((xx / w + yy / h) - 1) * 7                               # sol üst aydınlık, sağ alt gölgeli
    f[..., :3] = np.clip(f[..., :3] + doku[..., None], 0, 255)
    f[..., :3] *= f[..., 3:] / 255
    kat = f.astype(np.uint8)
    y, x = np.mgrid[0:h, 0:w]
    kose = (np.minimum(x, w - 1 - x) < 2) | (np.minimum(y, h - 1 - y) < 2)
    kat[kose] = (kat[kose] * 0.6).astype(np.uint8)
    return kat


def _golge(hare, blur=9, alfa=0.34, pay=None):
    """Hare alfasından yumuşak temas gölgesi (ön çarpılmış)."""
    pay = int(blur * 3) if pay is None else pay
    a = cv2.copyMakeBorder(hare[..., 3], pay, pay, pay, pay, cv2.BORDER_CONSTANT, value=0).astype(np.float32) / 255
    a = np.clip(cv2.GaussianBlur(a, (0, 0), blur) * 0.75 + cv2.GaussianBlur(a, (0, 0), max(1.5, blur * 0.3)) * 0.25, 0, 1) * alfa
    out = np.empty(a.shape + (4,), np.float32)
    out[..., :3] = np.array((58, 60, 48), np.float32) * a[..., None]
    out[..., 3] = a * 255
    return out.astype(np.uint8)


def _isiklandir(kat, guc=10):
    """Sol üstten yönlü ışık: hare üzerinde yumuşak açık-koyu geçiş + ince doku."""
    h, w = kat.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = (1 - (xx / w + yy / h)) * guc + _rng(h * 7 + w).standard_normal((h, w)).astype(np.float32) * 1.6
    f = kat.astype(np.float32)
    a = f[..., 3:] / 255
    f[..., :3] = np.clip(f[..., :3] + g[..., None] * a, 0, 255 * a)
    return f.astype(np.uint8)


# ------------------------------------------------------------------ nesneler (üstten görünüm)
def _bardak():
    k = O.bos_katman((120, 120))
    _el(k, 60, 60, 52, 52, (255, 252, 246))
    _el(k, 60, 60, 44, 44, (236, 228, 216))
    _el(k, 60, 60, 40, 40, KAHVE)
    _el(k, 54, 55, 26, 22, (150, 104, 72), 0.8)
    _el(k, 50, 52, 12, 9, (196, 156, 118), 0.75)
    a = np.linspace(np.pi * 1.05, np.pi * 1.45, 12)
    O.cizgi_ince(k, np.stack([60 + 48 * np.cos(a), 60 + 48 * np.sin(a)], 1), O.BEYAZ, 4, 0.9)
    return k


def _termos():
    k = O.bos_katman((190, 80))
    O.kutu(k, 8, 12, 150, 68, 26, O.ADACAYI)
    O.kutu(k, 140, 16, 182, 64, 12, (232, 226, 214))
    O.kutu(k, 132, 12, 146, 68, 4, (70, 98, 80))
    O.kutu(k, 22, 20, 128, 30, 5, (150, 180, 158), 0.8)
    return _isiklandir(k, 8)


def _blok(renk, harf_renk):
    k = O.bos_katman((92, 96))
    O.kutu(k, 8, 16, 84, 92, 10, tuple(int(c * 0.72) for c in renk))     # yan yüz (yükseklik)
    O.kutu(k, 4, 4, 80, 80, 10, renk)
    O.kutu(k, 16, 16, 68, 68, 8, harf_renk, 0.5)
    return _isiklandir(k, 9)


def _top():
    k = O.bos_katman((112, 112))
    _el(k, 56, 56, 50, 50, (255, 250, 240))
    for a0 in (0, 120, 240):
        a = np.radians(np.linspace(a0, a0 + 60, 12))
        O.cokgen(k, np.concatenate([[[56, 56]], np.stack([56 + 50 * np.cos(a), 56 + 50 * np.sin(a)], 1)]), O.MERCAN)
    _el(k, 56, 56, 9, 9, (255, 250, 240))
    f = k.astype(np.float32)
    yy, xx = np.mgrid[0:112, 0:112].astype(np.float32)
    d = np.hypot(xx - 40, yy - 38) / 70
    a = f[..., 3:] / 255
    f[..., :3] = np.clip(f[..., :3] * (1.08 - 0.30 * d[..., None] ** 1.5), 0, 255 * a)
    k = f.astype(np.uint8)
    _el(k, 38, 34, 11, 7, O.BEYAZ, 0.7, aci=-30)
    return k


def _araba(renk, ic=(250, 243, 232)):
    """Bebek arabası, üstten: tekerlekler, gövde, tente, tutamak."""
    k = O.bos_katman((190, 280))
    for x, y in ((26, 96), (164, 96), (30, 214), (160, 214)):
        O.kutu(k, x - 13, y - 24, x + 13, y + 24, 9, KOYU)
    O.kutu(k, 34, 44, 156, 236, 44, tuple(int(c * 0.8) for c in renk))
    O.kutu(k, 42, 52, 148, 228, 38, ic)
    O.kutu(k, 52, 132, 138, 220, 28, (255, 250, 242))                      # battaniye
    for i in range(3):
        O.kutu(k, 52, 150 + i * 22, 138, 158 + i * 22, 3, renk, 0.35)
    a = np.radians(np.linspace(180, 360, 30))                              # tente (yarım kubbe)
    O.cokgen(k, np.stack([95 + 60 * np.cos(a), 122 + 84 * np.sin(a)], 1), renk)
    for dx in (-30, 0, 30):
        O.cizgi_ince(k, np.array([(95 + dx, 122), (95 + dx * 0.55, 44)]), tuple(int(c * 0.86) for c in renk), 2.5)
    O.cizgi_ince(k, np.array([(52, 248), (52, 266), (138, 266), (138, 248)]), KOYU, 7)
    return _isiklandir(k, 12)


def _kalem(renk):
    k = O.bos_katman((150, 30))
    O.kutu(k, 4, 6, 118, 24, 4, renk)
    O.kutu(k, 30, 6, 64, 24, 0, (255, 250, 240), 0.55)
    O.cokgen(k, np.array([(118, 6), (146, 15), (118, 24)]), tuple(int(c * 0.82) for c in renk))
    return _isiklandir(k, 7)


def _kagit():
    k = O.bos_katman((200, 250))
    O.kutu(k, 4, 4, 196, 246, 4, (255, 252, 246))
    _el(k, 140, 62, 26, 26, O.BAL, 0.9)
    for a in range(0, 360, 45):
        r = np.radians(a)
        O.cizgi_ince(k, np.array([(140 + 34 * np.cos(r), 62 + 34 * np.sin(r)), (140 + 46 * np.cos(r), 62 + 46 * np.sin(r))]), O.BAL, 5)
    x = np.linspace(24, 176, 40)
    O.cizgi_ince(k, np.stack([x, 190 + 14 * np.sin(x / 13)], 1), (127, 168, 142), 9)
    O.cizgi_ince(k, O.egri([(40, 150), (60, 110), (84, 150), (104, 112), (126, 150)], 10), O.LAVANTA, 7)
    return _isiklandir(k, 6)


def _firildak():
    k = O.bos_katman((170, 170))
    for i, renk in enumerate((O.SEFTALI, O.LAVANTA, O.BAL, (127, 168, 142))):
        a = np.radians(i * 90)
        c, s = np.cos(a), np.sin(a)
        P = np.array([(0, 0), (72, 0), (36, -40)], np.float32)
        Q = np.array([(0, 0), (36, -40), (0, -36)], np.float32)
        for poly, ton in ((P, renk), (Q, tuple(int(v * 0.82) for v in renk))):
            O.cokgen(k, np.stack([85 + poly[:, 0] * c - poly[:, 1] * s, 85 + poly[:, 0] * s + poly[:, 1] * c], 1), ton)
    _el(k, 85, 85, 9, 9, O.BEYAZ)
    _el(k, 85, 85, 4, 4, KOYU)
    return k


def _tabak():
    k = O.bos_katman((170, 170))
    _el(k, 85, 85, 80, 80, (255, 252, 246))
    _el(k, 85, 85, 60, 60, (243, 236, 224))
    r = _rng(31)
    for cx, cy in ((62, 70), (108, 66), (86, 106), (120, 104), (56, 108)):
        _el(k, cx, cy, 21, 21, (206, 154, 98))
        _el(k, cx - 3, cy - 4, 15, 14, (220, 172, 116), 0.8)
        for _ in range(4):
            a, d = r.uniform(0, 6.28), r.uniform(3, 13)
            _el(k, cx + d * np.cos(a), cy + d * np.sin(a), 3, 3, (104, 66, 44), n=10)
    return _isiklandir(k, 8)


def _ayakkabi():
    k = O.bos_katman((150, 120))
    for x, aci in ((42, -8), (104, 8)):
        _el(k, x, 60, 27, 48, O.SEFTALI, aci=aci)
        _el(k, x, 72, 17, 26, (214, 150, 118), aci=aci)
        _el(k, x, 28, 15, 11, (255, 246, 236), 0.8, aci=aci)
    return _isiklandir(k, 8)


def _boya():
    k = O.bos_katman((230, 190))
    for i, renk in enumerate((O.MERCAN_ACIK, O.LAVANTA, O.BAL, (127, 168, 142))):
        hr = _kalem(renk if i else (236, 150, 128))
        O.yapistir(k, hr, 112 + i * 4, 36 + i * 38, aci=-24 + i * 15)
    return k


@functools.lru_cache(maxsize=None)
def _nesne(ad):
    hr = {"bardak": _bardak, "termos": _termos, "top": _top, "kagit": _kagit, "firildak": _firildak,
          "tabak": _tabak, "ayakkabi": _ayakkabi, "boya": _boya,
          "blok1": lambda: _blok(AHSAP, O.MERCAN), "blok2": lambda: _blok(O.BAL, (255, 250, 240)),
          "blok3": lambda: _blok(O.SEFTALI, (127, 168, 142)),
          "araba": lambda: _araba((127, 168, 142)), "araba2": lambda: _araba(O.LAVANTA)}[ad]()
    yuksek = ad.startswith("araba") or ad in ("termos", "top")
    return hr, _golge(hr, 11 if yuksek else 7, 0.36 if yuksek else 0.30)


# (ad, örtü merkezine göre x, y, açı, beliriş anı, ölçek)
def _dizi():
    return [
        ("bardak", -200, -150, 0, T_KAHVE, 1.0), ("bardak", -92, -196, 0, T_KAHVE + 0.10, 1.0),
        ("termos", -210, -30, 24, T_KAHVE + 0.22, 1.0),
        ("blok1", 120, -190, -12, T_OYUN, 1.0), ("blok2", 196, -150, 10, T_OYUN + 0.07, 1.0), ("blok3", 146, -110, 24, T_OYUN + 0.14, 1.0),
        ("top", 250, -20, 0, T_OYUN + 0.20, 1.0), ("araba", 420, 70, 14, T_OYUN + 0.30, 1.0),
        ("kagit", 40, 130, -10, T_ATOLYE, 1.0), ("boya", -150, 150, 0, T_ATOLYE + 0.08, 1.0), ("firildak", 210, 170, 0, T_ATOLYE + 0.18, 1.0),
        ("tabak", 10, -40, 0, T_MAHALLE - 0.05, 1.0), ("araba2", -420, 30, -16, T_MAHALLE + 0.10, 1.0), ("ayakkabi", -250, 230, 20, T_MAHALLE + 0.24, 1.0),
    ]


# ------------------------------------------------------------------ kamera
def _kamera(t, K):
    """Hiç durmayan yavaş itme + kayma. Dönüş: arka plan (paylı) -> ekran afini."""
    p = O.ara(t, T0, T_SON)
    z = 1.0 + 0.05 * p
    dx, dy = 12 * np.sin(t * 0.55 + 1.0), -16 * p + 5 * np.cos(t * 0.4)
    cx, cy = K.cx + M, K.cy + M
    return np.array([[z, 0, K.cx - z * cx + dx], [0, z, K.cy - z * cy + dy]], np.float32)


# ------------------------------------------------------------------ süs ipi
def _ip(t, K, Y, n=64):
    A, B = np.array(Y["A"], np.float32), np.array(Y["B"], np.float32)
    u = np.linspace(0, 1, n)[:, None]
    P = A + (B - A) * u
    sal = 5 * np.sin(t * 1.7 + u[:, 0] * 5) * np.sin(np.pi * u[:, 0])
    P[:, 1] += Y["sark"] * 4 * u[:, 0] * (1 - u[:, 0]) + sal + 3 * np.sin(t * 0.9)
    return P


def _giris_yolu(t, K, Y):
    ip = _ip(t, K, Y)
    g = np.array(K.etk_giris, np.float32)
    A = ip[0]
    on = O.bezier(g, g + (-70, -70), A + (-50, 120), A, 24)
    return np.concatenate([on[:-1], ip]), len(on) - 1


def _cikis_yolu(t, K, Y):
    ip = _ip(t, K, Y)[::-1]                                   # B -> A
    A = ip[-1]
    s = np.array(K.s6_bas, np.float32)
    son = O.bezier(A, A + (-40, 130), s + (-150, -150), s, 28)
    return np.concatenate([ip[:-1], son])


def _yol_oran(P, i):
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return d[i] / d[-1]


def _cizgi_yolu(t, K, Y):
    """Sahnenin her anında çizginin yolu (alt-zaman örneklemesi için saf fonksiyon)."""
    if t < Z.ETK_GECIS:
        P, i0 = _giris_yolu(t, K, Y)
        r0 = _yol_oran(P, i0)
        bas = O.gir_cik(O.ara(t, T0, T0 + 0.80))
        kuy = r0 * O.gir_cik(O.ara(t, T0 + 0.25, T0 + 0.95))
        return O.yol_parca(P, kuy, bas)
    P = _cikis_yolu(t, K, Y)
    x = O.ara(t, Z.ETK_GECIS, BITIS - 0.02)
    ip_oran = _yol_oran(P, 63)
    bas = O.kar(ip_oran, 1.0, O.gir_cik(x))
    kuy = O.kar(0.0, 0.93, O.gir_cik(O.ara(t, Z.ETK_GECIS + 0.04, BITIS - 0.02)) ** 1.15)
    return O.yol_parca(P, kuy, bas)


def _bayraklar(tuval, t, K, Y):
    ip = _ip(t, K, Y)
    n, (bw, bh) = Y["n"], Y["by"]
    dalga_t = t - Z.ETK_DOKUNUS
    dus = O.ara(t, Z.ETK_GECIS, Z.ETK_GECIS + 0.33)
    for i in range(n):
        u = (i + 0.5) / n
        j = u * (len(ip) - 1)
        j0 = int(j)
        p = ip[j0] + (ip[min(j0 + 1, len(ip) - 1)] - ip[j0]) * (j - j0)
        teget = ip[min(j0 + 1, len(ip) - 1)] - ip[max(j0 - 1, 0)]
        egim = np.degrees(np.arctan2(teget[1], teget[0]))
        cik = O.yay(O.ara(t, T0 + 0.62 + i * 0.6 / n, T0 + 0.62 + i * 0.6 / n + 0.55), 5.5, 1.8)
        if cik <= 0.01:
            continue
        acik = 6 * np.sin(t * 2.3 + i * 0.9) + 2.5 * np.sin(t * 5.1 + i * 2.1)
        if dalga_t > 0:                                       # dokunuşta soldan sağa giden dalga
            faz = dalga_t - u * 0.45
            if faz > 0:
                acik += 26 * np.exp(-faz * 3.2) * np.sin(faz * 13)
        a = np.radians(egim + acik)
        c, s = np.cos(a), np.sin(a)
        boy = bh * min(cik, 1.15)
        yerel = np.array([(-bw / 2, 0), (bw / 2, 0), (0, boy)], np.float32)
        P = np.stack([p[0] + yerel[:, 0] * c - yerel[:, 1] * s, p[1] + yerel[:, 0] * s + yerel[:, 1] * c], 1)
        alfa = 1.0
        if dus > 0:                                           # ip bayrakları bırakır
            P = P + (0, 520 * dus ** 2 * (0.7 + 0.5 * ((i * 37) % 10) / 10))
            alfa = 1 - dus
        renk = BAYRAK[i % len(BAYRAK)]
        O.cokgen(tuval, P + (16, 30), (60, 78, 60), 0.16 * alfa, bulanik=7)          # çimdeki gölge
        O.cokgen(tuval, P, renk, alfa)
        O.cokgen(tuval, np.array([P[0] * 0.5 + P[1] * 0.5, P[1], P[2]]), tuple(int(v * 0.90) for v in renk), alfa)
        O.cizgi_ince(tuval, P[:2], tuple(int(v * 0.8) for v in renk), 5, alfa)


# ------------------------------------------------------------------ etkinlik kartı
KW, KH = 908, 484


@functools.lru_cache(maxsize=1)
def _kart_taban():
    k = O.bos_katman((KW, KH))
    O.kutu(k, 0, 0, KW, KH, 34, O.BEYAZ)
    O.kutu(k, 40, 40, 196, 196, 30, O.MERCAN_ACIK)
    O.kutu(k, 40, 40, 196, 92, 0, O.MERCAN, 0.0)
    O.yaz(k, "CMT", 118, 104, "nunito", 40, O.TUGLA, 900, hiza="orta", aralik=2)
    O.yaz(k, "10.30", 118, 164, "nunito", 44, O.METIN, 900, hiza="orta")
    bas = "Parkta kahve & oyun"
    boyut = 60
    while O.yazi_olcu(bas, "nunito", boyut, 800)[0] > KW - 228 - 36 and boyut > 56:
        boyut -= 1
    O.yaz(k, bas, 228, 112, "nunito", boyut, O.METIN, 800)
    O.yaz(k, "Beşiktaş · mahalle parkı", 230, 176, "nunito", 39, O.METIN2, 700)
    for i, renk in enumerate((O.SEFTALI, O.LAVANTA, O.BAL, (127, 168, 142), O.MERCAN_ACIK)):
        _el(k, 66 + i * 36, 268, 27, 27, O.BEYAZ)
        _el(k, 66 + i * 36, 268, 23, 23, renk)
    O.yaz(k, "anne katılıyor", 334, 283, "nunito", 42, O.METIN, 800)
    return k


def _kart(t):
    k = _kart_taban().copy()
    d = t - Z.ETK_DOKUNUS
    # sayaç 8 -> 9
    g = O.gir_cik(O.ara(d, 0.08, 0.42))
    if g < 1:
        O.yaz(k, "8", 296, 283 - 34 * g, "nunito", 42, O.METIN, 800, alfa=1 - g)
    if g > 0:
        O.yaz(k, "9", 296, 283 + 34 * (1 - g), "nunito", 42, O.TUGLA, 900, alfa=g)
        yeni = O.yay(O.ara(d, 0.10, 0.60), 5, 2)
        if yeni > 0.02:                                       # sıraya yeni bir daire eklenir
            _el(k, 66 + 5 * 36, 268, 27 * min(yeni, 1.2), 27 * min(yeni, 1.2), O.BEYAZ)
            _el(k, 66 + 5 * 36, 268, 23 * min(yeni, 1.2), 23 * min(yeni, 1.2), O.MERCAN)
    # buton
    bas = 1 - 0.04 * np.exp(-((d - 0.02) / 0.07) ** 2) if d > -0.1 else 1.0
    dol = O.cik(O.ara(d, 0.0, 0.22))
    cx, cy, bw, bh = KW / 2, 388, (KW - 80) * bas, 108 * bas
    O.kutu(k, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, bh / 2, None, cerceve=O.MERCAN, cerceve_kalinlik=4)
    if dol > 0:
        O.kutu(k, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, bh / 2, O.MERCAN, dol)
    if dol < 1:
        O.yaz(k, "Katılıyorum", cx, cy + 16, "nunito", 46, O.MERCAN, 800, alfa=1 - dol, hiza="orta")
    if dol > 0:
        w = O.yazi_olcu("Katılıyorsun", "nunito", 46, 800)[0]
        O.yaz(k, "Katılıyorsun", cx + 30, cy + 16, "nunito", 46, O.BEYAZ, 800, alfa=dol, hiza="orta")
        ok = O.ara(d, 0.12, 0.36)
        if ok > 0:
            tik = np.array([(-16, 0), (-5, 12), (17, -13)], np.float32) + (cx - w / 2 - 14, cy)
            O.cizgi_ince(k, O.yol_parca(tik, 0, ok), O.BEYAZ, 7, dol)
    return k


def _kart_ciz(tuval, t, K, Y):
    p_ham = O.ara(t, Z.ETK_KART, Z.ETK_KART + 0.55)
    if p_ham <= 0:
        return None
    p = O.geri(p_ham, 1.3)
    s = Y["ks"] * O.kar(0.80, 1.0, p)
    x0, y0 = Y["kart"]
    w, h = KW * Y["ks"], KH * Y["ks"]
    cx, cy = x0 + w / 2 + 4 * np.sin(t * 0.8), y0 + h / 2 + (1 - p) * 170 + 6 * np.sin(t * 1.25 + 0.5)
    ww, hh = KW * s, KH * s
    dar = (1 - min(p, 1)) * 0.10 * ww + 3 * np.sin(t * 0.7)        # öne kalkarken üst kenar dar (perspektif)
    egik = 4 * np.sin(t * 0.9 + 1)
    kose = np.array([(cx - ww / 2 + dar, cy - hh / 2 + egik), (cx + ww / 2 - dar, cy - hh / 2 - egik),
                     (cx + ww / 2, cy + hh / 2), (cx - ww / 2, cy + hh / 2)], np.float32)
    alfa = min(1.0, p_ham * 3)
    pp = min(max(p, 0), 1)
    O.golge(tuval, kose[3][0] + 10, kose[0][1] + 10, kose[2][0] + 10, kose[2][1], 34 * Y["ks"],
            bulanik=O.kar(10, 34, pp), alfa=0.26 * alfa, dy=O.kar(6, 30, pp), renk=(60, 70, 52))
    Mq = O.dortgen((KW, KH), kose)
    O.donustur(tuval, _kart(t), Mq, alfa)
    return Mq


# ------------------------------------------------------------------ buhar
def _buhar(tuval, x, y, t, guc, olcek):
    if guc <= 0.02:
        return
    kat = O.bos_katman((150, 210))
    for j in range(2):
        yy = np.linspace(0, 1, 16)
        xx = 75 + (j * 34 - 17) + 16 * np.sin(yy * 5 + t * 2.4 + j * 2) * yy - 26 * yy
        P = np.stack([xx, 190 - yy * 170], 1)
        for i in range(len(P) - 1):
            O.cizgi_ince(kat, P[i:i + 2], O.BEYAZ, 13 - 7 * yy[i], 0.55 * (1 - yy[i]) ** 1.2)
    kat = O.bulanik(kat, 5)
    O.yapistir(tuval, kat, x - 8 * olcek, y - 86 * olcek, alfa=guc * 0.8, olcek=olcek)


# ------------------------------------------------------------------ sahne
def ciz(t, K, tuval):
    Y = _yer(K)
    zemin = tuval.copy()
    Mk = _kamera(t, K)
    z = Mk[0, 0]
    sahne = cv2.warpAffine(_arka(K), Mk, (K.W, K.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    ox, oy = Y["ortu"][0] + M, Y["ortu"][1] + M
    s0 = Y["s"]
    for n, (ad, dx, dy, aci, tb, ol) in enumerate(_dizi()):
        x = O.ara(t, tb, tb + 0.60)
        if x <= 0:
            continue
        hr, gl = _nesne(ad)
        sc = O.yay(x, 5.2, 1.9) * ol * s0 * z
        if sc < 0.03:
            continue
        px, py = O.nokta(Mk, [(ox + dx * s0, oy + dy * s0)])[0]
        py -= (1 - min(x * 2.2, 1)) ** 2 * 46                              # hafifçe yukarıdan konar
        don = aci + (1 - O.cik(x)) * (18 if n % 2 else -18)
        if ad == "firildak":
            don = t * 70
        elif ad == "top":
            px += 5 * np.sin(t * 1.3) * z
            don = 14 * np.sin(t * 1.3)
        yuk = 1.0 + (1 - min(x * 1.6, 1)) * 1.4                             # konarken gölge uzakta
        O.yapistir(sahne, gl, px + ISIK[0] * yuk * z, py + ISIK[1] * yuk * z, alfa=min(1, x * 3), olcek=sc, aci=don)
        O.yapistir(sahne, hr, px, py, alfa=min(1, x * 4), olcek=sc, aci=don)
        if ad == "bardak":
            _buhar(sahne, px, py, t + n * 1.7, O.ara(t, tb + 0.3, tb + 0.9), s0 * z)

    # ağaç tepeleri: ön plan, biraz daha hızlı kayar (paralaks) ve hafif bulanık
    Mt = Mk.copy()
    Mt[:, :2] *= 1.035
    Mt[0, 2] = K.cx - Mt[0, 0] * (K.cx + M) + (Mk[0, 2] - (K.cx - z * (K.cx + M))) * 1.5 + 6 * np.sin(t * 0.8)
    Mt[1, 2] = K.cy - Mt[1, 1] * (K.cy + M) + (Mk[1, 2] - (K.cy - z * (K.cy + M))) * 1.5
    O.donustur(sahne, _taclar(K), Mt)

    # yaprak arası ışık benekleri + sahnenin nefes alan sıcak ışığı
    Mb = np.array([[1, 0, -M + 22 * np.sin(t * 0.5)], [0, 1, -M + 16 * np.cos(t * 0.37)]], np.float32)
    benek = cv2.warpAffine(_benek(K), Mb, (K.W, K.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    sahne = cv2.multiply(sahne, benek, scale=1 / 255)
    sahne = cv2.multiply(sahne, zemin, scale=1 / 244)

    _bayraklar(sahne, t, K, Y)
    ip = _ip(t, K, Y)
    pa = O.yay(O.ara(t, T0 + 0.30, T0 + 0.85), 5, 2)
    pb = O.yay(O.ara(t, T0 + 0.70, T0 + 1.25), 5, 2)
    if t < Z.ETK_GECIS + 0.02:
        if pa > 0.03:
            O.pin(sahne, ip[0][0], ip[0][1] + 4, 66 * min(pa, 1.2), O.MERCAN, delik=(255, 250, 244))
        if pb > 0.03:
            O.pin(sahne, ip[-1][0], ip[-1][1] + 4, 66 * min(pb, 1.2), O.ADACAYI, delik=(255, 250, 244))

    Mq = _kart_ciz(sahne, t, K, Y)
    d = t - Z.ETK_DOKUNUS
    if Mq is not None and -0.22 < d < 0.9:                                   # dokunuş: yarı saydam daire + halkalar
        bx, by = O.nokta(Mq, [(KW / 2 + 150, 392)])[0]
        ks = Y["ks"]
        gel = O.cik(O.ara(d, -0.22, -0.02))
        git = O.ara(d, 0.10, 0.34)
        if git < 1:
            r = O.kar(58, 40, O.ara(d, -0.05, 0.03)) * ks
            O.daire(sahne, bx + (1 - gel) * 60, by + (1 - gel) * 90, r, (255, 255, 255), 0.55 * gel * (1 - git))
            O.daire(sahne, bx + (1 - gel) * 60, by + (1 - gel) * 90, r, (120, 90, 80), 0.30 * gel * (1 - git), kalinlik=3)
        for i in range(3):
            h = O.ara(d, i * 0.09, i * 0.09 + 0.62)
            if 0 < h < 1:
                O.halka(sahne, bx, by, (40 + 190 * O.cik(h)) * ks, (1 - h) * 0.7, O.MERCAN, 5)

    # giriş: çıplak zeminden belirir; çıkış: geriye bulanıklaşıp zemine söner
    gir = O.yumusak(O.ara(t, T0, T0 + 0.34))
    if gir < 1:
        sahne = O.karistir(zemin, O.bulanik(sahne, 14 * (1 - gir)), gir)
    cik = O.ara(t, Z.ETK_GECIS, BITIS - 0.06)
    if cik > 0:
        sahne = zemin if cik >= 1 else O.karistir(O.bulanik(sahne, 26 * cik), zemin, O.yumusak(cik))
    tuval[:] = sahne

    # çizgi: girer, süs ipi olur; sonunda bayrakları bırakıp K.s6_bas'a akar
    hareketli = t < T0 + 1.0 or t >= Z.ETK_GECIS - 0.02
    yol = O.alt_zaman(lambda u: _cizgi_yolu(u, K, Y), t, 4) if hareketli else _cizgi_yolu(t, K, Y)
    kal = O.kar(12, 9, O.ara(t, T0 + 0.3, T0 + 0.9)) if t < Z.ETK_GECIS else O.kar(9, 12, O.ara(t, Z.ETK_GECIS, BITIS))
    O.cizgi(tuval, yol, kal, parlama=0.8)
    return tuval

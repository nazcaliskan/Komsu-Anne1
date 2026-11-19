"""Ortak çizim çekirdeği: kadraj, renkler, yazı, şekiller, katmanlar ve
filmin TEK çizgi fonksiyonu `cizgi()`.

Sözleşmeler
- Tuval: uint8 (H, W, 3) RGB. Katman / hare (sprite): uint8 (h, w, 4), ÖN ÇARPILMIŞ alfa (premultiplied RGBA).
- Koordinatlar piksel, (x, y), kayan noktalı olabilir; yavaş hareketler alt-piksel çizilir.
- Zaman saniye (float). Kare = saniye * 60. Zaman sabitleri ses/zaman.py içinde.
"""
import os, sys, functools
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{KOK}/ses")
import zaman as Z  # noqa: E402

# ------------------------------------------------------------------ renkler (RGB)
KREM = (251, 246, 241)      # #FBF6F1 sahne zemini
KREM2 = (245, 236, 228)     # #F5ECE4
# Palet (kullanıcı kararı): nil yeşili (mint) + somon / yavruağzı. Logo da bu renklere çevrilir.
SOMON = (250, 128, 114)         # #FA8072 çizgi, mercan pin, vurgu
SOMON_KOYU = (214, 84, 70)      # #D65446 küçük vurgu yazıları
YAVRUAGZI = (255, 184, 166)     # #FFB8A6
YAVRUAGZI_ACIK = (255, 232, 224)
NIL = (92, 191, 158)            # #5CBF9E logo yeşili, yeşil pin
NIL_KOYU = (47, 143, 114)       # #2F8F72 küçük yeşil yazılar
NIL_ACIK = (223, 244, 235)      # #DFF4EB
# Eski adlar yeni palete bağlanır (sahne kodları bu adları kullanıyor).
MERCAN, MERCAN_ACIK = SOMON, YAVRUAGZI_ACIK
ADACAYI, ADACAYI_ACIK = NIL, NIL_ACIK
TUGLA = SOMON_KOYU
METIN = (59, 47, 42)        # #3B2F2A
METIN2 = (122, 107, 99)     # #7A6B63
METIN3 = (165, 150, 141)
CIZGI_GRI = (239, 227, 216)  # #EFE3D8 arayüz çizgisi
BEYAZ = (255, 255, 255)
SEFTALI = YAVRUAGZI
BAL = (233, 169, 58)
LAVANTA = (143, 124, 199)
KRAFT = (214, 176, 132)

FONT = {"serif": f"{KOK}/assets/fonts/DMSerifDisplay-Regular.ttf",
        "karla": f"{KOK}/assets/fonts/Karla[wght].ttf",
        "nunito": f"{KOK}/assets/fonts/Nunito[wght].ttf"}


# ------------------------------------------------------------------ kadraj
class Kadraj:
    """Biçim ve sahneler arası devir noktaları. dikey: 1080x1920, yatay: 1920x1080."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.dikey = H > W
        self.cx, self.cy = W / 2, H / 2
        d = self.dikey
        # Çizginin sahneler arası devir noktaları (ekran pikseli). İki komşu sahne aynı noktayı kullanır.
        self.pin1 = (540, 900) if d else (640, 600)            # S1: pin ucunun indiği yer
        self.s2_giris = (540, 1180) if d else (640, 760)       # t=2.00: çizgi ucu S1'den S2'ye geçer
        self.dugme = (540, 1228, 760, 104) if d else (640, 742, 470, 66)   # t=6.60: "Gruba katıl" (cx, cy, en, boy)
        self.evA = (300, 1330) if d else (500, 760)            # t=7.20–7.60: açılan deliğin merkezi = A evi
        self.s5_giris = (540, 900) if d else (960, 520)        # t=10.00: çizgi ucu S4'ten S5'e geçer
        self.etk_giris = (200, 520) if d else (360, 330)       # t=14.20: çizgi ucu S5'ten etkinlik sahnesine geçer
        self.oneri_giris = (160, 560) if d else (300, 300)     # t=ONERI_BAS: çizgi ucu etkinlikten öneri sahnesine geçer
        self.slogan_giris = (140, 700) if d else (420, 330)    # t=SLOGAN_BAS: çizgi ucu öneriden slogan sahnesine geçer
        self.s6_bas = (150, 760) if d else (480, 400)          # söz satırlarının girişi (son sahnenin iç noktası)
        self.logo_merkez = (540, 900) if d else (960, 500)     # t=NEFES: çizgi burada toplanır; ikon merkezi

    def __repr__(self):
        return f"Kadraj({self.W}x{self.H})"


DIKEY, YATAY = Kadraj(1080, 1920), Kadraj(1920, 1080)


# ------------------------------------------------------------------ zaman eğrileri
def ara(t, t0, t1):
    """t'yi [t0, t1] aralığında 0..1'e sıkıştırır."""
    return float(np.clip((t - t0) / max(t1 - t0, 1e-9), 0, 1))


def yumusak(x):
    return x * x * (3 - 2 * x)


def cik(x, us=3):
    """Yavaşlayarak bitiş (ease-out)."""
    return 1 - (1 - x) ** us


def gir(x, us=3):
    return x ** us


def gir_cik(x):
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def geri(x, s=1.6):
    """Hedefi biraz aşıp oturan bitiş (back-out)."""
    x -= 1
    return 1 + x * x * ((s + 1) * x + s)


def yay(x, sonum=5.0, frekans=2.2):
    """Yaylı oturma: 0 -> 1, hafif salınım."""
    return 1 - np.exp(-sonum * x) * np.cos(2 * np.pi * frekans * x) if x > 0 else 0.0


def kar(a, b, x):
    return a + (b - a) * x


# ------------------------------------------------------------------ alçak düzey bileşim
def _birlestir(dst, rgb, a, x0, y0):
    """dst'nin (x0, y0) köşesine, alfa a (h, w float 0..1) ile rgb (renk ya da (h, w, 3)) bileştirir."""
    h, w = a.shape[:2]
    H, W = dst.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = min(w, W - x0), min(h, H - y0)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    a = a[sy0:sy1, sx0:sx1, None].astype(np.float32)
    roi = dst[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1]
    if isinstance(rgb, np.ndarray) and rgb.ndim == 3:
        rgb = rgb[sy0:sy1, sx0:sx1].astype(np.float32)
    else:
        rgb = np.array(rgb, np.float32)
    out = roi.astype(np.float32)
    out[..., :3] = out[..., :3] * (1 - a) + rgb * a
    if dst.shape[2] == 4:
        out[..., 3:] = out[..., 3:] * (1 - a) + 255 * a
    roi[:] = np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _maske(noktalar_listesi, pay=3, kapali=True, kalinlik=0):
    """Çokgen(ler)i kenar yumuşatmalı maskeye çizer. Dönüş: (maske float, x0, y0)."""
    P = [np.asarray(p, np.float32).reshape(-1, 2) for p in noktalar_listesi]
    tum = np.concatenate(P)
    m = pay + int(np.ceil(kalinlik))
    x0, y0 = int(np.floor(tum[:, 0].min())) - m, int(np.floor(tum[:, 1].min())) - m
    x1, y1 = int(np.ceil(tum[:, 0].max())) + m, int(np.ceil(tum[:, 1].max())) + m
    mk = np.zeros((max(1, y1 - y0), max(1, x1 - x0)), np.uint8)
    Q = [np.round((p - (x0, y0)) * 16).astype(np.int32) for p in P]
    if kalinlik:
        cv2.polylines(mk, Q, kapali, 255, max(1, int(round(kalinlik))), cv2.LINE_AA, 4)
    else:
        cv2.fillPoly(mk, Q, 255, cv2.LINE_AA, 4)
    return mk.astype(np.float32) / 255, x0, y0


def cokgen(dst, noktalar, renk, alfa=1.0, bulanik=0):
    """Dolu çokgen. noktalar: (N, 2) ya da çokgen listesi (delikli şekiller için çift-tek dolgu)."""
    L = noktalar if isinstance(noktalar, list) else [noktalar]
    mk, x0, y0 = _maske(L, pay=3 + int(bulanik * 3))
    if bulanik:
        mk = cv2.GaussianBlur(mk, (0, 0), bulanik)
    _birlestir(dst, renk, mk * alfa, x0, y0)


def cizgi_ince(dst, noktalar, renk, kalinlik=2, alfa=1.0, kapali=False):
    """Süs/arayüz için düz çoklu çizgi (filmin mercan çizgisi DEĞİL; onun için cizgi())."""
    mk, x0, y0 = _maske([noktalar], kapali=kapali, kalinlik=kalinlik)
    _birlestir(dst, renk, mk * alfa, x0, y0)


def _yd_noktalar(x0, y0, x1, y1, r, n=10):
    r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    if r < 0.5:
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    P = []
    for cx, cy, a0 in [(x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)]:
        a = np.radians(np.linspace(a0, a0 + 90, n))
        P.append(np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1))
    return np.concatenate(P).astype(np.float32)


def kutu(dst, x0, y0, x1, y1, r, renk, alfa=1.0, cerceve=None, cerceve_kalinlik=2):
    """Yuvarlak köşeli dikdörtgen; isteğe bağlı çerçeve rengi."""
    P = _yd_noktalar(x0, y0, x1, y1, r)
    if renk is not None:
        cokgen(dst, P, renk, alfa)
    if cerceve is not None:
        cizgi_ince(dst, P, cerceve, cerceve_kalinlik, alfa, kapali=True)


def daire(dst, x, y, r, renk, alfa=1.0, kalinlik=0):
    """Dolu daire; kalinlik > 0 ise halka."""
    a = np.linspace(0, 2 * np.pi, max(24, int(r * 1.2)), endpoint=False)
    P = np.stack([x + r * np.cos(a), y + r * np.sin(a)], 1)
    if kalinlik:
        cizgi_ince(dst, P, renk, kalinlik, alfa, kapali=True)
    else:
        cokgen(dst, P, renk, alfa)


def golge(dst, x0, y0, x1, y1, r, bulanik=24, alfa=0.16, dy=10, renk=(110, 70, 45)):
    """Yumuşak düşen gölge (uygulamadaki sıcak gölge tonu)."""
    cokgen(dst, _yd_noktalar(x0, y0 + dy, x1, y1 + dy, r), renk, alfa, bulanik=bulanik)


def bulanik(img, sigma):
    """Gauss bulanıklığı; büyük sigmada küçültüp büyüterek hızlanır. Katmanlarda da çalışır."""
    if sigma < 0.3:
        return img
    if sigma > 6:
        k = int(min(8, sigma // 3))
        h, w = img.shape[:2]
        kucuk = cv2.resize(img, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
        kucuk = cv2.GaussianBlur(kucuk, (0, 0), sigma / k)
        return cv2.resize(kucuk, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(img, (0, 0), sigma)


def radyal_bulanik(img, merkez, guc, adim=6):
    """Kamera itişi için yakınlaşma bulanıklığı. guc: 0..0.2 (ölçek farkı)."""
    if guc < 0.002:
        return img
    h, w = img.shape[:2]
    acc = np.zeros(img.shape, np.float32)
    for i in range(adim):
        s = 1 + guc * i / (adim - 1)
        M = np.array([[s, 0, merkez[0] * (1 - s)], [0, s, merkez[1] * (1 - s)]], np.float32)
        acc += cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return (acc / adim + 0.5).astype(np.uint8)


def karistir(a, b, x):
    """İki tuvali x (0..1) oranında karıştırır (uzun yumuşak geçişler)."""
    if x <= 0:
        return a
    if x >= 1:
        return b
    return cv2.addWeighted(a, 1 - x, b, x, 0)


# ------------------------------------------------------------------ katmanlar ve hareler
def bos_katman(K_or_wh):
    w, h = (K_or_wh.W, K_or_wh.H) if isinstance(K_or_wh, Kadraj) else K_or_wh
    return np.zeros((int(h), int(w), 4), np.uint8)


def katman_bas(dst, kat, alfa=1.0, x=0, y=0):
    """Ön çarpılmış katmanı tam piksel konumuna basar (dönüşümsüz, hızlı)."""
    h, w = kat.shape[:2]
    H, W = dst.shape[:2]
    x, y = int(round(x)), int(round(y))
    sx0, sy0, sx1, sy1 = max(0, -x), max(0, -y), min(w, W - x), min(h, H - y)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    k = kat[sy0:sy1, sx0:sx1].astype(np.float32) * alfa
    roi = dst[y + sy0:y + sy1, x + sx0:x + sx1]
    out = roi.astype(np.float32) * (1 - k[..., 3:] / 255)
    out[..., :3] += k[..., :3]
    if dst.shape[2] == 4:
        out[..., 3] += k[..., 3]
    roi[:] = np.clip(out + 0.5, 0, 255).astype(np.uint8)


def afin(olcek=1.0, aci=0.0, capa=(0, 0), hedef=(0, 0), olcek_y=None):
    """Hare içindeki `capa` noktasını ekrandaki `hedef`e taşıyan 2x3 dönüşüm (ölçek, derece cinsinden açı)."""
    sy = olcek if olcek_y is None else olcek_y
    c, s = np.cos(np.radians(aci)), np.sin(np.radians(aci))
    A = np.array([[c * olcek, -s * sy], [s * olcek, c * sy]], np.float32)
    t = np.array(hedef, np.float32) - A @ np.array(capa, np.float32)
    return np.concatenate([A, t[:, None]], 1)


def nokta(M, pts):
    """Noktaları 2x3 (afin) ya da 3x3 (perspektif) dönüşümden geçirir."""
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    M = np.asarray(M, np.float32)
    if M.shape == (2, 3):
        return p @ M[:, :2].T + M[:, 2]
    q = np.concatenate([p, np.ones((len(p), 1), np.float32)], 1) @ M.T
    return q[:, :2] / q[:, 2:]


def yapistir(dst, hare, x, y, alfa=1.0, olcek=1.0, aci=0.0, capa=(0.5, 0.5), olcek_y=None):
    """Hareyi alt-piksel doğrulukla basar. (x, y): harenin `capa` (0..1 oranlı) noktasının ekran konumu."""
    h, w = hare.shape[:2]
    if olcek < 0.55:                       # büyük küçültmelerde önce alan ortalaması
        k = max(olcek, 0.02) / 0.9
        hare = cv2.resize(hare, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
        sx, sy_ = hare.shape[1] / w, hare.shape[0] / h
        h, w = hare.shape[:2]
        olcek_y = (olcek if olcek_y is None else olcek_y) / sy_
        olcek = olcek / sx
    M = afin(olcek, aci, (capa[0] * w, capa[1] * h), (x, y), olcek_y)
    donustur(dst, hare, M, alfa)


def donustur(dst, hare, M, alfa=1.0):
    """Hareyi 2x3 afin ya da 3x3 perspektif M ile dst'ye basar (yalnız kapladığı bölgede)."""
    h, w = hare.shape[:2]
    H, W = dst.shape[:2]
    M = np.asarray(M, np.float32)
    kose = nokta(M, [[0, 0], [w, 0], [w, h], [0, h]])
    x0, y0 = max(0, int(np.floor(kose[:, 0].min())) - 1), max(0, int(np.floor(kose[:, 1].min())) - 1)
    x1, y1 = min(W, int(np.ceil(kose[:, 0].max())) + 1), min(H, int(np.ceil(kose[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    if M.shape == (2, 3):
        M2 = M.copy(); M2[:, 2] -= (x0, y0)
        k = cv2.warpAffine(hare, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float32)
        k = cv2.warpPerspective(hare, T @ M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    katman_bas(dst, k, alfa, x0, y0)


def dortgen(kaynak_wh, kose):
    """(w, h) boyutlu hareyi dört köşeye (sol-üst, sağ-üst, sağ-alt, sol-alt) oturtan 3x3 perspektif."""
    w, h = kaynak_wh
    return cv2.getPerspectiveTransform(np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32), np.asarray(kose, np.float32))


def onceden_carp(rgba):
    """Düz alfalı RGBA -> ön çarpılmış."""
    o = rgba.astype(np.float32)
    o[..., :3] *= o[..., 3:] / 255
    return (o + 0.5).astype(np.uint8)


# ------------------------------------------------------------------ yazı
@functools.lru_cache(maxsize=None)
def _font(ad, boyut, agirlik):
    f = ImageFont.truetype(FONT[ad], int(round(boyut)))
    if ad != "serif":
        f.set_variation_by_axes([agirlik])
    return f


@functools.lru_cache(maxsize=4096)
def yazi(metin, ad="karla", boyut=40, renk=METIN, agirlik=500, aralik=0):
    """Metni ön çarpılmış hareye çizer. ad: 'serif' (DM Serif Display), 'karla', 'nunito'.
    Dönen harenin sol kenarı metnin başlangıcı; satır taban çizgisi için yazi_olcu() kullan."""
    f = _font(ad, boyut, agirlik)
    asc, desc = f.getmetrics()
    w = int(np.ceil(f.getlength(metin) + aralik * max(0, len(metin) - 1))) + 8
    im = Image.new("L", (max(w, 2), asc + desc + 8), 0)
    d = ImageDraw.Draw(im)
    if aralik:
        x = 4.0
        for ch in metin:
            d.text((x, 4), ch, font=f, fill=255)
            x += f.getlength(ch) + aralik
    else:
        d.text((4, 4), metin, font=f, fill=255)
    a = np.asarray(im, np.float32) / 255
    out = np.empty(a.shape + (4,), np.float32)
    out[..., :3] = np.array(renk, np.float32) * a[..., None]
    out[..., 3] = a * 255
    return (out + 0.5).astype(np.uint8)


def yazi_olcu(metin, ad="karla", boyut=40, agirlik=500, aralik=0):
    """(genişlik, çıkış yüksekliği, iniş) — hare içinde metin x=4'te başlar, taban çizgisi y=4+çıkış."""
    f = _font(ad, boyut, agirlik)
    asc, desc = f.getmetrics()
    return f.getlength(metin) + aralik * max(0, len(metin) - 1), asc, desc


def yaz(dst, metin, x, y, ad="karla", boyut=40, renk=METIN, agirlik=500, alfa=1.0, hiza="sol", olcek=1.0, aralik=0):
    """Metni basar. (x, y): hizaya göre sol / orta / sağ ucu ve TABAN ÇİZGİSİ. Dönüş: metin genişliği."""
    hr = yazi(metin, ad, boyut, tuple(renk), agirlik, aralik)
    w, asc, _ = yazi_olcu(metin, ad, boyut, agirlik, aralik)
    kay = {"sol": 0, "orta": w / 2, "sag": w}[hiza]
    M = afin(olcek, 0, (4 + kay, 4 + asc), (x, y))
    donustur(dst, hr, M, alfa)
    return w * olcek


# ------------------------------------------------------------------ marka (resmî dosyalar)
@functools.lru_cache(maxsize=None)
def marka(ad, yukseklik):
    """Resmî logo katmanı (ön çarpılmış); biçim resmî dosyadan, renkler yeni palete eşlenir (adaçayı→NIL, mercan→SOMON). ad: 'ikon', 'ikon_mercan', 'ikon_yesil', 'yazi', 'yatay'.
    Yalnızca ölçeklenir; ikon katmanları aynı 1024x1024 tuvali paylaşır (üst üste konunca resmî ikon).
    Yatay logoda oranlar: ikon 138 birim kare; yazı ikonun solundan 162, üstünden 14 birim içeride, 449x110."""
    im = cv2.imread(f"{KOK}/assets/marka/png/{ad}.png", cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
    h, w = im.shape[:2]
    yeni = (max(1, int(round(w * yukseklik / h))), int(round(yukseklik)))
    rgb = im[..., :3].astype(np.int32)
    kirmizi = (rgb[..., 0] - rgb[..., 1]) > 40                      # mercan pikseller (R >> G); gerisi adaçayı
    im[..., :3] = np.where(kirmizi[..., None], np.array(SOMON, np.uint8), np.array(NIL, np.uint8))
    return onceden_carp(cv2.resize(im, yeni, interpolation=cv2.INTER_AREA if yukseklik < h else cv2.INTER_CUBIC))


# ------------------------------------------------------------------ zemin, ışık, gren
@functools.lru_cache(maxsize=4)
def _isik_tabani(W, H):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    return x / W, y / H


def zemin(t, K):
    """Krem sahne: yavaşça nefes alan sıcak ışık + hafif kenar kararması."""
    x, y = _isik_tabani(K.W // 8, K.H // 8)
    nefes = 0.5 + 0.5 * np.sin(2 * np.pi * t / 6.5)
    cx, cy = 0.5 + 0.16 * np.sin(2 * np.pi * t / 17), 0.36 + 0.07 * np.cos(2 * np.pi * t / 13)
    d = np.sqrt(((x - cx) * (K.W / K.H if not K.dikey else 1)) ** 2 + ((y - cy) * (K.H / K.W if K.dikey else 1)) ** 2)
    isik = np.exp(-(d / 0.55) ** 2) * (0.55 + 0.45 * nefes)
    kenar = np.clip(np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2) * 1.6 - 0.55, 0, 1) ** 2
    img = np.empty((K.H // 8, K.W // 8, 3), np.float32)
    img[:] = KREM
    img += isik[..., None] * np.array([4, 2, -3], np.float32) * 1.4
    img -= kenar[..., None] * np.array([16, 20, 24], np.float32)
    img = cv2.resize(img, (K.W, K.H), interpolation=cv2.INTER_CUBIC)
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


_GREN = None


def gren(img, kare, guc=3.2):
    """İnce film greni (kare numarasına göre değişir, tekrar üretilebilir)."""
    global _GREN
    H, W = img.shape[:2]
    if _GREN is None or _GREN.shape[1:3] != (H, W):
        r = np.random.default_rng(3)
        _GREN = np.stack([cv2.GaussianBlur(r.standard_normal((H, W)).astype(np.float32), (0, 0), 0.7) for _ in range(6)])
        _GREN /= _GREN.std()
    g = _GREN[kare % 6]
    if (kare // 6) % 2:
        g = g[::-1, ::-1]
    return np.clip(img.astype(np.float32) + g[..., None] * guc, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ eğriler
def bezier(p0, p1, p2, p3, n=48):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, np.float32) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def egri(noktalar, n=24):
    """Verilen noktalardan geçen yumuşak eğri (Catmull-Rom). n: parça başına örnek."""
    P = np.asarray(noktalar, np.float32)
    P = np.concatenate([P[:1] * 2 - P[1:2], P, P[-1:] * 2 - P[-2:-1]])
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    out = []
    for i in range(1, len(P) - 2):
        a, b, c, d = P[i - 1], P[i], P[i + 1], P[i + 2]
        out.append(0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t ** 2 + (3 * b - a - 3 * c + d) * t ** 3))
    return np.concatenate(out + [P[-2:-1]])


def yol_parca(P, s0, s1):
    """Çoklu çizginin yay uzunluğuna göre [s0, s1] (0..1) arasındaki parçası. 'Çizilen' ve 'kuyruklu' çizgi için."""
    P = np.asarray(P, np.float32)
    s0, s1 = float(np.clip(s0, 0, 1)), float(np.clip(s1, 0, 1))
    if s1 <= s0:
        s1 = min(1.0, s0 + 1e-4)
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    if d[-1] <= 0:
        return P[:1].repeat(2, 0)
    d /= d[-1]
    def at(s):
        i = int(np.clip(np.searchsorted(d, s) - 1, 0, len(P) - 2))
        u = (s - d[i]) / max(d[i + 1] - d[i], 1e-9)
        return P[i] + (P[i + 1] - P[i]) * u
    ic = P[(d > s0) & (d < s1)]
    return np.concatenate([[at(s0)], ic, [at(s1)]]).astype(np.float32)


def yol_nokta(P, s):
    """Çoklu çizgide yay uzunluğu oranı s'deki nokta."""
    return yol_parca(P, s, s)[0]


# ------------------------------------------------------------------ FİLMİN TEK ÇİZGİSİ
def cizgi(dst, yol, kalinlik=10, alfa=1.0, renk=MERCAN, parlama=1.0, ornekler=None):
    """Filmdeki kesintisiz mercan ışık çizgisi. HER sahne çizgiyi yalnızca bu fonksiyonla çizer.

    yol       : (N, 2) noktalar — ya da hareket bulanıklığı için aynı karenin alt-zaman örneklerinden
                oluşan [yol_t0, yol_t1, ...] listesi (örn. t-1/120, t-1/240, t anları). Örnekler ortalanır.
    kalinlik  : piksel; yuvarlak uçlu. Liste verilirse her alt-zaman örneği için ayrı kalınlık.
    parlama   : çevresindeki yumuşak ışık (0 kapalı). Dolgu gibi çok kalın kullanımlarda 0.3–0.5 yeter.
    Aynı görünüm: logo mercanı, yuvarlak uç, sıcak hale, içte hafif açık çekirdek.
    """
    yollar = yol if isinstance(yol, list) else [yol]
    yollar = [np.asarray(p, np.float32).reshape(-1, 2) for p in yollar]
    kal = kalinlik if isinstance(kalinlik, (list, tuple)) else [kalinlik] * len(yollar)
    kmax = float(max(kal))
    hale = (6 + kmax * 0.9) * (1 if parlama > 0 else 0)
    pay = int(kmax / 2 + hale * 2.5 + 4)
    tum = np.concatenate(yollar)
    H, W = dst.shape[:2]
    x0, y0 = max(0, int(tum[:, 0].min()) - pay), max(0, int(tum[:, 1].min()) - pay)
    x1, y1 = min(W, int(tum[:, 0].max()) + pay + 1), min(H, int(tum[:, 1].max()) + pay + 1)
    if x1 <= x0 or y1 <= y0 or alfa <= 0:
        return
    acc = np.zeros((y1 - y0, x1 - x0), np.float32)
    for P, k in zip(yollar, kal):
        mk = np.zeros(acc.shape, np.uint8)
        if len(P) == 1:
            P = P.repeat(2, 0)
        Q = np.round((P - (x0, y0)) * 16).astype(np.int32)
        cv2.polylines(mk, [Q], False, 255, max(1, int(round(k))), cv2.LINE_AA, 4)
        acc += mk
    acc /= 255 * len(yollar)
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    if parlama > 0:                                   # sıcak hale: zemini mercana doğru hafifçe aydınlatır
        h = bulanik(acc, hale) * 0.42 * parlama * alfa
        roi[..., :3] += (np.array([255, 150, 120], np.float32) - roi[..., :3]) * h[..., None]
    a = acc * alfa
    cekirdek = bulanik(acc, max(1.0, min(kmax, 120) * 0.16)) ** 3 * 0.16 * min(1.0, 60 / kmax)   # içte hafif açık çekirdek (kalın dolguda söner)
    rgb = np.array(renk, np.float32) + (255 - np.array(renk, np.float32)) * cekirdek[..., None]
    roi[..., :3] = roi[..., :3] * (1 - a[..., None]) + rgb * a[..., None]
    if dst.shape[2] == 4:
        roi[..., 3] = np.maximum(roi[..., 3], a * 255)
    dst[y0:y1, x0:x1] = np.clip(roi + 0.5, 0, 255).astype(np.uint8)


def alt_zaman(fn, t, n=4, aciklik=0.5):
    """Hareket bulanıklığı için: fn(t) -> yol; obtüratör açıklığı kadar geriye n örnek döndürür.
    Kullanım: cizgi(dst, alt_zaman(lambda u: yolum(u), t), kalinlik)"""
    return [fn(t - aciklik / Z.FPS * i / max(1, n - 1)) for i in range(n - 1, -1, -1)]


# ------------------------------------------------------------------ hazır parçalar
def pin(dst, x, y, boy, renk=MERCAN, alfa=1.0, olcek_x=1.0, olcek_y=1.0, delik=KREM):
    """Dik duran genel harita pini; ucu (x, y)'de. (Logo değil: logodaki pinler marka() ile gelir.)"""
    r = boy * 0.36
    cy = -boy + r
    a = np.linspace(np.radians(-218), np.radians(38), 40)
    bas = np.stack([r * np.cos(a), cy + r * np.sin(a)], 1)
    sag = bezier(bas[-1], (r * 0.62, cy + r * 1.25), (r * 0.10, -boy * 0.10), (0, 0), 14)
    sol = bezier((0, 0), (-r * 0.10, -boy * 0.10), (-r * 0.62, cy + r * 1.25), bas[0], 14)
    P = np.concatenate([bas, sag, sol]) * (olcek_x, olcek_y) + (x, y)
    cokgen(dst, P, renk, alfa)
    if delik is not None:
        b = np.linspace(0, 2 * np.pi, 28, endpoint=False)
        D = np.stack([r * 0.36 * np.cos(b), cy + r * 0.36 * np.sin(b)], 1) * (olcek_x, olcek_y) + (x, y)
        cokgen(dst, D, delik, alfa)


def halka(dst, x, y, r, alfa, renk=MERCAN, kalinlik=4):
    """Vuruş / dokunuş dalgası halkası."""
    if alfa > 0.01 and r > 1:
        daire(dst, x, y, r, renk, alfa, kalinlik=kalinlik)


def cam_kart(dst, x0, y0, x1, y1, r=28, alfa=1.0, ton=BEYAZ, opak=0.72):
    """Buzlu cam kart: arkayı bulandırır, beyaz perde ve ince parlak kenar ekler, yumuşak gölge düşürür."""
    if alfa <= 0.01:
        return
    H, W = dst.shape[:2]
    golge(dst, x0, y0, x1, y1, r, bulanik=26, alfa=0.14 * alfa, dy=12)
    ix0, iy0, ix1, iy1 = max(0, int(x0)), max(0, int(y0)), min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1)))
    if ix1 <= ix0 or iy1 <= iy0:
        return
    arka = bulanik(dst[iy0:iy1, ix0:ix1].copy(), 14).astype(np.float32)
    arka = arka * (1 - opak) + np.array(ton, np.float32) * opak
    mk, mx, my = _maske([_yd_noktalar(x0, y0, x1, y1, r)])
    tam = np.zeros(mk.shape + (3,), np.float32)
    tam[iy0 - my:iy1 - my, ix0 - mx:ix1 - mx] = arka
    _birlestir(dst, tam, mk * alfa, mx, my)
    cizgi_ince(dst, _yd_noktalar(x0, y0, x1, y1, r), BEYAZ, 2, 0.8 * alfa, kapali=True)


# Uygulamadaki avatar dilinde (düz, yumuşak) küçük portreler. Fotoğraf yok.
AVATAR = {
    "a": dict(zemin=(253, 227, 216), ten=(249, 220, 200), sac=(107, 62, 38), ust=MERCAN, tip="uzun"),
    "b": dict(zemin=(227, 239, 230), ten=(227, 170, 130), sac=(196, 81, 58), ust=(127, 168, 142), tip="ortu"),
    "c": dict(zemin=(238, 232, 250), ten=(242, 198, 164), sac=(58, 42, 34), ust=LAVANTA, tip="topuz"),
    "d": dict(zemin=(253, 241, 216), ten=(201, 139, 98), sac=(34, 27, 24), ust=BAL, tip="kivircik"),
    "e": dict(zemin=(251, 226, 234), ten=(249, 220, 200), sac=(214, 162, 79), ust=(127, 168, 142), tip="uzun"),
    "f": dict(zemin=(226, 238, 247), ten=(227, 170, 130), sac=(58, 42, 34), ust=MERCAN, tip="kisa"),
}


@functools.lru_cache(maxsize=64)
def avatar(boyut, tip="a"):
    """Yuvarlak avatar haresi (ön çarpılmış). tip: 'a'..'f'."""
    v = AVATAR[tip]
    S = 4
    n = int(boyut * S)
    u = n / 100
    k = np.zeros((n, n, 4), np.uint8)
    daire(k, 50 * u, 50 * u, 50 * u, v["zemin"])
    def el(cx, cy, rx, ry, renk):
        a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        cokgen(k, np.stack([(cx + rx * np.cos(a)) * u, (cy + ry * np.sin(a)) * u], 1), renk)
    if v["tip"] == "uzun":
        el(50, 52, 25, 34, v["sac"])
    elif v["tip"] == "kivircik":
        for cx, cy, r in [(32, 30, 11), (43, 22, 12), (57, 22, 12), (68, 30, 11), (72, 44, 10), (28, 44, 10)]:
            el(cx, cy, r, r, v["sac"])
    elif v["tip"] == "topuz":
        el(50, 16, 10, 10, v["sac"])
    el(50, 96, 30, 24, v["ust"])                      # omuzlar
    el(50, 70, 8, 10, v["ten"])                       # boyun
    if v["tip"] == "ortu":
        el(50, 50, 26, 33, v["sac"])
    el(50, 46, 18, 20, v["ten"])                      # yüz
    if v["tip"] != "ortu":
        a = np.linspace(np.pi, 2 * np.pi, 24)
        ust = np.stack([(50 + 19 * np.cos(a)) * u, (40 + 20 * np.sin(a)) * u], 1)
        alt = np.stack([(50 + 19 * np.cos(a[::-1])) * u, (41 + 9 * np.sin(a[::-1])) * u], 1)
        cokgen(k, np.concatenate([ust, alt]), v["sac"])
    for gx in (43, 57):                               # gülümseyen kapalı gözler
        a = np.linspace(np.pi * 1.1, np.pi * 1.9, 10)
        cizgi_ince(k, np.stack([(gx + 3 * np.cos(a)) * u, (48 + 2.4 * np.sin(a)) * u], 1), (58, 42, 34), 1.8 * u)
        el(gx + (-4 if gx < 50 else 4), 54, 3.4, 3.0, (242, 150, 140))
    a = np.linspace(np.pi * 0.15, np.pi * 0.85, 12)
    cizgi_ince(k, np.stack([(50 + 6 * np.cos(a)) * u, (53 + 5 * np.sin(a)) * u], 1), (156, 74, 58), 1.9 * u)
    y, x = np.mgrid[0:n, 0:n]
    k[np.hypot(x - n / 2 + 0.5, y - n / 2 + 0.5) > n / 2] = 0
    return cv2.resize(k, (int(boyut), int(boyut)), interpolation=cv2.INTER_AREA)


def kaydet(img, yol):
    os.makedirs(os.path.dirname(yol), exist_ok=True)
    cv2.imwrite(yol, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

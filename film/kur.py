"""Birleştirici ve render hattı: sahneleri tek kesintisiz plana bağlar.

Kullanım:
  python3 film/kur.py --bicim dikey                 # tam film (60 fps) + ses
  python3 film/kur.py --bicim yatay --adim 2 --olcek 0.5 --ad onizleme
  python3 film/kur.py --bicim dikey --kareler 10,144,362 --klasor /tmp/kareler
"""
import argparse, os, subprocess, sys, time
import multiprocessing as mp
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ortak as O
from ortak import Z

SES = f"{O.KOK}/cikti/ses/annea_ses_lilyv3_5.wav"
TOPLAM = int(round(Z.SURE * Z.FPS))          # 1272 kare

_mod = {}


def sahne(ad):
    if ad not in _mod:
        try:
            _mod[ad] = __import__(f"sahne_{ad}")
        except ModuleNotFoundError:                    # henüz kurulmamış sahne: boş geç
            import types
            _mod[ad] = types.SimpleNamespace(ARALIK=(0, 0))
    return _mod[ad]


def icinde(m, t):
    return m.ARALIK[0] <= t < m.ARALIK[1]


# ------------------------------------------------------------------ S3 -> S4: çizgi kareyi doldurur, kamera içinden geçer
G_BAS, G_DOLU, G_AC, G_ACIK, G_SON = Z.FIRLAMA, 7.05, 7.08, 7.52, 7.60


def _firlama_yolu(t, K):
    """Buton dolgusu olan çizgi kameraya doğru büyür: uçlar merkeze toplanır, kalınlık kareyi kaplar."""
    cx, cy, en, boy = K.dugme
    p = O.gir(O.ara(t, G_BAS, G_DOLU), 2.2)
    kaplama = 2.6 * max(K.W, K.H)
    kal = boy * (kaplama / boy) ** p
    yarim = (en - boy) / 2 * (1 - O.cik(O.ara(t, G_BAS, G_BAS + 0.25)))
    mx, my = O.kar(cx, K.cx, p), O.kar(cy, K.cy, p)
    return np.array([(mx - yarim, my), (mx + yarim, my)], np.float32), kal


def _delik_yolu(t, K):
    """Mercan kare, A evini merkez alan bir halka olarak açılır (içinden geçiş)."""
    x, y = K.evA
    r = 1.25 * np.hypot(K.W, K.H) * O.yumusak(O.ara(t, G_AC, G_ACIK)) ** 1.6
    w = 2.4 * np.hypot(K.W, K.H)
    a = np.linspace(0, 2 * np.pi, 96)
    return np.stack([x + (r + w / 2) * np.cos(a), y + (r + w / 2) * np.sin(a)], 1).astype(np.float32), w


def gecis_s3_s4(t, K, tuval):
    if t < G_BAS or t >= G_SON:
        return tuval
    if t < G_DOLU:
        hiz = O.ara(t, G_BAS + 0.1, G_DOLU)
        tuval = O.radyal_bulanik(tuval, (K.cx, K.cy), 0.10 * hiz)
        alt = [_firlama_yolu(t - 0.5 / Z.FPS * j / 7, K) for j in range(7, -1, -1)]
        O.cizgi(tuval, [a[0] for a in alt], [a[1] for a in alt], parlama=max(0.0, 0.5 - hiz))
    elif t < G_AC:
        tuval[:] = O.MERCAN
    else:
        if t < G_ACIK:
            alt = [_delik_yolu(t - 0.5 / Z.FPS * j / 7, K) for j in range(7, -1, -1)]
            O.cizgi(tuval, [a[0] for a in alt], [a[1] for a in alt], parlama=0.4)
        # deliğin ortasında çizginin ucu kalır; S4 7.60'ta devralır
        kal = O.kar(46, 14, O.cik(O.ara(t, G_AC, G_SON)))
        O.cizgi(tuval, np.array([K.evA, K.evA], np.float32), kal)
    return tuval


# ------------------------------------------------------------------ mercek: gerçekçilik için son işlem
_EGRI = None


def son_isleme(img):
    """Hafif ışık taşması (bloom) + yumuşak kontrast eğrisi: düz çizimi 'çekilmiş' gibi gösterir."""
    global _EGRI
    if _EGRI is None:
        x = np.arange(256, dtype=np.float32) / 255
        _EGRI = np.clip((x + 0.06 * np.sin(2 * np.pi * (x - 0.5)) * -0.5) * 255 + 0.5, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    k = cv2.resize(img, (w // 6, h // 6), interpolation=cv2.INTER_AREA).astype(np.float32)
    parlak = np.clip(k - 205, 0, None) * 2.2
    parlak = cv2.GaussianBlur(parlak, (0, 0), 7)
    tasma = cv2.resize(parlak, (w, h), interpolation=cv2.INTER_LINEAR)
    out = img.astype(np.float32)
    out += (255 - out) * np.clip(tasma / 255, 0, 1) * 0.35
    return cv2.LUT(np.clip(out + 0.5, 0, 255).astype(np.uint8), _EGRI)


# ------------------------------------------------------------------ kare
S45_BAS, S45_SON = 9.85, 10.20


def kare(i, K, gren=True):
    t = i / Z.FPS
    tuval = O.zemin(t, K)
    pin, tel, sokak, canta, etk, son = (sahne(a) for a in ("pin", "telefon", "sokak", "canta", "etkinlik", "son"))
    if icinde(pin, t):
        tuval = pin.ciz(t, K, tuval)
    if icinde(tel, t) and t < G_AC:
        tuval = tel.ciz(t, K, tuval)
    if t >= G_AC and icinde(sokak, t):
        tuval = sokak.ciz(t, K, tuval)
        if t < G_ACIK:
            tuval = O.radyal_bulanik(tuval, K.evA, 0.08 * (1 - O.ara(t, G_AC, G_ACIK)))
    if icinde(canta, t):
        if t < S45_SON:                               # S4 -> S5: uzun, yumuşak karışım
            b = canta.ciz(t, K, O.zemin(t, K))
            tuval = O.karistir(tuval, b, O.yumusak(O.ara(t, S45_BAS, S45_SON)))
        else:
            tuval = canta.ciz(t, K, O.zemin(t, K))
    tuval = gecis_s3_s4(t, K, tuval)
    if icinde(etk, t):
        tuval = etk.ciz(t, K, tuval)
    if icinde(son, t):
        tuval = son.ciz(t, K, tuval)
    tuval = son_isleme(tuval)
    return O.gren(tuval, i) if gren else tuval


# ------------------------------------------------------------------ render
_K = None


def _hazirla(bicim):
    global _K
    cv2.setNumThreads(1)
    _K = O.DIKEY if bicim == "dikey" else O.YATAY


def _is(i):
    return i, kare(i, _K).tobytes()


def render(bicim, ad, adim=1, olcek=1.0, bas=0, son=TOPLAM, crf=15, cekirdek=None):
    K = O.DIKEY if bicim == "dikey" else O.YATAY
    os.makedirs(f"{O.KOK}/cikti", exist_ok=True)
    cikti = f"{O.KOK}/cikti/{ad}.mp4"
    fps = Z.FPS // adim
    vf = ["-vf", f"scale={int(K.W * olcek) // 2 * 2}:{int(K.H * olcek) // 2 * 2}:flags=lanczos"] if olcek != 1 else []
    komut = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{K.W}x{K.H}", "-r", str(fps), "-i", "-",
             "-ss", f"{bas / Z.FPS:.4f}", "-t", f"{(son - bas) / Z.FPS:.4f}", "-i", SES, *vf,
             "-c:v", "libx264", "-preset", "slow" if olcek == 1 else "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
             "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart", "-shortest", cikti]
    ff = subprocess.Popen(komut, stdin=subprocess.PIPE)
    kareler = list(range(bas, son, adim))
    t0 = time.time()
    with mp.Pool(cekirdek or mp.cpu_count(), initializer=_hazirla, initargs=(bicim,)) as havuz:
        for n, (i, veri) in enumerate(havuz.imap(_is, kareler, chunksize=4)):
            ff.stdin.write(veri)
            if n % 120 == 0:
                print(f"  kare {i}/{son}  {time.time() - t0:.0f} sn", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"{cikti}  ({len(kareler)} kare, {time.time() - t0:.0f} sn)")
    return cikti


def kontak(bicim, kareler, yol, sutun=6, en=360):
    """Kontak föyü: seçili kareler, altında kare numarası ve saniye."""
    K = O.DIKEY if bicim == "dikey" else O.YATAY
    boy = int(en * K.H / K.W)
    satir = -(-len(kareler) // sutun)
    foy = np.full((satir * (boy + 40) + 10, sutun * (en + 10) + 10, 3), 250, np.uint8)
    for n, i in enumerate(kareler):
        k = cv2.resize(kare(i, K), (en, boy), interpolation=cv2.INTER_AREA)
        x, y = 10 + (n % sutun) * (en + 10), 10 + (n // sutun) * (boy + 40)
        foy[y:y + boy, x:x + en] = k
        O.yaz(foy, f"{i}  ·  {i / Z.FPS:.2f} sn", x + 4, y + boy + 28, "karla", 22, O.METIN, 600)
    O.kaydet(foy, yol)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bicim", default="dikey", choices=["dikey", "yatay"])
    ap.add_argument("--ad", default=None)
    ap.add_argument("--adim", type=int, default=1)
    ap.add_argument("--olcek", type=float, default=1.0)
    ap.add_argument("--bas", type=int, default=0)
    ap.add_argument("--son", type=int, default=TOPLAM)
    ap.add_argument("--kareler", default=None, help="virgülle kare numaraları: PNG olarak yazar")
    ap.add_argument("--kontak", default=None, help="kontak föyü dosya yolu (--kareler ile)")
    ap.add_argument("--klasor", default="/tmp/kareler")
    a = ap.parse_args()
    if a.kareler:
        L = [int(x) for x in a.kareler.split(",")]
        if a.kontak:
            kontak(a.bicim, L, a.kontak)
        else:
            K = O.DIKEY if a.bicim == "dikey" else O.YATAY
            for i in L:
                O.kaydet(kare(i, K), f"{a.klasor}/{a.bicim}_{i:04d}.png")
    else:
        render(a.bicim, a.ad or f"annea_{a.bicim}", a.adim, a.olcek, a.bas, a.son)

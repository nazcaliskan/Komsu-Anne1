"""S1 · Pin (0.00–2.60). Mercan çizgi üstten süzülür, krem mahalle haritasına pin olarak iner;
1.20'den sonra pinin ucundan yeniden çıkıp t=2.00'de K.s2_giris'e varır (telefon sahnesi devralır).
1.60–2.60 arasında harita ve pin geriye düşüp bulanarak zemine karışır."""
import functools
import numpy as np, cv2
from ortak import (Z, KREM2, NIL_ACIK, BEYAZ, SOMON, afin, nokta, cokgen, bos_katman, donustur, katman_bas,
                   bulanik, cizgi, alt_zaman, yol_parca, egri, bezier, pin, halka, ara, yumusak, cik, gir, gir_cik,
                   yay, kar, _yd_noktalar)

ARALIK = (0.0, 2.6)
CIZGI_SON = 2.0            # bu andan sonra çizgi telefon sahnesinin
ACI = -13.0                # harita ızgarasının eğimi (derece)


def _capa(K):
    """Pinin harita katmanı içindeki yeri (katman her yöne taşar ki kamera hareketinde kenar görünmesin)."""
    return (K.pin1[0] + 0.25 * K.W, K.pin1[1] + 0.25 * K.H)


@functools.lru_cache(maxsize=2)
def _harita(W, H, ax, ay):
    """Mahalle haritası: (keskin, bulanık) iki ön çarpılmış katman. Bir kez üretilir."""
    r = np.random.default_rng(21)
    wm, hm = int(W * 1.5), int(H * 1.5)
    k = bos_katman((wm, hm))
    k[:] = (np.array(BEYAZ + (255,), np.float32) * 0.55).astype(np.uint8)       # sokaklar: beyazımsı perde
    c, s = np.cos(np.radians(ACI)), np.sin(np.radians(ACI))
    R = np.array([[c, -s], [s, c]], np.float32)
    def d(P):
        return np.asarray(P, np.float32) @ R.T + (ax, ay)
    def serit(ilk, son, orta):
        """Pin, (0,0) çevresindeki adanın içinde kalsın diye sokaklar sabit bir çekirdekten dışa doğru dizilir."""
        xs = list(orta)
        while xs[-1] < son:
            xs.append(xs[-1] + r.uniform(270, 430))
        while xs[0] > ilk:
            xs.insert(0, xs[0] - r.uniform(270, 430))
        return xs
    us, vs = serit(-2600, 2600, [-230, 190]), serit(-3200, 3200, [-270, 170])
    tonlar = [KREM2, (241, 230, 219), (238, 226, 214), (247, 239, 231)]
    for i in range(len(us) - 1):
        for j in range(len(vs) - 1):
            g = 21 if (i + j) % 5 == 0 else 14                 # ana cadde / ara sokak yarı genişliği
            u0, u1, v0, v1 = us[i] + g, us[i + 1] - g, vs[j] + g, vs[j + 1] - g
            park = r.random() < 0.16 and not (u0 < 0 < u1 and v0 < 0 < v1)
            cokgen(k, d(_yd_noktalar(u0, v0, u1, v1, 26)), NIL_ACIK if park else tonlar[r.integers(4)], 0.94)
            if park:
                for _ in range(int((u1 - u0) * (v1 - v0) / 9000)):
                    x, y, rr = r.uniform(u0 + 30, u1 - 30), r.uniform(v0 + 30, v1 - 30), r.uniform(13, 24)
                    a = np.linspace(0, 2 * np.pi, 20, endpoint=False)
                    cokgen(k, d(np.stack([x + rr * np.cos(a), y + rr * np.sin(a)], 1)), (186, 228, 210), 0.95)
                continue
            # ada içi yapı izleri: hafif gölgeli küçük bloklar (2.5B hissi)
            nx, ny = max(1, int((u1 - u0) // 120)), max(1, int((v1 - v0) // 130))
            bw, bh = (u1 - u0 - 28) / nx, (v1 - v0 - 28) / ny
            for a_ in range(nx):
                for b_ in range(ny):
                    if r.random() < 0.18:
                        continue
                    x0, y0 = u0 + 14 + a_ * bw + 7, v0 + 14 + b_ * bh + 7
                    x1, y1 = x0 + bw - 14 - r.uniform(0, bw * 0.25), y0 + bh - 14 - r.uniform(0, bh * 0.25)
                    cokgen(k, d(_yd_noktalar(x0 + 5, y0 + 8, x1 + 5, y1 + 8, 9)), (150, 110, 80), 0.10, bulanik=4)
                    ton = [(250, 243, 236), (244, 232, 221), (252, 247, 242)][r.integers(3)]
                    cokgen(k, d(_yd_noktalar(x0, y0, x1, y1, 9)), ton, 0.96)
    keskin = cv2.GaussianBlur(k, (0, 0), 1.1)
    bulut = bulanik(k, 9)
    # sığ alan derinliği: pin çevresi net, uzaklaştıkça bulanık
    y, x = np.mgrid[0:hm, 0:wm].astype(np.float32)
    m = np.clip((np.hypot(x - ax, (y - ay) * 0.9) - 330) / 760, 0, 1)[..., None]
    return (keskin * (1 - m) + bulut * m + 0.5).astype(np.uint8)


def _kamera(t, K):
    """Pine doğru hiç durmayan yavaş itme; 1.60'tan sonra geriye düşüş. Dönüş: (M, ölçek, geri_çekilme 0..1)."""
    f = gir_cik(ara(t, Z.SAHNE["S2"], 2.6))
    olcek = (1.0 + 0.042 * t) * kar(1.0, 0.80, f)
    aci = 0.5 * np.sin(t * 0.7) - 1.2 * f
    hedef = (K.pin1[0], K.pin1[1] - (70 if K.dikey else 40) * f)
    return afin(olcek, aci, _capa(K), hedef), olcek, f


def _uc(t, K):
    """Pin ucunun ekrandaki yeri."""
    M, _, _ = _kamera(t, K)
    return nokta(M, [_capa(K)])[0]


def _inis_yolu(K):
    x, y = K.pin1
    b = 1.0 if K.dikey else 0.62
    return egri([(x + 330 * b, -260), (x + 250 * b, y - 640 * b), (x + 40 * b, y - 400 * b), (x - 26 * b, y - 170 * b), (x, y)], 20)


def _cikis_yolu(t, K):
    """Pin ucundan K.s2_giris'e uzanan kavis (ucu t=2.00'de tam devir noktasında)."""
    p0 = _uc(t, K)
    p3 = np.array(K.s2_giris, np.float32)
    b = 1.0 if K.dikey else 0.8
    return bezier(p0, p0 + (-230 * b, 60 * b), p3 + (250 * b, -150 * b), p3, 40)


def ciz(t, K, tuval):
    M, olcek, f = _kamera(t, K)
    soluk = 1 - yumusak(ara(t, 1.72, 2.55))
    uc = nokta(M, [_capa(K)])[0]
    boy = (212 if K.dikey else 168)
    if soluk > 0.004:
        kat = bos_katman(K)
        donustur(kat, _harita(K.W, K.H, *_capa(K)), M)
        # vuruş halkaları (harita düzleminde)
        for i in range(3):
            p = ara(t, Z.PIN_VURUS + i * 0.11, Z.PIN_VURUS + i * 0.11 + 40 / 60)
            if 0 < p < 1:
                halka(kat, uc[0], uc[1], (18 + cik(p, 2) * (190 if K.dikey else 150)) * olcek, (1 - p) ** 1.6 * 0.75, kalinlik=max(1, 5 * (1 - p) + 1))
        # pin: çizgi vurduğu an ölçek aşımıyla büyür
        q = ara(t, Z.PIN_VURUS, Z.PIN_VURUS + 0.62)
        if q > 0:
            sy = float(yay(q, 5.5, 2.3)) * olcek
            sx = (1 + (1 - float(yay(q, 5.5, 2.3))) * 0.35) * min(1.0, q * 9) * olcek
            a = np.linspace(0, 2 * np.pi, 28, endpoint=False)
            g = min(1.0, q * 5)
            cokgen(kat, np.stack([uc[0] + 4 + boy * 0.30 * g * olcek * np.cos(a), uc[1] + 5 + boy * 0.10 * g * olcek * np.sin(a)], 1), (120, 70, 45), 0.26, bulanik=6)
            pin(kat, uc[0], uc[1], boy, SOMON, olcek_x=sx, olcek_y=max(sy, 0.02))
        if f > 0:
            kat = bulanik(kat, 13 * f)
        katman_bas(tuval, kat, soluk)
    kal = 11 if K.dikey else 10
    # 0.00–0.17: çizgi süzülerek iner (kare 0'da yolun üçte biri aşılmış), sonra kuyruk pinin içine çekilir
    if t < Z.PIN_VURUS + 0.16:
        P = _inis_yolu(K)
        def inis(u):
            bas = 0.36 + 0.64 * gir(ara(u, -0.02, Z.PIN_VURUS), 1.25) if u < Z.PIN_VURUS else 1.0
            kuy = max(0.0, bas - 0.62) if u < Z.PIN_VURUS else kar(0.38, 1.0, cik(ara(u, Z.PIN_VURUS, Z.PIN_VURUS + 0.15), 2))
            return yol_parca(P, kuy, bas)
        cizgi(tuval, alt_zaman(inis, t, 5, 0.7), kal, alfa=1 - ara(t, Z.PIN_VURUS + 0.10, Z.PIN_VURUS + 0.16))
    # 1.20–2.00: çizgi pinin ucundan yeniden çıkar, aşağı uzar; kuyruğu ~250 px
    if 1.2 <= t < CIZGI_SON:
        def cikis(u):
            P = _cikis_yolu(u, K)
            L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
            bas = gir_cik(ara(u, 1.2, CIZGI_SON))
            return yol_parca(P, max(0.0, bas - 250 / L), bas)
        cizgi(tuval, alt_zaman(cikis, t, 4, 0.6), kal, alfa=min(1.0, (t - 1.2) * 20))
    return tuval

"""S6 · Söz, S7 · Logo, S8 · Kapanış kartı (13.30–21.20).

S6: çizgi K.s6_bas'tan girer, üç satırın altından yılan gibi akan tek bir alt çizgi olur; kelimeler
    seslendirmenin kilit dosyasındaki anlarda oturur. Sonunda çizgi toplanıp mercan pinin göz deliğine iner.
S7: vuruşta (16.80) çizgi resmî ikonun mercan yarısına dönüşür (dairesel maske), yeşil yarı soldan gelip
    yaslanır, "Annea" yazısı açılır. Logo katmanları yalnızca taşınır, ölçeklenir, soldurulur, maskelenir.
S8: kapanış kartı."""
import json, functools
import numpy as np
from scipy.interpolate import PchipInterpolator
from ortak import (Z, KOK, METIN, METIN2, MERCAN, BEYAZ, afin, nokta, cizgi, alt_zaman, yol_parca, bezier, yazi_olcu,
                   yaz, marka, yapistir, katman_bas, kutu, golge, halka, ara, yumusak, cik, gir_cik, geri, kar)

ARALIK = (Z.SAHNE["S6"], Z.SURE)

_KILIT = json.load(open(f"{KOK}/cikti/ses/kilit_lilyv3_1.json"))
KB = [k["bas"] for k in _KILIT["satirlar"][5]["kelimeler"]]          # altı kelimenin oturma anları (sn)
SATIR = [["Anneden", "anneye,"], ["elden", "ele,"], ["gönülden", "gönüle."]]
TOPLAN = Z.LOGO_VURUS - 0.10                                          # çizginin toplanmayı bitirdiği an
BIRIM = 1.5                                                            # yatay logodaki 1 birim = 1,5 px (ikon 207, yazı 165 px)
DELIK = (0.7297, 0.3759)                                               # mercan pinin göz deliği (ikon tuvaline oranla)


def _kamera(t, K):
    """Bölüm boyunca hiç durmayan çok yavaş itme (kadraj merkezine)."""
    s = 1 + 0.006 * (t - ARALIK[0])
    return afin(s, 0, (K.cx, K.cy), (K.cx, K.cy)), s


# ------------------------------------------------------------------ S6
@functools.lru_cache(maxsize=2)
def _duzen(K):
    d = K.dikey
    # punto: dikeyde en uzun satır kare genişliğinin ~%82'si; yatayda üç satırlık blok kare yüksekliğinin ~%45'i
    en104 = max(yazi_olcu(" ".join(s), "serif", 104)[0] for s in SATIR)
    punto = int(104 * 0.82 * K.W / en104) if d else 128
    aralik = round(punto * (1.5 if d else 1.42))
    orta = 1060 if d else 590                                        # ikinci satırın taban çizgisi
    taban = [orta - aralik, orta, orta + aralik]
    alt = round(punto * 0.44)                                         # alt çizginin taban çizgisinden uzaklığı
    R = aralik / 2
    rx = min(R, 44) if d else R                                       # dönüşlerin yatay yarıçapı (dikeyde kenara taşmasın)
    en = [yazi_olcu(" ".join(s), "serif", punto)[0] for s in SATIR]
    ilk = [yazi_olcu(s[0] + " ", "serif", punto)[0] for s in SATIR]
    x0 = (K.W - max(en)) / 2
    xr = x0 + max(en)
    cs, cg = (x0 - 20, xr + 8) if d else (x0 - 34, xr + 12)           # çizginin sol / sağ dönüş hizası (yazıdan pay)
    u = [y + alt for y in taban]
    p0 = np.array(K.s6_bas, np.float32)
    solo = 138 * BIRIM * (1.5 if d else 1.6)
    G = np.array(K.logo_merkez, np.float32) + ((DELIK[0] - 0.5) * solo, (DELIK[1] - 0.5) * solo)
    a = np.radians(np.linspace(-90, 90, 28))
    b = np.radians(np.linspace(-90, -270, 28))
    parcalar = [
        bezier(p0, p0 + ((-60, 60) if d else (-10, 70)), (cs - 70, u[0]), (cs, u[0]), 24),
        np.array([(cs, u[0]), (cg, u[0])], np.float32),
        np.stack([cg + rx * np.cos(a), u[0] + R + R * np.sin(a)], 1),
        np.array([(cg, u[1]), (cs, u[1])], np.float32),
        np.stack([cs + rx * np.cos(b), u[1] + R + R * np.sin(b)], 1),
        np.array([(cs, u[2]), (x0 + en[2], u[2])], np.float32),
        bezier((x0 + en[2], u[2]), (x0 + en[2] + 80, u[2]), G + (300, 150), G + (26, 0), 40),
    ]
    P = np.concatenate([np.asarray(p, np.float32) for p in parcalar])
    uz = np.concatenate([[0], np.cumsum([np.linalg.norm(np.diff(p, axis=0), axis=1).sum() + (np.linalg.norm(parcalar[i + 1][0] - p[-1]) if i + 1 < len(parcalar) else 0) for i, p in enumerate(parcalar)])])
    # çizgi ucunun zaman -> yay uzunluğu anahtarları: her kelime otururken altından geçer
    anahtar = [
        (ARALIK[0], 0), (KB[0], uz[1]), (KB[1], uz[1] + ilk[0] + (x0 - cs)), (KB[1] + 0.40, uz[2]),
        (KB[2], uz[3]), (KB[3], uz[3] + (cg - x0 - en[1])), (KB[3] + 0.35, uz[4]),
        (KB[4], uz[5]), (KB[5], uz[5] + ilk[2] + (x0 - cs)), (KB[5] + 0.20, uz[6]), (TOPLAN, uz[7]),
    ]
    T, S = zip(*anahtar)
    assert all(np.diff(T) > 0) and all(np.diff(S) >= 0), anahtar
    uc = PchipInterpolator(T, S)
    geri_cek = KB[5] - 0.05                                           # kuyruk bu andan sonra ucu yakalar
    kuyruk = PchipInterpolator([ARALIK[0], KB[0] - 0.06, KB[0] + 0.10, geri_cek, TOPLAN], [0, 0, uz[1], uz[1], uz[7] - 44])
    kelime = []                                                       # (metin, x, taban, renk, an, satır)
    for i, s in enumerate(SATIR):
        kelime.append((s[0], x0, taban[i], MERCAN if s[0] == "gönülden" else METIN, KB[2 * i], i))
        kelime.append((s[1], x0 + ilk[i], taban[i], METIN, KB[2 * i + 1], i))
    return dict(P=P, L=float(uz[7]), uc=uc, kuyruk=kuyruk, kelime=kelime, punto=punto, G=G, solo=solo, kal=punto * 0.10)


def _soz_yolu(u, K, D):
    u = float(np.clip(u, ARALIK[0], TOPLAN))
    h = float(D["uc"](u))
    k = min(float(D["kuyruk"](u)), h - 2)
    return yol_parca(D["P"], max(k, 0) / D["L"], h / D["L"])


def _soz(t, K, tuval, M, s):
    D = _duzen(K)
    kal = D["kal"] * s
    # yazı: ilk iki satır son kelimeden hemen sonra, son satır vuruştan önce solar
    son1 = 1 - yumusak(ara(t, KB[5] - 0.02, KB[5] + 0.30))
    son3 = 1 - yumusak(ara(t, KB[5] + 0.24, Z.LOGO_VURUS - 0.03))
    for metin, x, y, renk, an, satir in D["kelime"]:
        p = cik(ara(t, an, an + 10 / 60))
        a = p * (son3 if satir == 2 else son1)
        if a > 0.004:
            q = nokta(M, [(x, y + (1 - p) * 0.24 * D["punto"])])[0]
            yaz(tuval, metin, q[0], q[1], "serif", D["punto"], renk, alfa=a, olcek=s)
    if t < TOPLAN:
        cizgi(tuval, alt_zaman(lambda u: nokta(M, _soz_yolu(u, K, D)), t, 4, 0.6), kal)


# ------------------------------------------------------------------ S7 + S8
def _kilit(t, K):
    """Yatay logo kilidinin merkezi ve ölçeği (S8'de yerini değiştirir)."""
    e = gir_cik(ara(t, Z.KAPANIS_SLOGAN, Z.KAPANIS_SLOGAN + 0.50))
    c0 = np.array(K.logo_merkez, np.float32)
    if K.dikey:
        return kar(c0, np.array((540, 640), np.float32), e), 1.0
    return kar(c0, np.array((500, 540), np.float32), e), kar(1.0, 0.9, e)


def _maskeli(hare, mk):
    return (hare.astype(np.float32) * mk[..., None] + 0.5).astype(np.uint8)


def _logo(t, K, tuval, M, s):
    D = _duzen(K)
    b = BIRIM
    Lc, g = _kilit(t, K)
    ik = int(round(138 * b))                                           # ikon tuvali (px)
    solo_k = D["solo"] / ik                                            # tek başınayken ikon büyütmesi
    e = gir_cik(ara(t, Z.YAZI_ACILIS - 0.22, Z.YAZI_ACILIS + 0.20))    # ikon merkezden kilit yerine kayar
    L0 = Lc - g * np.array((305.5 * b, 69 * b), np.float32)
    C = kar(Lc, L0 + g * np.array((69 * b, 69 * b), np.float32), e)
    rel = kar(solo_k, 1.0, e) * g
    x = t - Z.YESIL_YASLAN - 0.13                                      # kalp tamamlanınca hafif ortak esneme
    if x > 0:
        rel *= 1 + 0.045 * np.exp(-7 * x) * np.sin(2 * np.pi * 2.4 * x)
    Cq = nokta(M, [C])[0]
    # yeşil yarı: soldan kayarak + solarak gelir, mercanın ARKASINA oturur
    py = ara(t, Z.YESIL_YASLAN - 0.20, Z.YESIL_YASLAN + 0.13)
    if py > 0:
        kay = (1 - geri(py, 1.3)) * 0.42 * ik * rel
        yapistir(tuval, marka("ikon_yesil", ik), Cq[0] - kay * s, Cq[1], alfa=cik(ara(py, 0, 0.55)), olcek=rel * s)
    # mercan yarı: çizginin durduğu göz deliğinden büyüyen dairesel maskeyle açılır
    pm = ara(t, Z.LOGO_VURUS, Z.LOGO_VURUS + 0.24)
    if pm > 0:
        hm = marka("ikon_mercan", ik)
        if pm < 1:
            yy, xx = np.mgrid[0:ik, 0:ik].astype(np.float32)
            d = np.hypot(xx - DELIK[0] * ik, yy - DELIK[1] * ik)
            hm = _maskeli(hm, np.clip((cik(pm, 2) * ik * 0.86 - d) / (ik * 0.05) + 0.5, 0, 1))
        pop = rel * (0.86 + 0.14 * geri(ara(t, Z.LOGO_VURUS, Z.LOGO_VURUS + 0.30), 2.2))
        Gq = Cq + (np.array(DELIK) - 0.5) * ik * rel * s               # delik sabit kalsın diye ondan ölçekle
        yapistir(tuval, hm, Gq[0], Gq[1], olcek=pop * s, capa=DELIK)
        for i in range(2):
            ph = ara(t, Z.LOGO_VURUS + i * 0.10, Z.LOGO_VURUS + i * 0.10 + 0.6)
            if 0 < ph < 1:
                halka(tuval, Gq[0], Gq[1], (30 + cik(ph, 2) * ik * 1.1) * s, (1 - ph) ** 1.8 * 0.4, kalinlik=max(1, 4 * (1 - ph)))
    # "Annea" yazısı: soldan sağa maskeyle açılır; ikon üzerinden çekildikçe ortaya çıkar
    pw = ara(t, Z.YAZI_ACILIS, Z.YAZI_ACILIS + 0.55)
    if pw > 0:
        hy = marka("yazi", int(round(110 * b)))
        wy = hy.shape[1]
        if pw < 1 or e < 1:
            xs = np.arange(wy, dtype=np.float32)
            sag = np.clip((cik(pw, 2) * (wy + 40) - xs) / 40, 0, 1)
            ikon_sag = (C[0] + 69 * b * rel + 24 * g - (L0[0] + 162 * b * g)) / g      # ikonun sağ kenarı (yazı pikseli)
            sol = np.clip((xs - ikon_sag) / 18, 0, 1)
            hy = _maskeli(hy, np.broadcast_to(sag * sol, hy.shape[:2]))
        # resmî yatay logoda yazı tam 449 birim eninde ve 110 birimlik kutuda dikey ortalı (0,04 birim pay)
        q = nokta(M, [L0 + g * np.array((162 * b, 14.04 * b), np.float32)])[0]
        yapistir(tuval, hy, q[0], q[1], olcek=g * s * 449 * b / wy, capa=(0, 0))
    # nefes: çizgi göz deliğinde asılı kısa parlak bir çizgi; vuruşta şişip pine karışır
    if TOPLAN <= t < Z.LOGO_VURUS + 0.14:
        Gq = nokta(M, [D["G"]])[0]
        n = ara(t, TOPLAN, Z.LOGO_VURUS)
        yarim = kar(22, 3, yumusak(n)) * s
        sis = cik(ara(t, Z.LOGO_VURUS - 0.03, Z.LOGO_VURUS + 0.06))
        kal = kar(D["kal"], 2 * 0.083 * D["solo"] * 1.25, sis) * s
        dy = 2.5 * np.sin((t - TOPLAN) * 9) * (1 - sis)
        cizgi(tuval, np.array([(Gq[0] - yarim + 13 * s, Gq[1] + dy), (Gq[0] + yarim + 13 * s * (1 - yumusak(n)), Gq[1] + dy)]), kal,
              alfa=1 - ara(t, Z.LOGO_VURUS + 0.05, Z.LOGO_VURUS + 0.14), parlama=1 + 0.8 * n * (1 - sis))


def _kapanis(t, K, tuval, M, s):
    d = K.dikey
    cx = 540 if d else 1385
    punto = 84 if d else 104
    y1, y2 = (900, 1000) if d else (388, 510)
    yb, yu = (1160, 1300) if d else (655, 786)
    bw, bh = (480, 112) if d else (520, 126)
    for i, (metin, y) in enumerate([("Annelik, birlikte", y1), ("daha kolay.", y2)]):
        p = cik(ara(t, Z.KAPANIS_SLOGAN + 0.27 + i * 0.10, Z.KAPANIS_SLOGAN + 0.57 + i * 0.10))
        if p > 0:
            q = nokta(M, [(cx, y + (1 - p) * 26)])[0]
            yaz(tuval, metin, q[0], q[1], "serif", punto, METIN, alfa=p, hiza="orta", olcek=s)
    p = ara(t, Z.KAPANIS_BUTON, Z.KAPANIS_BUTON + 0.34)
    if p > 0:
        k = (0.80 + 0.20 * geri(p, 1.8)) * s
        q = nokta(M, [(cx, yb)])[0]
        a = cik(ara(p, 0, 0.5))
        golge(tuval, q[0] - bw / 2 * k, q[1] - bh / 2 * k, q[0] + bw / 2 * k, q[1] + bh / 2 * k, bh / 2 * k, bulanik=18, alfa=0.20 * a, dy=10, renk=(170, 70, 50))
        kutu(tuval, q[0] - bw / 2 * k, q[1] - bh / 2 * k, q[0] + bw / 2 * k, q[1] + bh / 2 * k, bh / 2 * k, MERCAN, a)
        yaz(tuval, "Hemen katıl", q[0], q[1] + (16 if d else 20) * k, "karla", 48 if d else 58, BEYAZ, 700, alfa=a, hiza="orta", olcek=k)
    p = cik(ara(t, Z.KAPANIS_BUTON + 0.12, Z.KAPANIS_BUTON + 0.40))
    if p > 0:
        q = nokta(M, [(cx, yu + (1 - p) * 16)])[0]
        yaz(tuval, "annea.app", q[0], q[1], "karla", 46 if d else 60, METIN2, 500, alfa=p, hiza="orta", olcek=s, aralik=1)


def ciz(t, K, tuval):
    M, s = _kamera(t, K)
    if t < Z.LOGO_VURUS:
        _soz(t, K, tuval, M, s)
    if t >= TOPLAN:
        _logo(t, K, tuval, M, s)
    if t >= Z.KAPANIS_SLOGAN:
        _kapanis(t, K, tuval, M, s)
    return tuval

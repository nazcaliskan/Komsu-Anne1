"""Slogan, logo ve kapanış kartı (Z.SLOGAN_BAS – Z.SURE). Tüm anlar ses/zaman.py sabitlerinden ve
seslendirme kilidinden gelir; modülde mutlak saniye yoktur.

Slogan: çizgi K.slogan_giris'ten girer; "Bebeğin sosyal büyüsün."ün altını çizer. Sonra sokak zemini olarak yatar,
        üstüne beş küçük ev konar ("Bizim mahallemiz vardı;"). "onun Annea'sı var." gelirken evler çekilir,
        yerlerinden iki pin yükselir; çizgi "Annea'sı"nın altına sıcak bir yay olarak oturur.
        Ekranda aynı anda en çok iki satır yazı bulunur. Sonra hepsi solar, çizgi Z.NEFES'te logo noktasında toplanır.
Logo  : Z.LOGO_VURUS'ta çizgi resmî ikonun pembe yarısına dönüşür (dairesel maske), yeşil yarı soldan gelip
        yaslanır, "Annea" yazısı açılır. Logo katmanları yalnızca taşınır, ölçeklenir, soldurulur, maskelenir.
Kart  : slogan, uygulamanın dört artısı (hap etiketler), "Hemen katıl" ve adres."""
import json, functools
import numpy as np
from scipy.interpolate import PchipInterpolator
from ortak import (Z, KOK, METIN, METIN2, SOMON, NIL, NIL_KOYU, NIL_ACIK, YAVRUAGZI, YAVRUAGZI_KOYU, YAVRUAGZI_ACIK, BEYAZ,
                   cizgi_ince, egri, afin, nokta, cizgi, alt_zaman, yol_parca, bezier, yazi_olcu, yaz, marka, yapistir,
                   katman_bas, kutu, golge, halka, cokgen, pin, ara, yumusak, cik, gir, gir_cik, geri, yay, kar)

ARALIK = (Z.SLOGAN_BAS, Z.SURE)

_KILIT = json.load(open(f"{KOK}/cikti/ses/kilit_lilyv3_1.json"))["satirlar"]


def _satir(bas):
    return next(x for x in _KILIT if x["metin"].startswith(bas))


_SL1, _SL2 = _satir("Bebeğin"), _satir("Bizim")
A = [k["bas"] for k in _SL1["kelimeler"]]                              # Bebeğin, sosyal, büyüsün
B = [k["bas"] for k in _SL2["kelimeler"]]                              # Bizim, mahallemiz, vardı, onun, Annea'sı, var
A_SON = _SL1["son"]
SON = Z.SOZ_BAS                                                        # slogan bu anda söner
TOPLAN = Z.NEFES                                                       # çizginin logo noktasında toplandığı an
BIRIM = 1.5                                                            # yatay logodaki 1 birim = 1,5 px (ikon 207, yazı 165 px)
DELIK = (0.7297, 0.3759)                                               # pembe pinin göz deliği (ikon tuvaline oranla)
EV_BAS = B[1]                                                          # evler "mahallemiz" ile konmaya başlar
EV_ARA = (B[2] - B[1]) / 5                                             # son ev "vardı"dan hemen önce oturur
PIN_BAS = B[3] + 0.16                                                  # pinler "onun"dan sonra yükselir


def _kamera(t, K):
    """Bölüm boyunca hiç durmayan çok yavaş itme (kadraj merkezine)."""
    s = 1 + 0.004 * (t - ARALIK[0])
    return afin(s, 0, (K.cx, K.cy), (K.cx, K.cy)), s


# ------------------------------------------------------------------ slogan
# (genişlik, gövde yüksekliği, gövde, çatı, kapı)
EVLER = [
    (104, 78, YAVRUAGZI_ACIK, YAVRUAGZI, YAVRUAGZI_KOYU),
    (122, 106, (255, 252, 248), NIL, NIL_KOYU),
    (96, 68, NIL_ACIK, YAVRUAGZI_KOYU, YAVRUAGZI_KOYU),
    (126, 96, (255, 240, 243), NIL_KOYU, NIL_KOYU),
    (106, 82, (255, 252, 248), YAVRUAGZI, YAVRUAGZI_KOYU),
]


@functools.lru_cache(maxsize=2)
def _duzen(K):
    d = K.dikey
    def sigdir(metin, punto, oran):
        return int(min(punto, punto * oran * K.W / yazi_olcu(metin, "serif", punto)[0]))
    if d:
        pA = sigdir("Bebeğin sosyal", 128, 0.82)
        satA = [(["Bebeğin", "sosyal"], 900), (["büyüsün."], 900 + round(pA * 1.22))]
        pB, pC, yB, yC = sigdir("Bizim mahallemiz vardı;", 96, 0.86), sigdir("onun Annea'sı var.", 132, 0.86), 1030, 1200
        eo = 1.12
    else:
        pA = 136
        satA = [(["Bebeğin", "sosyal", "büyüsün."], 580)]
        pB, pC, yB, yC = 108, 172, 540, 770
        eo = 1.5
    kelime = []                                                       # (metin, x, taban, renk, an, grup, punto)
    ai = 0
    for kel, y in satA:
        x = (K.W - yazi_olcu(" ".join(kel), "serif", pA)[0]) / 2
        ax0 = x
        for w in kel:
            kelime.append((w, x, y, SOMON if w == "sosyal" else METIN, A[ai], "A", pA)); ai += 1
            x += yazi_olcu(w + " ", "serif", pA)[0]
        ax1, ya = ax0 + yazi_olcu(" ".join(kel), "serif", pA)[0], y
    ua = ya + round(pA * 0.42)
    konum = {}
    for metinler, y, pp, anlar, renkler, grup in ((["Bizim", "mahallemiz", "vardı;"], yB, pB, B[:3], [METIN2] * 3, "B"),
                                                 (["onun", "Annea'sı", "var."], yC, pC, B[3:], [METIN, NIL_KOYU, METIN], "B")):
        x = (K.W - yazi_olcu(" ".join(metinler), "serif", pp)[0]) / 2
        for w, an, renk in zip(metinler, anlar, renkler):
            kelime.append((w, x, y, renk, an, grup, pp))
            konum[w] = (x, x + yazi_olcu(w, "serif", pp)[0])
            x += yazi_olcu(w + " ", "serif", pp)[0]
    # sokak: "Bizim mahallemiz vardı;" satırının üstünde yatay zemin; üstüne beş ev
    gy = yB - pB * 1.30
    ev_en = sum(e[0] for e in EVLER) * eo + 4 * 34 * eo
    sx0, sx1 = K.cx - ev_en / 2 - 46, K.cx + ev_en / 2 + 46
    evler, x = [], K.cx - ev_en / 2
    for w, h, govde, cati, kapi in EVLER:
        evler.append((x + w * eo / 2, w * eo, h * eo, govde, cati, kapi)); x += (w + 34) * eo
    c0, c1 = konum["Annea'sı"]
    uc_ = yC + pC * 0.30
    tt_ = np.linspace(0, 1, 30)[:, None]
    yay_ = np.concatenate([c1 + 6 - (c1 - c0 + 12) * tt_, uc_ + pC * 0.16 * 4 * tt_ * (1 - tt_)], 1).astype(np.float32)   # sağdan sola, ortası sarkık
    g0 = np.array(K.slogan_giris, np.float32)
    sag = min(K.W - 60, max(konum["vardı;"][1], konum["var."][1], sx1) + 64)      # yazının sağından dolanma hizası
    solo = 138 * BIRIM * (1.5 if d else 1.6)
    G = np.array(K.logo_merkez, np.float32) + ((DELIK[0] - 0.5) * solo, (DELIK[1] - 0.5) * solo)
    parcalar = [
        (egri([g0, (52, g0[1] + 130), (58, ya - pA * 0.5), (ax0 - 120, ua - 14), (ax0, ua)], 14) if d      # dikeyde yazının solundan dolanır
         else bezier(g0, g0 + (0, 140), (ax0 - 90, ua), (ax0, ua), 24)),
        np.array([(ax0, ua), (ax1, ua)], np.float32),
        egri([(ax1, ua), ((ax0 + ax1) / 2, ua + pA * 0.55), (max(sx0 - 70, 64), ua - 20), (max(sx0 - 70, 60), gy + 90), (sx0, gy)], 14),
        np.array([(sx0, gy), (sx1, gy)], np.float32),
        egri([(sx1, gy), (sag - 14, gy + 16), (sag, (gy + uc_) / 2), (min(c1 + 100, sag), uc_ + 30), yay_[0]], 14),
        yay_,
        bezier(yay_[-1], yay_[-1] + (-150, 70), G + (300, 220), G + (26, 0), 40),
    ]
    P = np.concatenate([np.asarray(p, np.float32) for p in parcalar])
    U = np.concatenate([[0], np.cumsum([np.linalg.norm(np.diff(p, axis=0), axis=1).sum() + (np.linalg.norm(parcalar[i + 1][0] - p[-1]) if i + 1 < len(parcalar) else 0) for i, p in enumerate(parcalar)])])
    tA = A[-1] - 0.08 if d else A[0]                                  # dikeyde alt çizgi ikinci satırla (büyüsün.) birlikte başlar
    anahtar = [(ARALIK[0], 0), (tA, U[1]), ((tA + A_SON) / 2, U[1] + 0.6 * (U[2] - U[1])), (A_SON - 0.06, U[2]),
               (B[0] + 0.10, U[3]), (EV_BAS + 0.02, U[4]), (B[3] + 0.10, U[4]), (B[4], U[5]), (B[4] + 0.32, U[6]),
               (B[5] + 0.03, U[6]), (TOPLAN, U[7])]
    kuy = [(ARALIK[0], 0), (tA - 0.05, 0), (tA + 0.12, U[1]), (A_SON - 0.04, U[1]), (B[0] + 0.14, U[3]), (B[3] + 0.06, U[3]),
           (B[4] + 0.06, U[5]), (B[5] + 0.01, U[5]), (TOPLAN, U[7] - 44)]
    for liste in (anahtar, kuy):
        T, S = zip(*liste)
        assert all(np.diff(T) > 0) and all(np.diff(S) >= 0), liste
    return dict(P=P, L=float(U[7]), uc=PchipInterpolator(*zip(*anahtar)), kuyruk=PchipInterpolator(*zip(*kuy)), kelime=kelime,
                G=G, solo=solo, kal=12.0, gy=gy, evler=evler, pin_boy=(150 if d else 170))


def _yol(u, K, D):
    u = float(np.clip(u, ARALIK[0], TOPLAN))
    h = float(D["uc"](u))
    k = min(float(D["kuyruk"](u)), h - 2)
    return yol_parca(D["P"], max(k, 0) / D["L"], h / D["L"])


def _ev(tuval, M, s, cx, gy, w, h, govde, cati, kapi, k, alfa):
    """Düz, dolu küçük ev: gövde + üçgen çatı + kapı + pencere; zemine (cx, gy) oturur, k ile zeminden büyür."""
    if k <= 0.01 or alfa <= 0.01:
        return
    def p(noktalar):
        q = np.asarray(noktalar, np.float32)
        return nokta(M, np.stack([cx + q[:, 0] * w * k, gy - q[:, 1] * k], 1))
    a = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    cokgen(tuval, p(np.stack([0.56 * np.cos(a) + 0.04, 7 * np.sin(a) - 4], 1)), (140, 90, 80), 0.20 * alfa, bulanik=5 * s)   # temas gölgesi
    cokgen(tuval, p([(-0.5, 0), (0.5, 0), (0.5, h), (-0.5, h)]), govde, alfa)
    cokgen(tuval, p([(0.18, 0), (0.5, 0), (0.5, h), (0.18, h)]), (150, 100, 90), 0.07 * alfa)                               # yan yüzde hafif gölge
    cokgen(tuval, p([(-0.62, h - 2), (0.62, h - 2), (0.0, h + w * 0.46)]), cati, alfa)
    cokgen(tuval, p([(-0.11, 0), (0.11, 0), (0.11, h * 0.46), (-0.11, h * 0.46)]), kapi, alfa)
    for wx in (-0.32, 0.30):
        if h > 74 or wx > 0:
            cokgen(tuval, p([(wx - 0.09, h * 0.52), (wx + 0.09, h * 0.52), (wx + 0.09, h * 0.52 + w * 0.18), (wx - 0.09, h * 0.52 + w * 0.18)]), (255, 246, 214), alfa)


def _slogan(t, K, tuval, M, s):
    D = _duzen(K)
    sol = {"A": 1 - yumusak(ara(t, A_SON, B[0] + 0.04)), "B": 1 - yumusak(ara(t, SON - 0.08, SON + 0.16))}
    for metin, x, y, renk, an, grup, punto in D["kelime"]:
        p = cik(ara(t, an, an + 12 / 60))
        a = p * sol[grup]
        if a > 0.004:
            kay = (1 - p) * 0.22 * punto - ((1 - sol[grup]) * 0.35 * punto if grup == "A" else 0)     # ilk slogan yukarı çekilerek yer açar
            q = nokta(M, [(x, y + kay)])[0]
            yaz(tuval, metin, q[0] + 2 * s, q[1] + (3 + 5 * (1 - p)) * s, "serif", punto, (120, 80, 70), alfa=0.13 * a, olcek=s)   # otururken hafif gölge
            yaz(tuval, metin, q[0], q[1], "serif", punto, renk, alfa=a, olcek=s)
    if t < TOPLAN:
        cizgi(tuval, alt_zaman(lambda u: nokta(M, _yol(u, K, D)), t, 4, 0.6), D["kal"] * s)
    # mahalle: evler sokağa sırayla konar; "onun"da uçlardan ortaya doğru çekilir
    gy = D["gy"] - D["kal"] / 2 + 1
    for i, (cx, w, h, govde, cati, kapi) in enumerate(D["evler"]):
        gel = ara(t, EV_BAS + i * EV_ARA, EV_BAS + i * EV_ARA + 0.42)
        git = gir(ara(t, B[3] - 0.02 + (2 - abs(i - 2)) * 0.07, B[3] + 0.24 + (2 - abs(i - 2)) * 0.07), 2)
        if gel > 0 and git < 1:
            _ev(tuval, M, s, cx, gy, w, h, govde, cati, kapi, float(yay(gel, 6.0, 1.7)) * (1 - git) * s / s, 1 - git ** 2)
    # iki pin: mahalleden Annea'ya
    for i, (dx, renk) in enumerate(((-0.30, NIL), (0.30, SOMON))):
        p = ara(t, PIN_BAS + i * 0.10, PIN_BAS + i * 0.10 + 0.5)
        if p > 0 and sol["B"] > 0.01:
            boy = D["pin_boy"]
            q = nokta(M, [(K.cx + dx * boy, gy)])[0]
            k = float(yay(p, 5.5, 2.0)) * s
            a = np.linspace(0, 2 * np.pi, 20, endpoint=False)
            cokgen(tuval, np.stack([q[0] + boy * 0.24 * k * np.cos(a), q[1] + 3 + boy * 0.06 * k * np.sin(a)], 1), (140, 90, 80), 0.20 * sol["B"], bulanik=5)
            pin(tuval, q[0], q[1], boy, renk, alfa=sol["B"], olcek_x=k, olcek_y=max(k, 0.02))


# ------------------------------------------------------------------ logo + kapanış
def _kilit(t, K):
    """Yatay logo kilidinin merkezi ve ölçeği (S8'de yerini değiştirir)."""
    e = gir_cik(ara(t, Z.KAPANIS_SLOGAN, Z.KAPANIS_SLOGAN + 0.32))
    c0 = np.array(K.logo_merkez, np.float32)
    if K.dikey:
        return kar(c0, np.array((540, 600), np.float32), e), 1.0
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
        yarim = kar(22, 5, yumusak(ara(t, Z.LOGO_VURUS - 0.12, Z.LOGO_VURUS))) * s
        sis = cik(ara(t, Z.LOGO_VURUS - 0.03, Z.LOGO_VURUS + 0.06))
        kal = kar(D["kal"], 2 * 0.083 * D["solo"] * 1.25, sis) * s
        dy = 2.5 * np.sin((t - TOPLAN) * 9) * (1 - sis)
        cizgi(tuval, np.array([(Gq[0] - yarim + 13 * s, Gq[1] + dy), (Gq[0] + yarim + 13 * s, Gq[1] + dy)]), kal,
              alfa=1 - ara(t, Z.LOGO_VURUS + 0.05, Z.LOGO_VURUS + 0.14), parlama=1 + 0.8 * n * (1 - sis))


ETIKET = [("Sor", NIL), ("Paylaş", YAVRUAGZI), ("Takas et", NIL), ("Buluş", YAVRUAGZI)]


def _hap(tuval, cx, cy, w, h, renk, alfa, golge_renk=(120, 80, 75), golge_alfa=0.16):
    """Hap biçimli yüzey: yumuşak temas gölgesi + üst kenarda ince ışık (hafif fiziksel ipucu)."""
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    golge(tuval, x0 + 3, y0, x1 - 3, y1, h / 2, bulanik=max(3, h * 0.10), alfa=golge_alfa * alfa, dy=h * 0.07, renk=golge_renk)
    kutu(tuval, x0, y0, x1, y1, h / 2, renk, alfa)
    if w > h * 1.2:
        cizgi_ince(tuval, np.array([(x0 + h * 0.42, y0 + 2.2), (x1 - h * 0.42, y0 + 2.2)], np.float32), BEYAZ, 2, 0.5 * alfa)


def _kapanis(t, K, tuval, M, s):
    d = K.dikey
    cx = 540 if d else 1385
    punto = 84 if d else 104
    y1, y2 = (850, 950) if d else (330, 452)
    ye, eh, ep = (1064, 96, 47) if d else (548, 84, 42)                 # etiket sırası: merkez y, yükseklik, punto
    yb, yu = (1230, 1366) if d else (688, 814)
    bw, bh = (480, 112) if d else (520, 126)
    for i, (metin, y) in enumerate([("Annelik, birlikte", y1), ("daha kolay.", y2)]):
        p = cik(ara(t, Z.KAPANIS_SLOGAN + 0.13 + i * 0.06, Z.KAPANIS_SLOGAN + 0.40 + i * 0.06))
        if p > 0:
            q = nokta(M, [(cx, y + (1 - p) * 26)])[0]
            yaz(tuval, metin, q[0], q[1], "serif", punto, METIN, alfa=p, hiza="orta", olcek=s)
    # uygulamanın dört artısı: slogan ile buton arasında sırayla, yaylı girer
    enler = [yazi_olcu(m, "karla", ep, 800)[0] + eh * 0.72 for m, _ in ETIKET]
    bosluk = 16 if d else 14
    x = cx - (sum(enler) + bosluk * 3) / 2
    for i, ((metin, renk), w) in enumerate(zip(ETIKET, enler)):
        p = ara(t, Z.KAPANIS_SLOGAN + 0.15 + i * 0.07, Z.KAPANIS_SLOGAN + 0.45 + i * 0.07)
        if p > 0:
            k = (0.72 + 0.28 * geri(p, 2.0)) * s
            a = cik(ara(p, 0, 0.45))
            q = nokta(M, [(x + w / 2, ye + (1 - cik(p)) * 18)])[0]
            _hap(tuval, q[0], q[1], w * k, eh * k, renk, a, golge_alfa=0.13)
            yaz(tuval, metin, q[0], q[1] + ep * 0.33 * k, "karla", ep, METIN, 800, alfa=a, hiza="orta", olcek=k)
        x += w + bosluk
    p = ara(t, Z.KAPANIS_BUTON, Z.KAPANIS_BUTON + 0.34)
    if p > 0:
        k = (0.80 + 0.20 * geri(p, 1.8)) * s
        q = nokta(M, [(cx, yb)])[0]
        a = cik(ara(p, 0, 0.5))
        golge(tuval, q[0] - bw / 2 * k, q[1] - bh / 2 * k, q[0] + bw / 2 * k, q[1] + bh / 2 * k, bh / 2 * k, bulanik=20, alfa=0.16 * a, dy=14, renk=(170, 60, 90))
        _hap(tuval, q[0], q[1], bw * k, bh * k, YAVRUAGZI_KOYU, a, golge_renk=(150, 45, 75), golge_alfa=0.30)
        yaz(tuval, "Hemen katıl", q[0], q[1] + (16 if d else 20) * k, "karla", 48 if d else 58, BEYAZ, 700, alfa=a, hiza="orta", olcek=k)
    p = cik(ara(t, Z.KAPANIS_BUTON + 0.12, Z.KAPANIS_BUTON + 0.40))
    if p > 0:
        q = nokta(M, [(cx, yu + (1 - p) * 16)])[0]
        yaz(tuval, "annea.app", q[0], q[1], "karla", 46 if d else 60, METIN2, 500, alfa=p, hiza="orta", olcek=s, aralik=1)


def ciz(t, K, tuval):
    M, s = _kamera(t, K)
    if t < Z.LOGO_VURUS:
        _slogan(t, K, tuval, M, s)
    if t >= TOPLAN:
        _logo(t, K, tuval, M, s)
    if t >= Z.KAPANIS_SLOGAN:
        _kapanis(t, K, tuval, M, s)
    return tuval

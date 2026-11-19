"""Filmin tek zaman çizelgesi. Ses ve görüntü aynı sabitleri kullanır.
Tüm değerler saniye; kare = saniye * FPS."""
FPS = 60
SURE = 21.2
BPM = 100
VURUS = 60 / BPM          # 0.6 sn = 36 kare
OLCU = 4 * VURUS          # 2.4 sn = 144 kare

SAHNE = {"S1": 0.0, "S2": 1.6, "S3": 4.8, "S4": 7.2, "S5": 10.0, "S6": 13.3, "S7": 16.5, "S8": 19.2}

PIN_VURUS = 0.17
TELEFON_OTUR = 2.40
ODAK_CIZGI = 2.40
YAZI_BAS, YAZI_SON = 2.50, 3.30
YAZI = "Bebeğim gece uyumuyor…"
KARTLAR = [3.40, 3.72, 4.03]      # kalkış; her biri 0.33 sn sürer, sonuncusu 4.37'de oturur
GRUP_GECIS = 4.50
DOLGU_BAS, DOLGU_SON = 5.50, 5.97
DOKUNUS = 6.03
FIRLAMA = 6.60
KAMERA_GECIS = 6.80
EVLER = [7.30, 7.45, 7.60, 7.75, 7.90, 8.05]
ROTA_BAS, ROTA_SON = 8.00, 9.60
YESIL_PIN = 9.60
CAM_KART = 9.70
CANTA = 10.00
KURDELE_BAS, KURDELE_SON = 10.40, 10.83
EL_DEGISTIRME = 12.00
BILDIRIM = 12.40
S6_GECIS = 12.90
NEFES = 16.50
LOGO_VURUS = 16.80
YESIL_YASLAN = 17.20
YAZI_ACILIS = 17.50
KAPANIS_SLOGAN = 19.20
KAPANIS_BUTON = 19.60

# Seslendirme: satırın ilk sesi bu ana oturur.
VO_SATIR = [
    ("Aklına takılan her şey…", 0.50),
    ("bir sokak ötede cevabını bulur.", 2.67),
    ("Tek dokunuş…", 5.40),
    ("ve mahallen yanında.", 6.60),
    ("Küçülen tulumu atma; bir sokak ötede Fatma.", 10.40),
    ("Anneden anneye, elden ele, gönülden gönüle.", 13.50),
    ("Annea.", 17.50),
]

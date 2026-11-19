"""Filmin tek zaman çizelgesi. Ses ve görüntü aynı sabitleri kullanır.
Tüm değerler saniye; kare = saniye * FPS."""
FPS = 60
SURE = 25.9
BPM = 100
VURUS = 60 / BPM          # 0.6 sn = 36 kare
OLCU = 4 * VURUS          # 2.4 sn = 144 kare

SAHNE = {"S1": 0.0, "S2": 1.6, "S3": 4.8, "S4": 7.2, "S5": 10.0, "ETK": 14.4, "SOZ": 19.15, "LOGO": 22.4, "KART": 23.95}

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
CAM_KART = 9.20
CANTA = 10.00
KATLAMA = [9.95, 10.12, 10.28]      # tulum katlanır, 10.36'da çantaya düşer
ADIMLAR = [11.10, 11.40, 11.70]      # çanta bir sokak öteye taşınır
KURDELE_BAS, KURDELE_SON = 10.40, 10.83
EL_DEGISTIRME = 12.00
BILDIRIM = 12.60
S5_GECIS = 14.00              # takas sahnesi söner (14.00–14.40)
S6_GECIS = S5_GECIS
ETK_BAS = 14.40               # etkinlik sahnesi (ölçü başı); çizgi K.etk_giris'ten girer
ETK_KART = 14.70              # etkinlik kartı öne kalkar
ETK_DOKUNUS = 16.80           # "Katılıyorum" dokunuşu (ölçü başı)
ETK_GECIS = 18.75             # etkinlik sahnesi söner (18.75–19.15)
SOZ_BAS = 19.15               # söz sahnesi; çizgi K.s6_bas'tan girer
NEFES = 22.40
LOGO_VURUS = 22.65
YESIL_YASLAN = 23.00
YAZI_ACILIS = 23.25
KAPANIS_SLOGAN = 23.95
KAPANIS_BUTON = 24.35

# Seslendirme: satırın ilk sesi bu ana oturur.
VO_SATIR = [
    ("Aklına takılan her şey…", 0.50),
    ("bir sokak ötede cevabını bulur.", 2.67),
    ("Tek dokunuş…", 5.40),
    ("ve mahallen yanında.", 6.60),
    ("Küçülen tulum dolapta durmasın;", 9.55),
    ("bir sokak ötede bir bebeği sarsın.", 11.75),
    ("Parkta kahve, oyun, atölye…", 14.50),
    ("Etkinliğini kur, mahalle gelsin.", 16.95),
    ("Anneden anneye, elden ele, gönülden gönüle.", 19.25),
    ("Annea.", 23.25),
]

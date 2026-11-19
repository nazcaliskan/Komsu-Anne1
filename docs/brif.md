# Annea reklam filmi — zamanlama brifi (Aşama 2)

"Bir sokak ötede." · 21,2 sn · 60 fps · 1272 kare · tek kesintisiz plan
Ana format 1080×1920 (9:16); ikinci kurulum 1920×1080 (16:9, yeniden yerleşim).

Kare = saniye × 60. Müzik 100 BPM: bir vuruş 36 kare, bir ölçü 144 kare.
Ölçü başları: 0 · 144 · 288 · 432 · 576 · 720 · 864 · 1008 · 1152.
Brifteki ana anların dördü tam ölçü başına düşüyor (7.20, 9.60, 12.00, 16.80).

## Sabitler

| Öğe | Değer | Kaynak |
|---|---|---|
| Sahne zemini | `#FBF6F1` | brif = uygulama `--zemin` |
| Mercan (çizgi, pin, "gönülden") | `#E8705A` | logo dosyası |
| Adaçayı (yeşil pin, "Annea" yazısı) | `#5F7F6A` | logo dosyası |
| Tuğla kırmızısı (küçük vurgu yazıları, etiketler) | `#AB3A2A` | brif |
| Metin | `#3B2F2A`, ikincil `#7A6B63` | uygulama |
| Ekran içi arayüz | Nunito; kart yarıçapı 22, alan yarıçapı 14, çizgi `#EFE3D8` | uygulama CSS'i |
| Başlıklar | DM Serif Display | brif |
| Küçük yazılar, kapanış | Karla | brif |
| Logo | `assets/marka/*.svg`, yalnızca taşı / ölçekle / soldur / maskele | resmî dosya |

Çizgi: tek ortak fonksiyon `cizgi(t)`; kalınlık, uç yuvarlaklığı, parıltı ve
hareket bulanıklığı her sahnede aynı koddan gelir. Film boyunca hiç kopmaz;
sahneden sahneye yalnızca yolu ve kalınlığı değişir.

Kamera: hiç durmaz (en yavaş anında bile ~%0,3/sn itme + hafif kayma).
Geçişlerin tamamı biçim dönüşümü ya da kamera hareketidir; sert kesme yok.

## Sahne sahne

### S1 · Pin — kare 0–96 (0.00–1.60)
| Kare | Sn | Görüntü | Ses |
|---|---|---|---|
| 0 | 0.00 | Hareket başlamış: mercan çizgi üstten süzülüyor, altta krem mahalle haritası (sokaklar, ada blokları, bulanık) | Müzik: tuşlar + pad; hafif hışırtı |
| 10 | 0.17 | Çizgi pine dönüşüp iner; yumuşak esneme, halka dalgası (3 halka, 40 kare) | Pin vuruşu |
| 30 | 0.50 | Kamera pine doğru yavaş iter | VO: "Aklına takılan her şey…" |
| 72–96 | 1.20–1.60 | Pinin ucundan çizgi yeniden çıkar, aşağı doğru uzar; harita geriye düşüp bulanıklaşır | Geçiş hışırtısı |

### S2 · Sor — kare 96–288 (1.60–4.80)
| Kare | Sn | Görüntü | Ses |
|---|---|---|---|
| 96–144 | 1.60–2.40 | Telefon alttan yükselir, hafif eğik (3B), 144'te yerine oturur | Yükselme sesi; 144 = ölçü başı |
| 138–150 | 2.30–2.50 | Çizgi ekrana girer, soru alanının altına yatıp odak çizgisi olur | Küçük tık |
| 150–198 | 2.50–3.30 | "Bebeğim gece uyumuyor…" harf harf (22 karakter, ~2,2 kare/harf), imleç yanıp söner | Harf başına tık |
| 160 | 2.67 | — | VO: "…bir sokak ötede cevabını bulur." |
| 204 / 223 / 242 | 3.40 / 3.72 / 4.03 | Üç cevap kartı sırayla ekrandan öne kalkar (her biri 20 kare), derinlikte ayrışır; her kartta avatar + kısa cevap + "2 sokak ötede" etiketi | Kart kaydırma ×3 |
| 262 | 4.37 | Üçüncü kart oturur | — |
| 262–288 | 4.37–4.80 | Kartlar yana açılır, ekran grup sayfasına kayar (aynı telefon, içerik dönüşür) | Yumuşak geçiş |

Kart metinleri (kurgusal): "Bizde beyaz gürültü işe yaradı." · "Akşam banyosu
mucize oldu." · "Geçecek, yalnız değilsin."

### S3 · Tek dokunuş — kare 288–432 (4.80–7.20)
| Kare | Sn | Görüntü | Ses |
|---|---|---|---|
| 288 | 4.80 | Grup sayfası: "Beşiktaş · 0–6 ay anneleri", "Akran grubu · 128 anne", üye avatarları, boş çerçeveli "Gruba katıl" butonu | Ölçü başı |
| 324 | 5.40 | Kamera butona yaklaşır | VO: "Tek dokunuş…" |
| 330–358 | 5.50–5.97 | Çizgi butonun içine akar, soldan sağa dolgusu olur; yazı beyaza döner | Dolma sesi (yükselen) |
| 362 | 6.03 | DOKUNUŞ: dalga halkası, buton hafifçe basılır; "Hoş geldin!" belirir | Dokunuş + küçük başarı sesi |
| 396 | 6.60 | Çizgi butondan kopmadan ekranın dışına, kameraya doğru fırlar | VO: "ve mahallen yanında."; hışırtı |
| 408–432 | 6.80–7.20 | Kamera çizginin içinden geçer (mercan kareyi doldurur, sonra açılır) | Geçiş hışırtısı |

### S4 · Sokak — kare 432–600 (7.20–10.00)
| Kare | Sn | Görüntü | Ses |
|---|---|---|---|
| 432 | 7.20 | Mercan açılır: yukarıdan görülen mahalle (telefon ekranı haritaya dönüşmüş) | Ritim girer (ölçü başı) |
| 432–492 | 7.20–8.20 | Evler sırayla yükselir (2.5B, gölgeli) | Küçük yükselme sesleri |
| 480–576 | 8.00–9.60 | Çizgi sokak boyunca bir evden diğerine rotayı çizer, köşeyi döner | Çizim sesi |
| 576 | 9.60 | Varış: yeşil pin bekliyor, çizgi yanına ulaşır; halka dalgası | Pin sesi (ölçü başı) |
| 582–600 | 9.70–10.00 | Cam kart: "Fatma · bir sokak ötede"; kamera kapıya doğru alçalır | Kart sesi |

### S5 · Elden ele — kare 600–798 (10.00–13.30)
| Kare | Sn | Görüntü | Ses |
|---|---|---|---|
| 600 | 10.00 | Kapı önü; kâğıt çanta, içinde katlanmış bebek tulumu. İki düz, yumuşak illüstrasyon figür | Kâğıt hışırtısı |
| 624 | 10.40 | Çizgi çantaya yaklaşır | VO: "Küçülen tulumu atma; bir sokak ötede Fatma." |
| 624–650 | 10.40–10.83 | Çizgi çantayı bir kez sarar, 650'de fiyonklu kurdele olur | Kurdele sesi |
| 720 | 12.00 | Çanta el değiştirir | Müzik yükselir (ölçü başı); çanta sesi |
| 744 | 12.40 | Bildirim yukarıdan iner: "Aslı: Çok yakıştı, teşekkürler!" | Bildirim sesi |
| 774–798 | 12.90–13.30 | Sahne geriye bulanıklaşır, kurdele çözülüp yeniden çizgi olur | Yumuşak geçiş |

### S6 · Söz — kare 798–990 (13.30–16.50)
| Kare | Sn | Görüntü | Ses |
|---|---|---|---|
| 810 | 13.50 | Krem zemin; kelimeler VO ile aynı anda oturur | VO: "Anneden anneye, elden ele, gönülden gönüle." |
| ~810 / ~870 / ~930 | — | Üç satır: "Anneden anneye," / "elden ele," / "gönülden gönüle." — "gönülden" mercan; çizgi her satırın altından geçer | Kelime başına yumuşak tık |
| 960–990 | 16.00–16.50 | Yazı solar, çizgi ortada toplanır | — |

Kelime kareleri seslendirme kaydından ölçülüp Aşama 3'te kesinleşir.

### S7 · Logo — kare 990–1152 (16.50–19.20)
| Kare | Sn | Görüntü | Ses |
|---|---|---|---|
| 990–1008 | 16.50–16.80 | Nefes: çizgi tek noktada asılı, kamera çok yavaş | Müzik susar |
| 1008 | 16.80 | VURUŞ: çizgi mercan pine dönüşür (resmî ikonun mercan yarısı maskeyle açılır) | Dolu sıcak akor (ölçü başı) |
| 1014–1040 | 16.90–17.33 | Yeşil pin soldan gelip yaslanır; kalp tamamlanır, hafif esneme | Yumuşak dokunuş |
| 1050 | 17.50 | "Annea" yazısı soldan sağa maskeyle açılır | VO: "Annea." |
| 1090–1152 | 18.17–19.20 | Logo yerine oturur, kamera hâlâ hafifçe iter | Akor çınlar |

Logo yalnızca resmî SVG'den gelir: iki yarı ayrı maskelenip taşınır, biçimi ve
rengi değişmez. Çizginin mercan tonu logo mercanıyla aynıdır, dönüşüm bu yüzden
dikişsiz görünür.

### S8 · Kapanış kartı — kare 1152–1272 (19.20–21.20)
| Kare | Sn | Görüntü | Ses |
|---|---|---|---|
| 1152–1176 | 19.20–19.60 | Logo yukarı kayar; "Annelik, birlikte daha kolay." (DM Serif) | Akorun kuyruğu |
| 1176–1200 | 19.60–20.00 | "annea.app" (Karla) + mercan "Hemen katıl" butonu | Küçük tık |
| 1200–1272 | 20.00–21.20 | Sabit kart, nefes alan ışık; son 12 karede ses kısılır | — |

## Seslendirme

Tek metin, tek seferde, 2 kayıt; iyi olan satır satır kesilip kendi karesine oturur.

| Satır | Sn | Kare |
|---|---|---|
| "Aklına takılan her şey…" | 0.50 | 30 |
| "…bir sokak ötede cevabını bulur." | 2.67 | 160 |
| "Tek dokunuş…" | 5.40 | 324 |
| "ve mahallen yanında." | 6.60 | 396 |
| "Küçülen tulumu atma; bir sokak ötede Fatma." | 10.40 | 624 |
| "Anneden anneye, elden ele, gönülden gönüle." | 13.50 | 810 |
| "Annea." | 17.50 | 1050 |

## Müzik ve efekt

Fa majör, 100 BPM, kodla sentezlenir. Ölçü 0–3: yumuşak tuşlar + pad.
Ölçü 3'ten (7.20) itibaren ritim. Ölçü 5'te (12.00) yükseliş. 16.50'de nefes
(sessizlik, yalnız kuyruk). 16.80'de dolu Fa majör akoru, kapanışın altında çınlar.

Efektler de kodla üretilir; yukarıdaki tablolardaki her satırın bir sesi var.
Miks: ses −16 LUFS, müzik her kelimenin en az 12 dB altında (yan zincir),
master −14 LUFS / −1,25 dBTP.

## 16:9 yeniden yerleşim

Aynı zaman çizelgesi, ayrı kompozisyon:
- S1, S4: harita yatayda genişler, pin/rota sol üçte birden sağ üçte bire akar.
- S2, S3: telefon solda (kare yüksekliğinin ~%86'sı), cevap kartları ve
  "Hoş geldin!" sağdaki boşluğa açılır.
- S5: figürler kare genişliğine yayılır, bildirim sağ üstte.
- S6: söz tek blok, ortada; satır aralığı daralır.
- S7, S8: yatay logo ortada; kapanışta logo solda, yazı ve buton sağda.

## Onay bekleyen noktalar

1. **"ele" anı.** Çanta 12.00'de el değiştiriyor, ama "elden ele" satırı 13.50'de
   başlıyor; 12.00'de seslendirme "…Fatma." satırının sonunda. Öneri: el
   değiştirme 12.00'de müzik yükselişine otursun, "elden ele" S6'da yazıyla
   yankılansın.
2. **10.40 ve 13.50 satırları sıkışık.** "Küçülen tulumu atma…" doğal hızda
   ~3 sn sürer ve 13.50'ye dayanır. Kayıttan sonra gerekirse S6 satırı en çok
   0,2 sn kayar; karar Aşama 3'te.
3. **"2 sokak ötede" etiketi.** Uygulama grupları ilçe ve yaş bandına göre
   kuruyor; sokak mesafesi gösteren bir özellik pakette yok. Etiket reklamda
   bir vaat gibi okunabilir.
4. **Uygulama içi ekranlar.** Ekran görüntüsü ulaşmadı; soru alanı ve grup
   sayfası paketteki bileşen, renk ve metinlerle kurulacak.
5. **Ses.** Hesapta yalnızca İngilizce hazır sesler var. Türkçe için ses
   kütüphanesinden bir ses hesaba eklenmeli (aday: "Gözde Arıkan — Soft, Warm
   and Clear", "Duygu Eriş — Warm, Encouraging and Soft", "Ahu — Cheerful,
   Encouraging and Inviting").

"""ElevenLabs ile seslendirme: tüm metin tek seferde, karakter zaman damgalarıyla.
Anahtar ortamın ağ vekili tarafından eklenir; burada tutulmaz."""
import sys, json, base64, urllib.request, os
METIN = ("Aklına takılan her şey… bir sokak ötede cevabını bulur. "
         "Tek dokunuş… ve mahallen yanında. "
         "Bir puset, dört bebek. Küçülen tulum dolapta durmasın; bir sokak ötede bir bebeği sarsın. "
         "Parkta kahve, oyun, atölye… Etkinliğini kur; çocuklar arkadaş bulsun, sen de. "
         "Kreş, doktor, pişik kremi… Doğrulanmış anneler önersin, güvenle. "
         "Bebeğin sosyal büyüsün. Bizim mahallemiz vardı; onun Annea'sı var. "
         "Anneden anneye, elden ele, gönülden gönüle. "
         "Annea.")
# v3 modeli için duygu yönergeli metin: köşeli etiketler okunmaz, tonu yönlendirir.
METIN_V3 = ("[warmly, gently] Aklına takılan her şey… bir sokak ötede cevabını bulur. "
            "[smiling] Tek dokunuş… ve mahallen yanında. "
            "[playful] Bir puset, dört bebek. [tenderly] Küçülen tulum dolapta durmasın; bir sokak ötede bir bebeği sarsın. "
            "[cheerful, inviting] Parkta kahve, oyun, atölye… Etkinliğini kur; çocuklar arkadaş bulsun, sen de. "
            "[friendly] Kreş, doktor, pişik kremi… [reassuring] Doğrulanmış anneler önersin, güvenle. "
            "[warm, sincere] Bebeğin sosyal büyüsün. Bizim mahallemiz vardı; onun Annea'sı var. "
            "[heartfelt] Anneden anneye, elden ele, gönülden gönüle. "
            "[softly] Annea.")
def uret(voice_id, ad, seed, model="eleven_multilingual_v2"):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps?output_format=mp3_44100_128"
    body = {"text": METIN, "model_id": model, "language_code": "tr", "seed": seed,
            "voice_settings": {"stability": 0.45, "similarity_boost": 0.8, "style": 0.25, "use_speaker_boost": True, "speed": 1.0}}
    if model == "eleven_v3":
        body.update(text=METIN_V3, voice_settings={"stability": 0.5, "similarity_boost": 0.75})
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", "xi-api-key": "proxy-injected"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=120))
    except urllib.error.HTTPError as e:
        print(ad, seed, "HATA", e.code, e.read()[:300].decode("utf8", "replace")); return False
    open(f"vo3/{ad}_{seed}.mp3", "wb").write(base64.b64decode(d["audio_base64"]))
    json.dump(d["alignment"], open(f"vo3/{ad}_{seed}.json", "w"), ensure_ascii=False)
    print(ad, seed, "tamam", round(d["alignment"]["character_end_times_seconds"][-1], 2), "sn"); return True
if __name__ == "__main__":
    uret(sys.argv[1], sys.argv[2], int(sys.argv[3]), *(sys.argv[4:5]))

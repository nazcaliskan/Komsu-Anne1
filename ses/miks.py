"""Miks: seslendirme satırları kendi karesine oturur, müzik her kelimenin
en az 12 dB altına kısılır, master -14 LUFS / -1,25 dBTP.
Kullanım: python3 miks.py <kayıt adı, ör. sarah_1> [...]"""
import sys, json, subprocess, os
import numpy as np, soundfile as sf, pyloudnorm as pyln
from scipy import signal
import zaman as Z, sentez as S
from vo_olc import satirlar

SR = S.SR
CIKTI = "../cikti/ses"
olcer = pyln.Meter(SR)


def db(x): return 20 * np.log10(max(x, 1e-9))
def rms(x): return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def vo_yukle(ad):
    wav = f"/tmp/vo_{ad}.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"vo/{ad}.mp3", "-ar", str(SR), "-ac", "1", wav], check=True)
    x, _ = sf.read(wav); os.remove(wav)
    x = signal.sosfilt(signal.butter(2, 85, "high", fs=SR, output="sos"), x)
    return x, json.load(open(f"vo/{ad}.json"))


def kelimeler(a, i0, i1):
    """[i0,i1] karakter aralığındaki kelimelerin (metin, baş, son) listesi."""
    ch, s, e = a["characters"], a["character_start_times_seconds"], a["character_end_times_seconds"]
    out, bas = [], None
    for i in range(i0, i1 + 2):
        harf = i <= i1 and (ch[i].isalpha() or ch[i] in "'’")
        if harf and bas is None: bas = i
        if not harf and bas is not None:
            out.append(("".join(ch[bas:i]), s[bas], e[i - 1])); bas = None
    return out


def kur(ad, mz, fx):
    x, a = vo_yukle(ad)
    S_ = satirlar(a)
    txt = "".join(a["characters"])
    # satırları kes: 40 ms ön pay, kuyruk bir sonraki satıra kadar (en çok 0,35 sn)
    parcalar, pos = [], 0
    for k, ((bs, sn), (metin, hedef)) in enumerate(zip(S_, Z.VO_SATIR)):
        son = min(sn + 0.35, S_[k + 1][0] - 0.03) if k + 1 < len(S_) else sn + 0.5
        p = x[max(0, int((bs - 0.04) * SR)): int(son * SR)].copy()
        f = int(0.03 * SR); p[:f] *= np.linspace(0, 1, f); g = int(0.08 * SR); p[-g:] *= np.linspace(1, 0, g)
        i0 = txt.index(metin, pos); pos = i0 + len(metin)
        kel = [(w, hedef + (b - bs), hedef + (e - bs)) for w, b, e in kelimeler(a, i0, pos - 1)]
        parcalar.append((p, hedef - 0.04, kel, hedef + (sn - bs)))
    # seslendirme -16 LUFS (yalnız konuşma üzerinden ölçülür)
    kazanc = 10 ** ((-16 - olcer.integrated_loudness(np.concatenate([p for p, *_ in parcalar]))) / 20)
    vo = np.zeros(S.N)
    for p, t0, _, _ in parcalar:
        i = int(t0 * SR); vo[i:i + len(p)] += p * kazanc
    # satır çakışması denetimi
    for k in range(len(parcalar) - 1):
        bosluk = Z.VO_SATIR[k + 1][1] - parcalar[k][3]
        assert bosluk > 0.05, f"satır {k+1} bir sonrakine taşıyor ({bosluk:.2f} sn)"
    # müzik kısma: her kelimede müzik RMS <= kelime RMS - 13 dB (1 dB pay)
    kis = np.ones(S.N); mono = mz.mean(1)
    for p, t0, kel, son in parcalar:
        g = 1.0
        for w, b, e in kel:
            i, j = int(b * SR), int(e * SR)
            g = min(g, 10 ** ((db(rms(vo[i:j])) - 13 - db(rms(mono[i:j]))) / 20))
        i, j = int((t0 - 0.10) * SR), int((son + 0.15) * SR)
        kis[max(i, 0):j] = np.minimum(kis[max(i, 0):j], g)
    n = int(0.25 * SR); kis = np.minimum(kis, signal.fftconvolve(kis, np.hanning(n) / np.hanning(n).sum(), "same"))
    kis = np.minimum.reduce([np.roll(kis, s) for s in (0, n // 4, -n // 4)])
    fxk = 0.6 + 0.4 * kis ** 0.5                     # efektler seslendirme altında hafifçe geri çekilir
    # S6 kelime tıkları
    fx = fx.copy(); S.soz_tiklari(fx, [b for w, b, e in parcalar[5][2]])
    miks = np.stack([vo, vo], 1) + mz * kis[:, None] + fx * fxk[:, None] * 0.65
    son_n = int(0.2 * SR); miks[-son_n:] *= np.linspace(1, 0, son_n)[:, None] ** 1.5
    # master: -14 LUFS, tavan -1,25 dBTP (4x örneklemeyle gerçek tepe)
    tavan = 10 ** (-1.25 / 20)
    for _ in range(4):
        miks *= 10 ** ((-14 - olcer.integrated_loudness(miks)) / 20)
        ust = signal.resample_poly(miks, 4, 1, axis=0)
        tepe = np.abs(ust).max(1).reshape(-1, 4).max(1)
        if tepe.max() <= tavan * 0.995: break
        gr = np.minimum(1, tavan * 0.97 / np.maximum(tepe, 1e-9))
        w = int(0.004 * SR); gr = -signal.fftconvolve(np.pad(-(np.array([gr[max(0, i - w):i + w].min() for i in range(len(gr))])), w, mode="edge"), np.hanning(2 * w) / np.hanning(2 * w).sum(), "same")[w:-w]
        miks *= gr[:, None]
    # ölçüm raporu
    fark = []
    for p, t0, kel, son in parcalar:
        for w, b, e in kel:
            i, j = int(b * SR), int(e * SR)
            fark.append((db(rms(vo[i:j])) - db(rms((mono * kis)[i:j])), w, b))
    en_az = min(fark)
    os.makedirs(CIKTI, exist_ok=True)
    sf.write(f"{CIKTI}/annea_ses_{ad}.wav", miks, SR, subtype="PCM_24")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{CIKTI}/annea_ses_{ad}.wav", "-b:a", "256k", f"{CIKTI}/annea_ses_{ad}.mp3"], check=True)
    kilit = {"kayit": ad, "satirlar": [{"metin": m, "bas": round(h, 3), "son": round(pp[3], 3),
             "kelimeler": [{"k": w, "bas": round(b, 3), "kare": round(b * Z.FPS)} for w, b, e in pp[2]]}
             for (m, h), pp in zip(Z.VO_SATIR, parcalar)]}
    json.dump(kilit, open(f"{CIKTI}/kilit_{ad}.json", "w"), ensure_ascii=False, indent=1)
    print(f"{ad}: master {olcer.integrated_loudness(miks):.2f} LUFS | müzik en yakın kelimede {en_az[0]:.1f} dB altta ('{en_az[1]}' {en_az[2]:.2f} sn)")


if __name__ == "__main__":
    mz, fx = S.muzik(), S.efektler()
    mz *= 10 ** ((-21 - olcer.integrated_loudness(mz)) / 20)      # müzik yatağı: -21 LUFS
    print("müzik tepe", round(db(np.abs(mz).max()), 1), "dB | efekt tepe", round(db(np.abs(fx).max()), 1), "dB")
    os.makedirs(CIKTI, exist_ok=True)
    sf.write(f"{CIKTI}/muzik.wav", mz / max(1, np.abs(mz).max() * 1.05), SR, subtype="PCM_24")
    sf.write(f"{CIKTI}/efekt.wav", fx / max(1, np.abs(fx).max() * 1.05), SR, subtype="PCM_24")
    for ad in sys.argv[1:]: kur(ad, mz, fx)

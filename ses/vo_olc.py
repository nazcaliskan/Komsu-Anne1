import json, glob, sys
sys.path.insert(0, "."); from vo_uret import METIN
import zaman as Z
SATIR = [s for s, _ in Z.VO_SATIR]
BAS = [b for _, b in Z.VO_SATIR]
def satirlar(a):
    ch, s, e = a["characters"], a["character_start_times_seconds"], a["character_end_times_seconds"]
    txt = "".join(ch); out = []; pos = 0
    for L in SATIR:
        i = txt.index(L, pos); j = i + len(L) - 1
        while ch[j] in " .,…;": j -= 1          # noktalama kuyruğunu sayma
        out.append((s[i], e[j])); pos = i + len(L)
    return out
if __name__ == "__main__":
    yer = [BAS[i+1] - BAS[i] for i in range(len(BAS) - 1)] + [1.7]
    print("yer     ", " ".join(f"{y:5.2f}" for y in yer))
    for f in sorted(glob.glob("vo3/*.json")):
        try: S = satirlar(json.load(open(f)))
        except ValueError: print(f, "metin eşleşmedi"); continue
        d = [b - a for a, b in S]
        tasma = [f"L{i+1}+{d[i]-yer[i]:.2f}" for i in range(len(BAS)) if d[i] > yer[i] - 0.10]
        print(f"{f[4:-5]:10s}", " ".join(f"{x:5.2f}" for x in d), "| taşan:", tasma or "-")

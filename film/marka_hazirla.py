"""Resmî logo SVG'lerini şeffaf PNG'ye çevirir (Chromium ile, birebir).
Yeniden çizim yok: iki pin, ikon SVG'sindeki grupların biri gizlenerek ayrılır (maske)."""
import re, subprocess, json, os
KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = f"{KOK}/assets/marka"
ikon = open(f"{M}/annea-ikon.svg").read()
gruplar = re.findall(r'<g transform="rotate.*?</g>', ikon)
assert len(gruplar) == 2
isler = {
    "ikon": (ikon, 1024, 1024),
    "ikon_yesil": (ikon.replace(gruplar[1], ""), 1024, 1024),
    "ikon_mercan": (ikon.replace(gruplar[0], ""), 1024, 1024),
    "yazi": (open(f"{M}/annea-yazi.svg").read(), 2847, 697),
    "yatay": (open(f"{M}/annea-logo-yatay.svg").read(), 2444, 552),
}
js = """
const { chromium } = require('playwright'); const fs = require('fs');
(async () => { const b = await chromium.launch(); const isler = JSON.parse(fs.readFileSync(process.argv[2]));
  for (const [ad, [svg, w, h]] of Object.entries(isler)) {
    const p = await b.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
    await p.setContent(`<body style="margin:0;background:transparent">${svg.replace(/<svg /, `<svg style="display:block;width:${w}px;height:${h}px" `)}</body>`);
    await p.screenshot({ path: `${process.argv[3]}/${ad}.png`, omitBackground: true }); await p.close(); }
  await b.close(); })();
"""
json.dump(isler, open("/tmp/_isler.json", "w")); open("/tmp/_marka.js", "w").write(js)
subprocess.run(["node", "/tmp/_marka.js", "/tmp/_isler.json", f"{M}/png"], check=True, env={**os.environ, "NODE_PATH": subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()})
print(sorted(os.listdir(f"{M}/png")))

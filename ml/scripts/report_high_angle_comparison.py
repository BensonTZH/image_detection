#!/usr/bin/env python3
"""Publish the completed local experiment comparison, without changing the Site."""
import base64
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'artifacts/high-angle-boundary-review-20260917'
v4=json.loads((ROOT/'runs/high-angle-classifier/v4-framing/comparison.json').read_text())
v5=json.loads((ROOT/'runs/high-angle-classifier/v5-framing-balanced/comparison.json').read_text())
data=ROOT/'dataset/high-angle-classifier-v4-framing'
models=[('Current website',v4,'previous'),('Candidate 1: varied framing',v4,'candidate'),('Candidate 2: balanced cohorts',v5,'candidate')]

def fraction(rows,key,label):
    good=sum((r[key]>=.75)==(label=='high_angle') for r in rows)
    return f'{good}/{len(rows)}'

table=[];summary=[]
for name,report,key in models:
    rows=report['images']
    acceptable=[r for r in rows if r['label']=='acceptable' and r['variant']=='full']
    high=[r for r in rows if r['label']=='high_angle' and r['split']!='train']
    original=[r for r in high if r['variant']=='full']
    closer=[r for r in high if r['variant']!='full']
    entry={'model':name,'acceptable_original_clear':fraction(acceptable,key,'acceptable'),
           'high_original_detected':fraction(original,key,'high_angle'),'high_closer_detected':fraction(closer,key,'high_angle')}
    summary.append(entry)
    table.append('<tr>'+''.join(f'<td>{html.escape(v)}</td>' for v in entry.values())+'</tr>')

example=next(r for r in v4['images'] if r['label']=='high_angle' and r['variant']=='close' and r['previous']<.01 and r['candidate']>=.75)
pair=[next(r for r in v4['images'] if r['source']==example['source'] and r['variant']=='full'),example]
border=next(r for r in v4['images'] if Path(r['source']).name=='IMG20260917115707.jpg' and r['variant']=='full')

def card(r,text):
    image=base64.b64encode((data/r['prepared']).read_bytes()).decode()
    return f'<article><img src="data:image/png;base64,{image}" alt="{html.escape(text)}"><h3>{html.escape(text)}</h3><p>{html.escape(Path(r["source"]).name)}</p><p>Current high-angle score: {r["previous"]:.1%}<br>Candidate 1: {r["candidate"]:.1%}</p></article>'

page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>High-angle model review</title><style>
body{font:16px/1.6 system-ui,sans-serif;color:#18283a;background:#f4f6f9;max-width:1050px;margin:30px auto;padding:0 20px}h1,h2,h3{line-height:1.2}h1{font-size:34px}h2{margin-top:36px}table{border-collapse:collapse;background:white;width:100%}td,th{text-align:left;padding:14px;border-bottom:1px solid #dce2e9}th{background:#e8eef4}.table{overflow:auto}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px}article,.notice{padding:20px;background:white;border-radius:12px}article img{width:100%;max-height:400px;object-fit:contain}article p{overflow-wrap:anywhere}a{color:#165ab5}.notice{border-left:5px solid #cc8930}.muted{color:#526074}li{margin:10px 0}</style>
<h1>High-angle review: neither candidate deployed</h1><p>Completed 18 September 2026. The current website and its 300 ms alignment hold remain unchanged.</p>
<div class="notice"><strong>Finding:</strong> the current model is sensitive to framing. Cropping closer while retaining the complete slot can remove its high-angle warning even though the camera viewpoint has not changed. Training with these crops helped, but neither candidate preserved all required acceptable and high-angle results.</div>
<h2>Measured results at the 75% warning threshold</h2><div class="table"><table><tr><th>Model</th><th>Original acceptable photos clear</th><th>Original held-out high photos detected</th><th>Closer held-out high views detected</th></tr>'''
page+=''.join(table)+'''</table></div><p class="muted">The 20 closer views come from two crops of the same 10 held-out high-angle photos. These are correlated development checks, not 20 independent captures. The acceptable total includes training photos and is a regression check, not a generalization estimate.</p>
<h2>Same high-angle photo, different framing</h2><p>Both views retain the full annotated slot. The crop changes composition, not viewpoint.</p><div class="cards">'''
page+=''.join(card(r,'Original framing' if r['variant']=='full' else 'Closer framing, same viewpoint') for r in pair)+'</div>'
page+='<h2>Acceptable example the candidates still struggle with</h2><p>This photo remains labeled acceptable, as you requested. Candidate 1 incorrectly warns on it; candidate 2 clears the original but still incorrectly warns on its closer crops.</p><div class="cards">'+card(border,'User-labeled acceptable — not relabeled')+'</div>'
page+='''<h2>Capture-flow check</h2><p>Both candidates retain 13 of 18 original Addon photos reaching capture in a static replay, matching the current deployment. The other five are stopped by pose/detection checks. Candidate 1 removes the existing high-angle capture failures in the tested closer views, but introduces an acceptable-view warning. Candidate 2 loses too much detection on original high views. Neither passed all release criteria.</p>
<h2>Recommended next step</h2><p>Keep the existing labels and collect matched framings of this same machine: at each acceptable or high camera height, take a wider and closer view while maintaining the viewpoint as closely as possible. Include the transition around the acceptable/high boundary, and keep a separate capture session for evaluation. The current tests show why merely changing the threshold or repeating the same acceptable photos is insufficient.</p>
<p>Fresh live-camera testing is still needed before claiming improved handheld capture behavior. The replay repeats static photos; it does not reproduce phone movement or browser resizing exactly.</p>
<h2>Detailed evidence</h2><ul><li><a href="framing-report.html">Original framing sensitivity audit</a></li><li><a href="candidate-report.html">Candidate 1 results and errors</a></li><li><a href="v5-balanced/candidate-report.html">Candidate 2 results and errors</a></li><li><a href="v5-balanced/data-quality.json">Source and split checks</a></li></ul></html>'''
(AUDIT/'review-summary.html').write_text(page)
(AUDIT/'completed-review.json').write_text(json.dumps({'status':'completed_not_deployed','date':'2026-09-18','models':summary,
    'deployment':'unchanged version 98','alignment_hold_ms':300,
    'reason':'Neither trained candidate passed all predeclared development release checks.'},indent=2))
print(json.dumps(summary,indent=2))

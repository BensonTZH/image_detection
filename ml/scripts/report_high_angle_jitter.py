from pathlib import Path
import json,html
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/high-angle-v8-jitter-20260925';OLD=ROOT/'artifacts/high-angle-v7-six-20260923'
d=json.loads((OUT/'comparison.json').read_text());decision=json.loads((OUT/'release-decision.json').read_text())
esc=lambda x:html.escape(str(x),quote=True)
page='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Crop-robustness training results</title><style>body{font:16px system-ui;background:#f5f6f7;color:#17232a;max-width:1200px;padding:24px;margin:auto}h1{font-size:30px}p{line-height:1.55}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}article{padding:18px;background:white;border-radius:10px}h3{overflow-wrap:anywhere;font-size:15px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}img{width:100%;object-fit:contain;height:300px}.notice{padding:18px;background:#fff0d4;border-radius:10px}.bad{color:#ad330f}.good{color:#08714f}small{color:#53606a}</style></head><body><h1>Training with small slot-crop variations</h1>'''
status='Passed the recorded development checks; live testing is still needed.' if decision['ready'] else 'Some development checks still fail; see the remaining errors below.'
page+=f'<p class="notice"><strong>Candidate only — not deployed.</strong><br>{status}</p><p>The labels, class balance, 40% warning threshold, and source-image splits were preserved. Training randomly chooses among the original crop and eight small box-position/size variations, plus color variation. The deployed v7 model is the baseline. The best checkpoint was selected by validation balanced accuracy at 40%, including the crop grid. Test photos and screenshots were excluded from training and checkpoint selection.</p><p>Existing validation/test sessions overlap with training collection sessions and were reviewed previously. This is a development comparison, not a claim of independent live-camera accuracy.</p><h2>Original photos and framings</h2><table><tr><th>Group</th><th>Deployed: high warned</th><th>Candidate: high warned</th><th>Deployed: acceptable clear</th><th>Candidate: acceptable clear</th></tr>'
for name,g in d['groups'].items():
 if name.startswith('train'):continue
 a,b=g['current'],g['candidate'];page+=f'<tr><td>{name}</td><td>{a["high_detected"]}/{a["high_total"]}</td><td>{b["high_detected"]}/{b["high_total"]}</td><td>{a["acceptable_clear"]}/{a["acceptable_total"]}</td><td>{b["acceptable_clear"]}/{b["acceptable_total"]}</td></tr>'
page+='</table><h2>Small crop changes</h2><p>For every existing validation/test framing, the box was shifted vertically by −5%, 0%, or +5% of slot height and scaled to 95%, 100%, or 105%. Nine views per framing are correlated, not nine independent photos.</p><table><tr><th>Group</th><th>Deployed: high warned</th><th>Candidate: high warned</th><th>Deployed: acceptable clear</th><th>Candidate: acceptable clear</th></tr>'
for name,g in d['grid_groups'].items():
 a,b=g['current'],g['candidate'];page+=f'<tr><td>{name}</td><td>{a["high_detected"]}/{a["high_total"]}</td><td>{b["high_detected"]}/{b["high_total"]}</td><td>{a["acceptable_clear"]}/{a["acceptable_total"]}</td><td>{b["acceptable_clear"]}/{b["acceptable_total"]}</td></tr>'
page+='</table><h2>Your six training additions</h2><p>All six are training examples. Matching these labels is a fit check, not independent accuracy.</p><table><tr><th>Photo</th><th>Label</th><th>Candidate high score (full / near / close)</th></tr>'
new=[r for r in d['images'] if r.get('cohort')=='new_six'];sources=list(dict.fromkeys(r['source'] for r in new))
for src in sources:
 rr=[r for r in new if r['source']==src];score=' / '.join(f'{r["candidate"]*100:.2f}%' for r in rr);page+=f'<tr><td>{esc(Path(src).name)}</td><td>{rr[0]["label"]}</td><td>{score}</td></tr>'
page+='</table><h2>Previously failed screenshots</h2><p>These were not used for training or checkpoint selection. Screenshot crops cannot reproduce the original raw camera frame, so these scores are observational checks.</p><div class="grid">'
for r in d['external']:
 correct=r['candidate']>=.4;page+=f'<article><h3>{esc(r["name"])}</h3><img src="file://{esc(r["path"])}"><p>Deployed: {r["current"]*100:.2f}% high<br>Candidate: {r["candidate"]*100:.2f}% high</p><strong class="{"good" if correct else "bad"}">{"Warned" if correct else "Still missed"}</strong></article>'
page+='</div><h2>Remaining original-view mistakes</h2><div class="grid">'
for r in d['images']:
 if r['candidate'] is None or (r['candidate']>=.4)==(r['label']=='high_angle'):continue
 page+=f'<article><h3>{esc(Path(r["source"]).name)}</h3><p>{r["label"]} · {r["split"]} · {r["variant"]}</p><img src="file://{esc(OLD/"roi"/r["prepared"])}"><p>Candidate: {r["candidate"]*100:.2f}% high</p></article>'
page+='</div><h2>Check outcomes</h2><ul>'
for k,v in decision['checks'].items():page+=f'<li class="{"good" if v else "bad"}">{esc(k)}: {"PASS" if v else "FAIL"}</li>'
page+='</ul><p>No labels or live-site model files were changed.</p></body></html>'
(OUT/'report.html').write_text(page);print(OUT/'report.html')

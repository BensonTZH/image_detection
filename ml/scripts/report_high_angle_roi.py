"""Create a review gallery for the isolated high-angle ROI trial."""
import html
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('ROI_AUDIT_DIR', ROOT/'artifacts/high-angle-roi-20260923'))
d=json.loads((OUT/'comparison.json').read_text());decision=json.loads((OUT/'release-decision.json').read_text())
rows=[]
for group,mm in d['groups'].items():
    if group.startswith('train'):continue
    b,c=mm['current'],mm['candidate']
    rows.append(f'<tr><td>{group}</td><td>{b["acceptable_clear"]}/{b["acceptable_total"]}</td><td>{c["acceptable_clear"]}/{c["acceptable_total"]}</td><td>{b["high_detected"]}/{b["high_total"]}</td><td>{c["high_detected"]}/{c["high_total"]}</td></tr>')
cards=[]
for r in d['images']:
    expected=r['label']=='high_angle';candidate=r['candidate'];current=r['current']
    wrong=candidate is None or (candidate>=.4)!=expected
    score=lambda s:'No detection' if s is None else f'{s*100:.1f}%'
    prepared=html.escape(r['prepared'],quote=True)
    roi_image=f'<img loading="lazy" src="roi/{prepared}">' if r['box'] else '<p>No reliable detection; no ROI prepared.</p>'
    cards.append(f'''<article data-label="{r['label']}" data-split="{r['split']}" data-wrong="{str(wrong).lower()}">
<h3>{html.escape(Path(r['source']).name)}</h3><p>{r['label']} · {r['split']} · {r['variant']}</p>
<div class="images"><figure><img loading="lazy" src="full/{prepared}"><figcaption>Current: {score(current)} high</figcaption></figure><figure>{roi_image}<figcaption>Candidate: {score(candidate)} high</figcaption></figure></div>
<b class="{'bad' if wrong else 'good'}">{'Needs review' if wrong else 'Matches folder label'}</b></article>''')
status='Passed development checks — still requires live verification' if decision['ready'] else 'Not released — candidate failed development checks'
page='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>High-angle ROI trial</title><style>
body{font:16px system-ui;background:#f5f6f7;color:#17232a;margin:32px auto;max-width:1250px;padding:0 20px}h1{font-size:30px}p{line-height:1.5}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;text-align:left;border-bottom:1px solid #dde2e4}select,label{font:inherit;margin:10px 18px 10px 0}#gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}article{background:white;border:1px solid #dde2e4;border-radius:10px;padding:16px;overflow:hidden}h3{font-size:13px;overflow-wrap:anywhere}.images{display:flex;gap:8px}figure{margin:0;width:50%}img{width:100%}figcaption{font-size:13px}.bad{color:#ad330f}.good{color:#08714f}article[hidden]{display:none}.notice{background:#fff2d7;padding:16px;border-radius:10px}
</style></head><body>'''
page+=f'<h1>High-angle classifier: normalized slot region</h1><p class="notice"><strong>{status}</strong></p><p>Threshold fixed at 40%. Existing folder labels and source splits preserved. Cropped variants share a source; they are not independent samples. High-angle and Addon holdouts come from the same sessions as their training photos. This report cannot establish new-session or live-camera accuracy.</p>'
new=[r for r in d['images'] if r.get('cohort')=='new_six' and r['variant']=='full']
if new:
    page+='<h2>Your six new training photos</h2><p>All six were used for training as requested. These results check whether the model learned your labels; they are not independent test accuracy. Each photo also has two crop variants in the gallery below.</p><table><tr><th>Photo</th><th>Your label</th><th>Current high-angle score</th><th>Candidate high-angle score</th></tr>'
    for r in new:
        score='No detection' if r['candidate'] is None else f'{r["candidate"]*100:.2f}%'
        page+=f'<tr><td>{html.escape(Path(r["source"]).name)}</td><td>{r["label"]}</td><td>{r["current"]*100:.2f}%</td><td>{score}</td></tr>'
    page+='</table>'
page+='<h2>Validation and test comparison</h2><p>Acceptable: clear of high-angle warning. High angle: correctly warned.</p><table><thead><tr><th>Group</th><th>Current acceptable clear</th><th>Candidate acceptable clear</th><th>Current high detected</th><th>Candidate high detected</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
external=OUT/'external/result.json'
if external.exists():
    e=json.loads(external.read_text())
    page+=f'<h2>Your latest rejected screenshot</h2><p>Current: {e["current"]*100:.2f}% high. Candidate: {e["candidate"]*100:.2f}% high. This screenshot was excluded from training and checkpoint selection. Its visible-camera crop still contains the top UI overlay, so this is an observational check rather than exact raw-frame replay.</p><div class="images" style="max-width:640px"><figure><img src="external/full/external/high_angle/latest-rejected-view.png"><figcaption>Visible-camera crop</figcaption></figure><figure><img src="external/roi/external/high_angle/latest-rejected-view.png"><figcaption>Detected slot region</figcaption></figure></div>'
page+='<h2>Review images</h2><select id="label"><option value="all">All labels</option><option>acceptable</option><option>high_angle</option></select><select id="split"><option value="all">All splits</option><option>train</option><option>val</option><option>test</option></select><label><input type="checkbox" id="wrong" checked> Candidate mistakes only</label><p id="count"></p><div id="gallery">'+''.join(cards)+'</div>'
page+='''<script>function filter(){let n=0;for(const a of document.querySelectorAll('article')){const show=(label.value==='all'||a.dataset.label===label.value)&&(split.value==='all'||a.dataset.split===split.value)&&(!wrong.checked||a.dataset.wrong==='true');a.hidden=!show;if(show)n++}document.getElementById('count').textContent=n+' views shown'}const label=document.getElementById('label'),split=document.getElementById('split'),wrong=document.getElementById('wrong');for(const e of [label,split,wrong])e.onchange=filter;filter();</script></body></html>'''
(OUT/'report.html').write_text(page)
print(OUT/'report.html')

#!/usr/bin/env python3
"""Report the candidate and enforce the recorded development release checks."""
import base64
import html
import json
from pathlib import Path
from high_angle_framing import RUN,DATA,AUDIT
from case_classifier_utils import write_json

report=json.loads((RUN/'comparison.json').read_text());groups=report['groups'];rows=report['images']
pipeline=json.loads((AUDIT/'pipeline-results.json').read_text())
checks={}
full=[r for r in rows if r['variant']=='full' and r['label']=='acceptable']
checks['all_original_acceptable_clear']=all(r['candidate']<.75 for r in full)
for split in ('val','test'):
    for variant in ('full','near','close'):
        group=groups[f'{split}_{variant}_all'];new=group['candidate'];old=group['previous']
        checks[f'{split}_{variant}_high_detected']=new['high_detected'] >= (old['high_detected'] if variant=='full' else 4)
        checks[f'{split}_{variant}_acceptable_clear']=new['acceptable_pass_rate']>=.95
        addon=groups[f'{split}_{variant}_addon']['candidate']
        checks[f'{split}_{variant}_addon_clear']=addon['acceptable_blocked']==0
addon=[r for r in pipeline['results'] if r['cohort']=='addon' and r['variant']=='full']
checks['addon_capture_preserved']=sum(r['candidateResult']['captured'] for r in addon)>=13
write_json(AUDIT/'release-decision.json',{'ready':all(checks.values()),'checks':checks,'failed':[k for k,v in checks.items() if not v],
        'limits':'Existing correlated development photos and framing stress tests; fresh live-camera verification still required.'})
table=[]
for split in ('train','val','test'):
    for variant in ('full','near','close'):
        g=groups[f'{split}_{variant}_all'];old=g['previous'];new=g['candidate']
        table.append(f'<tr><td>{split}</td><td>{variant}</td><td>{old["high_detected"]}/{old["high_detected"]+old["high_missed"]} → {new["high_detected"]}/{new["high_detected"]+new["high_missed"]}</td><td>{old["acceptable_blocked"]} → {new["acceptable_blocked"]}</td></tr>')
failures=[r for r in rows if (r['candidate']>=.75)!=(r['label']=='high_angle')]
cards=[]
for r in failures:
    img=base64.b64encode((DATA/r['prepared']).read_bytes()).decode()
    cards.append(f'<article><img src="data:image/png;base64,{img}"><p>{html.escape(Path(r["source"]).name)}<br>{r["split"]} · {r["variant"]} · labeled {r["label"]}<br>High-angle score: {r["previous"]:.1%} → {r["candidate"]:.1%}</p></article>')
status='Passed development release checks' if all(checks.values()) else 'Not ready to deploy'
page='''<!doctype html><meta charset="utf-8"><title>High-angle candidate evaluation</title><style>body{font:16px system-ui;max-width:1100px;margin:35px auto;padding:0 20px;background:#f4f6f8;color:#162738}p{line-height:1.6}table{border-collapse:collapse;background:white;width:100%}th,td{text-align:left;padding:12px;border-bottom:1px solid #ccd5df}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:20px}article{background:white;padding:14px;border-radius:10px}img{width:100%}article p{overflow-wrap:anywhere}</style>'''
page+=f'<h1>{status}</h1><p>The candidate was trained on three framings per source without changing labels. Threshold75%; source splits preserved. These are correlated development checks, not independent live-camera accuracy.</p><table><tr><th>Split</th><th>Framing</th><th>High views detected: previous → candidate</th><th>Acceptable views wrongly warned: previous → candidate</th></tr>'+''.join(table)+'</table>'
page+=f'<h2>Remaining classifier errors ({len(failures)})</h2><div class="cards">'+''.join(cards)+'</div>'
(AUDIT/'candidate-report.html').write_text(page)
print(json.dumps({'ready':all(checks.values()),'failed_checks':[k for k,v in checks.items() if not v],
    'full_acceptable_blocked':sum(r['candidate']>=.75 for r in full),
    'addon_captured':sum(r['candidateResult']['captured'] for r in addon),'remaining_classifier_errors':len(failures)},indent=2))

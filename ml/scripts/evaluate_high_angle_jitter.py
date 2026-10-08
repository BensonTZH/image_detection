"""Evaluate candidate against the deployed v7 at an unchanged 40% threshold."""
import json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from ultralytics import YOLO
from high_angle_roi_trial import session,tensor,write
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/high-angle-v8-jitter-20260925'
OLD=ROOT/'artifacts/high-angle-v7-six-20260923'
RUN=ROOT/'runs/high-angle-classifier/v8-jitter'

model=YOLO(str(RUN/'weights/best.pt'));assert model.names=={0:'acceptable',1:'high_angle'}
exported=Path(model.export(format='onnx',imgsz=320,opset=17,simplify=False,dynamic=False,device='cpu'))
current=session(ROOT/'web/public/models/slot-high-angle.onnx');candidate=session(exported)
views=json.loads((OUT/'views.json').read_text());rows=json.loads((OLD/'comparison.json').read_text())['images']
original=[];parity=[];parity_counts=defaultdict(int);model.model.eval().to('cpu')
for idx,r in enumerate(rows):
    result={**r,'current':r['candidate'],'candidate':None}
    if r['box']:
        t=tensor(OLD/'roi'/r['prepared']);result['candidate']=float(candidate.run(None,{'images':t})[0][0,1])
        key=(r['label'],r['cohort'],r['split'])
        if parity_counts[key]<2:
            with torch.no_grad():p=model.model(torch.from_numpy(t))
            if isinstance(p,tuple):p=p[0]
            parity.append(abs(float(p[0,1])-result['candidate']));parity_counts[key]+=1
    original.append(result)
    if idx%200==0:print('Original views',idx,len(rows),flush=True)
assert max(parity)<.001
base={r['prepared']:r for r in original}
grid=[]
cached_path=OUT/'baseline-grid-scores.json'
cached=json.loads(cached_path.read_text()) if cached_path.exists() else {}
for i,r in enumerate(v for v in views if v['split'] in ('val','test')):
    result={**r}
    if r['jitter']=='base':result.update({n:base[r['prepared']][n] for n in ['current','candidate']})
    else:
        t=tensor(r['path']);result['current']=cached[r['path']] if r['path'] in cached else float(current.run(None,{'images':t})[0][0,1]);result['candidate']=float(candidate.run(None,{'images':t})[0][0,1])
    grid.append(result)
    if i%200==0:print('Crop-grid views',i,flush=True)

def counts(rr,key):
    a=[r for r in rr if r['label']=='acceptable'];h=[r for r in rr if r['label']=='high_angle'];addon=[r for r in a if r.get('cohort')=='addon']
    return {'acceptable_clear':sum(r[key] is not None and r[key]<.4 for r in a),'acceptable_total':len(a),'high_detected':sum(r[key] is not None and r[key]>=.4 for r in h),'high_total':len(h),'addon_clear':sum(r[key] is not None and r[key]<.4 for r in addon),'addon_total':len(addon),'no_detection':sum(r[key] is None for r in rr)}
groups={};checks={}
for split in ['train','val','test']:
    for variant in ['full','near','close']:
        subset=[r for r in original if r['split']==split and r['variant']==variant];g={n:counts(subset,n) for n in ['current','candidate']};groups[split+'_'+variant]=g
        if split!='train':
            a,b=g['current'],g['candidate'];key=split+'_'+variant
            checks[key+'_acceptable']=b['acceptable_clear']/b['acceptable_total']>=.95
            checks[key+'_high_no_regression']=b['high_detected']>=a['high_detected']
            checks[key+'_addon_all_clear']=b['addon_clear']==b['addon_total']
grid_groups={}
for split in ['val','test']:
    rr=[r for r in grid if r['split']==split];m={n:counts(rr,n) for n in ['current','candidate']};grid_groups[split]=m
    checks[split+'_grid_high_improved']=m['candidate']['high_detected']>m['current']['high_detected']
    checks[split+'_grid_acceptable_no_regression']=m['candidate']['acceptable_clear']>=m['current']['acceptable_clear']
new=[r for r in original if r['cohort']=='new_six']
checks['new_training_fit']=all(r['candidate'] is not None and (r['candidate']>=.4)==(r['label']=='high_angle') for r in new)
external=[]
for name,p in [('earlier-high-view',OLD/'external/roi/external/high_angle/latest-rejected-view.png'),('latest-visible-camera',ROOT/'artifacts/high-angle-boundary-audit-20260925/visible-camera-roi.png'),('latest-whole-camera-with-ui',ROOT/'artifacts/high-angle-boundary-audit-20260925/whole-camera-with-ui-roi.png')]:
    t=tensor(p);external.append({'name':name,'path':str(p),'current':float(current.run(None,{'images':t})[0][0,1]),'candidate':float(candidate.run(None,{'images':t})[0][0,1])})
write(OUT/'comparison.json',{'groups':groups,'grid_groups':grid_groups,'images':original,'grid':grid,'external':external,'parity_max':max(parity)})
write(OUT/'release-decision.json',{'ready':all(checks.values()),'checks':checks,'failed':[k for k,v in checks.items() if not v],'threshold':.4,'sha256':hashlib.sha256(exported.read_bytes()).hexdigest(),'limitations':'Previously inspected development data with collection-session overlap. Screenshot observations are not exact raw-frame replays. No automatic deployment.'})
before=json.loads((OUT/'production-before.json').read_text());assert all(hashlib.sha256((ROOT/'web/public/models'/n).read_bytes()).hexdigest()==h for n,h in before.items())
print(json.dumps({'groups':groups,'grid':grid_groups,'checks':checks,'external':external},indent=2),flush=True)

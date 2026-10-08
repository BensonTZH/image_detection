"""Add the six user-labeled Sept 23 images to the ROI classifier training trial."""
import json
import os
from pathlib import Path
from collections import Counter
import high_angle_roi_trial as base

ROOT=base.ROOT
base.AUDIT=ROOT/'artifacts/high-angle-v7-six-20260923'
base.DATA=ROOT/'dataset/high-angle-classifier-v7-six'
base.RUN=ROOT/'runs/high-angle-classifier/v7-six'
AUDIT,DATA,RUN=base.AUDIT,base.DATA,base.RUN
OLD=ROOT/'artifacts/high-angle-roi-20260923'

def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():assert os.path.samefile(src,dst)
    else:os.link(src,dst)

def prepare():
    assert not (DATA/'manifest.json').exists()
    old=json.loads((OLD/'inputs.json').read_text())
    new=json.loads((AUDIT/'new/inputs.json').read_text())
    assert len(new)==18 and len({r['source'] for r in new})==6
    assert all(r['split']=='train' and r['box'] is not None for r in new), 'New photo needs detection review'
    for source,rows in [(OLD,old),(AUDIT/'new',new)]:
        for r in rows:
            for kind in ('full','roi'):
                if kind=='roi' and not r['box']:continue
                link(source/kind/r['prepared'],AUDIT/kind/r['prepared'])
    rows=old+new;hashes={}
    for r in rows:
        hashes.setdefault(r['sha256'],set()).add(r['split'])
        if r['box']:link(AUDIT/'roi'/r['prepared'],DATA/r['prepared'])
    assert all(len(s)==1 for s in hashes.values())
    cohorts={}
    for r in rows:
        if r['split']=='train' and r['box']:
            k='high_angle' if r['label']=='high_angle' else r.get('cohort','original')
            cohorts.setdefault(k,[]).append(r)
    n=len(cohorts['original']);targets={'original':n,'addon':n,'new_six':96,'high_angle':2*n+96}
    for cohort,items in cohorts.items():
        for i in range(targets[cohort]-len(items)):
            r=items[i%len(items)];src=DATA/r['prepared'];link(src,src.with_name(src.stem+f'__repeat_{i}.png'))
    base.write(AUDIT/'inputs.json',rows)
    base.write(DATA/'manifest.json',{'images':rows,'train_weights':targets,'unique_sources':dict(Counter(r['label'] for r in rows if r['variant']=='full')),'limitations':'All six new photos used for training, not held out. Crop variants are correlated. Existing val/test assignments preserved; high-angle/Addon held-out data share collection sessions with training.'})
    criteria=json.loads((OLD/'release-criteria.json').read_text())
    criteria['rules']+=['All six new training additions must fit their user labels in all framings (fit check, not independent accuracy).']
    base.write(AUDIT/'release-criteria.json',criteria)
    print(json.dumps({'sources':len(hashes),'views':len(rows),'train_weights':targets}),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','evaluate']);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='train':
        os.environ['ROI_INITIAL_WEIGHTS']=str(ROOT/'runs/high-angle-classifier/v6-roi/weights/best.pt')
        base.train()
    else:
        base.evaluate()
        comparison=json.loads((AUDIT/'comparison.json').read_text())
        new=[r for r in comparison['images'] if r.get('cohort')=='new_six']
        failed=[r['prepared'] for r in new if r['candidate'] is None or ((r['candidate']>=.4)!=(r['label']=='high_angle'))]
        decision=json.loads((AUDIT/'release-decision.json').read_text())
        decision['checks']['new_training_fit']=not failed
        decision['new_training_fit_failures']=failed
        decision['ready']=all(decision['checks'].values())
        decision['failed']=[k for k,v in decision['checks'].items() if not v]
        base.write(AUDIT/'release-decision.json',decision)
        base.write(AUDIT/'new-six-results.json',new)
        print('New training addition mismatches:',failed,flush=True)

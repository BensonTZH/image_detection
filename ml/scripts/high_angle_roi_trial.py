"""Isolated ROI classifier experiment; never writes production models or labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[2]
AUDIT=Path(os.environ.get('ROI_AUDIT_DIR', ROOT/'artifacts/high-angle-roi-20260923'))
RUN=Path(os.environ.get('ROI_RUN_DIR', ROOT/'runs/high-angle-classifier/v6-roi'))
DATA=Path(os.environ.get('ROI_DATA_DIR', ROOT/'dataset/high-angle-classifier-v6-roi'))

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))

def tensor(p):
    with Image.open(p) as im:return np.asarray(im.convert('RGB'),dtype=np.float32).transpose(2,0,1)[None]/255

def session(p):
    o=ort.SessionOptions();o.intra_op_num_threads=2
    return ort.InferenceSession(str(p),sess_options=o,providers=['CPUExecutionProvider'])

def collect():
    models={n:session(ROOT/'web/public/models'/f'slot-{n}.onnx') for n in ('pose','high-angle','acceptability')}
    rows=json.loads((AUDIT/'manifest.json').read_text())
    for i,r in enumerate(rows):
        t=tensor(AUDIT/'full'/r['prepared'])
        outputs={n:s.run(None,{'images':t})[0] for n,s in models.items()}
        pose=outputs['pose'][0];idx=int(pose[4:7].max(axis=0).argmax());v=pose[:,idx]
        r['pose']=v.tolist();r['confidence']=float(v[4:7].max())
        w,h=r['width'],r['height'];scale=min(320/w,320/h);px=(320-round(w*scale))/2;py=(320-round(h*scale))/2
        r['box']=[max(0,min(w,float((v[0]-v[2]/2-px)/scale))),max(0,min(h,float((v[1]-v[3]/2-py)/scale))),max(0,min(w,float((v[0]+v[2]/2-px)/scale))),max(0,min(h,float((v[1]+v[3]/2-py)/scale)))] if r['confidence']>=.35 else None
        r['current']=float(outputs['high-angle'][0,1]);r['acceptability']=float(outputs['acceptability'][0,1])
        if i%100==0:print(f'Inference {i}/{len(rows)}',flush=True)
    write(AUDIT/'inputs.json',rows)

def prepare():
    assert not (DATA/'manifest.json').exists(),'Refusing to overwrite completed trial'
    rows=json.loads((AUDIT/'inputs.json').read_text());counts={};splits={}
    for r in rows:
        splits.setdefault(r['sha256'],set()).add(r['split'])
        if not r['box']:continue
        src=AUDIT/'roi'/r['prepared'];dst=DATA/r['prepared'];dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert os.path.samefile(src,dst)
        else:os.link(src,dst)
        key=(r['split'],r['label']);counts[key]=counts.get(key,0)+1
    assert all(len(s)==1 for s in splits.values())
    # Balance classes but cap Addon cohort at the original cohort's total weight.
    cohorts={}
    for r in rows:
        if r['split']=='train' and r['box']:
            key='high_angle' if r['label']=='high_angle' else r.get('cohort','original')
            cohorts.setdefault(key,[]).append(r)
    # 50/50 classes; acceptable is half original, half Addon for this machine.
    target=max(len(cohorts['original']),len(cohorts['addon']))
    targets={'original':target,'addon':target,'high_angle':target*2}
    for key,items in cohorts.items():
        for i in range(targets[key]-len(items)):
            r=items[i%len(items)];src=DATA/r['prepared'];os.link(src,src.with_name(src.stem+f'__repeat_{i}.png'))
    write(DATA/'manifest.json',{'images':rows,'unique_counts':{str(k):v for k,v in counts.items()},'train_weight':targets,'limitations':'Same capture sessions across high-angle and Addon development splits; framing variants are correlated. No new independent test set.'})
    write(AUDIT/'release-criteria.json',{'threshold':.4,'fixed_before_training':True,'rules':['At least 95% of acceptable val and test views clear in each framing.','All six Addon val/test sources clear in all framings.','High-angle val and test recall at least 80% in each framing.','Full-frame high-angle recall no worse than current model in val and test.'], 'deployment':'Only consider replacing production if all checks pass; no thresholds selected on test.'})
    print(counts,flush=True)

def train():
    from ultralytics import YOLO
    from ultralytics.models.yolo.classify import ClassificationTrainer
    from torchvision import transforms as T
    import torch
    class RoiTrainer(ClassificationTrainer):
        def build_dataset(self,img_path,mode='train',batch=None):
            d=super().build_dataset(img_path,mode,batch)
            ops=[T.ColorJitter(brightness=.2,contrast=.2,saturation=.15,hue=.01)] if mode=='train' else []
            d.torch_transforms=T.Compose(ops+[T.ToTensor()]);return d
    assert not (RUN/'weights').exists()
    # Generic pretrained weights avoid inheriting background shortcuts from v3/v4.
    init=Path(os.environ.get('ROI_INITIAL_WEIGHTS', ROOT/'yolo11n-cls.pt'))
    write(RUN/'experiment.json',{'initial':str(init),'epochs':25,'seed':23,'input':'Chromium canvas 320 ROI from current detected box; unchanged perspective','threshold':.4,'selection':'validation top1 only; test excluded','frozen_layers':5})
    YOLO(str(init)).train(trainer=RoiTrainer,data=str(DATA),project=str(RUN.parent),name=RUN.name,exist_ok=True,epochs=25,imgsz=320,batch=32,device='mps' if torch.backends.mps.is_available() else 'cpu',workers=0,patience=7,optimizer='AdamW',lr0=.0003,lrf=.1,warmup_epochs=1,seed=23,deterministic=True,freeze=5,cache=False,plots=False,auto_augment=None,erasing=0,fliplr=0,flipud=0,scale=0)

def evaluate():
    import torch
    from ultralytics import YOLO
    model=YOLO(str(RUN/'weights/best.pt'));assert model.names=={0:'acceptable',1:'high_angle'}
    exported=Path(model.export(format='onnx',imgsz=320,opset=17,simplify=False,dynamic=False,device='cpu'))
    s=session(exported);rows=json.loads((AUDIT/'inputs.json').read_text());parity=[];parity_groups={};model.model.eval().to('cpu')
    for i,r in enumerate(rows):
        r['candidate']=None
        if r['box']:
            t=tensor(AUDIT/'roi'/r['prepared']);r['candidate']=float(s.run(None,{'images':t})[0][0,1])
            parity_key=(r['label'],r.get('cohort','original'),r['split'])
            if parity_groups.get(parity_key,0)<2:
                with torch.no_grad():p=model.model(torch.from_numpy(t))
                if isinstance(p,tuple):p=p[0]
                parity.append(abs(float(p[0,1])-r['candidate']))
                parity_groups[parity_key]=parity_groups.get(parity_key,0)+1
        if i%100==0:print(f'Evaluate {i}/{len(rows)}',flush=True)
    assert max(parity)<.001
    groups={};checks={}
    for split in ('train','val','test'):
        for variant in ('full','near','close'):
            rr=[r for r in rows if r['split']==split and r['variant']==variant]
            m={}
            for name in ('current','candidate'):
                a=[r for r in rr if r['label']=='acceptable'];h=[r for r in rr if r['label']=='high_angle'];addon=[r for r in a if r.get('cohort')=='addon']
                m[name]={'acceptable_clear':sum(r[name] is not None and r[name]<.4 for r in a),'acceptable_total':len(a),'high_detected':sum(r[name] is not None and r[name]>=.4 for r in h),'high_total':len(h),'addon_clear':sum(r[name] is not None and r[name]<.4 for r in addon),'addon_total':len(addon),'no_detection':sum(r[name] is None for r in rr)}
            groups[f'{split}_{variant}']=m
            if split!='train':
                c=m['candidate'];b=m['current'];k=f'{split}_{variant}'
                checks[k+'_acceptable']=c['acceptable_clear']/c['acceptable_total']>=.95
                checks[k+'_addon']=c['addon_clear']==c['addon_total']
                checks[k+'_high']=c['high_detected']/c['high_total']>=.8
                if variant=='full':checks[k+'_baseline_high']=c['high_detected']>=b['high_detected']
    write(AUDIT/'comparison.json',{'groups':groups,'images':rows,'parity_max':max(parity)})
    write(AUDIT/'release-decision.json',{'ready':all(checks.values()),'checks':checks,'failed':[k for k,v in checks.items() if not v],'threshold':.4,'sha256':hashlib.sha256(exported.read_bytes()).hexdigest(),'limitations':'Correlated existing development splits, not new live-camera accuracy. No detection counts as unavailable, not a classifier success.'})
    print(json.dumps({'groups':groups,'checks':checks},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['collect','prepare','train','evaluate']);a=p.parse_args();globals()[a.command]()

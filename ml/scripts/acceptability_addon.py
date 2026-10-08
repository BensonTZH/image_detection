#!/usr/bin/env python3
"""Preserve reviewed splits, add Addon, and evaluate a separate acceptability candidate."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps
from case_classifier_utils import letterbox_square, write_json

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'dataset/reviewed-acceptability-v3-addon'
RUN = ROOT / 'runs/case-classifier/reviewed-acceptability-v3-addon'
AUDIT = ROOT / 'artifacts/acceptability-v3-addon'
OLD_DATA = ROOT / 'dataset/reviewed-acceptability-classifier'
OLD_WEIGHTS = ROOT / 'runs/case-classifier/reviewed-acceptability-v2/weights/best.pt'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    assert not DATA.exists() and not AUDIT.exists(), 'Refusing to overwrite previous work'
    original = json.loads((OLD_DATA/'manifest.json').read_text())
    addon = json.loads((ROOT/'artifacts/high-angle-v3-addon/import.json').read_text())['imported']
    rows = [{**r, 'cohort':'original'} for r in original['images'] if not r['replica']]
    assert len(rows)==980 and len(addon)==18
    for r in addon:
        rows.append({'source_path':r['source'], 'filename':Path(r['source']).name,
                     'source_id':r['source_id'], 'label':'success', 'split':r['split'],
                     'replica':False, 'cohort':'addon'})
    hashes = {}
    for r in rows:
        source = Path(r['source_path'])
        assert source.exists(), source
        print(f'Checking {source}', flush=True)
        sha = digest(source)
        assert sha not in hashes or hashes[sha]==r['split'], 'Duplicate crosses splits'
        hashes[sha]=r['split']; r['sha256']=sha
    AUDIT.mkdir(parents=True)
    backup=AUDIT/'rollback'; backup.mkdir()
    for p in (ROOT/'web/public/models').glob('*.onnx'):
        if p.name=='slot-acceptability.onnx': shutil.copy2(p, backup/p.name)
    write_json(backup/'model-hashes.json', {p.name:digest(p) for p in (ROOT/'web/public/models').glob('*.onnx')})
    shutil.copy2(ROOT/'web/public/models/slot-acceptability.metadata.json', backup/'slot-acceptability.metadata.json')
    for r in rows:
        dest=DATA/r['split']/r['label']/(Path(r['filename']).stem+'.png')
        assert not dest.exists()
        dest.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(r['source_path']) as im: letterbox_square(ImageOps.exif_transpose(im),320).save(dest)
        r['prepared_path']=str(dest.relative_to(DATA))
    replicas=[]
    train={label:[r for r in rows if r['split']=='train' and r['label']==label] for label in ('error','success')}
    for label,items in train.items():
        for i in range(max(map(len,train.values()))-len(items)):
            r=items[i%len(items)]; src=DATA/r['prepared_path']; dst=src.with_name(f'{src.stem}__replica_{i}.png')
            shutil.copy2(src,dst); replicas.append({**r,'prepared_path':str(dst.relative_to(DATA)),'replica':True})
    counts={s:dict(Counter(r['label'] for r in rows if r['split']==s)) for s in ('train','val','test')}
    write_json(DATA/'manifest.json',{'images':rows+replicas,'counts':counts,'preprocessing':'full-frame RGB letterbox320 fill114 /255',
        'limitations':'Old splits unchanged. Addon12/3/3 matches high-angle training; held-out Addon shares one capture session.'})
    print(json.dumps(counts),flush=True)

def train():
    from ultralytics import YOLO
    from ultralytics.models.yolo.classify import ClassificationTrainer
    from torchvision import transforms as T
    from training_utils import choose_device
    class FullFrameTrainer(ClassificationTrainer):
        def build_dataset(self,img_path,mode='train',batch=None):
            data=super().build_dataset(img_path,mode,batch)
            ops=[T.ColorJitter(brightness=.15,contrast=.15,saturation=.15,hue=.01)] if mode=='train' else []
            data.torch_transforms=T.Compose(ops+[T.ToTensor()])
            return data
    assert not (RUN/'weights').exists(), 'Refusing to overwrite trained weights'
    write_json(RUN/'experiment.json',{'source_weights':str(OLD_WEIGHTS),'source_sha256':digest(OLD_WEIGHTS),
        'dataset':str(DATA),'epochs':25,'seed':17,'threshold':.75,'augmentation':'Color jitter only; preserve full frame',
        'strategy':'Fine-tune deployed v2 with all old training images plus12 Addon; unchanged holdout splits.'})
    YOLO(str(OLD_WEIGHTS)).train(trainer=FullFrameTrainer,data=str(DATA),project=str(RUN.parent),name=RUN.name,
        exist_ok=True,epochs=25,imgsz=320,batch=16,device=choose_device('auto'),workers=0,patience=8,
        optimizer='AdamW',lr0=.0003,lrf=.1,warmup_epochs=1,seed=17,deterministic=True,
        cache=False,plots=True,auto_augment=None,erasing=0,fliplr=0,flipud=0,scale=0)

def evaluate():
    import numpy as np
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO
    model=YOLO(str(RUN/'weights/best.pt'))
    assert model.names=={0:'error',1:'success'}
    exported=Path(model.export(format='onnx',imgsz=320,opset=17,simplify=False,dynamic=False,device='cpu'))
    opts=ort.SessionOptions();opts.intra_op_num_threads=2
    sessions={n:ort.InferenceSession(str(p),sess_options=opts,providers=['CPUExecutionProvider']) for n,p in
              [('previous',AUDIT/'rollback/slot-acceptability.onnx'),('candidate',exported)]}
    manifest=json.loads((DATA/'manifest.json').read_text());rows=[];parity=[]
    model.model.eval().to('cpu')
    for r in manifest['images']:
        if r['replica']:continue
        # Lossless PNG contains the identical full-frame preprocessing used
        # above; avoid re-reading large original photos during evaluation.
        with Image.open(DATA/r['prepared_path']) as im: im=im.convert('RGB')
        tensor=np.asarray(im,dtype=np.float32).transpose(2,0,1)[None]/255
        scores={n:float(s.run(None,{s.get_inputs()[0].name:tensor})[0][0,1]) for n,s in sessions.items()}
        if len(parity)<8:
            with torch.no_grad(): result=model.model(torch.from_numpy(tensor))
            if isinstance(result,tuple):result=result[0]
            parity.append(abs(float(result[0,1])-scores['candidate']))
        rows.append({**r,**scores})
        if len(rows)%100==0: print(f'Evaluated {len(rows)} images',flush=True)
    def metrics(subset,key):
        accepted=[r for r in subset if r['label']=='success']; rejected=[r for r in subset if r['label']=='error']
        return {'accepted_passed':sum(r[key]>=.75 for r in accepted),'accepted_total':len(accepted),
                'rejected_passed':sum(r[key]>=.75 for r in rejected),'rejected_total':len(rejected)}
    groups={f'{c}_{s}':{n:metrics([r for r in rows if r['cohort']==c and r['split']==s],n) for n in sessions}
            for c in ('original','addon') for s in ('train','val','test')}
    assert max(parity)<.001
    write_json(RUN/'comparison.json',{'threshold':.75,'groups':groups,'parity_max':max(parity),'images':rows,'limitations':manifest['limitations']})
    write_json(RUN/'model.metadata.json',{'source_weights':str(RUN/'weights/best.pt'),'sha256':digest(exported),
        'classes':['error','success'],'success_threshold':.75,'input_size':320,'preprocessing':manifest['preprocessing'],
        'groups':groups,'onnx_max_probability_difference':max(parity),'limitations':manifest['limitations']})
    print(json.dumps({'groups':groups,'parity':max(parity)},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','train','evaluate'])
    globals()[parser.parse_args().command]()

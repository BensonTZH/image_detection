#!/usr/bin/env python3
"""Import reviewed Addon pairs and train an isolated high-angle v3 candidate."""
import argparse
import base64
import hashlib
import html
import io
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
import high_angle_classifier as base
from case_classifier_utils import letterbox_square, write_json

ROOT = base.ROOT
SOURCE = ROOT / 'dataset/Categorize/slot-angle-pose-v3-full-984'
ADDON = SOURCE / 'Addon'
DATA = ROOT / 'dataset/high-angle-classifier-v3-addon'
RUN = ROOT / 'runs/high-angle-classifier/v3-addon'
AUDIT = ROOT / 'artifacts/high-angle-v3-addon'
BASELINE = ROOT / 'web/public/models/slot-high-angle.onnx'
THRESHOLD = .75


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if DATA.exists() or AUDIT.exists():
        raise FileExistsError('Refusing to overwrite an existing import or prepared dataset')
    photos = {p.stem:p for p in ADDON.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}}
    labels_list = list(ADDON.glob('*/labels/train/*.txt'))
    labels = {p.stem:p for p in labels_list}
    assert len(labels) == len(labels_list), 'Duplicate annotation basename'
    matched = sorted(photos.keys() & labels.keys())
    orphan_photos = sorted(photos.keys()-labels.keys())
    orphan_labels = sorted(labels.keys()-photos.keys())
    assert len(matched)==18 and len(orphan_photos)==2 and not orphan_labels, 'Unexpected Addon contents'
    annotations = {}
    for stem in matched:
        lines = labels[stem].read_text().strip().splitlines()
        assert len(lines)==1, f'Expected one slot: {stem}'
        values = list(map(float, lines[0].split()))
        assert len(values)==17 and all(math.isfinite(v) for v in values) and values[0]==0
        assert all(0<=v<=1 for v in values[1:5]) and min(values[3:5])>0
        for i in range(5,17,3):
            assert 0<=values[i]<=1 and 0<=values[i+1]<=1 and values[i+2] in (0,1,2)
        with Image.open(photos[stem]) as im:
            im.verify()
        annotations[stem] = {'image_sha256':digest(photos[stem]), 'label_sha256':digest(labels[stem]), 'values':values}
        for dest in (SOURCE/'images/centered/acceptable'/photos[stem].name, SOURCE/'labels/centered/acceptable'/labels[stem].name):
            assert not dest.exists(), f'Destination already exists: {dest}'
    original = json.loads((ROOT/'dataset/high-angle-classifier-v2/manifest.json').read_text())
    items = [{**x,'cohort':'original'} for x in original['images'] if not x['replica']]
    for item in items:
        assert digest(Path(item['source'])) == item['sha256'], f'Original source changed: {item["source"]}'
    AUDIT.mkdir(parents=True)
    public_models = {str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'web/public/models').glob('*.onnx')}
    write_json(AUDIT/'production-model-hashes.json',public_models)
    shutil.copy2(BASELINE,AUDIT/'baseline-high-angle.onnx')
    backup = AUDIT/'removed-unmatched'; backup.mkdir()
    removed=[]
    for stem in orphan_photos:
        src=photos[stem]; dst=backup/src.name
        shutil.copy2(src,dst)
        assert digest(src)==digest(dst)
        removed.append({'source':str(src),'recovery_copy':str(dst),'sha256':digest(src)})
        src.unlink()
    imported=[]
    for index,stem in enumerate(matched):
        image_dest=SOURCE/'images/centered/acceptable'/photos[stem].name
        label_dest=SOURCE/'labels/centered/acceptable'/labels[stem].name
        image_dest.parent.mkdir(parents=True,exist_ok=True);label_dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(photos[stem],image_dest);shutil.copy2(labels[stem],label_dest)
        assert digest(image_dest)==annotations[stem]['image_sha256'] and digest(label_dest)==annotations[stem]['label_sha256']
        split='train' if index<12 else 'val' if index<15 else 'test'
        item={'source':str(image_dest),'source_id':f'addon-session-{split}', 'sha256':digest(image_dest),'split':split,'label':'acceptable','replica':False,'cohort':'addon', 'pose_label':str(label_dest)}
        items.append(item);imported.append(item)
    write_json(AUDIT/'import.json',{'matched':18,'removed_unmatched':removed,'annotations':annotations,'imported':imported,'note':'Matching originals retained in Addon. Canonical copies added to centered/acceptable. Historical baseline manifests unchanged.'})
    hash_splits={}
    for item in items:
        sha=item['sha256'];split=item['split']
        assert sha not in hash_splits or hash_splits[sha]==split, 'Identical image crosses splits'
        hash_splits[sha]=split
        dest=DATA/split/item['label']/(Path(item['source']).stem+'.png')
        assert not dest.exists(), 'Prepared name collision'
        dest.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(item['source']) as im:letterbox_square(ImageOps.exif_transpose(im),320).save(dest)
        item['prepared']=str(dest.relative_to(DATA))
    train={label:[x for x in items if x['split']=='train' and x['label']==label] for label in base.LABELS}
    target=max(map(len,train.values()));replicas=[]
    for label, originals in train.items():
        for i in range(target-len(originals)):
            item=originals[i%len(originals)];src=DATA/item['prepared'];dest=src.with_name(f'{src.stem}__replica_{i}.png')
            shutil.copy2(src,dest);replicas.append({**item,'prepared':str(dest.relative_to(DATA)),'replica':True})
    counts={split:dict(Counter(x['label'] for x in items if x['split']==split)) for split in ('train','val','test')}
    write_json(DATA/'manifest.json',{'classes':base.LABELS,'input_size':320,'preprocessing':original['preprocessing'],'original_counts':counts,'training_replicas':len(replicas),'images':items+replicas,'limitations':'Original splits preserved. Addon uses chronological blocks:12 train,3 validation,3 test. All 18 are one short session, so held-out Addon and high-angle photos are correlated development checks, not independent generalization estimates.'})
    print(json.dumps({'counts':counts,'removed_unmatched':removed,'training_replicas':len(replicas)},indent=2),flush=True)


def train(epochs):
    base.DATA=DATA;base.RUN=RUN
    base.train(epochs)


def evaluate():
    import numpy as np
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO
    model=YOLO(str(RUN/'weights/best.pt'))
    assert model.names=={0:'acceptable',1:'high_angle'}
    exported=Path(model.export(format='onnx',imgsz=320,opset=17,simplify=False,dynamic=False,device='cpu'))
    options=ort.SessionOptions();options.intra_op_num_threads=2
    sessions={name:ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']) for name,path in [('previous',AUDIT/'baseline-high-angle.onnx'),('candidate',exported)]}
    manifest=json.loads((DATA/'manifest.json').read_text())
    rows=[];errors=[];model.model.eval();model.model.to('cpu')
    for item in manifest['images']:
        if item['replica']:continue
        with Image.open(item['source']) as im:prepared=letterbox_square(ImageOps.exif_transpose(im),320)
        tensor=np.asarray(prepared,dtype=np.float32).transpose(2,0,1)[None]/255
        scores={name:float(session.run(None,{session.get_inputs()[0].name:tensor})[0][0,1]) for name,session in sessions.items()}
        if len(errors)<8:
            with torch.no_grad(): result=model.model(torch.from_numpy(tensor))
            if isinstance(result,tuple):result=result[0]
            errors.append(abs(float(result[0,1])-scores['candidate']))
        rows.append({**item,**scores})
    assert max(errors)<.001, 'ONNX parity failed'
    groups={}
    for split in ('train','val','test'):
        for cohort in ('original','addon'):
            subset=[r for r in rows if r['split']==split and r['cohort']==cohort]
            groups[f'{cohort}_{split}']={name:base.metrics([{**r,'probability':r[name]} for r in subset],THRESHOLD) for name in sessions}
    report={'threshold':THRESHOLD,'threshold_selection':'Fixed at current deployed75%; not selected on these evaluation results.','groups':groups,'onnx_max_probability_difference':max(errors),'limitations':manifest['limitations'],'images':rows}
    write_json(RUN/'comparison.json',report)
    write_json(RUN/'model.metadata.json',{'model':str(exported),'sha256':digest(exported),'classes':model.names,'high_angle_threshold':THRESHOLD,'input_size':320,'preprocessing':manifest['preprocessing'],'onnx_max_probability_difference':max(errors),'status':'evaluated candidate; not deployed','limitations':manifest['limitations'],'groups':groups})
    parts=[]
    for r in sorted(rows,key=lambda r:(r['cohort']!='addon',-r['candidate'])):
        with Image.open(r['source']) as im:
            im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((220,300));buf=io.BytesIO();im.save(buf,format='JPEG',quality=75)
        parts.append(f'<article><img loading="lazy" src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"><div><b>{html.escape(Path(r["source"]).name)}</b><p>{r["cohort"]} · {r["split"]} · label: {r["label"]}</p><p>High-angle score<br>Previous: {r["previous"]*100:.3f}%<br>Candidate: {r["candidate"]*100:.3f}%</p><strong>{"Lower phone" if r["candidate"]>=THRESHOLD else "No high-angle warning"}</strong></div></article>')
    page='<!doctype html><meta charset="utf-8"><title>High-angle v3 comparison</title><style>body{font:15px system-ui;background:#f2f4f7;color:#172638;margin:30px}section{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:16px}article{background:white;border-radius:12px;overflow:hidden}img{width:100%;height:300px;object-fit:contain;background:#dde3e9}article div{padding:15px}b{overflow-wrap:anywhere}p{line-height:1.5}</style><h1>High-angle classifier: previous vs candidate</h1><p>Threshold75%. Addon photos first.12 Addon images trained,3 validated,3 tested. Same-session held-out images are correlated; this does not establish live-camera accuracy. Only the high-angle gate is evaluated. Candidate has not been deployed.</p><section>'+''.join(parts)+'</section>'
    (AUDIT/'comparison.html').write_text(page)
    hashes=json.loads((AUDIT/'production-model-hashes.json').read_text())
    assert all(digest(ROOT/path)==sha for path,sha in hashes.items()), 'Production model changed during training'
    print(json.dumps({'groups':groups,'parity_max':max(errors),'report':str(AUDIT/'comparison.html')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','evaluate']);p.add_argument('--epochs',type=int,default=30);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='train':train(a.epochs)
    else:evaluate()

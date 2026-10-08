#!/usr/bin/env python3
"""Train high-angle detection across framings without changing pose or labels."""
import argparse
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageOps
from case_classifier_utils import letterbox_square, write_json
from high_angle_classifier import digest, metrics

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'dataset/Categorize/slot-angle-pose-v3-full-984'
DATA=ROOT/'dataset/high-angle-classifier-v4-framing'
RUN=ROOT/'runs/high-angle-classifier/v4-framing'
AUDIT=ROOT/'artifacts/high-angle-boundary-review-20260917'

def frame_rect(size,box,top,bottom):
    w,h=size;cx,cy,bw,bh=box
    y1=max(0,int((cy-bh/2-bh*top)*h));y2=min(h,int((cy+bh/2+bh*bottom)*h+1))
    cw=min(w,max(int((y2-y1)*w/h+.5),int(bw*w+2)))
    x1=min(w-cw,max(0,int(cx*w-cw/2)))
    assert x1 <= (cx-bw/2)*w+1 and x1+cw >= (cx+bw/2)*w-1
    assert y1 <= (cy-bh/2)*h+1 and y2 >= (cy+bh/2)*h-1
    return [x1,y1,x1+cw,y2]

def prepare(resume=False):
    assert not (DATA/'manifest.json').exists(), 'Prepared dataset already complete'
    if not resume: assert not DATA.exists() and not (AUDIT/'rollback').exists()
    labels={p.stem:p for p in (SOURCE/'labels/centered/acceptable').glob('*.txt')}
    labels.update({p.stem:p for p in (SOURCE/'images/high_angle').glob('*/labels/train/*.txt')})
    records=[r for r in json.loads((ROOT/'dataset/high-angle-classifier-v3-addon/manifest.json').read_text())['images'] if not r['replica']]
    assert len(records)==275
    for r in records: assert Path(r['source']).stem in labels, r['source']
    backup=AUDIT/'rollback'
    if not resume:
        backup.mkdir()
        shutil.copy2(ROOT/'web/public/models/slot-high-angle.onnx',backup/'slot-high-angle.onnx')
        shutil.copy2(ROOT/'web/public/models/slot-high-angle.metadata.json',backup/'slot-high-angle.metadata.json')
        write_json(backup/'model-hashes.json',{p.name:digest(p) for p in (ROOT/'web/public/models').glob('*.onnx')})
        write_json(backup/'deployment.json',{'version':98,'commit':'2e5c0d42320290a5e8a08a16213b1b1af2167628',
            'project_id':'appgprj_6a82cdce34a481919643dba6df61c620',
            'version_id':'appgprj_6a82cdce34a481919643dba6df61c620~appgver_d2032960256081918ae948d9c84c5576'})
    else:
        assert all(digest(ROOT/'web/public/models'/name)==sha for name,sha in json.loads((backup/'model-hashes.json').read_text()).items())
    def prepare_record(r):
        variants=[]
        src=Path(r['source']);values=list(map(float,labels[src.stem].read_text().split()));assert len(values)==17
        with Image.open(src) as photo:
            photo=ImageOps.exif_transpose(photo).convert('RGB');w,h=photo.size
            for variant,rect in [('full',[0,0,w,h]),('near',frame_rect((w,h),values[1:5],.30,.15)),('close',frame_rect((w,h),values[1:5],.08,.08))]:
                dest=DATA/r['split']/r['label']/f'{src.stem}__{variant}.png';dest.parent.mkdir(parents=True,exist_ok=True)
                cropped=photo.crop(rect);letterbox_square(cropped,320).save(dest)
                variants.append({**r,'prepared':str(dest.relative_to(DATA)),'variant':variant,'rect':rect,'width':cropped.width,'height':cropped.height,'pose_label':str(labels[src.stem])})
        return variants
    rows=[]
    with ThreadPoolExecutor(max_workers=6) as executor:
        for i,variants in enumerate(executor.map(prepare_record,records)):
            rows.extend(variants)
            if i%25==0:print(f'Prepared {i}/275 sources',flush=True)
    train={label:[r for r in rows if r['label']==label and r['split']=='train'] for label in ('acceptable','high_angle')}
    replicas=[]
    for label,items in train.items():
        for i in range(max(map(len,train.values()))-len(items)):
            r=items[i%len(items)];src=DATA/r['prepared'];dst=src.with_name(f'{src.stem}__replica_{i}.png')
            shutil.copy2(src,dst);replicas.append({**r,'prepared':str(dst.relative_to(DATA)),'replica':True})
    counts={s:dict(Counter(r['label'] for r in rows if r['split']==s)) for s in ('train','val','test')}
    write_json(DATA/'manifest.json',{'counts':counts,'images':rows+replicas,'limitations':'All source splits preserved; crops stay with their source. Crops are correlated stress tests, not new independent photos. Addon and high-angle holdouts share their respective capture sessions.'})
    print(json.dumps(counts),flush=True)

def train():
    from ultralytics import YOLO
    from ultralytics.models.yolo.classify import ClassificationTrainer
    from torchvision import transforms as T
    from training_utils import choose_device
    class FramingTrainer(ClassificationTrainer):
        def build_dataset(self,img_path,mode='train',batch=None):
            dataset=super().build_dataset(img_path,mode,batch)
            ops=[T.ColorJitter(brightness=.15,contrast=.15,saturation=.15,hue=.01)] if mode=='train' else []
            dataset.torch_transforms=T.Compose(ops+[T.ToTensor()])
            return dataset
    assert not (RUN/'weights').exists()
    init=ROOT/'runs/high-angle-classifier/v3-addon/weights/best.pt'
    write_json(RUN/'experiment.json',{'initial_weights':str(init),'dataset':str(DATA),'epochs':25,'lr0':.0003,'threshold':.75,'seed':17,
        'augmentation':'Three fixed framings per source, each retaining complete labeled slot and original perspective; color jitter only.'})
    YOLO(str(init)).train(trainer=FramingTrainer,data=str(DATA),project=str(RUN.parent),name=RUN.name,exist_ok=True,
        epochs=25,imgsz=320,batch=16,device=choose_device('auto'),workers=0,patience=8,
        optimizer='AdamW',lr0=.0003,lrf=.1,warmup_epochs=1,seed=17,deterministic=True,
        cache=False,plots=True,auto_augment=None,erasing=0,fliplr=0,flipud=0,scale=0)

def evaluate():
    import numpy as np
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO
    model=YOLO(str(RUN/'weights/best.pt'));assert model.names=={0:'acceptable',1:'high_angle'}
    exported=Path(model.export(format='onnx',imgsz=320,opset=17,simplify=False,dynamic=False,device='cpu'))
    opt=ort.SessionOptions();opt.intra_op_num_threads=2
    sessions={n:ort.InferenceSession(str(p),sess_options=opt,providers=['CPUExecutionProvider']) for n,p in
              [('previous',AUDIT/'rollback/slot-high-angle.onnx'),('candidate',exported)]}
    manifest=json.loads((DATA/'manifest.json').read_text());rows=[];parity=[];model.model.eval().to('cpu')
    for r in manifest['images']:
        if r['replica']:continue
        with Image.open(DATA/r['prepared']) as im:tensor=np.asarray(im.convert('RGB'),dtype=np.float32).transpose(2,0,1)[None]/255
        scores={n:float(s.run(None,{s.get_inputs()[0].name:tensor})[0][0,1]) for n,s in sessions.items()}
        if len(parity)<8:
            with torch.no_grad():result=model.model(torch.from_numpy(tensor))
            if isinstance(result,tuple):result=result[0]
            parity.append(abs(float(result[0,1])-scores['candidate']))
        rows.append({**r,**scores})
        if len(rows)%100==0:print(f'Evaluated {len(rows)} views',flush=True)
    assert max(parity)<.001
    groups={}
    for split in ('train','val','test'):
        for variant in ('full','near','close'):
            for cohort in ('all','addon'):
                subset=[r for r in rows if r['split']==split and r['variant']==variant and (cohort=='all' or r.get('cohort')=='addon')]
                groups[f'{split}_{variant}_{cohort}']={n:metrics([{**r,'probability':r[n]} for r in subset],.75) for n in sessions}
    write_json(RUN/'comparison.json',{'groups':groups,'images':rows,'parity_max':max(parity),'threshold':.75,'limitations':manifest['limitations']})
    write_json(RUN/'model.metadata.json',{'model':str(exported),'sha256':digest(exported),'classes':model.names,
        'input_size':320,'high_angle_threshold':.75,'preprocessing':'full-frame RGB letterbox320 fill114 divide255',
        'augmentation':'Framing variants retain the complete slot; no pose warp or rotation','groups':groups,'limitations':manifest['limitations']})
    print(json.dumps({k:v for k,v in groups.items() if not k.startswith('train')},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','resume-prepare','train','evaluate']);a=p.parse_args()
    if a.command=='resume-prepare':prepare(True)
    else:globals()[a.command]()

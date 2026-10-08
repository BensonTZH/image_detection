"""Candidate-only crop-robustness experiment; never changes labels or web models."""
import argparse, json, os, random, hashlib
from pathlib import Path
from copy import copy
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/high-angle-v8-jitter-20260925'
DATA=ROOT/'dataset/high-angle-classifier-v8-jitter'
RUN=ROOT/'runs/high-angle-classifier/v8-jitter'
OLD=ROOT/'dataset/high-angle-classifier-v7-six'

def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n')

def prepare():
    assert not (DATA/'manifest.json').exists()
    views=json.loads((OUT/'views.json').read_text());pools=json.loads((OUT/'training-pools.json').read_text())
    splits={}
    for r in views:splits.setdefault(r['sha256'],set()).add(r['split'])
    assert all(len(v)==1 for v in splits.values())
    n=0
    for src in (OLD/'train').rglob('*.png'):
        dst=DATA/src.relative_to(OLD);dst.parent.mkdir(parents=True,exist_ok=True)
        key=src.parent.name+'/'+src.stem.split('__repeat_')[0]
        assert key in pools,key
        os.link(src,dst);n+=1
    for r in views:
        if r['split']!='val':continue
        dst=DATA/'val'/r['label']/(Path(r['prepared']).stem+'__'+r['jitter']+'.png')
        dst.parent.mkdir(parents=True,exist_ok=True);os.link(r['path'],dst)
    write(DATA/'manifest.json',{'training_samples_per_epoch':n,'training_crop_pools':len(pools),'validation_views':sum(r['split']=='val' for r in views),'labels_unchanged':True,'splits_unchanged':True,'train_weights':json.loads((OLD/'manifest.json').read_text())['train_weights'],'seed':25,'selection':'validation balanced accuracy at 0.4 over original and crop-grid inputs','views_manifest':str(OUT/'views.json')})
    print((DATA/'manifest.json').read_text(),flush=True)

def train():
    import torch
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator
    from torchvision import transforms as T
    pools=json.loads((OUT/'training-pools.json').read_text())
    class JitterDataset(torch.utils.data.Dataset):
        def __init__(self,base):self.base=base
        def __len__(self):return len(self.base)
        def __getattr__(self,name):return getattr(self.base,name)
        def __getitem__(self,index):
            f,label,_,_=self.base.samples[index]
            f=Path(f);key=f.parent.name+'/'+f.stem.split('__repeat_')[0]
            with Image.open(random.choice(pools[key])) as im:img=self.base.torch_transforms(im.convert('RGB'))
            return {'img':img,'cls':label}
    class ThresholdValidator(ClassificationValidator):
        def init_metrics(self,model):
            super().init_metrics(model);self.high_scores=[]
        def update_metrics(self,preds,batch):
            super().update_metrics(preds,batch);self.high_scores.append(preds[:,1].detach().cpu())
        def get_stats(self):
            stats=super().get_stats();scores=torch.cat(self.high_scores);targets=torch.cat(self.targets)
            high=float((scores[targets==1]>=.4).float().mean());clear=float((scores[targets==0]<.4).float().mean())
            stats['fitness']=(high+clear)/2
            stats['metrics/high_recall_at40']=high;stats['metrics/acceptable_clear_at40']=clear
            return stats
    class JitterTrainer(ClassificationTrainer):
        def build_dataset(self,img_path,mode='train',batch=None):
            d=super().build_dataset(img_path,mode,batch)
            ops=[T.ColorJitter(brightness=.2,contrast=.2,saturation=.15,hue=.01)] if mode=='train' else []
            d.torch_transforms=T.Compose(ops+[T.ToTensor()])
            return JitterDataset(d) if mode=='train' else d
        def get_validator(self):return ThresholdValidator(self.test_loader,self.save_dir,args=copy(self.args),_callbacks=self.callbacks)
    assert not (RUN/'weights').exists()
    initial=ROOT/'runs/high-angle-classifier/v7-six/weights/best.pt'
    write(RUN/'experiment.json',{'initial':str(initial),'epochs':25,'patience':8,'freeze':5,'seed':25,'threshold':.4,'selection':'Validation balanced accuracy at 40% over 9 box perturbations per source framing; test and screenshots excluded','train_augmentation':'Choose one of 9 browser-rendered crops per sample on each epoch, plus color jitter','crop_ranges':{'horizontal_shift_box_width':[-.15,.15],'vertical_shift_box_height':[-.07,.07],'box_scale':[.92,1.08]},'torch':torch.__version__})
    model=YOLO(str(initial));assert model.names=={0:'acceptable',1:'high_angle'}
    model.train(trainer=JitterTrainer,data=str(DATA),project=str(RUN.parent),name=RUN.name,exist_ok=True,epochs=25,imgsz=320,batch=32,device='mps' if torch.backends.mps.is_available() else 'cpu',workers=0,patience=8,optimizer='AdamW',lr0=.0003,lrf=.1,warmup_epochs=1,seed=25,deterministic=True,freeze=5,cache=False,plots=False,auto_augment=None,erasing=0,fliplr=0,flipud=0,scale=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train']);a=p.parse_args();globals()[a.command]()

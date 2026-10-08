#!/usr/bin/env python3
"""Balance capture cohorts while preserving v4 source splits and framing variants."""
import argparse
import json
import os
import runpy
import shutil
from collections import Counter
import high_angle_framing as base
from case_classifier_utils import write_json

OLD_DATA=base.DATA
OLD_RUN=base.RUN
OLD_AUDIT=base.AUDIT
base.DATA=base.ROOT/'dataset/high-angle-classifier-v5-framing-balanced'
base.RUN=base.ROOT/'runs/high-angle-classifier/v5-framing-balanced'
base.AUDIT=OLD_AUDIT/'v5-balanced'
DATA,RUN,AUDIT=base.DATA,base.RUN,base.AUDIT

def prepare():
    assert not DATA.exists() and not AUDIT.exists()
    AUDIT.mkdir()
    shutil.copytree(OLD_AUDIT/'rollback',AUDIT/'rollback')
    shutil.copy2(OLD_AUDIT/'release-checks.json',AUDIT/'release-checks.json')
    manifest=json.loads((OLD_DATA/'manifest.json').read_text())
    rows=[r for r in manifest['images'] if not r['replica']]
    for r in rows:
        dst=DATA/r['prepared'];dst.parent.mkdir(parents=True,exist_ok=True)
        os.link(OLD_DATA/r['prepared'],dst)
    cohorts={}
    for r in rows:
        if r['split']!='train':continue
        cohort='high_angle' if r['label']=='high_angle' else r.get('cohort','original')
        cohorts.setdefault(cohort,[]).append(r)
    assert sorted(map(len,cohorts.values()))==[36,57,480], {k:len(v) for k,v in cohorts.items()}
    replicas=[]
    for cohort,items in cohorts.items():
        for i in range(480-len(items)):
            r=items[i%len(items)];src=DATA/r['prepared'];dst=src.with_name(f'{src.stem}__replica_{i}.png')
            os.link(src,dst);replicas.append({**r,'prepared':str(dst.relative_to(DATA)),'replica':True})
    manifest.update(images=rows+replicas,cohort_training_counts={k:480 for k in cohorts},
        training_counts=dict(Counter(r['label'] for r in rows+replicas if r['split']=='train')))
    write_json(DATA/'manifest.json',manifest)
    # Pose and acceptability inputs are identical. Reuse verified outputs, replace only height scores after evaluation.
    shutil.copy2(OLD_AUDIT/'pipeline-inputs.json',AUDIT/'pipeline-inputs.json')
    print(json.dumps(manifest['training_counts']))

def train():
    from ultralytics import YOLO
    from ultralytics.models.yolo.classify import ClassificationTrainer
    from torchvision import transforms as T
    from training_utils import choose_device
    class BalancedTrainer(ClassificationTrainer):
        def build_dataset(self,img_path,mode='train',batch=None):
            dataset=super().build_dataset(img_path,mode,batch)
            ops=[T.ColorJitter(brightness=.15,contrast=.15,saturation=.15,hue=.01)] if mode=='train' else []
            dataset.torch_transforms=T.Compose(ops+[T.ToTensor()])
            return dataset
    assert not (RUN/'weights').exists()
    init=OLD_RUN/'weights/best.pt'
    write_json(RUN/'experiment.json',{'initial_weights':str(init),'dataset':str(DATA),'epochs':20,'lr0':.0001,
        'seed':17,'threshold':.75,'cohorts':'480 original acceptable, 480 Addon acceptable, 480 high-angle training views including repeats; validation/test unchanged',
        'selection':'Ultralytics validation top1 fitness; test excluded from training and checkpoint selection'})
    YOLO(str(init)).train(trainer=BalancedTrainer,data=str(DATA),project=str(RUN.parent),name=RUN.name,exist_ok=True,
        epochs=20,imgsz=320,batch=16,device=choose_device('auto'),workers=0,patience=6,
        optimizer='AdamW',lr0=.0001,lrf=.1,warmup_epochs=1,seed=17,deterministic=True,
        cache=False,plots=True,auto_augment=None,erasing=0,fliplr=0,flipud=0,scale=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','evaluate','pipeline','report']);a=p.parse_args()
    if a.command=='evaluate':base.evaluate()
    elif a.command=='pipeline':
        import sys
        sys.argv=['high_angle_framing_pipeline.py','--attach-height']
        runpy.run_path(str(base.ROOT/'ml/scripts/high_angle_framing_pipeline.py'),run_name='__main__')
    elif a.command=='report':runpy.run_path(str(base.ROOT/'ml/scripts/report_high_angle_framing.py'),run_name='__main__')
    else:globals()[a.command]()

#!/usr/bin/env python3
"""Generate model outputs for an offline replay of the production capture gates."""
import json
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from case_classifier_utils import letterbox_square, write_json
from acceptability_addon import ROOT, DATA, RUN, AUDIT

options=ort.SessionOptions(); options.intra_op_num_threads=2
if '--score-candidate' in sys.argv:
    session=ort.InferenceSession(str(RUN/'weights/best.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    rows=json.loads((AUDIT/'pipeline-inputs.json').read_text())
    for row in rows:
        with Image.open(row['file']) as image:
            tensor=np.asarray(letterbox_square(ImageOps.exif_transpose(image),320),dtype=np.float32).transpose(2,0,1)[None]/255
        row['candidate']=float(session.run(None,{session.get_inputs()[0].name:tensor})[0][0,1])
    write_json(AUDIT/'pipeline-inputs.json',rows)
    print(f'Scored {len(rows)} candidate inputs',flush=True)
    raise SystemExit(0)
models={'pose':ROOT/'web/public/models/slot-pose.onnx',
        'high':ROOT/'web/public/models/slot-high-angle.onnx',
        'previous':AUDIT/'rollback/slot-acceptability.onnx'}
sessions={name:ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']) for name,path in models.items()}
items=[r for r in json.loads((DATA/'manifest.json').read_text())['images']
       if not r['replica'] and (r['cohort']=='addon' or r['split']!='train')]
items += [{'source_path':r['source'],'cohort':'high_angle','split':r['split'],'label':'high_angle'} for r in
          json.loads((ROOT/'dataset/high-angle-classifier-v3-addon/manifest.json').read_text())['images']
          if not r['replica'] and r['label']=='high_angle']
rows=[]
for i,r in enumerate(items):
    with Image.open(r['source_path']) as image:
        image=ImageOps.exif_transpose(image);width,height=image.size
        tensor=np.asarray(letterbox_square(image,320),dtype=np.float32).transpose(2,0,1)[None]/255
    outputs={name:session.run(None,{session.get_inputs()[0].name:tensor})[0] for name,session in sessions.items()}
    pose=outputs['pose'][0]
    assert pose.shape[0]==19
    # The first production NMS result is always the highest scoring candidate.
    best=int(pose[4:7].max(axis=0).argmax())
    rows.append({'file':r['source_path'],'cohort':r['cohort'],'split':r['split'],'label':r['label'],
                 'width':width,'height':height,'pose':pose[:,best].tolist(),
                 **{name:float(outputs[name][0,1]) for name in ('high','previous')}})
    if i%50==0:print(f'{i}/{len(items)}',flush=True)
write_json(AUDIT/'pipeline-inputs.json',rows)
print(f'Wrote {len(rows)} inputs',flush=True)

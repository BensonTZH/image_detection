#!/usr/bin/env python3
"""Collect pose/acceptability outputs once; attach candidate height scores later."""
import json
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image
from case_classifier_utils import write_json
from high_angle_framing import ROOT,DATA,RUN,AUDIT

if '--attach-height' in sys.argv:
    scores={r['prepared']:r for r in json.loads((RUN/'comparison.json').read_text())['images']}
    rows=json.loads((AUDIT/'pipeline-inputs.json').read_text())
    for row in rows:
        r=scores[row['prepared']];row['previousHigh']=r['previous'];row['candidateHigh']=r['candidate']
    write_json(AUDIT/'pipeline-inputs.json',rows)
    raise SystemExit(0)

opt=ort.SessionOptions();opt.intra_op_num_threads=2
sessions={n:ort.InferenceSession(str(ROOT/'web/public/models'/p),sess_options=opt,providers=['CPUExecutionProvider'])
          for n,p in [('pose','slot-pose.onnx'),('acceptability','slot-acceptability.onnx')]}
rows=[]
for r in json.loads((DATA/'manifest.json').read_text())['images']:
    if r['replica'] or (r['split']=='train' and r.get('cohort')!='addon' and r['label']!='high_angle'):continue
    with Image.open(DATA/r['prepared']) as im: tensor=np.asarray(im.convert('RGB'),dtype=np.float32).transpose(2,0,1)[None]/255
    outputs={n:s.run(None,{s.get_inputs()[0].name:tensor})[0] for n,s in sessions.items()}
    pose=outputs['pose'][0];best=int(pose[4:7].max(axis=0).argmax())
    rows.append({'file':r['source'],'prepared':r['prepared'],'cohort':'high_angle' if r['label']=='high_angle' else r.get('cohort','original'),
        'split':r['split'],'variant':r['variant'],'label':r['label'],'width':r['width'],'height':r['height'],
        'pose':pose[:,best].tolist(),'acceptability':float(outputs['acceptability'][0,1])})
    if len(rows)%100==0:print(f'Pipeline inputs {len(rows)}',flush=True)
write_json(AUDIT/'pipeline-inputs.json',rows)
print(f'Wrote {len(rows)} views',flush=True)

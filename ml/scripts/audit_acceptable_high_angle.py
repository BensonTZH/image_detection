#!/usr/bin/env python3
"""Fresh high-angle inference over every current acceptable image; visual report only."""
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from case_classifier_utils import IMAGE_SUFFIXES, letterbox_square

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'dataset/Categorize/slot-angle-pose-v3-full-984/images/centered/acceptable'
OUT=ROOT/'artifacts'/('acceptable-high-angle-audit-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
MODELS={
    'current':('Current website — high-angle v3',ROOT/'web/public/models/slot-high-angle.onnx'),
    'candidate1':('Candidate 1 — varied framing (not deployed)',ROOT/'runs/high-angle-classifier/v4-framing/weights/best.onnx'),
    'candidate2':('Candidate 2 — balanced cohorts (not deployed)',ROOT/'runs/high-angle-classifier/v5-framing-balanced/weights/best.onnx'),
}
THRESHOLD=.75
OUT.mkdir();(OUT/'thumbnails').mkdir()
paths=sorted(p for p in SOURCE.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
assert paths
options=ort.SessionOptions();options.intra_op_num_threads=2
sessions={key:ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']) for key,(_,path) in MODELS.items()}
metadata={key:{'name':name,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for key,(name,path) in MODELS.items()}

def prepare(item):
    index,path=item
    try:
        raw=path.read_bytes()
        with Image.open(io.BytesIO(raw)) as photo:
            photo=ImageOps.exif_transpose(photo).convert('RGB')
            tensor=np.asarray(letterbox_square(photo,320),dtype=np.float32).transpose(2,0,1)[None]/255
            thumb=ImageOps.contain(photo,(320,560),method=Image.Resampling.LANCZOS)
            thumbnail=f'thumbnails/{index:04d}.jpg';thumb.save(OUT/thumbnail,quality=85)
            return {'file':str(path),'name':path.name,'relative':str(path.relative_to(SOURCE)),
                'source_url':path.as_uri(),'sha256':hashlib.sha256(raw).hexdigest(),'width':photo.width,'height':photo.height,'thumbnail':thumbnail},tensor
    except Exception as error:
        return {'file':str(path),'error':str(error)},None

rows=[];errors=[]
with ThreadPoolExecutor(max_workers=6) as pool:
    for row,tensor in pool.map(prepare,enumerate(paths)):
        if tensor is None:errors.append(row);continue
        row['scores']={}
        for key,session in sessions.items():
            output=session.run(None,{session.get_inputs()[0].name:tensor})[0]
            assert output.shape==(1,2) and np.isfinite(output).all() and np.all((output>=0)&(output<=1))
            row['scores'][key]=float(output[0,1])
        rows.append(row)
        if len(rows)%25==0:print(f'Scored {len(rows)}/{len(paths)} original acceptable images with all three models',flush=True)

summary={key:{'images':len(rows),'warnings':sum(r['scores'][key]>=THRESHOLD for r in rows),
    'no_warning':sum(r['scores'][key]<THRESHOLD for r in rows),
    'maximum_score':max(r['scores'][key] for r in rows)} for key in MODELS}
payload={'source':str(SOURCE),'created':datetime.now().isoformat(),'threshold':THRESHOLD,'models':metadata,
    'summary':summary,'images':rows,'errors':errors,'files_found':len(paths),
    'scope':'Fresh CPU ONNX inference on original full photos, EXIF applied, RGB letterbox320 fill114 divide255. No crops or training. Scores are high-angle-only; no warning does not establish full capture eligibility. Live camera/browser resizing and temporal warning hysteresis can differ. This includes training photos, so it is not independent accuracy.'}
(OUT/'results.json').write_text(json.dumps(payload,indent=2))
template='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Acceptable photos — high-angle audit</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1320px;margin:28px auto;padding:0 20px;background:#f4f6f8;color:#19293c}h1{font-size:30px;line-height:1.2}.muted{color:#536274}.controls{display:flex;gap:14px;flex-wrap:wrap;padding:16px 0;position:sticky;top:0;background:#f4f6f8;z-index:1}select,input{font:inherit;padding:10px;border:1px solid #b8c5d2;border-radius:7px;max-width:100%}input{min-width:210px}#summary{background:#fff;border-radius:12px;padding:18px;margin:18px 0;font-size:19px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:18px}.card{background:white;border-radius:12px;overflow:hidden;border:1px solid #dce2e9}.card img{width:100%;height:350px;object-fit:contain;background:#e8edf2}.details{padding:14px}.name{font-size:12px;overflow-wrap:anywhere;color:#526174}.badge{display:inline-block;padding:4px 9px;border-radius:5px;font-weight:650}.clear{color:#166b41;background:#e1f3e9}.warn{color:#a03326;background:#ffe6df}.score{font-size:23px;font-weight:700;margin:9px 0}a{color:#155da8}#count{padding:8px 0}.other{font-size:12px;color:#536274}</style>
<h1>Every acceptable photo: high-angle classifier results</h1><p id="intro"></p><p class="muted">Original photos only. A score of 75% or above triggers the high-angle gate; a sustained warning asks you to lower the phone. “No high-angle warning” does not mean all capture checks pass. This folder includes training images, so these results are not independent accuracy.</p>
<div class="controls"><select id="model" aria-label="Choose classifier"></select><select id="filter" aria-label="Filter photos"><option value="all">All photos</option><option value="warnings">Warnings only</option><option value="clear">No warning only</option></select><input id="search" type="search" placeholder="Search filename" aria-label="Search filename"></div>
<div id="summary"></div><div id="count"></div><main id="grid" class="grid"></main><p><a href="results.json">Download detailed scores, model hashes and source records</a></p>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const model=document.getElementById('model'),filter=document.getElementById('filter'),search=document.getElementById('search');
Object.entries(data.models).forEach(([key,value])=>{const o=document.createElement('option');o.value=key;o.textContent=value.name;model.append(o)});
document.getElementById('intro').textContent=`${data.images.length} of ${data.files_found} images evaluated · ${data.errors.length} read errors · ${data.created.slice(0,19).replace('T',' ')}`;
function render(){const key=model.value,stats=data.summary[key],query=search.value.toLowerCase();
 document.getElementById('summary').textContent=`${stats.no_warning} no warning · ${stats.warnings} high-angle warning${stats.warnings===1?'':'s'} · ${stats.images} photos`;
 const rows=data.images.filter(r=>r.name.toLowerCase().includes(query)&&(filter.value==='all'||(filter.value==='warnings')===(r.scores[key]>=data.threshold))).sort((a,b)=>b.scores[key]-a.scores[key]);
 document.getElementById('count').textContent=`Showing ${rows.length} photos, highest high-angle score first.`;
 const grid=document.getElementById('grid');grid.replaceChildren();
 for(const r of rows){const card=document.createElement('article');card.className='card';const link=document.createElement('a');link.href=r.source_url;link.target='_blank';const img=document.createElement('img');img.src=r.thumbnail;img.alt=r.name;img.loading='lazy';link.append(img);card.append(link);
  const details=document.createElement('div');details.className='details';const warning=r.scores[key]>=data.threshold;const badge=document.createElement('span');badge.className='badge '+(warning?'warn':'clear');badge.textContent=warning?'High-angle warning':'No high-angle warning';details.append(badge);
  const score=document.createElement('div');score.className='score';score.textContent=(r.scores[key]*100).toFixed(2)+'% high angle';details.append(score);
  const name=document.createElement('div');name.className='name';name.textContent=r.name;details.append(name);
  const other=document.createElement('p');other.className='other';other.textContent=`Current ${(r.scores.current*100).toFixed(2)}% · Candidate 1 ${(r.scores.candidate1*100).toFixed(2)}% · Candidate 2 ${(r.scores.candidate2*100).toFixed(2)}%`;details.append(other);card.append(details);grid.append(card);
 }
}
model.addEventListener('change',render);filter.addEventListener('change',render);search.addEventListener('input',render);render();
</script></html>'''
(OUT/'report.html').write_text(template.replace('__DATA__',json.dumps(payload).replace('<','\\u003c')))
print(json.dumps({'output':str(OUT),'summary':summary,'errors':errors},indent=2),flush=True)

// Render box perturbations using the deployed browser preprocessing, preserving perspective.
const fs=require('node:fs'),path=require('node:path'),{createHash}=require('node:crypto');
const root=path.resolve(__dirname,'../..'),old=root+'/artifacts/high-angle-v7-six-20260923',out=root+'/artifacts/high-angle-v8-jitter-20260925';
const {chromium}=require('/Users/bensontan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const ts=require(root+'/web/node_modules/typescript');
const code=ts.transpileModule(fs.readFileSync(root+'/web/lib/high-angle-input.ts','utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText;
let seed=250923;function random(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;}
const rows=JSON.parse(fs.readFileSync(old+'/comparison.json')).images;
const models=Object.fromEntries(['slot-high-angle.onnx','slot-pose.onnx','slot-acceptability.onnx'].map(n=>[n,createHash('sha256').update(fs.readFileSync(root+'/web/public/models/'+n)).digest('hex')]));
fs.writeFileSync(out+'/production-before.json',JSON.stringify(models,null,2));
fs.writeFileSync(out+'/release-criteria.json',JSON.stringify({threshold:.4,selection:'Maximum validation balanced accuracy at the 40% threshold across base and fixed crop-grid views. Test excluded from checkpoint selection.',rules:['No loss of baseline high-angle rejection on the original full/near/close validation and test views.','At least 95% acceptable views pass in each framing and split.','All new-six labels still fit across their original three framings (training fit only).','Fewer high-angle crop-grid misses than v7 on validation and test without increasing acceptable crop-grid false warnings.','Document all remaining failed examples; no automatic deployment.'],limitations:'Validation/test sessions overlap training collection sessions and have been inspected before. These are development checks, not independent live accuracy.'},null,2));
(async()=>{const b=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});const all=[],pools={};try{const p=await b.newPage();await p.evaluate(code=>{window.crop={};new Function('exports',code)(window.crop);},code);let last='';
for(let i=0;i<rows.length;i++){const r=rows[i];if(!r.box)continue;
 if(last!==r.source){const data='data:image/jpeg;base64,'+fs.readFileSync(r.source).toString('base64');await p.evaluate(async data=>{window.sourceImage=new Image();sourceImage.src=data;await sourceImage.decode();},data);last=r.source;}
 const original={...r,jitter:'base',dx:0,dy:0,scale:1,path:old+'/roi/'+r.prepared};all.push(original);
 const changes=r.split==='train'?Array.from({length:8},(_,n)=>({id:'aug'+n,dx:(random()*2-1)*.15,dy:(random()*2-1)*.07,scale:.92+random()*.16})):Array.from({length:9},(_,n)=>({id:'grid'+n,dx:0,dy:[-.05,0,.05][Math.floor(n/3)],scale:[.95,1,1.05][n%3]})).filter(x=>x.id!=='grid4');
 const outputs=await p.evaluate(({r,changes})=>{const frame=document.createElement('canvas');frame.width=r.width;frame.height=r.height;const [x1,y1,x2,y2]=r.rect;frame.getContext('2d').drawImage(sourceImage,x1,y1,x2-x1,y2-y1,0,0,frame.width,frame.height);const canvas=document.createElement('canvas');return changes.map(c=>{const [l,t,rr,bb]=r.box,w=rr-l,h=bb-t,cx=(l+rr)/2+c.dx*w,cy=(t+bb)/2+c.dy*h;crop.highAngleInput(frame,frame.width,frame.height,[cx-w*c.scale/2,cy-h*c.scale/2,cx+w*c.scale/2,cy+h*c.scale/2],canvas);return canvas.toDataURL().split(',')[1];});},{r,changes});
 const pool=[original.path];for(let n=0;n<changes.length;n++){const c=changes[n],relative=r.prepared.replace(/\.png$/,'__'+c.id+'.png'),dst=out+'/jitter/'+relative;fs.mkdirSync(path.dirname(dst),{recursive:true});fs.writeFileSync(dst,Buffer.from(outputs[n],'base64'));all.push({...r,jitter:c.id,dx:c.dx,dy:c.dy,scale:c.scale,path:dst});pool.push(dst);}
 if(r.split==='train')pools[r.label+'/'+path.basename(r.prepared,'.png')]=pool;
 if(i%100===0)console.log('Rendered '+i+'/'+rows.length+' source framings');
}
fs.writeFileSync(out+'/views.json',JSON.stringify(all,null,2));fs.writeFileSync(out+'/training-pools.json',JSON.stringify(pools,null,2));console.log('Prepared '+all.length+' views, '+Object.keys(pools).length+' training crop pools.');
}finally{await b.close();}})().catch(e=>{console.error(e);process.exit(1)});

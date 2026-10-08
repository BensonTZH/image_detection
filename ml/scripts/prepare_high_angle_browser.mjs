// Render owned dataset photos with the same Chromium canvas path as the live app.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { chromium } = require('/Users/bensontan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = path.resolve(import.meta.dirname, '../..');
const out = process.env.ROI_AUDIT_DIR || path.join(root, 'artifacts/high-angle-roi-20260923');
const mode = process.argv[2] || 'full';
const rows = mode === 'full' ? JSON.parse(fs.readFileSync(process.env.ROI_INPUT_MANIFEST || path.join(root,'dataset/high-angle-classifier-v4-framing/manifest.json'))).images.filter(r=>!r.replica) : JSON.parse(fs.readFileSync(path.join(out,'inputs.json')));
const browser = await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
try {
 const page=await browser.newPage();
 let last='', data='';
 for(let i=0;i<rows.length;i++) {
  const r=rows[i];
  if(r.source!==last) { data='data:image/jpeg;base64,'+fs.readFileSync(r.source).toString('base64');last=r.source; }
  if(mode==='roi' && !r.box) continue;
  const png=await page.evaluate(async ({data,r,mode})=>{
   const img=new Image();img.src=data;await img.decode();
   const [x1,y1,x2,y2]=r.rect;
   const full=document.createElement('canvas');full.width=x2-x1;full.height=y2-y1;
   full.getContext('2d').drawImage(img,x1,y1,x2-x1,y2-y1,0,0,full.width,full.height);
   const canvas=document.createElement('canvas');canvas.width=canvas.height=320;
   const ctx=canvas.getContext('2d');ctx.fillStyle='rgb(114,114,114)';ctx.fillRect(0,0,320,320);
   if(mode==='full') {
    const scale=Math.min(320/full.width,320/full.height),w=Math.round(full.width*scale),h=Math.round(full.height*scale);
    ctx.drawImage(full,(320-w)/2,(320-h)/2,w,h);
   } else {
    const [bx1,by1,bx2,by2]=r.box;
    // Fixed aspect, 12.5% margin at each end. No perspective rectification.
    const h=Math.max((by2-by1)*1.25,(bx2-bx1)*3*16/9), w=h*9/16;
    const cx=(bx1+bx2)/2,cy=(by1+by2)/2,s=320/h;
    // Drawing the full image with a transform preserves fill beyond its edges.
    ctx.save();ctx.beginPath();ctx.rect(70,0,180,320);ctx.clip();
    ctx.drawImage(full,70-(cx-w/2)*s,-(cy-h/2)*s,full.width*s,full.height*s);ctx.restore();
   }
   return canvas.toDataURL('image/png').split(',')[1];
  },{data,r,mode});
  const dst=path.join(out,mode,r.prepared);fs.mkdirSync(path.dirname(dst),{recursive:true});fs.writeFileSync(dst,Buffer.from(png,'base64'));
  if(i%100===0) console.log(`${mode}: ${i}/${rows.length}`);
 }
 if(mode==='full') fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify(rows,null,2));
 console.log(`${mode}: finished ${rows.length} views`);
} finally { await browser.close(); }

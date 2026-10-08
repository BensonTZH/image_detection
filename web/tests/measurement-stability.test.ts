import test from 'node:test';
import assert from 'node:assert/strict';
import { stabilizeMeasurements, type MeasurementHistory } from '../lib/measurement-stability.ts';
import { DEFAULT_ALIGNMENT_REFERENCE, type Detection } from '../lib/pose.ts';
import { liveGeometryDecision, updateCaptureHold, emptyCaptureHold } from '../lib/live-guidance.ts';
function slot(vertical: number): Detection {
 const bottom=100*(1-vertical);
 return {box:[450,100,550,900],confidence:.9,angleClass:'centered',keypoints:[
 {x:450,y:100,confidence:1},{x:550,y:100,confidence:1},
 {x:500+bottom/2,y:900,confidence:1},{x:500-bottom/2,y:900,confidence:1}]};
}
function feed(values:number[], step=400) {
 let history:MeasurementHistory=[];
 return values.map((v,i)=>{const r=stabilizeMeasurements(history,slot(v),1000,1000,i*step,i>0);history=r.history;return r;});
}
test('small prediction jitter is filtered while the 16% limit remains unchanged',()=>{
 const results=feed([.13,.15,.14,.17,.13,.15,.14]);
 assert.equal(results[0].ready,false);assert.equal(results[1].ready,false);
 for(const r of results.slice(2)) {
  assert.equal(r.ready,true);
  const d=liveGeometryDecision(slot(r.raw!.widthDifference),1000,1000,DEFAULT_ALIGNMENT_REFERENCE,r.ready,true,false,r.measured!);
  assert.equal(d.raw,'aligned');assert.equal(d.veto,null);assert.ok(d.vertical!<=.15);
 }
 const high=feed([.17,.17,.17]).at(-1)!;
 assert.equal(liveGeometryDecision(slot(.17),1000,1000,DEFAULT_ALIGNMENT_REFERENCE,high.ready,true,false,high.measured!).veto,'position');
});
test('a video-sized 2.9% to 28.3% jump cannot change guidance or finish capture',()=>{
 const r=feed([.029,.029,.029,.029,.283]);const last=r.at(-1)!;
 assert.equal(last.ready,false);assert.ok(Math.abs(last.measured!.widthDifference-.029)<1e-6);
 const d=liveGeometryDecision(slot(.283),1000,1000,DEFAULT_ALIGNMENT_REFERENCE,last.ready,true,false,last.measured!);
 assert.equal(d.raw,'steady');assert.equal(d.veto,'corners');
 let hold=emptyCaptureHold();for(const t of [0,400])hold=updateCaptureHold(hold,t,true);
 assert.equal(updateCaptureHold(hold,800,d.veto===null).ready,false);
});
test('a sustained real change replaces the old estimate and still blocks high views',()=>{
 const r=feed([.03,.03,.03,.03,.28,.29,.27]);
 assert.equal(r[4].ready,false);assert.equal(r[5].ready,false);assert.equal(r[6].ready,true);
 const d=liveGeometryDecision(slot(.27),1000,1000,DEFAULT_ALIGNMENT_REFERENCE,true,true,false,r[6].measured!);
 assert.equal(d.raw,'lower_phone');assert.equal(d.veto,'position');
 const back=feed([.28,.28,.28,.28,.06,.05,.04]);assert.equal(back.at(-1)!.ready,true);
 assert.ok(back.at(-1)!.measured!.widthDifference<.16);
});
test('alternating large jumps never complete a capture hold',()=>{
 const values=[.029,.283,.027,.317,.071,.272,.029,.283,.027,.317];let hold=emptyCaptureHold();
 for(const [i,r] of feed(values).entries()) {
  const d=liveGeometryDecision(slot(values[i]),1000,1000,DEFAULT_ALIGNMENT_REFERENCE,r.ready,true,false,r.measured!);
  hold=updateCaptureHold(hold,i*400,d.veto===null);assert.equal(hold.ready,false);
 }
});
test('missing or weak frames never advance the filter or renew history expiry',()=>{
 const last=feed([.1,.1,.1]).at(-1)!;
 const weak=slot(.1);weak.keypoints[0].confidence=.1;
 let r=stabilizeMeasurements(last.history,weak,1000,1000,1100,true);assert.equal(r.ready,false);assert.equal(r.history.length,3);
 r=stabilizeMeasurements(r.history,null,1000,1000,1500,true);assert.equal(r.ready,false);
 r=stabilizeMeasurements(r.history,null,1000,1000,2100,true);assert.equal(r.history.length,0);
 const resumed=stabilizeMeasurements(last.history,slot(.1),1000,1000,1200,true);assert.equal(resumed.ready,true);
});
test('new tracks, time reversals, and long gaps require new evidence',()=>{
 const old=feed([.1,.1,.1]).at(-1)!.history;
 for(const [now,same] of [[1000,false],[800,true],[2400,true]] as const) {
  const r=stabilizeMeasurements(old,slot(.1),1000,1000,now,same);assert.equal(r.history.length,1);assert.equal(r.ready,false);
 }
});
test('fresh box position and area are never smoothed away',()=>{
 const old=feed([.1,.1,.1]).at(-1)!.history;const moved=slot(.1);moved.box=[600,100,700,900];
 const r=stabilizeMeasurements(old,moved,1000,1000,1200,true);
 assert.equal(r.measured!.centerX,.65);
 const d=liveGeometryDecision(moved,1000,1000,DEFAULT_ALIGNMENT_REFERENCE,r.ready,true,false,r.measured!);
 assert.equal(d.raw,'right');assert.equal(d.veto,'position');
});

import test from 'node:test';
import assert from 'node:assert/strict';
import { evidenceGapLimit, freshInference, type CompletedInference } from '../lib/inference-timing.ts';
import { updateTrackingEvidence, recoverPoseReliability, type TrackingEvidence } from '../lib/tracking-continuity.ts';
import { stabilizeMeasurements, type MeasurementHistory } from '../lib/measurement-stability.ts';
import { DEFAULT_ALIGNMENT_REFERENCE, type Detection } from '../lib/pose.ts';
import { liveGeometryDecision, updateInstruction, emptyInstructionState, updateCaptureHold, emptyCaptureHold, chooseLiveGuidance } from '../lib/live-guidance.ts';
import { updateAcceptabilityWindow } from '../lib/acceptability.ts';
import { updateSideAngleState, emptySideAngleState, confirmedSideGuidance } from '../lib/side-angle.ts';
function slot(v=.1):Detection { return {box:[450,100,550,900],confidence:.9,angleClass:'centered',keypoints:[
 {x:450,y:100,confidence:1},{x:550,y:100,confidence:1},{x:550-50*v,y:900,confidence:1},{x:450+50*v,y:900,confidence:1}]}; }

test('eight matching frames can capture at 1.1 and 1.8 seconds per inference',()=>{
 for(const duration of [1100,1800]) {
  let timing:CompletedInference=null,track:TrackingEvidence=null,history:MeasurementHistory=[],count=0;
  let instruction=emptyInstructionState(),hold=emptyCaptureHold(),accept:number[]=[],lastAccept:number|null=null;
  let everCaptured=false;
  for(let i=0;i<8;i++) {
   const now=i*(duration+30),gap=evidenceGapLimit(timing,now),d=slot();
   const t=updateTrackingEvidence(track,d,true,now,gap);track=t.state;
   if(i)assert.equal(t.sameTrack,true);
   const p=recoverPoseReliability(count,true,t.retain);count=p.count;
   const s=stabilizeMeasurements(history,d,1000,1000,now,t.sameTrack,gap);history=s.history;
   const g=liveGeometryDecision(d,1000,1000,DEFAULT_ALIGNMENT_REFERENCE,p.ready,true,false,s.measured!,s.ready);
   let ready=g.veto===null;
   if(ready) {
    if(lastAccept===null||now-lastAccept>gap)accept=[];
    const a=updateAcceptabilityWindow(accept,.99);accept=a.history;lastAccept=now;ready=a.accepted;
   }
   const next=chooseLiveGuidance({raw:g.raw,frameReady:ready,veto:g.veto,confirmedSide:null,angleCorrection:null,heightConfirmed:false,acceptabilityCorrection:null});
   instruction=updateInstruction(instruction,next,now,false,g.raw==='aligned'&&next==='steady',gap);
   hold=updateCaptureHold(hold,now,ready&&instruction.guidance==='aligned',gap);
   everCaptured ||= hold.ready;
   timing={sampledAt:now,completedAt:now+duration};
  }
  assert.equal(everCaptured,true,`${duration} ms frames must not reset forever`);
 }
});
test('processing allowance does not excuse idle gaps, background pauses, or stale results',()=>{
 assert.equal(evidenceGapLimit({sampledAt:0,completedAt:1800},1830),2050);
 assert.equal(evidenceGapLimit({sampledAt:0,completedAt:1800},2400),1200);
 assert.equal(evidenceGapLimit({sampledAt:0,completedAt:4000},4030),1200);
 assert.equal(evidenceGapLimit(null,1830),1200);
 assert.equal(freshInference(0,1800),true);assert.equal(freshInference(0,2600),false);
 let h=updateCaptureHold(emptyCaptureHold(),0,true,2050);h=updateCaptureHold(h,1830,true,2050);
 assert.equal(updateCaptureHold(h,5000,true,1200).readings,1);
 const track=updateTrackingEvidence(null,slot(),true,0).state;
 assert.equal(updateTrackingEvidence(track,null,false,1830,2050).retain,false);
 const moved=slot();moved.box=[650,100,750,900];
 assert.equal(updateTrackingEvidence(track,moved,true,1830,2050).sameTrack,false);
});
test('movement guidance is available before capture stability, without permitting capture',()=>{
 const d=slot(.25);
 const g=liveGeometryDecision(d,1000,1000,DEFAULT_ALIGNMENT_REFERENCE,true,true,false,undefined,false);
 assert.equal(g.raw,'lower_phone');assert.equal(g.veto,'measurement_stability');
 let instruction=emptyInstructionState();
 for(const t of [0,1830,3660]) instruction=updateInstruction(instruction,g.raw,t,false,false,2050);
 assert.equal(instruction.guidance,'lower_phone');
 assert.equal(updateCaptureHold(emptyCaptureHold(),3660,g.veto===null,2050).ready,false);
});
test('confirmed side instructions survive slow consecutive processing',()=>{
 let state=emptySideAngleState();const d=slot();d.angleClass='camera_right';
 for(const t of [0,1830,3660,5490])state=updateSideAngleState(state,d,true,t,true,2050);
 assert.equal(confirmedSideGuidance(state),'step_left');assert.equal(state.blocked,true);
});
test('a slow failing frame still cancels capture and requires a fresh hold',()=>{
 let h=emptyCaptureHold();for(const t of [0,1830])h=updateCaptureHold(h,t,true,2050);
 h=updateCaptureHold(h,3660,false,2050);assert.equal(h.startedAt,null);
 for(const t of [5490,7320]){h=updateCaptureHold(h,t,true,2050);assert.equal(h.ready,false);}
 h=updateCaptureHold(h,9150,true,2050);assert.equal(h.ready,true);
});

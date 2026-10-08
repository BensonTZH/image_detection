import assert from "node:assert/strict";
import test from "node:test";
import { canPauseCapture, chooseLiveGuidance, emptyCaptureHold, emptyInstructionState, movementGuidance, liveGeometryDecision, liveSideCorrection, waitingGuidanceCopy, updateCaptureHold, updateInstruction, interruptInstruction, verificationStatus } from "../lib/live-guidance.ts";
import { freezeCameraFrame, encodeValidatedFrame } from "../lib/capture-frame.ts";
import { alignmentMeasurements, captureAlignmentVeto, DEFAULT_ALIGNMENT_REFERENCE, smoothDetection, recoverTemporaryLoss, recoveredDisplayDetection, type Detection, type Guidance } from "../lib/pose.ts";
import { emptyHighAngleState, updateHighAngleState } from "../lib/high-angle.ts";
import { confirmedSideGuidance, emptySideAngleState, updateSideAngleState } from "../lib/side-angle.ts";
import { updateTrackingEvidence, recoverPoseReliability, preserveAcceptabilityHistory, type TrackingEvidence } from "../lib/tracking-continuity.ts";
import { captureAcceptabilityVeto, updateAcceptabilityWindow } from "../lib/acceptability.ts";

test("alternating left, tilt and waiting readings do not flash directions", () => {
  let state = emptyInstructionState();
  const sequence: Guidance[] = ["step_left", "tilt_down", "steady", "step_left", "tilt_down", "steady", "step_left", "tilt_down"];
  sequence.forEach((next, i) => { state = updateInstruction(state, next, i * 250); assert.equal(state.guidance, "steady"); });
});

test("directions need sustained evidence, ties and isolated reversals keep the confirmed instruction", () => {
  let state = emptyInstructionState();
  for (const time of [0, 250, 500]) state = updateInstruction(state, "step_left", time);
  assert.equal(state.guidance, "step_left");
  for (const [next,time] of [["step_right",750],["tilt_down",1000],["step_right",1250],["tilt_down",1500]] as const) {
    state = updateInstruction(state,next,time);assert.equal(state.guidance,"step_left");
  }
  state = updateInstruction(state,"tilt_down",1750);
  state = updateInstruction(state,"tilt_down",2000);
  assert.equal(state.guidance,"tilt_down");
  state = updateInstruction(state,"steady",2250);assert.equal(state.guidance,"tilt_down");
  state = updateInstruction(state,"steady",2500);assert.equal(state.guidance,"tilt_down");
  state = updateInstruction(state,"aligned",2550);assert.equal(state.guidance,"aligned");
  state = updateInstruction(state,"aligned",2800);assert.equal(state.guidance,"aligned");
  state = updateInstruction(state,"searching",2850);assert.equal(state.guidance,"searching");
});

test("a burst of rapid readings cannot trigger a new direction immediately", () => {
  let state=emptyInstructionState();
  for(const time of [0,10,20,30,40]) state=updateInstruction(state,"tilt_down",time);
  assert.equal(state.guidance,"steady");
  state=updateInstruction(state,"tilt_down",400);assert.equal(state.guidance,"tilt_down");
});

test("an unsupported old direction expires to checking, never hold steady", () => {
  let state=emptyInstructionState();
  for(const time of [0,250,500]) state=updateInstruction(state,"step_left",time);
  for(const [i,next] of (["step_right","tilt_down","step_right","tilt_down","step_right"] as Guidance[]).entries()) {
    state=updateInstruction(state,next,750+i*250);
  }
  assert.equal(state.guidance,"steady");
});

test("hold steady is exclusive to passing gates and disappears on the first failed check", () => {
  let state=emptyInstructionState();
  for(const time of [0,300,600,900]) {
    state=updateInstruction(state,"steady",time);
    assert.notEqual(state.guidance,"aligned");
  }
  state=updateInstruction(state,"aligned",1000);assert.equal(state.guidance,"aligned");
  state=updateInstruction(state,"aligned",1200);assert.equal(state.guidance,"aligned");
  state=updateInstruction(state,"step_left",1500);assert.notEqual(state.guidance,"aligned");
  state=updateInstruction(state,"step_left",1800);
  state=updateInstruction(state,"step_left",2100);assert.equal(state.guidance,"step_left");
  state=updateInstruction(state,"steady",2400);assert.equal(state.guidance,"step_left");
  state=updateInstruction(state,"steady",2700);assert.equal(state.guidance,"step_left");
});

test("strong right-side readings block frontal-looking geometry and then instruct left", () => {
  const detection=slot(514);detection.angleClass="camera_right";detection.confidence=.87;
  assert.equal(captureAlignmentVeto(detection,1000,1000),null);
  let side=emptySideAngleState();
  for(const time of [0,300,600,900]) {
    side=updateSideAngleState(side,detection,true,time);
    assert.equal(side.blocked,true);
    assert.equal(captureAcceptabilityVeto(side.blocked ? "side_class" : null,{accepted:true}),"side_class");
    if(time<900)assert.equal(confirmedSideGuidance(side),null);
  }
  assert.equal(confirmedSideGuidance(side),"step_left");
  const instruction=updateInstruction(emptyInstructionState(),confirmedSideGuidance(side)!,900,true);
  assert.equal(instruction.guidance,"step_left");
  detection.angleClass="centered";
  side=updateSideAngleState(side,detection,true,1200);assert.equal(side.blocked,true);
  side=updateSideAngleState(side,detection,true,1500);assert.equal(side.blocked,false);
});

test("weak, isolated, mixed and stale side evidence cannot produce an invented direction", () => {
  const detection=slot(500);detection.angleClass="camera_left";detection.confidence=.84;
  let side=emptySideAngleState();
  for(const time of [0,300,600,900,1200]) side=updateSideAngleState(side,detection,true,time);
  assert.equal(side.blocked,false);assert.equal(confirmedSideGuidance(side),null);
  detection.confidence=.9;
  for(const time of [1500,1800,2100,2400,2700]) {
    detection.angleClass=detection.angleClass==='camera_left'?'camera_right':'camera_left';
    side=updateSideAngleState(side,detection,true,time);
    assert.equal(confirmedSideGuidance(side),null);
  }
  side=updateSideAngleState(side,null,false,3000);assert.equal(side.blocked,false);
  detection.angleClass='camera_left';
  for(const time of [3300,3600,3900,4200])side=updateSideAngleState(side,detection,true,time);
  assert.equal(confirmedSideGuidance(side),'step_right');
  side=updateSideAngleState(side,detection,true,6000);assert.equal(confirmedSideGuidance(side),null);
  assert.equal(updateSideAngleState(side,detection,true,6300,false).blocked,false);
});

test("passing checks share one 800 ms hold and a side conflict resets it", () => {
  let instruction=emptyInstructionState(),hold=emptyCaptureHold();
  for(const time of [0,300,600,900]) {
    instruction=updateInstruction(instruction,"aligned",time);
    hold=updateCaptureHold(hold,time,instruction.guidance==='aligned');
    assert.equal(hold.ready,time>=800);
  }
  instruction=updateInstruction(instruction,"aligned",1100);
  hold=updateCaptureHold(hold,1100,instruction.guidance==='aligned');assert.equal(hold.ready,true);
  const detection=slot(500);detection.angleClass='camera_right';detection.confidence=.9;
  const side=updateSideAngleState(emptySideAngleState(),detection,true,1200);
  hold=updateCaptureHold(hold,1200,!side.blocked);assert.equal(hold.ready,false);
  instruction=updateInstruction(instruction,"steady",1200);assert.notEqual(instruction.guidance,'aligned');
  instruction=updateInstruction(instruction,"aligned",1500);assert.equal(instruction.guidance,'aligned');
  instruction=updateInstruction(instruction,"aligned",3000);assert.equal(instruction.guidance,'aligned');
});


function slot(cx:number):Detection {
  return {box:[cx-50,100,cx+50,900],confidence:.9,angleClass:"centered",keypoints:[
    {x:cx-50,y:100,confidence:1},{x:cx+50,y:100,confidence:1},{x:cx+50,y:900,confidence:1},{x:cx-50,y:900,confidence:1}]};
}
test("centering has distinct trigger and release limits without weakening capture limits", () => {
  assert.equal(movementGuidance(slot(610),1000,1000,"steady",true),"right");
  assert.equal(movementGuidance(slot(590),1000,1000,"right",true),"right");
  assert.equal(movementGuidance(slot(590),1000,1000,"left",true),"aligned");
  assert.equal(movementGuidance(slot(570),1000,1000,"right",true),"aligned");
  assert.equal(captureAlignmentVeto(slot(610),1000,1000),"position");
  assert.equal(captureAlignmentVeto(slot(590),1000,1000),null);
});

test("capture waits 800 ms with at least three passing frames and restarts after failure", () => {
  let hold=emptyCaptureHold();
  hold=updateCaptureHold(hold,0,true);
  hold=updateCaptureHold(hold,400,true);assert.equal(hold.ready,false);
  hold=updateCaptureHold(hold,799,true);assert.equal(hold.ready,false);
  hold=updateCaptureHold(hold,800,false);assert.equal(hold.startedAt,null);
  for(const time of [1000,1400,1799]) {hold=updateCaptureHold(hold,time,true);assert.equal(hold.ready,false);}
  hold=updateCaptureHold(hold,1800,true);assert.equal(hold.ready,true);
  let sparse=updateCaptureHold(emptyCaptureHold(),0,true);
  sparse=updateCaptureHold(sparse,800,true);assert.equal(sparse.ready,false);
  sparse=updateCaptureHold(sparse,3000,true);assert.equal(sparse.startedAt,3000);assert.equal(sparse.ready,false);
});

test("a new high-angle frame cancels a nearly complete hold even while the UI is steady", () => {
  let high=emptyHighAngleState(),hold=emptyCaptureHold();
  for(const time of [0,300,600,900]) {high=updateHighAngleState(high,.1);hold=updateCaptureHold(hold,time,!high.blocked);}
  assert.equal(hold.ready,false);
  high=updateHighAngleState(high,.41);
  hold=updateCaptureHold(hold,1200,!high.blocked);
  assert.equal(hold.ready,false);assert.equal(hold.startedAt,null);
});

test("motion during inference cannot change the pixels exported from the evaluated frame", async () => {
  const video={videoWidth:720,videoHeight:1280,pixels:"acceptable"};
  let pixels="",draws=0;
  const canvas={width:0,height:0,getContext:()=>({drawImage:(source:typeof video)=>{pixels=source.pixels;draws++;}}),toDataURL:()=>`jpeg:${pixels}`};
  const frame=freezeCameraFrame(video as unknown as HTMLVideoElement,()=>canvas as unknown as HTMLCanvasElement);
  const evaluated=pixels;
  await Promise.resolve();video.pixels="unacceptable new angle";
  assert.equal(evaluated,"acceptable");assert.equal(encodeValidatedFrame(frame),"jpeg:acceptable");
  assert.equal(draws,1);assert.equal(frame.width,720);assert.equal(frame.height,1280);
});

function verticalSlot(perspective: number): Detection {
  const result = slot(449);
  const top = perspective >= 0 ? 100 : 100 * (1 + perspective);
  const bottom = perspective >= 0 ? 100 * (1 - perspective) : 100;
  result.keypoints[0].x = 449 - top / 2;
  result.keypoints[1].x = 449 + top / 2;
  result.keypoints[2].x = 449 + bottom / 2;
  result.keypoints[3].x = 449 - bottom / 2;
  return result;
}

test("+21% perspective blocks capture even with clear classifiers", () => {
  const detection = verticalSlot(.21);
  const veto = captureAlignmentVeto(detection, 1000, 1000);
  assert.equal(veto, "position");
  assert.equal(captureAcceptabilityVeto(veto, { accepted: true }), "position");
  let highAngle = emptyHighAngleState();
  for (let i = 0; i < 3; i++) highAngle = updateHighAngleState(highAngle, 0);
  assert.equal(highAngle.blocked, false);
  let instruction = emptyInstructionState();
  let hold = emptyCaptureHold();
  for (const time of [0, 300, 600, 900, 1200]) {
    const direction = movementGuidance(detection, 1000, 1000, instruction.guidance, true);
    assert.equal(direction, "lower_phone");
    instruction = updateInstruction(instruction, direction, time);
    hold = updateCaptureHold(hold, time, veto === null);
    assert.equal(hold.ready, false);
  }
  assert.equal(instruction.guidance, "lower_phone");
});

test("positive limit is 16% while negative perspective retains the 28% limit", () => {
  assert.equal(captureAlignmentVeto(verticalSlot(.159), 1000, 1000), null);
  assert.equal(captureAlignmentVeto(verticalSlot(.161), 1000, 1000), "position");
  assert.equal(captureAlignmentVeto(verticalSlot(-.20), 1000, 1000), null);
  assert.equal(movementGuidance(verticalSlot(-.29), 1000, 1000, "steady", true), "tilt_up");
  assert.equal(movementGuidance(verticalSlot(.13), 1000, 1000, "lower_phone", true), "aligned");
  assert.equal(movementGuidance(verticalSlot(.11), 1000, 1000, "lower_phone", true), "aligned");
});

test("a renewed high view resets the hold and a corrected view must complete a fresh hold", () => {
  let hold = emptyCaptureHold();
  for (const time of [0, 300, 600]) hold = updateCaptureHold(hold, time, true);
  hold = updateCaptureHold(hold, 800, captureAlignmentVeto(verticalSlot(.21), 1000, 1000) === null);
  assert.equal(hold.startedAt, null);
  for (const time of [1000, 1300, 1600]) {
    hold = updateCaptureHold(hold, time, captureAlignmentVeto(verticalSlot(.09), 1000, 1000) === null);
    assert.equal(hold.ready, false);
  }
  hold = updateCaptureHold(hold, 1800, true);
  assert.equal(hold.ready, true);
});


test("the earlier +16.1% example is blocked by the 16% geometry gate", () => {
  assert.equal(captureAlignmentVeto(verticalSlot(.161), 1000, 1000), "position");
  assert.equal(movementGuidance(verticalSlot(.161), 1000, 1000, "steady", true), "lower_phone");
});


const guidanceInput = { raw: "aligned" as Guidance, frameReady: false, veto: "high_angle" as const,
  confirmedSide: null, angleCorrection: null, heightConfirmed: true, acceptabilityCorrection: null };

test("confirmed left/right corrections precede height guidance without allowing capture", () => {
  for (const side of ["step_left", "step_right"] as const) {
    const direction = chooseLiveGuidance({ ...guidanceInput, confirmedSide: side });
    assert.equal(direction, side);
    const instruction = updateInstruction(emptyInstructionState(), direction, 1000, true);
    assert.equal(instruction.guidance, side);
    let hold = emptyCaptureHold();
    for (const time of [1000, 1400, 1800, 2200]) {
      hold = updateCaptureHold(hold, time, guidanceInput.frameReady && (["aligned"] as Guidance[]).includes(instruction.guidance));
      assert.equal(hold.ready, false);
    }
    assert.equal(chooseLiveGuidance({ ...guidanceInput, raw: side }), side);
    assert.equal(chooseLiveGuidance({ ...guidanceInput, angleCorrection: side }), side);
  }
  assert.equal(chooseLiveGuidance(guidanceInput), "lower_phone");
  assert.equal(chooseLiveGuidance({ ...guidanceInput, frameReady: true, veto: null }), "aligned");
  assert.equal(chooseLiveGuidance({ ...guidanceInput, raw: "searching", confirmedSide: "step_left" }), "searching");
});

test("pending classifier checks keep useful framing instructions and unknown stays checking", () => {
  for (const veto of ["side_class", "angle_vote", "high_angle"] as const) {
    for (const raw of ["left", "right", "closer", "farther"] as const) {
      assert.equal(chooseLiveGuidance({ ...guidanceInput, veto, raw, heightConfirmed: false }), raw);
    }
    assert.equal(chooseLiveGuidance({ ...guidanceInput, veto, heightConfirmed: false }), "steady");
  }
});

test("one unknown reading preserves pending correction evidence but does not count as evidence", () => {
  let state = updateInstruction(emptyInstructionState(), "step_left", 0);
  state = updateInstruction(state, "step_left", 300);
  state = updateInstruction(state, "steady", 600);
  assert.equal(state.guidance, "steady");
  assert.equal(state.readings, 2);
  state = updateInstruction(state, "step_left", 900);
  assert.equal(state.guidance, "step_left");
  state = updateInstruction(state, "steady", 1200);
  assert.equal(state.guidance, "step_left");
  state = updateInstruction(state, "steady", 2201);
  assert.equal(state.guidance, "steady");
});

test("stale or contradictory pending evidence cannot confirm an instruction", () => {
  let state = updateInstruction(emptyInstructionState(), "step_left", 0);
  state = updateInstruction(state, "step_left", 300);
  state = updateInstruction(state, "steady", 901);
  state = updateInstruction(state, "step_left", 1000);
  assert.equal(state.guidance, "steady");
  assert.equal(state.readings, 1);
  state = updateInstruction(state, "step_right", 1300);
  state = updateInstruction(state, "step_left", 1600);
  assert.equal(state.readings, 1);
  state = updateInstruction(state, "step_left", 3000);
  assert.equal(state.readings, 1);
  state = updateInstruction(state, "searching", 3300);
  assert.equal(state.pending, null);
});

test("a passing reading shows hold steady and an unknown reading removes it", () => {
  let state = emptyInstructionState();
  for (const time of [0, 300, 600]) state = updateInstruction(state, "lower_phone", time);
  state = updateInstruction(state, "aligned", 900);
  assert.equal(state.guidance, "aligned");
  state = updateInstruction(state, "steady", 1200);
  assert.equal(state.guidance, "steady");
  state = updateInstruction(state, "aligned", 1500);
  assert.equal(state.guidance, "aligned");
  state = updateInstruction(state, "aligned", 1800);
  assert.equal(state.guidance, "aligned");
});


test("a retained display box cannot keep an imminent capture alive", () => {
  const previous = slot(500);
  let hold = emptyCaptureHold();
  for (const time of [0, 300, 600]) hold = updateCaptureHold(hold, time, true);
  const recovery = recoverTemporaryLoss(previous, null, 0, 600, 900);
  const display = recoveredDisplayDetection(previous, null, recovery, 600, 900);
  assert.equal(display, previous);
  assert.equal(recovery.state, "fading");
  const veto = captureAlignmentVeto(null, 1000, 1000);
  assert.equal(veto, "no_detection");
  hold = updateCaptureHold(hold, 900, veto === null);
  assert.equal(hold.ready, false); assert.equal(hold.startedAt, null);
  for (const time of [1100, 1400, 1700]) {
    hold = updateCaptureHold(hold, time, captureAlignmentVeto(previous, 1000, 1000) === null);
    assert.equal(hold.ready, false);
  }
  hold = updateCaptureHold(hold, 1900, true);
  assert.equal(hold.ready, true);
});


test("verification remains secondary to movement and is absent during accepted hold", () => {
  assert.equal(verificationStatus("step_left", "high_angle"), "Checking phone height…");
  assert.equal(verificationStatus("steady", "position"), "Verifying angle…");
  assert.equal(verificationStatus("steady", null), "Confirming alignment…");
  assert.equal(verificationStatus("aligned", null), null);
  assert.equal(verificationStatus("searching", "no_detection"), null);
  assert.equal(verificationStatus("searching", "no_detection", true), "Tracking interrupted — waiting for a fresh reading…");
});

// These sequences exercise recovery without ever treating a retained box as a
// new detection or allowing a missing/weak frame to advance capture.
test("brief same-slot corner failure retains evidence but cannot pass capture", () => {
  const detection = slot(500);
  let tracking: TrackingEvidence = null, count = 0;
  for (const time of [0, 400]) {
    const next = updateTrackingEvidence(tracking, detection, true, time);
    tracking = next.state;
    count = recoverPoseReliability(count, true, next.retain).count;
  }
  assert.equal(count, 2);
  const weak = { ...detection, keypoints: detection.keypoints.map(p => ({...p, confidence:.1})) as Detection['keypoints'] };
  const gap = updateTrackingEvidence(tracking, weak, false, 800);
  const suspended = recoverPoseReliability(count, false, gap.retain);
  assert.equal(suspended.count, 2);
  assert.equal(suspended.ready, false);
  assert.equal(captureAlignmentVeto(weak, 1000, 1000, undefined, suspended.ready), 'corners');
  const resumed = updateTrackingEvidence(gap.state, slot(505), true, 1200);
  assert.equal(resumed.retain, true);
  assert.equal(recoverPoseReliability(suspended.count, true, resumed.retain).ready, true);
});

test("missing frames never extend the evidence expiry and a returning different box starts over", () => {
  const original = updateTrackingEvidence(null, slot(500), true, 100).state;
  const gap = updateTrackingEvidence(original, null, false, 600);
  assert.equal(gap.retain, true);
  assert.equal(gap.state, original);
  assert.equal(updateTrackingEvidence(gap.state, null, false, 1101).retain, false);
  assert.equal(updateTrackingEvidence(gap.state, slot(750), true, 800).retain, false);
  const larger = slot(500); larger.box = [390,0,610,1000];
  assert.equal(updateTrackingEvidence(original, larger, true, 800).retain, false);
  assert.equal(updateTrackingEvidence(original, slot(500), true, 1500).retain, false);
  assert.equal(updateTrackingEvidence(original, slot(500), true, 100).retain, false);
});

test("repeated weak corners cannot keep old geometry evidence alive indefinitely", () => {
  let tracking = updateTrackingEvidence(null, slot(500), true, 100).state;
  for (const time of [400, 700, 1000]) {
    const next = updateTrackingEvidence(tracking, slot(500), false, time);
    assert.equal(next.retain, true); tracking = next.state;
  }
  const expired = updateTrackingEvidence(tracking, slot(500), false, 1200);
  assert.equal(expired.retain, false);
  assert.equal(recoverPoseReliability(2, false, expired.retain).count, 0);
  assert.equal(updateTrackingEvidence(expired.state, slot(500), true, 1400).retain, false);
});

test("tracking interruptions preserve a recent direction without renewing it or preserving accepted hold", () => {
  let state = emptyInstructionState();
  for (const time of [0, 250, 500]) state = updateInstruction(state, 'step_left', time);
  state = interruptInstruction(state, 900);
  assert.equal(state.guidance, 'step_left');
  assert.equal(state.lastSupportedAt, 500);
  assert.equal(interruptInstruction(state, 1800).guidance, 'steady');
  state = updateInstruction(state, 'aligned', 1000);
  state = interruptInstruction(state, 1200);
  assert.notEqual(state.pending, 'aligned');
  state = updateInstruction(state, 'aligned', 1400);
  assert.equal(state.guidance, 'aligned');
  state = updateInstruction(state, 'aligned', 1700);
  assert.equal(state.guidance, 'aligned');
  assert.equal(interruptInstruction(state, 1800).guidance, 'steady');
});

test("acceptability history survives only short uncertainty and a current rejection still blocks", () => {
  assert.equal(preserveAcceptabilityHistory(true, true, 100, 900), true);
  assert.equal(preserveAcceptabilityHistory(true, false, 100, 900), false);
  assert.equal(preserveAcceptabilityHistory(false, true, 100, 900), false);
  assert.equal(preserveAcceptabilityHistory(true, true, 100, 1101), false);
  assert.equal(preserveAcceptabilityHistory(true, true, null, 900), false);
  const resumed = updateAcceptabilityWindow([.95,.96,.98], .99);
  assert.equal(resumed.accepted, true);
  const rejected = updateAcceptabilityWindow([.95,.96,.98], .2);
  assert.equal(rejected.accepted, false);
  assert.equal(captureAcceptabilityVeto(null, rejected), 'acceptability');
});

test("retained classifier history resumes quickly but still requires a completely new capture hold", () => {
  let high = emptyHighAngleState();
  for (let i=0;i<3;i++) high = updateHighAngleState(high,.1);
  let hold = emptyCaptureHold();
  for (const time of [0,300,600]) hold = updateCaptureHold(hold,time,true);
  hold = updateCaptureHold(hold,800,false);
  assert.equal(hold.startedAt,null);
  // Resume with a fresh classifier result, not the saved probability alone.
  high = updateHighAngleState(high,.1);
  assert.equal(high.blocked,false);
  let instruction = interruptInstruction(updateInstruction(updateInstruction(emptyInstructionState(),'aligned',0),'aligned',300),800);
  for (const time of [1000,1300,1600,1900,2100]) {
    instruction = updateInstruction(instruction,'aligned',time);
    hold = updateCaptureHold(hold,time,!high.blocked && instruction.guidance==='aligned');
    assert.equal(hold.ready,time>=1800);
  }
  high=updateHighAngleState(high,.7);
  assert.equal(updateCaptureHold(hold,2200,!high.blocked).ready,false);
});


test("fresh geometry passes below 16% even when the display still contains an older high view", () => {
  const old = verticalSlot(.28), current = verticalSlot(.15);
  const display = smoothDetection(old, current, .3);
  assert.equal(movementGuidance(display,1000,1000,"lower_phone",true),"lower_phone");
  const decision = liveGeometryDecision(current,1000,1000,DEFAULT_ALIGNMENT_REFERENCE,true,true,false);
  assert.equal(decision.raw,"aligned");
  assert.equal(decision.veto,null);
  assert.ok(Math.abs(decision.vertical!-.15)<1e-8);
  const high = liveGeometryDecision(old,1000,1000,DEFAULT_ALIGNMENT_REFERENCE,true,true,false);
  assert.equal(high.raw,"lower_phone");
  assert.equal(high.veto,"position");
});

test("a resolved geometry correction clears while classifier confirmation is still pending", () => {
  let state=emptyInstructionState();
  for(const time of [0,300,600]) state=updateInstruction(state,"lower_phone",time);
  const geometry=liveGeometryDecision(verticalSlot(.15),1000,1000,DEFAULT_ALIGNMENT_REFERENCE,true,true,false);
  const next=chooseLiveGuidance({raw:geometry.raw,frameReady:false,veto:"acceptability",confirmedSide:null,angleCorrection:null,heightConfirmed:false,acceptabilityCorrection:null});
  state=updateInstruction(state,next,900,false,geometry.raw==="aligned" && next==="steady");
  assert.equal(state.guidance,"steady");
  assert.equal(state.pending,null);
  assert.equal(updateCaptureHold(emptyCaptureHold(),900,false).ready,false);
});

test("alternating passing and failing geometry cannot confirm hold or capture", () => {
  let instruction=emptyInstructionState(),hold=emptyCaptureHold();
  for(let i=0;i<12;i++) {
    const geometry=liveGeometryDecision(verticalSlot(i%2?.17:.15),1000,1000,DEFAULT_ALIGNMENT_REFERENCE,true,true,false);
    instruction=updateInstruction(instruction,geometry.raw,i*400);
    if(geometry.veto===null)assert.equal(instruction.guidance,"aligned");
    hold=updateCaptureHold(hold,i*400,geometry.veto===null&&instruction.guidance==="aligned");
    assert.equal(hold.ready,false);
  }
});

function sizedVerticalSlot(area: number, perspective: number): Detection {
  const result = verticalSlot(perspective);
  const cx = 500, halfWidth = area * 1000 * 1000 / 800 / 2;
  result.box = [cx - halfWidth, 100, cx + halfWidth, 900];
  result.keypoints[0].x = cx - halfWidth;
  result.keypoints[1].x = cx + halfWidth;
  result.keypoints[2].x = cx + halfWidth * (1 - perspective);
  result.keypoints[3].x = cx - halfWidth * (1 - perspective);
  return result;
}

test("oversized high views request distance first, then reassess height without bypassing capture", () => {
  const decide = (area: number, perspective: number) => liveGeometryDecision(
    sizedVerticalSlot(area, perspective), 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, true, false);
  const close = decide(.22, .25);
  assert.equal(close.raw, "farther");
  assert.equal(close.veto, "position");
  assert.equal(chooseLiveGuidance({ ...guidanceInput, raw: close.raw }), "farther");
  assert.equal(chooseLiveGuidance({ ...guidanceInput, raw: close.raw, confirmedSide: "step_left" }), "step_left");
  // Distance alone must not clear a remaining high view.
  const stillHigh = decide(.10, .25);
  assert.equal(stillHigh.raw, "lower_phone");
  assert.equal(stillHigh.veto, "position");
  // If backing away also resolves perspective, all geometry gates may pass.
  const corrected = decide(.10, .12);
  assert.equal(corrected.raw, "aligned");
  assert.equal(corrected.veto, null);
  // An independent confirmed height classifier must still prevent capture.
  assert.equal(chooseLiveGuidance({ ...guidanceInput, raw: corrected.raw }), "lower_phone");
  assert.equal(chooseLiveGuidance({ ...guidanceInput, raw: corrected.raw, frameReady: true, veto: null }), "aligned");
});

test("normal-size high views never invent a move-back instruction and keep the 16% boundary", () => {
  for (const area of [.04, .10, .19]) {
    const blocked = liveGeometryDecision(sizedVerticalSlot(area, .161), 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, true, false);
    assert.equal(blocked.raw, "lower_phone");
    assert.notEqual(blocked.veto, null);
  }
  assert.equal(DEFAULT_ALIGNMENT_REFERENCE.maximumPositiveVerticalPerspectiveDifference, .16);
});

test("distance correction requires sustained readings and cannot carry an old capture countdown", () => {
  let instruction = emptyInstructionState();
  let hold = emptyCaptureHold();
  for (const at of [0, 300, 600]) {
    instruction = updateInstruction(instruction, "farther", at);
    hold = updateCaptureHold(hold, at, false);
    assert.equal(hold.ready, false);
    if (at < 600) assert.notEqual(instruction.guidance, "farther");
  }
  assert.equal(instruction.guidance, "farther");
  for (const at of [900, 1200, 1500, 1800, 2100]) {
    instruction = updateInstruction(instruction, "aligned", at);
    hold = updateCaptureHold(hold, at, instruction.guidance === "aligned");
    assert.equal(hold.ready, at >= 1700);
  }
});

// Reproduce the video's 12% -> 0.1% change. Both poses pass even though the
// old median-agreement filter considered the change unstable.
test("different acceptable measurements complete one hold without a stability warmup", () => {
  let hold = emptyCaptureHold(), instruction = emptyInstructionState();
  for (const [i, vertical] of [.12, .001, .15].entries()) {
    const geometry = liveGeometryDecision(verticalSlot(vertical), 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, true, false);
    const acceptability = updateAcceptabilityWindow([], .99, 1, 1);
    const eligible = captureAcceptabilityVeto(geometry.veto, acceptability) === null;
    assert.equal(eligible, true);
    instruction = updateInstruction(instruction, "aligned", i * 400);
    assert.equal(instruction.guidance, "aligned");
    hold = updateCaptureHold(hold, i * 400, eligible);
    assert.equal(hold.ready, i === 2);
  }
  // An old filtered estimate must not let a new out-of-range frame pass.
  const failed = liveGeometryDecision(verticalSlot(.17), 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, true, false);
  assert.equal(updateCaptureHold(hold, 1200, failed.veto === null).ready, false);
});


test("a brief missing or disagreeing reading pauses without adding capture time", () => {
  let hold=emptyCaptureHold();
  hold=updateCaptureHold(hold,0,true);
  assert.equal(hold.confirmed,false);
  hold=updateCaptureHold(hold,400,true);
  assert.equal(hold.confirmed,true);
  hold=updateCaptureHold(hold,800,false,1200,true);
  assert.equal(hold.paused,true);assert.equal(hold.ready,false);
  assert.equal(hold.validMs,400);assert.equal(hold.readings,2);
  hold=updateCaptureHold(hold,1200,true);
  assert.equal(hold.ready,false);assert.equal(hold.validMs,400);
  hold=updateCaptureHold(hold,1600,true);
  assert.equal(hold.ready,true);assert.equal(hold.validMs,800);
});

test("uncertainty expires from the last passing frame and cannot renew itself", () => {
  let hold=emptyCaptureHold();
  for(const at of [0,400,800])hold=updateCaptureHold(hold,at,true);
  assert.equal(hold.ready,true);
  for(const at of [1000,1300,1600]) {
    hold=updateCaptureHold(hold,at,false,1200,true);
    assert.equal(hold.ready,false);assert.equal(hold.lastAt,800);
  }
  hold=updateCaptureHold(hold,2001,false,1200,true);
  assert.equal(hold.startedAt,null);assert.equal(hold.confirmed,false);
  assert.equal(updateCaptureHold(hold,2200,true).ready,false);
});

test("confirmed failures, new tracks and stale frames cannot use the uncertainty grace", () => {
  for(const veto of ['high_angle','side_class','acceptability','stale_frame'] as const)
    assert.equal(canPauseCapture(veto,null,true,true),false);
  assert.equal(canPauseCapture('position','position',true,true),false);
  assert.equal(canPauseCapture('position',null,true,true),true);
  assert.equal(canPauseCapture('no_detection','no_detection',true,true),true);
  assert.equal(canPauseCapture('corners','corners',true,true),true);
  assert.equal(canPauseCapture('no_detection',null,false,true),false);
  assert.equal(canPauseCapture('position',null,true,false),false);
});

test("one acceptable reading cannot replace a confirmed movement instruction", () => {
  let instruction=emptyInstructionState(),hold=emptyCaptureHold();
  for(const at of [0,400,800])instruction=updateInstruction(instruction,'lower_phone',at);
  hold=updateCaptureHold(hold,1200,true);
  instruction=updateInstruction(instruction,hold.confirmed?'aligned':'steady',1200);
  assert.equal(instruction.guidance,'lower_phone');
  hold=updateCaptureHold(hold,1600,true);
  instruction=updateInstruction(instruction,hold.confirmed?'aligned':'steady',1600);
  assert.equal(instruction.guidance,'aligned');assert.equal(hold.ready,false);
  hold=updateCaptureHold(hold,2000,true);assert.equal(hold.ready,true);
});

function sideSlot(difference: number): Detection {
  const detection=slot(500);
  detection.box=[400,100,600,900];
  detection.keypoints[0].x=detection.keypoints[3].x=400;
  detection.keypoints[1].x=detection.keypoints[2].x=600;
  if(difference<0) detection.keypoints[3].y+=800*difference;
  else detection.keypoints[2].y-=800*difference;
  return detection;
}

test('side vote waiting gives a sustained correction, then hands off to the shared photo hold',()=>{
  let instruction=emptyInstructionState(),hold=emptyCaptureHold();
  for(let i=0;i<16;i++) {
    const detection=sideSlot(i<12?-.10:0);
    const geometry=liveGeometryDecision(detection,1000,1000,DEFAULT_ALIGNMENT_REFERENCE,true,false,false);
    assert.equal(geometry.raw,'aligned');
    assert.equal(geometry.veto,i<12?'angle_vote':null);
    const correction=liveSideCorrection(detection,1000,1000,instruction.guidance,true,undefined,geometry.veto,geometry.veto,false);
    hold=updateCaptureHold(hold,i*400,geometry.veto===null);
    const next=chooseLiveGuidance({raw:geometry.raw,currentRaw:geometry.raw,frameReady:hold.confirmed,veto:geometry.veto,
      confirmedSide:null,angleCorrection:correction,heightConfirmed:false,acceptabilityCorrection:null});
    instruction=updateInstruction(instruction,next,i*400,false,false,1200,geometry.veto===null);
    if(i>=2&&i<12)assert.equal(instruction.guidance,'step_left','do not expire a still-supported correction');
    if(i<14)assert.equal(hold.ready,false);
    if(i>=13)assert.equal(instruction.guidance,'aligned');
    if(i>=14)assert.equal(hold.ready,true);
  }
});

test('raw geometry veto supplies a correction without allowing one-frame instruction flashes',()=>{
  let instruction=emptyInstructionState();
  const filtered=alignmentMeasurements(verticalSlot(.12),1000,1000);
  for(let i=0;i<8;i++) {
    const detection=verticalSlot(i<5?(i%2?.12:.21):.21);
    const current=liveGeometryDecision(detection,1000,1000,DEFAULT_ALIGNMENT_REFERENCE,true,true,false);
    const median=liveGeometryDecision(detection,1000,1000,DEFAULT_ALIGNMENT_REFERENCE,true,true,false,filtered);
    const next=chooseLiveGuidance({raw:median.raw,currentRaw:current.raw,frameReady:false,veto:current.veto,
      confirmedSide:null,angleCorrection:null,heightConfirmed:false,acceptabilityCorrection:null});
    instruction=updateInstruction(instruction,next,i*400,false,false,1200,current.veto===null);
    if(i<5)assert.equal(instruction.guidance,'steady');
    if(i===7)assert.equal(instruction.guidance,'lower_phone');
  }
});

test('side retention needs fresh corroboration and cannot keep pushing left after resolution or reversal',()=>{
  const near=sideSlot(-.07), resolved=sideSlot(0), opposite=sideSlot(.1);
  const filtered=alignmentMeasurements(near,1000,1000);
  assert.equal(liveSideCorrection(near,1000,1000,'step_left',true,filtered,null,null,false),'step_left');
  assert.equal(liveSideCorrection(resolved,1000,1000,'step_left',true,filtered,null,null,false),null);
  assert.equal(liveSideCorrection(opposite,1000,1000,'step_left',true,filtered,null,null,false),null);
  assert.equal(liveSideCorrection(near,1000,1000,'step_left',false,filtered,null,null,false),null);
  let state=updateInstruction(emptyInstructionState(),'step_left',0,true);
  state=updateInstruction(state,'steady',1300);
  assert.equal(state.guidance,'steady','no indefinite stale instruction');
});

test('visible-slot uncertainty never claims the opening is missing or the angle has passed',()=>{
  for(const veto of ['corners','stale_frame','side_class','angle_vote','acceptability','high_angle',null] as const) {
    const copy=waitingGuidanceCopy(veto,true);
    assert.notEqual(copy.title,'Keep the slot in view');
    assert.notEqual(copy.title,'Hold steady for photo');
    assert.doesNotMatch(copy.title,/checking|calculating/i);
  }
  assert.equal(waitingGuidanceCopy('no_detection',false).title,'Keep the slot in view');
  assert.equal(waitingGuidanceCopy('acceptability',true).title,'Face the slot straight on');
});

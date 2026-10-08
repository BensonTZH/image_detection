// Replay static photos through the actual decoder, guidance and capture gates.
// This verifies gate interactions; it cannot estimate handheld camera stability.
import fs from 'node:fs';
import { captureAlignmentVeto, DEFAULT_ALIGNMENT_REFERENCE, decodePoseOutput, letterboxGeometry,
  poseCornersReliable, guidanceFor, updateAngleCaptureWindow, updatePoseReliabilityStreak,
  updateAlignmentHold, stabilizeGuidance, acceptabilityCorrectionFor, sideAngleCorrectionFor,
  type AngleClass, type Guidance, type CaptureVeto } from '../../web/lib/pose.ts';
import { captureAcceptabilityVeto, updateAcceptabilityWindow } from '../../web/lib/acceptability.ts';
import { emptyHighAngleState, updateHighAngleState } from '../../web/lib/high-angle.ts';
import { emptySideAngleState, updateSideAngleState, confirmedSideGuidance } from '../../web/lib/side-angle.ts';

import { emptyCaptureHold, emptyInstructionState, movementGuidance, updateCaptureHold, updateInstruction } from '../../web/lib/live-guidance.ts';

const [input, output] = process.argv.slice(2);
const rows = JSON.parse(fs.readFileSync(input, 'utf8')).images;
function simulate(row: any, score: number, revised: boolean) {
  const detection = decodePoseOutput(new Float32Array(row.pose), 1, letterboxGeometry(row.width,row.height),row.width,row.height)[0] ?? null;
  let instruction = emptyInstructionState(), revisedHold = emptyCaptureHold();
  let angles: AngleClass[] = [], guides: Guidance[] = [], accepted: number[] = [];
  let corners = 0, high = emptyHighAngleState(), side = emptySideAngleState(), started: number | null = null;
  let veto: string | null = null, guide: Guidance = 'searching';
  for (let frame = 0; frame < 12; frame++) {
    const reliability = updatePoseReliabilityStreak(corners, !!detection && poseCornersReliable(detection));
    corners = reliability.count;
    const votes = updateAngleCaptureWindow(angles, detection?.angleClass ?? null); angles = votes.history;
    const raw = revised ? movementGuidance(detection,row.width,row.height,instruction.guidance,reliability.ready) : guidanceFor(detection,row.width,row.height,DEFAULT_ALIGNMENT_REFERENCE,reliability.ready);
    let current = captureAlignmentVeto(detection,row.width,row.height,DEFAULT_ALIGNMENT_REFERENCE,reliability.ready,votes.ready,true,!!votes.sideDirection);
    if(revised) {
      side=updateSideAngleState(side,detection,reliability.ready,frame*300);
      if(side.blocked)current='side_class';
    }
    let correction: Guidance | null = null;
    if (!current) {
      const window = updateAcceptabilityWindow(accepted,score); accepted = window.history;
      current = captureAcceptabilityVeto<CaptureVeto>(current,window);
      if (current === 'acceptability') correction = acceptabilityCorrectionFor(detection,row.width,row.height,revised ? instruction.guidance : "steady");
    } else accepted = [];
    high = updateHighAngleState(high,detection ? row.high : null);
    if (detection && high.blocked) current = 'high_angle';
    let next: Guidance = current === 'acceptability' && correction ? correction : raw === 'aligned' ? 'steady' : raw;
    if (current === 'angle_vote') next = sideAngleCorrectionFor(detection,row.width,row.height,revised ? instruction.guidance : "steady") ?? 'steady';
    if (current === 'side_class') next = confirmedSideGuidance(side) ?? 'steady';
    if (current === 'high_angle') next = high.lowerPhone ? 'lower_phone' : raw === 'aligned' ? 'steady' : raw;
    let ready = false;
    if(revised) {
      if(current===null) next='aligned';
      instruction=updateInstruction(instruction,next,frame*300,current==='side_class'&&side.direction!==null);guide=instruction.guidance;
      revisedHold=updateCaptureHold(revisedHold,frame*300,current===null&&guide==='aligned');
      ready=revisedHold.ready;
    } else {
      const stabilized=stabilizeGuidance(guides,next,5);guides=stabilized.history;guide=stabilized.guidance;
      const hold=updateAlignmentHold(started,frame*300,current===null&&(guide==='steady'||guide==='aligned'),300);
      started=hold.startedAt;ready=hold.ready;
    }
    veto=current;
    if (ready) return {captured:true,frame,guide,veto,angle:detection?.angleClass};
  }
  return {captured:false,guide,veto,angle:detection?.angleClass};
}
const results=rows.map((r:any)=>({...r,pose:undefined,
 currentResult:simulate({...r,high:r.current},r.acceptability,true),
 candidateResult:simulate({...r,high:r.candidate},r.acceptability,true)}));
const summary:Record<string,unknown>={};
for(const split of ['train','val','test']) for(const label of ['acceptable','high_angle']) for(const variant of ['full','near','close']) {
 const rr=results.filter((r:any)=>r.split===split&&r.label===label&&r.variant===variant);
 summary[`${split}_${label}_${variant}`]={total:rr.length,currentCapture:rr.filter((r:any)=>r.currentResult.captured).length,candidateCapture:rr.filter((r:any)=>r.candidateResult.captured).length};
}
fs.writeFileSync(output,JSON.stringify({summary,results,limitations:'Static repeated-frame replay; both paths use current 800ms capture hold, side gate and 40% high-angle threshold. Input images rendered by Chromium. This does not reproduce handheld temporal motion.'},null,2));
console.log(JSON.stringify(summary,null,2));

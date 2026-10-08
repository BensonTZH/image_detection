// Replay static photos through the actual decoder, guidance and capture gates.
// This verifies gate interactions; it cannot estimate handheld camera stability.
import fs from 'node:fs';
import { captureAlignmentVeto, DEFAULT_ALIGNMENT_REFERENCE, decodePoseOutput, letterboxGeometry,
  poseCornersReliable, guidanceFor, updateAngleCaptureWindow, updatePoseReliabilityStreak,
  updateAlignmentHold, stabilizeGuidance, acceptabilityCorrectionFor, sideAngleCorrectionFor,
  type AngleClass, type Guidance, type CaptureVeto } from '../lib/pose.ts';
import { captureAcceptabilityVeto, updateAcceptabilityWindow } from '../lib/acceptability.ts';
import { emptyHighAngleState, updateHighAngleState } from '../lib/high-angle.ts';

const [input, output] = process.argv.slice(2);
const rows = JSON.parse(fs.readFileSync(input, 'utf8'));
function simulate(row: any, score: number) {
  const detection = decodePoseOutput(new Float32Array(row.pose), 1, letterboxGeometry(row.width,row.height),row.width,row.height)[0] ?? null;
  let angles: AngleClass[] = [], guides: Guidance[] = [], accepted: number[] = [];
  let corners = 0, high = emptyHighAngleState(), started: number | null = null;
  let veto: string | null = null, guide: Guidance = 'searching';
  for (let frame = 0; frame < 12; frame++) {
    const reliability = updatePoseReliabilityStreak(corners, !!detection && poseCornersReliable(detection));
    corners = reliability.count;
    const votes = updateAngleCaptureWindow(angles, detection?.angleClass ?? null); angles = votes.history;
    const raw = guidanceFor(detection,row.width,row.height,DEFAULT_ALIGNMENT_REFERENCE,reliability.ready);
    let current = captureAlignmentVeto(detection,row.width,row.height,DEFAULT_ALIGNMENT_REFERENCE,reliability.ready,votes.ready,true,!!votes.sideDirection);
    let correction: Guidance | null = null;
    if (!current) {
      const window = updateAcceptabilityWindow(accepted,score); accepted = window.history;
      current = captureAcceptabilityVeto<CaptureVeto>(current,window);
      if (current === 'acceptability') correction = acceptabilityCorrectionFor(detection,row.width,row.height);
    } else accepted = [];
    high = updateHighAngleState(high,detection ? row.high : null);
    if (detection && high.blocked) current = 'high_angle';
    let next: Guidance = current === 'acceptability' && correction ? correction : raw === 'aligned' ? 'steady' : raw;
    if (current === 'angle_vote') next = sideAngleCorrectionFor(detection,row.width,row.height) ?? 'steady';
    if (current === 'high_angle') next = high.lowerPhone ? 'lower_phone' : raw === 'aligned' ? 'steady' : raw;
    const stabilized = stabilizeGuidance(guides,next,5); guides = stabilized.history; guide = stabilized.guidance;
    const hold = updateAlignmentHold(started,frame*300,current===null && (guide==='steady'||guide==='aligned'));
    started = hold.startedAt; veto = current;
    if (hold.ready) return {captured:true,frame,guide,veto,angle:detection?.angleClass};
  }
  return {captured:false,guide,veto,angle:detection?.angleClass};
}
const results = rows.map((r: any) => ({...r, pose:undefined, previousResult:simulate(r,r.previous), candidateResult:simulate(r,r.candidate)}));
const summary: Record<string, unknown> = {};
for (const cohort of ['original','addon','high_angle']) for (const split of ['train','val','test']) {
  for (const label of ['success','error','high_angle']) {
    const selected = results.filter((r: any) => r.cohort===cohort && r.split===split && r.label===label);
    if (selected.length) summary[`${cohort}_${split}_${label}`] = {total:selected.length,
      previousCaptured:selected.filter((r: any)=>r.previousResult.captured).length,
      candidateCaptured:selected.filter((r: any)=>r.candidateResult.captured).length};
  }
}
fs.writeFileSync(output,JSON.stringify({summary,limitations:'Static repeated-frame replay using CPU ONNX and Pillow letterbox; previous/candidate use the same corrected guidance rules. Browser resize and handheld motion may differ.',results},null,2));
console.log(JSON.stringify(summary,null,2));

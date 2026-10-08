import { alignmentMeasurements, poseCornerFailure, type Detection } from './pose.ts';

export type Measurements = ReturnType<typeof alignmentMeasurements>;
type Sample = { at: number; measured: Measurements };
export type MeasurementHistory = Sample[];
const WINDOW_MS = 2400;
const MAX_GAP_MS = 1200;
const median = (values: number[]) => {
  const sorted = [...values].sort((a,b) => a-b);
  const middle = Math.floor(sorted.length/2);
  return sorted.length%2 ? sorted[middle] : (sorted[middle-1]+sorted[middle])/2;
};
// Percentage-point tolerances identify large prediction jumps, not acceptable
// camera angles. The existing 16% capture limit applies to the filtered result.
function agrees(a: Measurements, b: Measurements) {
  return Math.abs(a.widthDifference-b.widthDifference) <= .08
    && Math.abs(a.heightDifference-b.heightDifference) <= .04
    && Math.abs(a.rollDegrees-b.rollDegrees) <= 3;
}

export function stabilizeMeasurements(previous: MeasurementHistory, detection: Detection|null,
  width: number, height: number, now: number, sameTrack: boolean, maxGapMs = MAX_GAP_MS) {
  // Keep enough actual observations on slow devices, but only within the
  // processing allowance supplied by the live loop. Idle pauses still expire.
  const latest = previous.at(-1);
  const recent = sameTrack && latest && now > latest.at && now-latest.at <= maxGapMs
    ? previous.filter(sample => now-sample.at <= Math.max(WINDOW_MS, 2*maxGapMs)) : [];
  const failure = detection ? poseCornerFailure(detection) : 'No slot detected';
  const raw = detection ? alignmentMeasurements(detection,width,height) : null;
  if (failure || !raw || !Object.values(raw).every(Number.isFinite)) {
    return { history: recent, raw, measured: raw, ready: false, status: failure ?? 'Invalid measurements' };
  }
  const history = [...recent,{at:now,measured:raw}].slice(-5);
  // Keep current box position/size: smoothing must not hide reframing or crops.
  const measured = { ...raw,
    widthDifference: median(history.map(s=>s.measured.widthDifference)),
    heightDifference: median(history.map(s=>s.measured.heightDifference)),
    rollDegrees: median(history.map(s=>s.measured.rollDegrees)) };
  measured.perspectiveDifference = Math.max(Math.abs(measured.widthDifference),Math.abs(measured.heightDifference));
  const support = history.filter(s=>agrees(s.measured,measured)).length;
  const ready = history.length >= 3 && support >= Math.max(3,Math.ceil(history.length*.6)) && agrees(raw,measured);
  return { history, raw, measured, ready,
    status: ready ? 'Stable' : history.length < 3 ? `Collecting readings (${history.length}/3)` : 'Corner measurements changing' };
}

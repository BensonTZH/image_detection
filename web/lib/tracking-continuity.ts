import { updatePoseReliabilityStreak, type Detection } from "./pose.ts";

export const EVIDENCE_GRACE_MS = 1000;
export type TrackingEvidence = { detection: Detection; sampledAt: number; reliableAt: number | null } | null;

// Box overlap is only permission to reuse recent history, never evidence that
// the current angle is acceptable. Current-frame capture checks remain required.
export function updateTrackingEvidence(previous: TrackingEvidence, current: Detection | null, reliable: boolean, now: number, maxGapMs = EVIDENCE_GRACE_MS) {
  const sameTrack = previous !== null && now > previous.sampledAt && now - previous.sampledAt <= (current ? maxGapMs : EVIDENCE_GRACE_MS)
    && (!current || matchingSlot(previous.detection, current));
  const retain = sameTrack && previous.reliableAt !== null && now - previous.reliableAt <= (current && reliable && previous.reliableAt === previous.sampledAt ? maxGapMs : EVIDENCE_GRACE_MS);
  const state: TrackingEvidence = current
    ? { detection: current, sampledAt: now, reliableAt: reliable ? now : retain ? previous.reliableAt : null }
    : sameTrack ? previous : null;
  return { state, sameTrack, retain };
}

function matchingSlot(previous: Detection, current: Detection) {
  const a = previous.box, b = current.box;
  if (![...a, ...b].every(Number.isFinite)) return false;
  const aw = a[2] - a[0], ah = a[3] - a[1], bw = b[2] - b[0], bh = b[3] - b[1];
  if (Math.min(aw, ah, bw, bh) <= 0) return false;
  if (bw / aw < .7 || bw / aw > 1.4 || bh / ah < .7 || bh / ah > 1.4) return false;
  const intersection = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]))
    * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  return intersection / (aw * ah + bw * bh - intersection) >= .55;
}

export function recoverPoseReliability(previous: number, reliable: boolean, retain: boolean) {
  if (!reliable) return { count: retain ? previous : 0, ready: false };
  return updatePoseReliabilityStreak(retain ? previous : 0, true);
}

export function preserveAcceptabilityHistory(retain: boolean, temporarilyUncertain: boolean, lastSampledAt: number | null, now: number) {
  return retain && temporarilyUncertain && lastSampledAt !== null
    && now > lastSampledAt && now - lastSampledAt <= EVIDENCE_GRACE_MS;
}

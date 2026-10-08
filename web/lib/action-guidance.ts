import { emptyInstructionState, updateInstruction, type InstructionState } from './live-guidance.ts';
import type { AlignmentMeasurements, AlignmentReference, Guidance } from './pose.ts';

export type ActionInstructionState = InstructionState & { lastReliableAt?: number; resolvedReadings?: number };

// Resolve the instructed axis using both the current frame and its filtered
// estimate. A different capture blocker is not evidence to continue that move.
export function correctionResolved(previous: Guidance, current: AlignmentMeasurements | null,
  filtered: AlignmentMeasurements | null, reference: AlignmentReference, sideBlocked: boolean, heightBlocked: boolean): boolean {
  if (!current || !filtered) return false;
  return [current, filtered].every(m => {
    switch (previous) {
      case 'left': return m.centerX >= reference.centerX - reference.centerTolerance;
      case 'right': return m.centerX <= reference.centerX + reference.centerTolerance;
      case 'step_left': return !sideBlocked && m.heightDifference >= -.06;
      case 'step_right': return !sideBlocked && m.heightDifference <= .06;
      case 'lower_phone': return !heightBlocked && m.widthDifference <= Math.min(reference.maximumPositiveVerticalPerspectiveDifference, reference.maximumVerticalPerspectiveDifference);
      case 'tilt_up': return m.widthDifference >= -reference.maximumVerticalPerspectiveDifference;
      case 'tilt_down': return m.widthDifference <= reference.maximumVerticalPerspectiveDifference;
      case 'farther': case 'framing': return m.area <= reference.targetArea + reference.areaTolerance;
      case 'closer': return m.area >= reference.targetArea - reference.areaTolerance;
      case 'rotate_left': return m.rollDegrees - reference.targetRollDegrees >= -reference.maximumRollDegrees;
      case 'rotate_right': return m.rollDegrees - reference.targetRollDegrees <= reference.maximumRollDegrees;
      default: return false;
    }
  });
}

export function updateActionInstruction(previous: ActionInstructionState, next: Guidance, now: number,
  evidence: { sameTrack: boolean; fresh: boolean; reliable: boolean; resolved: boolean;
    sideConfirmed: boolean; passingFrame: boolean; maxGapMs: number }): ActionInstructionState {
  const gap = previous.lastReliableAt === undefined ? Infinity : now - previous.lastReliableAt;
  const continuous = evidence.sameTrack && gap > 0 && gap <= evidence.maxGapMs;
  const base = evidence.sameTrack ? previous : emptyInstructionState();
  const reliable = evidence.fresh && evidence.reliable;
  const lastReliableAt = reliable ? now : continuous ? previous.lastReliableAt : undefined;
  const resolvedReadings = reliable && evidence.resolved ? (continuous ? previous.resolvedReadings ?? 0 : 0) + 1 : 0;
  // No instruction can live indefinitely on old/weak detections. A short gap
  // may preserve a confirmed action, but never counts as resolved or accepted.
  const retain = reliable || (continuous && gap <= 1200);
  const candidate = evidence.fresh ? next : 'steady';
  const resolved = resolvedReadings >= 2;
  const instruction = updateInstruction(base, candidate, now, evidence.sideConfirmed && reliable,
    resolved && candidate === 'steady', evidence.maxGapMs, evidence.passingFrame && reliable, retain);
  return { ...instruction, lastReliableAt,
    resolvedReadings: instruction.guidance === base.guidance ? resolvedReadings : 0 };
}

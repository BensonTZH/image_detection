import type { AngleClass, Detection, Guidance } from "./pose.ts";

// Roll back this additional check without changing any model or geometry gate.
export const ENABLE_STRONG_SIDE_GATE = true;
export const STRONG_SIDE_THRESHOLD = .85;
export const SIDE_RELEASE_THRESHOLD = .75;
type Side = Exclude<AngleClass, "centered">;
export type SideAngleState = { history: (Side | null)[]; direction: Side | null; blocked: boolean; clearReadings: number; lastAt: number | null };
export function emptySideAngleState(): SideAngleState {
  return { history: [], direction: null, blocked: false, clearReadings: 0, lastAt: null };
}

export function updateSideAngleState(previous: SideAngleState, detection: Detection | null, reliable: boolean, now: number, enabled = ENABLE_STRONG_SIDE_GATE, maxGapMs = 1200): SideAngleState {
  if (!enabled || !detection || !reliable || !Number.isFinite(detection.confidence)) return emptySideAngleState();
  if (previous.lastAt !== null && (now <= previous.lastAt || now - previous.lastAt > maxGapMs)) previous = emptySideAngleState();
  const side = detection.angleClass === "centered" ? null : detection.angleClass;
  const vote = side && detection.confidence >= STRONG_SIDE_THRESHOLD ? side : null;
  let history = [...previous.history, vote].slice(-5);
  const confirmed = side && detection.confidence >= SIDE_RELEASE_THRESHOLD && history.filter(value => value === side).length >= 4 ? side : null;
  let direction = confirmed ?? previous.direction;
  const supportsDirection = direction !== null && side === direction && detection.confidence >= SIDE_RELEASE_THRESHOLD;
  const clearReadings = vote || supportsDirection ? 0 : previous.clearReadings + 1;
  if (clearReadings >= 2) { direction = null; history = []; }
  // Stop an imminent capture on the first strong side reading. Show a movement
  // only after four matching readings; two clear readings release the block.
  return { history, direction, clearReadings, lastAt: now, blocked: vote !== null || direction !== null || (previous.blocked && clearReadings < 2) };
}

export function confirmedSideGuidance(state: SideAngleState): Guidance | null {
  return state.direction === "camera_right" ? "step_left" : state.direction === "camera_left" ? "step_right" : null;
}

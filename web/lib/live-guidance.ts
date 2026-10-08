import { ALIGNMENT_HOLD_MS, DEFAULT_ALIGNMENT_REFERENCE, alignmentMeasurements, captureAlignmentVeto, guidanceFor, poseCornerFailure, sideAngleCorrectionFor, type AlignmentMeasurements, type AlignmentReference, type CaptureVeto, type Detection, type Guidance } from "./pose.ts";

export type InstructionState = { guidance: Guidance; pending: Guidance | null; readings: number; since: number; lastSupportedAt: number; pendingUpdatedAt?: number };
export function emptyInstructionState(): InstructionState {
  return { guidance: "steady", pending: null, readings: 0, since: 0, lastSupportedAt: 0 };
}

// Unknown frames are not new votes. Keep a recent correction without renewing
// its support time, and always discard a pending/active accepted state.
export function interruptInstruction(previous: InstructionState, now: number): InstructionState {
  const state = updateInstruction(previous, "steady", now);
  return state.pending === "aligned" ? { ...state, pending: null, readings: 0 } : state;
}

// Only the caller's passing capture gates may request aligned. Neutral/unknown
// readings never become a hold-steady prompt or immediately erase a direction.
export function updateInstruction(previous: InstructionState, next: Guidance, now: number, sideAlreadyConfirmed = false, correctionResolved = false, maxGapMs = 1200, passingFrame = false, retainConfirmed = false): InstructionState {
  if (next === "searching") return { ...emptyInstructionState(), guidance: "searching" };
  if (sideAlreadyConfirmed && (next === "step_left" || next === "step_right")) return { guidance: next, pending: null, readings: 0, since: now, lastSupportedAt: now };
  // Passing current-frame gates start the shared capture hold immediately.
  // Showing this prompt is not permission to capture; updateCaptureHold owns
  // the minimum duration and consecutive-reading requirement.
  if (next === "aligned") return { guidance: "aligned", pending: null, readings: 0, since: now, lastSupportedAt: now };
  if (next === "steady" && correctionResolved) return emptyInstructionState();
  // A known passing frame breaks a pending correction's evidence. Preserve
  // the displayed instruction briefly until the shared hold confirms the view.
  if (passingFrame) previous = { ...previous, pending: null, readings: 0 };
  const unsupported = !retainConfirmed && previous.guidance !== next && now - previous.lastSupportedAt > 1200;
  const guidance = previous.guidance === "searching" || previous.guidance === "aligned" || unsupported ? "steady" : previous.guidance;
  if (next === "steady") {
    // A brief unknown reading is not evidence against the pending correction.
    // Preserve evidence for 600 ms without counting the unknown as a vote.
    if (previous.pending && previous.pending !== "aligned" && now - (previous.pendingUpdatedAt ?? previous.since) <= 600) return { ...previous, guidance };
    return { guidance, pending: null, readings: 0, since: now, lastSupportedAt: previous.lastSupportedAt };
  }
  if (guidance === next) return { guidance, pending: null, readings: 0, since: now, lastSupportedAt: now };
  const continues = previous.pending === next && now - (previous.pendingUpdatedAt ?? previous.since) <= maxGapMs;
  const readings = continues ? previous.readings + 1 : 1;
  const since = continues ? previous.since : now;
  if (readings >= 3 && now - since >= 400) {
    return { guidance: next, pending: null, readings: 0, since: now, lastSupportedAt: now };
  }
  return { guidance, pending: next, readings, since, lastSupportedAt: previous.lastSupportedAt, pendingUpdatedAt: now };
}

// These narrower release limits affect instructions only, never capture rules.
// Once a correction is shown, tiny changes across its trigger do not clear it.
export function movementGuidance(detection: Detection | null, width: number, height: number,
  previous: Guidance, reliable: boolean, reference: AlignmentReference = DEFAULT_ALIGNMENT_REFERENCE): Guidance {
  const release = { ...reference };
  if (previous === "step_left" || previous === "step_right") release.maximumPerspectiveDifference = .09;
  if (previous === "tilt_down" || previous === "tilt_up") release.maximumVerticalPerspectiveDifference *= .8;
  if (previous === "left" || previous === "right") release.centerTolerance *= .8;
  if (previous === "closer" || previous === "farther") release.areaTolerance *= .8;
  if (previous === "rotate_left" || previous === "rotate_right") release.maximumRollDegrees *= .8;
  const retained = guidanceFor(detection, width, height, release, reliable);
  return retained === previous ? retained : guidanceFor(detection, width, height, reference, reliable);
}

// Guidance uses reliable current corners and filtered measurements while
// capture additionally waits for temporal stability. A movement instruction
// is useful before the view is stable enough to save.
export function liveGeometryDecision(detection: Detection | null, width: number, height: number,
  reference: AlignmentReference, reliable: boolean, angleReady: boolean, sideConfirmed: boolean, stableMeasurements?: AlignmentMeasurements, captureStable = true) {
  const raw = guidanceFor(detection, width, height, reference, reliable, true, stableMeasurements);
  const veto = detection && reliable && !captureStable ? "measurement_stability" : captureAlignmentVeto(detection, width, height, reference, reliable, angleReady, true, sideConfirmed, stableMeasurements);
  const measured = stableMeasurements ?? (detection ? alignmentMeasurements(detection, width, height) : null);
  return { raw, veto, vertical: measured && Number.isFinite(measured.widthDifference) ? measured.widthDifference : null,
    cornerFailure: detection ? poseCornerFailure(detection) : "No slot detected" };
}

export type CaptureHold = { startedAt: number | null; lastAt: number | null; readings: number; ready: boolean;
  validMs: number; consecutive: number; paused: boolean; confirmed: boolean };
export function emptyCaptureHold(): CaptureHold {
  return { startedAt: null, lastAt: null, readings: 0, ready: false, validMs: 0, consecutive: 0, paused: false, confirmed: false };
}

// A brief uncertain sample can preserve progress, but cannot add time, votes,
// or permission to capture. Expiry is anchored to the last actual passing frame.
export function updateCaptureHold(previous: CaptureHold, sampledAt: number, eligible: boolean, maxGapMs = 1200, uncertain = false): CaptureHold {
  const pauseLimit = Math.min(1200, maxGapMs);
  const age = previous.lastAt === null ? Infinity : sampledAt - previous.lastAt;
  const withinWindow = previous.startedAt !== null && sampledAt - previous.startedAt <= Math.max(4000, 3 * maxGapMs);
  if (!eligible) {
    return uncertain && age > 0 && age <= pauseLimit && withinWindow
      ? { ...previous, paused: true, ready: false, consecutive: 0 } : emptyCaptureHold();
  }
  const continuous = age > 0 && age <= (previous.paused ? pauseLimit : maxGapMs) && withinWindow;
  const startedAt = continuous ? previous.startedAt! : sampledAt;
  const readings = continuous ? previous.readings + 1 : 1;
  const consecutive = continuous && !previous.paused ? previous.consecutive + 1 : 1;
  const validMs = continuous ? previous.validMs + (previous.paused ? 0 : age) : 0;
  const confirmed = (continuous && previous.confirmed) || (consecutive >= 2 && validMs >= 200);
  return { startedAt, lastAt: sampledAt, readings, validMs, consecutive, paused: false, confirmed,
    ready: readings >= 3 && consecutive >= 2 && validMs >= ALIGNMENT_HOLD_MS };
}

export function canPauseCapture(veto: CaptureVeto | null, filteredVeto: CaptureVeto | null,
  retainTrack: boolean, fresh: boolean): boolean {
  if (!retainTrack || !fresh) return false;
  if (veto === "no_detection" || veto === "corners") return true;
  // Disagreement between a current corner estimate and the recent median is
  // uncertainty, not permission to save that frame. Confirmed failures reset.
  return filteredVeto === null && (veto === "position" || veto === "side_perspective" || veto === "angle_vote");
}

// Instruction priority is independent of capture vetoes: a height failure still
// blocks capture while the user follows a better-supported side correction.
export function chooseLiveGuidance({ raw, currentRaw, frameReady, veto, confirmedSide, angleCorrection, heightConfirmed, acceptabilityCorrection }: {
  raw: Guidance; frameReady: boolean; veto: CaptureVeto | null;
  currentRaw?: Guidance;
  confirmedSide: Guidance | null; angleCorrection: Guidance | null;
  heightConfirmed: boolean; acceptabilityCorrection: Guidance | null;
}): Guidance {
  if (raw === "searching") return "searching";
  if (frameReady) return "aligned";
  if (confirmedSide === "step_left" || confirmedSide === "step_right") return confirmedSide;
  if (raw === "step_left" || raw === "step_right") return raw;
  if (angleCorrection === "step_left" || angleCorrection === "step_right") return angleCorrection;
  // Resolve an oversized slot before asking for a height correction. Height
  // vetoes still block capture and are reassessed on every new frame.
  if (raw === "farther") return "farther";
  if (veto === "high_angle" && heightConfirmed) return "lower_phone";
  if (veto === "acceptability" && acceptabilityCorrection) return acceptabilityCorrection;
  // The median may pass while the current frame blocks capture. Give that
  // frame's correction to the normal instruction debounce instead of leaving
  // the user without a direction. Isolated raw spikes still cannot flash it.
  if (raw === "aligned" && (veto === "position" || veto === "side_perspective" || veto === "angle_vote")
    && currentRaw && currentRaw !== "aligned" && currentRaw !== "steady" && currentRaw !== "searching") return currentRaw;
  // Pending side/height checks must not erase useful current framing guidance.
  return raw === "aligned" ? "steady" : raw;
}

// A blocked side check needs geometry-based guidance even before the class
// vote window agrees. Continue an existing side correction only while BOTH
// fresh and filtered corners still support it; never prolong it by time alone.
export function liveSideCorrection(detection: Detection | null, width: number, height: number,
  previous: Guidance, reliable: boolean, filtered: AlignmentMeasurements | undefined,
  filteredVeto: CaptureVeto | null, currentVeto: CaptureVeto | null, votedSide: boolean): Guidance | null {
  if (!detection || !reliable) return null;
  const recent = sideAngleCorrectionFor(detection, width, height, previous, filtered);
  const current = sideAngleCorrectionFor(detection, width, height, previous);
  if (filteredVeto === "angle_vote" || votedSide) return recent;
  if (currentVeto === "angle_vote") return current;
  return (previous === "step_left" || previous === "step_right") && recent === previous && current === previous ? previous : null;
}

export function waitingGuidanceCopy(veto: CaptureVeto | null, slotDetected: boolean) {
  if (!slotDetected || veto === "no_detection") return {
    eyebrow: "Find the slot", title: "Keep the slot in view", detail: "Point your camera at the whole opening.",
  };
  if (veto === "side_class" || veto === "angle_vote" || veto === "side_perspective" || veto === "acceptability") return {
    eyebrow: "Slot found", title: "Face the slot straight on", detail: "Position yourself directly in front of the opening.",
  };
  return { eyebrow: "Slot found", title: "Pause movement briefly", detail: "Keep this view while the readings settle." };
}


// Verification is secondary copy, never a replacement movement instruction.
export function verificationStatus(guide: Guidance, veto: CaptureVeto | null, interrupted = false): string | null {
  if (interrupted) return "Tracking interrupted — waiting for a fresh reading…";
  if (guide === "aligned" || guide === "searching") return null;
  if (veto === "measurement_stability") return "Confirming stable measurements…";
  if (veto === "stale_frame") return "Waiting for a fresh camera reading…";
  if (veto === "corners") return "Slot detected — corner tracking is uncertain…";
  if (veto === "high_angle") return "Checking phone height…";
  if (veto === "side_class" || veto === "angle_vote") return "Checking side angle…";
  if (veto === "acceptability") return "Verifying viewing angle…";
  if (veto === null) return "Confirming alignment…";
  return guide === "steady" ? "Verifying angle…" : null;
}

export const MODEL_SIZE = 320;
export const OUTPUT_CHANNELS = 19;
// Close portrait views can produce a modest box/class score even when all four
// slot corners are exceptionally confident. Admit those candidates here; the
// independent corner, pose, angle-history, and alignment gates still decide
// whether automatic capture is safe.
export const DETECTION_CONFIDENCE_THRESHOLD = 0.35;
export const KEYPOINT_CONFIDENCE_THRESHOLD = 0.35;
export const POSE_RELIABILITY_REQUIRED_READINGS = 2;
export const ALIGNMENT_HOLD_MS = 800;
// Guidance can remain forgiving while the user is moving, but automatic
// capture should only happen from a noticeably more frontal view.
export const CAPTURE_SIDE_PERSPECTIVE_TOLERANCE = .12;
// The learned angle label is noisy on narrow frontal slots. Use a sustained
// side vote only to reinforce borderline measured perspective, never as a
// standalone capture veto.
export const ANGLE_VOTE_CORROBORATION_TOLERANCE = .08;
// The binary acceptability model cannot say how to correct a rejected frame.
// Use corner geometry only to choose a helpful direction. A rejection still
// blocks capture when no specific correction is available.
export const ACCEPTABILITY_SIDE_CORROBORATION_TOLERANCE = ANGLE_VOTE_CORROBORATION_TOLERANCE;
export const ACCEPTABILITY_VERTICAL_CORROBORATION_TOLERANCE = .20;
export const CLASSIFIER_SIDE_PERSPECTIVE_TOLERANCE = CAPTURE_SIDE_PERSPECTIVE_TOLERANCE;
// Set to false to restore the previous behavior where one unreliable pose
// reading immediately displays the `steady` guidance.
export const STABILIZE_UNRELIABLE_POSE_GUIDANCE = true;

export type Point = { x: number; y: number; confidence: number };
export type AngleClass = "centered" | "camera_left" | "camera_right";
export type Detection = {
  box: [number, number, number, number];
  confidence: number;
  angleClass: AngleClass;
  keypoints: [Point, Point, Point, Point];
};

const ANGLE_CLASSES: AngleClass[] = ["centered", "camera_left", "camera_right"];

export type Letterbox = { scale: number; padX: number; padY: number; width: number; height: number };

export function letterboxGeometry(width: number, height: number, size = MODEL_SIZE): Letterbox {
  const scale = Math.min(size / width, size / height);
  const resizedWidth = Math.round(width * scale);
  const resizedHeight = Math.round(height * scale);
  return { scale, padX: (size - resizedWidth) / 2, padY: (size - resizedHeight) / 2, width: resizedWidth, height: resizedHeight };
}

function clamp(value: number, maximum: number) { return Math.max(0, Math.min(maximum, value)); }

function iou(a: Detection, b: Detection) {
  const x1 = Math.max(a.box[0], b.box[0]);
  const y1 = Math.max(a.box[1], b.box[1]);
  const x2 = Math.min(a.box[2], b.box[2]);
  const y2 = Math.min(a.box[3], b.box[3]);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.box[2] - a.box[0]) * Math.max(0, a.box[3] - a.box[1]);
  const areaB = Math.max(0, b.box[2] - b.box[0]) * Math.max(0, b.box[3] - b.box[1]);
  return intersection / Math.max(areaA + areaB - intersection, 1e-6);
}

export function nonMaximumSuppression(detections: Detection[], threshold = 0.7, limit = 1) {
  const ordered = [...detections].sort((a, b) => b.confidence - a.confidence);
  const kept: Detection[] = [];
  while (ordered.length && kept.length < limit) {
    const current = ordered.shift()!;
    kept.push(current);
    for (let index = ordered.length - 1; index >= 0; index--) if (iou(current, ordered[index]) > threshold) ordered.splice(index, 1);
  }
  return kept;
}

export function smoothDetection(previous: Detection | null, current: Detection, currentWeight = .7): Detection {
  if (!previous) return current;
  const weight = Math.max(0, Math.min(1, currentWeight));
  const blend = (oldValue: number, newValue: number) => oldValue * (1 - weight) + newValue * weight;
  return {
    box: current.box.map((value, index) => blend(previous.box[index], value)) as Detection["box"],
    confidence: blend(previous.confidence, current.confidence),
    angleClass: current.angleClass,
    keypoints: current.keypoints.map((point, index) => ({
      x: blend(previous.keypoints[index].x, point.x),
      y: blend(previous.keypoints[index].y, point.y),
      confidence: blend(previous.keypoints[index].confidence, point.confidence),
    })) as Detection["keypoints"],
  };
}

export function decodePoseOutput(
  output: Float32Array,
  candidates: number,
  geometry: Letterbox,
  originalWidth: number,
  originalHeight: number,
  confidenceThreshold = DETECTION_CONFIDENCE_THRESHOLD,
) {
  if (output.length !== OUTPUT_CHANNELS * candidates) throw new Error(`Unexpected model output length: ${output.length}`);
  const at = (channel: number, index: number) => output[channel * candidates + index];
  const detections: Detection[] = [];
  const mapX = (value: number) => clamp((value - geometry.padX) / geometry.scale, originalWidth);
  const mapY = (value: number) => clamp((value - geometry.padY) / geometry.scale, originalHeight);
  for (let index = 0; index < candidates; index++) {
    const classScores = ANGLE_CLASSES.map((_, classIndex) => at(4 + classIndex, index));
    const classIndex = classScores.indexOf(Math.max(...classScores));
    const confidence = classScores[classIndex];
    if (confidence < confidenceThreshold) continue;
    const cx = at(0, index), cy = at(1, index), width = at(2, index), height = at(3, index);
    const points = Array.from({ length: 4 }, (_, point) => ({
      x: mapX(at(7 + point * 3, index)),
      y: mapY(at(8 + point * 3, index)),
      confidence: at(9 + point * 3, index),
    })) as Detection["keypoints"];
    detections.push({
      box: [mapX(cx - width / 2), mapY(cy - height / 2), mapX(cx + width / 2), mapY(cy + height / 2)],
      confidence,
      angleClass: ANGLE_CLASSES[classIndex],
      keypoints: points,
    });
  }
  return nonMaximumSuppression(detections);
}

export type Guidance = "searching" | "steady" | "framing" | "left" | "right" | "step_left" | "step_right" | "closer" | "farther" | "rotate_left" | "rotate_right" | "tilt_up" | "tilt_down" | "lower_phone" | "aligned";

export type AlignmentReference = {
  centerX: number; centerTolerance: number; targetArea: number; areaTolerance: number;
  targetRollDegrees: number; targetWidthDifference: number; targetHeightDifference: number;
  maximumRollDegrees: number; maximumPerspectiveDifference: number; maximumVerticalPerspectiveDifference: number;
  maximumPositiveVerticalPerspectiveDifference: number;
};

// Safe POC defaults. Calibrate the reference pose at the real machine for each phone/site.
export const DEFAULT_ALIGNMENT_REFERENCE: AlignmentReference = {
  // Treat the slot as horizontally centred anywhere in the middle 20% of the
  // frame. The previous 14% band produced left/right prompts for visually
  // centred, narrow slots when the detector box shifted by only a few pixels.
  centerX: .5, centerTolerance: .10, targetArea: .115, areaTolerance: .085,
  targetRollDegrees: 0, targetWidthDifference: 0, targetHeightDifference: 0,
  maximumRollDegrees: 7, maximumPerspectiveDifference: .22, maximumVerticalPerspectiveDifference: .28,
  // Stricter above-slot view limit; the opposite perspective retains its 28% limit.
  maximumPositiveVerticalPerspectiveDifference: .16,
};

export function alignmentReferenceFromStored(stored: Partial<AlignmentReference>): AlignmentReference {
  const savedNumber = (value: number | undefined, fallback: number) => typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return {
    ...DEFAULT_ALIGNMENT_REFERENCE,
    targetArea: savedNumber(stored.targetArea, DEFAULT_ALIGNMENT_REFERENCE.targetArea),
    targetRollDegrees: savedNumber(stored.targetRollDegrees, DEFAULT_ALIGNMENT_REFERENCE.targetRollDegrees),
    // Calibration may tune distance and minor phone roll, but a side-on or
    // tilted view must never become the site's accepted reference shape.
    targetWidthDifference: 0,
    targetHeightDifference: 0,
  };
}

export function alignmentExitReference(reference = DEFAULT_ALIGNMENT_REFERENCE): AlignmentReference {
  return {
    ...reference,
    centerTolerance: reference.centerTolerance + .02,
    areaTolerance: reference.areaTolerance + .01,
    maximumRollDegrees: reference.maximumRollDegrees + 2,
    maximumPerspectiveDifference: reference.maximumPerspectiveDifference + .04,
    maximumVerticalPerspectiveDifference: reference.maximumVerticalPerspectiveDifference + .04,
  };
}

function distance(a: Point, b: Point) { return Math.hypot(b.x - a.x, b.y - a.y); }

function lineOrientation(a: Point, b: Point) {
  const degrees = Math.atan2(b.y - a.y, b.x - a.x) * 180 / Math.PI;
  return (degrees + 180) % 180;
}

function orientationDifference(first: number, second: number) {
  const difference = Math.abs(first - second);
  return Math.min(difference, 180 - difference);
}

export function poseCornerFailure(detection: Detection, keypointConfidenceThreshold = KEYPOINT_CONFIDENCE_THRESHOLD): string | null {
  const [tl, tr, br, bl] = detection.keypoints;
  if (detection.keypoints.some((point) => !Number.isFinite(point.x) || !Number.isFinite(point.y) || !Number.isFinite(point.confidence))) return "Invalid corner coordinates or confidence";
  const weakCorner = detection.keypoints.findIndex(point => point.confidence < keypointConfidenceThreshold);
  if (weakCorner !== -1) return `Low ${["top-left", "top-right", "bottom-right", "bottom-left"][weakCorner]} confidence (${Math.round(detection.keypoints[weakCorner].confidence * 100)}%)`;
  const boxWidth = detection.box[2] - detection.box[0], boxHeight = detection.box[3] - detection.box[1];
  if (!detection.box.every(Number.isFinite) || boxWidth <= 0 || boxHeight <= 0) return "Invalid slot box";
  if ((tl.y + tr.y) / 2 >= (bl.y + br.y) / 2 || (tl.x + bl.x) / 2 >= (tr.x + br.x) / 2) return "Corners out of order";
  const topWidth = distance(tl, tr), bottomWidth = distance(bl, br);
  const leftHeight = distance(tl, bl), rightHeight = distance(tr, br);
  if (Math.min(topWidth, bottomWidth) < boxWidth * .15 || Math.min(leftHeight, rightHeight) < boxHeight * .30) return "Corner edges too short";
  // A single misplaced corner produces opposing edges with very different
  // directions, as seen when a bottom corner falls below the actual slot.
  if (orientationDifference(lineOrientation(tl, tr), lineOrientation(bl, br)) > 25) return "Top/bottom edges disagree";
  if (orientationDifference(lineOrientation(tl, bl), lineOrientation(tr, br)) > 25) return "Left/right edges disagree";
  return null;
}

export function poseCornersReliable(detection: Detection, keypointConfidenceThreshold = KEYPOINT_CONFIDENCE_THRESHOLD) {
  return poseCornerFailure(detection, keypointConfidenceThreshold) === null;
}

export function updatePoseReliabilityStreak(previous: number, reliable: boolean, required = POSE_RELIABILITY_REQUIRED_READINGS) {
  const count = reliable ? Math.min(required, previous + 1) : 0;
  return { count, ready: count >= required };
}

export function updateAngleCaptureWindow(previous: AngleClass[], next: AngleClass | null, windowSize = 5, minimumCenteredCount = 3, minimumSideCount = 4) {
  if (!next) return { history: [] as AngleClass[], ready: false, sideDirection: null as AngleClass | null };
  const history = [...previous, next].slice(-windowSize);
  const centered = history.filter((value) => value === "centered").length;
  const cameraLeft = history.filter((value) => value === "camera_left").length;
  const cameraRight = history.filter((value) => value === "camera_right").length;
  const sideDirection = cameraLeft >= minimumSideCount ? "camera_left" : cameraRight >= minimumSideCount ? "camera_right" : null;
  // A centered majority can survive one noisy current reading. A side view
  // needs four matching votes before it blocks capture, preventing a frontal
  // view from becoming stuck after a short classifier wobble.
  return { history, ready: history.length >= windowSize && centered >= minimumCenteredCount, sideDirection };
}

export function alignmentMeasurements(detection: Detection, width: number, height: number) {
  const [tl, tr, br, bl] = detection.keypoints;
  const centerX = (detection.box[0] + detection.box[2]) / 2 / width;
  const area = ((detection.box[2] - detection.box[0]) * (detection.box[3] - detection.box[1])) / (width * height);
  const rollDegrees = Math.atan2(tr.y - tl.y, tr.x - tl.x) * 180 / Math.PI;
  const topWidth = distance(tl, tr), bottomWidth = distance(bl, br);
  const leftHeight = distance(tl, bl), rightHeight = distance(tr, br);
  const widthDifference = (topWidth - bottomWidth) / Math.max(topWidth, bottomWidth, 1);
  const heightDifference = (leftHeight - rightHeight) / Math.max(leftHeight, rightHeight, 1);
  return { centerX, area, rollDegrees, widthDifference, heightDifference, perspectiveDifference: Math.max(Math.abs(widthDifference), Math.abs(heightDifference)) };
}

export type AlignmentMeasurements = ReturnType<typeof alignmentMeasurements>;

export function acceptabilityCorrectionFor(detection: Detection | null, width: number, height: number, previous: Guidance = "steady", stableMeasurements?: AlignmentMeasurements): Guidance | null {
  if (!detection || !poseCornersReliable(detection)) return null;
  const measured = stableMeasurements ?? alignmentMeasurements(detection, width, height);
  const sideDirection = measured.heightDifference > 0 ? "step_right" : "step_left";
  const verticalDirection = measured.widthDifference > 0 ? "lower_phone" : "tilt_up";
  const sideLimit = ACCEPTABILITY_SIDE_CORROBORATION_TOLERANCE * (previous === sideDirection ? .75 : 1);
  const verticalLimit = ACCEPTABILITY_VERTICAL_CORROBORATION_TOLERANCE * (previous === verticalDirection ? .8 : 1);
  if (Math.abs(measured.heightDifference) > sideLimit) return measured.heightDifference > 0 ? "step_right" : "step_left";
  if (Math.abs(measured.widthDifference) > verticalLimit) return measured.widthDifference > 0 ? "lower_phone" : "tilt_up";
  return null;
}

export function sideAngleCorrectionFor(detection: Detection | null, width: number, height: number, previous: Guidance = "steady", stableMeasurements?: AlignmentMeasurements): Guidance | null {
  if (!detection || !poseCornersReliable(detection)) return null;
  const { heightDifference } = stableMeasurements ?? alignmentMeasurements(detection, width, height);
  const direction = heightDifference > 0 ? "step_right" : "step_left";
  const limit = ANGLE_VOTE_CORROBORATION_TOLERANCE * (previous === direction ? .75 : 1);
  if (Math.abs(heightDifference) <= limit) return null;
  return heightDifference > 0 ? "step_right" : "step_left";
}

export function referenceAtCurrentPose(detection: Detection, width: number, height: number, reference = DEFAULT_ALIGNMENT_REFERENCE): AlignmentReference {
  const measured = alignmentMeasurements(detection, width, height);
  return {
    ...reference,
    targetArea: measured.area,
    targetRollDegrees: measured.rollDegrees,
    targetWidthDifference: 0,
    targetHeightDifference: 0,
  };
}

export function guidanceFor(detection: Detection | null, width: number, height: number, reference = DEFAULT_ALIGNMENT_REFERENCE, poseReliable = detection ? poseCornersReliable(detection) : false, useLegacyAngleGuidance = true, stableMeasurements?: AlignmentMeasurements): Guidance {
  if (!detection) return "searching";
  const measured = stableMeasurements ?? alignmentMeasurements(detection, width, height);
  const rollDifference = measured.rollDegrees - reference.targetRollDegrees;
  const widthDifference = measured.widthDifference;
  const heightDifference = measured.heightDifference;
  // Box-based centring and distance remain useful even when one of the four
  // corner predictions is weak. Pose instructions and capture wait until the
  // corners are consistently trustworthy.
  if (!poseReliable) {
    // A weak corner frame cannot corroborate a side label. Keep useful box
    // framing guidance, but wait for reliable geometry before guessing a side.
    if (measured.centerX < reference.centerX - reference.centerTolerance) return "left";
    if (measured.centerX > reference.centerX + reference.centerTolerance) return "right";
    if (measured.area < reference.targetArea - reference.areaTolerance) return "closer";
    if (measured.area > reference.targetArea + reference.areaTolerance) return "farther";
    return "steady";
  }
  // Correct side-on and vertical perspective before centering the slot in the
  // frame. Otherwise a user standing left of the machine can be told to move
  // farther left merely because the slot also appears left of centre.
  const sidePerspectiveTolerance = Math.min(reference.maximumPerspectiveDifference, CAPTURE_SIDE_PERSPECTIVE_TOLERANCE);
  if (useLegacyAngleGuidance && Math.abs(heightDifference) > sidePerspectiveTolerance) return heightDifference > 0 ? "step_right" : "step_left";
  // Excessive image size is direct evidence to step back. Reassess vertical
  // perspective at that distance before attributing it to phone height.
  if (measured.area > reference.targetArea + reference.areaTolerance) return "farther";
  if (useLegacyAngleGuidance && widthDifference > Math.min(reference.maximumPositiveVerticalPerspectiveDifference, reference.maximumVerticalPerspectiveDifference)) return "lower_phone";
  if (useLegacyAngleGuidance && widthDifference < -reference.maximumVerticalPerspectiveDifference) return "tilt_up";
  if (measured.centerX < reference.centerX - reference.centerTolerance) return "left";
  if (measured.centerX > reference.centerX + reference.centerTolerance) return "right";
  if (measured.area < reference.targetArea - reference.areaTolerance) return "closer";
  if (Math.abs(rollDifference) > reference.maximumRollDegrees) return rollDifference > 0 ? "rotate_right" : "rotate_left";
  return "aligned";
}

export type CaptureVeto = "measurement_stability" | "stale_frame" | "no_detection" | "corners" | "position" | "side_perspective" | "angle_vote" | "side_class" | "acceptability" | "high_angle";

export function captureAlignmentVeto(detection: Detection | null, width: number, height: number, reference = DEFAULT_ALIGNMENT_REFERENCE, poseReliable = detection ? poseCornersReliable(detection) : false, angleCaptureReady = detection?.angleClass === "centered", useLegacyAngleGates = true, sideAngleConfirmed = false, stableMeasurements?: AlignmentMeasurements): CaptureVeto | null {
  if (!detection) return "no_detection";
  if (!poseReliable) return "corners";
  const measured = stableMeasurements ?? alignmentMeasurements(detection, width, height);
  const captureSideTolerance = Math.min(reference.maximumPerspectiveDifference, CAPTURE_SIDE_PERSPECTIVE_TOLERANCE);
  const sidePerspective = Math.abs(measured.heightDifference);
  if (sidePerspective > captureSideTolerance) return "side_perspective";
  if (guidanceFor(detection, width, height, reference, poseReliable, useLegacyAngleGates, stableMeasurements) !== "aligned") return "position";
  if ((sideAngleConfirmed || !angleCaptureReady) && sidePerspective > ANGLE_VOTE_CORROBORATION_TOLERANCE) return "angle_vote";
  return null;
}

export function captureAlignmentReady(detection: Detection | null, width: number, height: number, reference = DEFAULT_ALIGNMENT_REFERENCE, poseReliable = detection ? poseCornersReliable(detection) : false, angleCaptureReady = detection?.angleClass === "centered") {
  return captureAlignmentVeto(detection, width, height, reference, poseReliable, angleCaptureReady) === null;
}

export function stabilizeGuidance(history: Guidance[], next: Guidance, windowSize = 3): { history: Guidance[]; guidance: Guidance } {
  if (next === "searching" || (next === "steady" && !STABILIZE_UNRELIABLE_POSE_GUIDANCE)) return { history: [next], guidance: next };
  const updated = [...history, next].slice(-windowSize);
  const counts = new Map<Guidance, number>();
  updated.forEach((value) => counts.set(value, (counts.get(value) ?? 0) + 1));
  let guidance = updated[updated.length - 1];
  let best = 0;
  for (const value of [...updated].reverse()) {
    const count = counts.get(value) ?? 0;
    if (count > best) { guidance = value; best = count; }
  }
  return { history: updated, guidance };
}

export function updateAlignmentHold(startedAt: number | null, now: number, aligned: boolean, holdMs = ALIGNMENT_HOLD_MS) {
  const nextStartedAt = aligned ? startedAt ?? now : null;
  return { startedAt: nextStartedAt, ready: nextStartedAt !== null && now - nextStartedAt >= holdMs };
}

// Display-only grace: missing frames must still veto capture immediately.
export const DETECTION_DISPLAY_GRACE_MS = 1000;
export type DetectionRecovery = {
  detection: Detection | null;
  misses: number;
  lastSeenAt: number | null;
  state: "visible" | "fading" | "lost";
};

export function recoverTemporaryLoss(previous: Detection | null, current: Detection | null, misses: number, lastSeenAt: number | null, now: number): DetectionRecovery {
  if (current) return { detection: current, misses: 0, lastSeenAt: now, state: "visible" };
  const nextMisses = misses + 1;
  if (previous && lastSeenAt !== null && now >= lastSeenAt && now - lastSeenAt < DETECTION_DISPLAY_GRACE_MS) {
    return { detection: previous, misses: nextMisses, lastSeenAt, state: "fading" };
  }
  return { detection: null, misses: nextMisses, lastSeenAt: null, state: "lost" };
}

// Retain the exact displayed box during a gap; only fresh observations update
// it. Restart smoothing after a long gap rather than blending a stale position.
export function recoveredDisplayDetection(previous: Detection | null, current: Detection | null, recovery: DetectionRecovery, previousSeenAt: number | null, now: number): Detection | null {
  if (!current) return recovery.detection ? previous ?? recovery.detection : null;
  const recent = previousSeenAt !== null && now >= previousSeenAt && now - previousSeenAt < DETECTION_DISPLAY_GRACE_MS;
  return smoothDetection(recent ? previous : null, current);
}

export type PerformanceSample = { inferenceMs: number; completedAt: number; detected: boolean };

export function balancedDelay(inferenceMs: number) {
  return Math.max(30, Math.min(100, 250 - inferenceMs));
}

export function summarizePerformance(samples: PerformanceSample[]) {
  if (!samples.length) return { averageMs: 0, p95Ms: 0, updatesPerMinute: 0, detectionRate: 0, count: 0 };
  const times = samples.map((sample) => sample.inferenceMs).sort((a, b) => a - b);
  const averageMs = times.reduce((total, value) => total + value, 0) / times.length;
  const p95Ms = times[Math.min(times.length - 1, Math.ceil(times.length * .95) - 1)];
  const duration = samples.length > 1 ? samples[samples.length - 1].completedAt - samples[0].completedAt : 0;
  const updatesPerMinute = duration > 0 ? ((samples.length - 1) * 60000) / duration : 0;
  const detectionRate = samples.filter((sample) => sample.detected).length / samples.length;
  return { averageMs, p95Ms, updatesPerMinute, detectionRate, count: samples.length };
}

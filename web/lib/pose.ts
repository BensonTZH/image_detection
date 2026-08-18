export const MODEL_SIZE = 640;
export const OUTPUT_CHANNELS = 17;

export type Point = { x: number; y: number; confidence: number };
export type Detection = {
  box: [number, number, number, number];
  confidence: number;
  keypoints: [Point, Point, Point, Point];
};

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

export function decodePoseOutput(
  output: Float32Array,
  candidates: number,
  geometry: Letterbox,
  originalWidth: number,
  originalHeight: number,
  confidenceThreshold = 0.25,
) {
  if (output.length !== OUTPUT_CHANNELS * candidates) throw new Error(`Unexpected model output length: ${output.length}`);
  const at = (channel: number, index: number) => output[channel * candidates + index];
  const detections: Detection[] = [];
  const mapX = (value: number) => clamp((value - geometry.padX) / geometry.scale, originalWidth);
  const mapY = (value: number) => clamp((value - geometry.padY) / geometry.scale, originalHeight);
  for (let index = 0; index < candidates; index++) {
    const confidence = at(4, index);
    if (confidence < confidenceThreshold) continue;
    const cx = at(0, index), cy = at(1, index), width = at(2, index), height = at(3, index);
    const points = Array.from({ length: 4 }, (_, point) => ({
      x: mapX(at(5 + point * 3, index)),
      y: mapY(at(6 + point * 3, index)),
      confidence: at(7 + point * 3, index),
    })) as Detection["keypoints"];
    detections.push({
      box: [mapX(cx - width / 2), mapY(cy - height / 2), mapX(cx + width / 2), mapY(cy + height / 2)],
      confidence,
      keypoints: points,
    });
  }
  return nonMaximumSuppression(detections);
}

export type Guidance = "searching" | "left" | "right" | "closer" | "farther" | "straighten" | "aligned";

export type AlignmentReference = {
  centerX: number; centerTolerance: number; targetArea: number; areaTolerance: number;
  maximumRollDegrees: number; maximumPerspectiveDifference: number;
};

// Safe POC defaults. Calibrate targetArea at the real machine for each phone/site.
export const DEFAULT_ALIGNMENT_REFERENCE: AlignmentReference = {
  centerX: .5, centerTolerance: .08, targetArea: .18, areaTolerance: .07,
  maximumRollDegrees: 7, maximumPerspectiveDifference: .25,
};

function distance(a: Point, b: Point) { return Math.hypot(b.x - a.x, b.y - a.y); }

export function alignmentMeasurements(detection: Detection, width: number, height: number) {
  const [tl, tr, br, bl] = detection.keypoints;
  const centerX = (detection.box[0] + detection.box[2]) / 2 / width;
  const area = ((detection.box[2] - detection.box[0]) * (detection.box[3] - detection.box[1])) / (width * height);
  const rollDegrees = Math.atan2(tr.y - tl.y, tr.x - tl.x) * 180 / Math.PI;
  const topWidth = distance(tl, tr), bottomWidth = distance(bl, br);
  const leftHeight = distance(tl, bl), rightHeight = distance(tr, br);
  const widthDifference = Math.abs(topWidth - bottomWidth) / Math.max(topWidth, bottomWidth, 1);
  const heightDifference = Math.abs(leftHeight - rightHeight) / Math.max(leftHeight, rightHeight, 1);
  return { centerX, area, rollDegrees, perspectiveDifference: Math.max(widthDifference, heightDifference) };
}

export function referenceAtCurrentDistance(detection: Detection, width: number, height: number, reference = DEFAULT_ALIGNMENT_REFERENCE): AlignmentReference {
  return { ...reference, targetArea: alignmentMeasurements(detection, width, height).area };
}

export function guidanceFor(detection: Detection | null, width: number, height: number, reference = DEFAULT_ALIGNMENT_REFERENCE): Guidance {
  if (!detection) return "searching";
  const measured = alignmentMeasurements(detection, width, height);
  if (measured.centerX < reference.centerX - reference.centerTolerance) return "left";
  if (measured.centerX > reference.centerX + reference.centerTolerance) return "right";
  if (measured.area < reference.targetArea - reference.areaTolerance) return "closer";
  if (measured.area > reference.targetArea + reference.areaTolerance) return "farther";
  if (Math.abs(measured.rollDegrees) > reference.maximumRollDegrees || measured.perspectiveDifference > reference.maximumPerspectiveDifference) return "straighten";
  return "aligned";
}

export function stabilizeGuidance(history: Guidance[], next: Guidance, windowSize = 3): { history: Guidance[]; guidance: Guidance } {
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

export type DetectionRecovery = {
  detection: Detection | null;
  misses: number;
  state: "visible" | "fading" | "lost";
};

export function recoverTemporaryLoss(previous: Detection | null, current: Detection | null, misses: number): DetectionRecovery {
  if (current) return { detection: current, misses: 0, state: "visible" };
  const nextMisses = misses + 1;
  if (previous && nextMisses === 1) return { detection: previous, misses: nextMisses, state: "visible" };
  if (previous && nextMisses === 2) return { detection: previous, misses: nextMisses, state: "fading" };
  return { detection: null, misses: nextMisses, state: "lost" };
}

export type PerformanceSample = { inferenceMs: number; completedAt: number; detected: boolean };

export function balancedDelay(inferenceMs: number) {
  return Math.max(250, Math.min(900, 1000 - inferenceMs));
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

import assert from "node:assert/strict";
import test from "node:test";
import { alignmentMeasurements, balancedDelay, decodePoseOutput, guidanceFor, letterboxGeometry, nonMaximumSuppression, recoverTemporaryLoss, referenceAtCurrentDistance, stabilizeGuidance, summarizePerformance, type Detection } from "../lib/pose.ts";

function detection(box: [number, number, number, number], confidence: number): Detection {
  const [x1, y1, x2, y2] = box;
  return { box, confidence, keypoints: [{ x:x1,y:y1,confidence:1 }, { x:x2,y:y1,confidence:1 }, { x:x2,y:y2,confidence:1 }, { x:x1,y:y2,confidence:1 }] };
}

test("letterbox preserves aspect ratio and centers padding", () => {
  assert.deepEqual(letterboxGeometry(1280, 720), { scale: .5, padX: 0, padY: 140, width: 640, height: 360 });
});

test("decoder maps a candidate back to original coordinates", () => {
  const output = new Float32Array(17);
  output.set([320, 320, 320, 180, .9, 200, 275, 1, 440, 275, 1, 440, 365, 1, 200, 365, 1]);
  const result = decodePoseOutput(output, 1, letterboxGeometry(1280, 720), 1280, 720);
  assert.equal(result.length, 1); assert.deepEqual(result[0].box, [320, 180, 960, 540]);
  assert.deepEqual(result[0].keypoints.map(({ x, y }) => [x, y]), [[400,270],[880,270],[880,450],[400,450]]);
});

test("decoder rejects low-confidence candidates", () => {
  assert.equal(decodePoseOutput(new Float32Array(17), 1, letterboxGeometry(640, 640), 640, 640).length, 0);
});

test("NMS keeps the highest-confidence overlapping box", () => {
  const result = nonMaximumSuppression([detection([0,0,100,100], .7), detection([2,2,99,99], .9)]);
  assert.equal(result.length, 1); assert.equal(result[0].confidence, .9);
});

test("guidance covers direction, both distances, tilt, perspective, alignment, and missing detection", () => {
  assert.equal(guidanceFor(null, 1000, 1000), "searching");
  assert.equal(guidanceFor(detection([0,300,200,700], .9), 1000, 1000), "left");
  assert.equal(guidanceFor(detection([800,300,1000,700], .9), 1000, 1000), "right");
  assert.equal(guidanceFor(detection([400,400,600,600], .9), 1000, 1000), "closer");
  assert.equal(guidanceFor(detection([300,250,700,750], .9), 1000, 1000), "aligned");
  assert.equal(guidanceFor(detection([200,150,800,850], .9), 1000, 1000), "farther");
  const tilted = detection([300,250,700,750], .9); tilted.keypoints[1].y = 340; tilted.keypoints[2].y = 840;
  assert.equal(guidanceFor(tilted, 1000, 1000), "straighten");
  const skewed = detection([300,250,700,750], .9); skewed.keypoints[2].x = 560;
  assert.equal(guidanceFor(skewed, 1000, 1000), "straighten");
});

test("site calibration records the current slot size as its distance reference", () => {
  const slot = detection([300,250,700,750], .9);
  assert.equal(alignmentMeasurements(slot, 1000, 1000).area, .2);
  assert.equal(referenceAtCurrentDistance(slot, 1000, 1000).targetArea, .2);
});

test("guidance stabilization uses the majority of the latest three results", () => {
  const first = stabilizeGuidance([], "left");
  const second = stabilizeGuidance(first.history, "right");
  const third = stabilizeGuidance(second.history, "left");
  assert.equal(third.guidance, "left");
  assert.deepEqual(third.history, ["left", "right", "left"]);
});

test("temporary detection loss retains, fades, and then removes the last result", () => {
  const previous = detection([100, 100, 500, 500], .9);
  const first = recoverTemporaryLoss(previous, null, 0);
  assert.equal(first.detection, previous); assert.equal(first.state, "visible");
  const second = recoverTemporaryLoss(first.detection, null, first.misses);
  assert.equal(second.detection, previous); assert.equal(second.state, "fading");
  const third = recoverTemporaryLoss(second.detection, null, second.misses);
  assert.equal(third.detection, null); assert.equal(third.state, "lost");
  const recovered = recoverTemporaryLoss(previous, detection([200, 200, 600, 600], .8), 2);
  assert.equal(recovered.misses, 0); assert.equal(recovered.state, "visible");
});

test("automatic performance balancing returns bounded cooldowns", () => {
  assert.equal(balancedDelay(400), 600);
  assert.equal(balancedDelay(900), 250);
  assert.equal(balancedDelay(50), 900);
});

test("performance summary reports timing, update rate, and detection rate", () => {
  const summary = summarizePerformance([
    { inferenceMs: 100, completedAt: 0, detected: true },
    { inferenceMs: 200, completedAt: 1000, detected: false },
    { inferenceMs: 300, completedAt: 2000, detected: true },
  ]);
  assert.equal(summary.averageMs, 200); assert.equal(summary.p95Ms, 300);
  assert.equal(summary.updatesPerMinute, 60); assert.equal(summary.detectionRate, 2 / 3); assert.equal(summary.count, 3);
});

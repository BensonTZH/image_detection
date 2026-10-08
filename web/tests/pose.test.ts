import assert from "node:assert/strict";
import test from "node:test";
import { ACCEPTABILITY_MODEL_SIZE, ACCEPTABILITY_SUCCESS_THRESHOLD, captureAcceptabilityVeto, decodeAcceptabilityOutput, ENABLE_ACCEPTABILITY_CLASSIFIER, updateAcceptabilityWindow } from "../lib/acceptability.ts";
import { ACCEPTABILITY_SIDE_CORROBORATION_TOLERANCE, ACCEPTABILITY_VERTICAL_CORROBORATION_TOLERANCE, acceptabilityCorrectionFor, ALIGNMENT_HOLD_MS, alignmentExitReference, alignmentMeasurements, alignmentReferenceFromStored, ANGLE_VOTE_CORROBORATION_TOLERANCE, balancedDelay, CAPTURE_SIDE_PERSPECTIVE_TOLERANCE, CLASSIFIER_SIDE_PERSPECTIVE_TOLERANCE, captureAlignmentReady, captureAlignmentVeto, decodePoseOutput, DEFAULT_ALIGNMENT_REFERENCE, DETECTION_CONFIDENCE_THRESHOLD, guidanceFor, KEYPOINT_CONFIDENCE_THRESHOLD, letterboxGeometry, MODEL_SIZE, nonMaximumSuppression, poseCornerFailure, poseCornersReliable, recoverTemporaryLoss, recoveredDisplayDetection, referenceAtCurrentPose, sideAngleCorrectionFor, smoothDetection, stabilizeGuidance, summarizePerformance, updateAlignmentHold, updateAngleCaptureWindow, updatePoseReliabilityStreak, type Detection } from "../lib/pose.ts";

function detection(box: [number, number, number, number], confidence: number): Detection {
  const [x1, y1, x2, y2] = box;
  return { box, confidence, angleClass: "centered", keypoints: [{ x:x1,y:y1,confidence:1 }, { x:x2,y:y1,confidence:1 }, { x:x2,y:y2,confidence:1 }, { x:x1,y:y2,confidence:1 }] };
}

test("letterbox preserves aspect ratio and centers padding", () => {
  assert.equal(MODEL_SIZE, 320);
  assert.deepEqual(letterboxGeometry(1280, 720), { scale: .25, padX: 0, padY: 70, width: 320, height: 180 });
});

test("acceptability classifier uses the reviewed model contract and validated threshold", () => {
  assert.equal(ENABLE_ACCEPTABILITY_CLASSIFIER, true);
  assert.equal(ACCEPTABILITY_MODEL_SIZE, 320);
  assert.equal(ACCEPTABILITY_SUCCESS_THRESHOLD, .75);
  assert.deepEqual(decodeAcceptabilityOutput(new Float32Array([.1, .9])), {
    errorProbability: .10000000149011612,
    successProbability: .8999999761581421,
    accepted: true,
  });
  assert.equal(decodeAcceptabilityOutput(new Float32Array([.6, .4])).accepted, false);
  assert.throws(() => decodeAcceptabilityOutput(new Float32Array([1])), /Unexpected acceptability output length/);
});

test("acceptability uses the median of five readings after a three-reading warmup", () => {
  let state = updateAcceptabilityWindow([], .8);
  assert.equal(state.accepted, false);
  state = updateAcceptabilityWindow(state.history, .4);
  assert.equal(state.accepted, false);
  state = updateAcceptabilityWindow(state.history, .9);
  assert.equal(state.probability, .8);
  assert.equal(state.accepted, true);
  state = updateAcceptabilityWindow(state.history, .2);
  state = updateAcceptabilityWindow(state.history, .7);
  assert.equal(state.probability, .7);
  assert.equal(state.accepted, false);
});

test("19% acceptability blocks frontal geometry even without a correction direction", () => {
  const frontal = detection([350,300,650,700], .63);
  const poseVeto = captureAlignmentVeto(frontal, 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, true);
  assert.equal(poseVeto, null);
  assert.equal(acceptabilityCorrectionFor(frontal, 1000, 1000), null);
  const stable = updateAcceptabilityWindow([.19, .19], .19);
  const veto = captureAcceptabilityVeto(poseVeto, stable);
  assert.equal(veto, "acceptability");
  assert.deepEqual(updateAlignmentHold(1000, 2000, veto === null), { startedAt: null, ready: false });
});

test("acceptability blocks missing results and warmup, and preserves other vetoes", () => {
  assert.equal(captureAcceptabilityVeto(null, null), "acceptability");
  assert.equal(captureAcceptabilityVeto(null, updateAcceptabilityWindow([], .9)), "acceptability");
  assert.equal(captureAcceptabilityVeto(null, updateAcceptabilityWindow([.9], .9)), "acceptability");
  assert.equal(captureAcceptabilityVeto(null, updateAcceptabilityWindow([.75, .75], .75)), null);
  assert.equal(captureAcceptabilityVeto(null, updateAcceptabilityWindow([.749, .749], .749)), "acceptability");
  assert.equal(captureAcceptabilityVeto("corners", { accepted: true }), "corners");
  assert.equal(captureAcceptabilityVeto(null, null, false), null);
});

test("a fresh rejected frame cannot pass using an older passing median", () => {
  const stable = updateAcceptabilityWindow([.9, .9, .9, .9], .19);
  assert.equal(stable.probability, .9);
  assert.equal(captureAcceptabilityVeto(null, stable), "acceptability");
});

test("decoder maps a candidate back to original coordinates", () => {
  const output = new Float32Array(19);
  output.set([320, 320, 320, 180, .9, .05, .05, 200, 275, 1, 440, 275, 1, 440, 365, 1, 200, 365, 1]);
  const result = decodePoseOutput(output, 1, letterboxGeometry(1280, 720, 640), 1280, 720);
  assert.equal(result.length, 1); assert.deepEqual(result[0].box, [320, 180, 960, 540]);
  assert.equal(result[0].angleClass, "centered");
  assert.deepEqual(result[0].keypoints.map(({ x, y }) => [x, y]), [[400,270],[880,270],[880,450],[400,450]]);
});

test("decoder rejects low-confidence candidates", () => {
  assert.equal(DETECTION_CONFIDENCE_THRESHOLD, .35);
  const output = new Float32Array(19);
  output[4] = DETECTION_CONFIDENCE_THRESHOLD - .01;
  assert.equal(decodePoseOutput(output, 1, letterboxGeometry(640, 640), 640, 640).length, 0);
  output[4] = DETECTION_CONFIDENCE_THRESHOLD + .01;
  assert.equal(decodePoseOutput(output, 1, letterboxGeometry(640, 640), 640, 640).length, 1);
});

test("weak corners do not turn uncorroborated angle labels into side instructions", () => {
  const fromLeft = detection([350, 300, 650, 700], .9);
  fromLeft.angleClass = "camera_left";
  fromLeft.keypoints[0].confidence = .2;
  assert.equal(guidanceFor(fromLeft, 1000, 1000), "steady");

  const fromRight = detection([350, 300, 650, 700], .9);
  fromRight.angleClass = "camera_right";
  fromRight.keypoints[0].confidence = .2;
  assert.equal(guidanceFor(fromRight, 1000, 1000), "steady");

  const staleRightClassAtFront = detection([350, 300, 650, 700], .9);
  staleRightClassAtFront.angleClass = "camera_right";
  assert.equal(guidanceFor(staleRightClassAtFront, 1000, 1000), "aligned");
});

test("reliable corner perspective overrides an opposing angle class", () => {
  const viewedFromRight = detection([350,300,650,700], .9);
  viewedFromRight.angleClass = "camera_left";
  viewedFromRight.keypoints[3].y = 590;
  assert.equal(guidanceFor(viewedFromRight, 1000, 1000), "step_left");
});

test("NMS keeps the highest-confidence overlapping box", () => {
  const result = nonMaximumSuppression([detection([0,0,100,100], .7), detection([2,2,99,99], .9)]);
  assert.equal(result.length, 1); assert.equal(result[0].confidence, .9);
});

test("detection smoothing dampens opposing corner jitter", () => {
  const previous = detection([350,300,650,700], .8);
  previous.keypoints[1].y = 332; previous.keypoints[2].y = 732;
  const current = detection([350,300,650,700], 1);
  current.keypoints[1].y = 268; current.keypoints[2].y = 668;
  const smoothed = smoothDetection(previous, current);
  assert.equal(smoothed.keypoints[1].y, 287.2);
  assert.equal(smoothed.confidence, .94);
  assert.equal(guidanceFor(smoothed, 1000, 1000), "aligned");
});

test("guidance covers direction, both distances, tilt, perspective, alignment, and missing detection", () => {
  assert.equal(guidanceFor(null, 1000, 1000), "searching");
  assert.equal(guidanceFor(detection([0,300,200,700], .9), 1000, 1000), "left");
  assert.equal(guidanceFor(detection([800,300,1000,700], .9), 1000, 1000), "right");
  assert.equal(guidanceFor(detection([450,400,550,600], .9), 1000, 1000), "closer");
  assert.equal(guidanceFor(detection([300,250,700,750], .9), 1000, 1000), "aligned");
  assert.equal(guidanceFor(detection([200,150,800,850], .9), 1000, 1000), "farther");
  const tilted = detection([300,250,700,750], .9); tilted.keypoints[1].y = 340; tilted.keypoints[2].y = 840;
  assert.equal(guidanceFor(tilted, 1000, 1000), "rotate_right");
  const skewed = detection([300,250,700,750], .9); skewed.keypoints[2].x = 560;
  assert.equal(guidanceFor(skewed, 1000, 1000), "lower_phone");
});

test("default distance guidance accepts approximately three to twenty percent of the frame", () => {
  assert.equal(guidanceFor(detection([480,250,520,750], .9), 1000, 1000), "closer");
  assert.equal(guidanceFor(detection([460,250,540,750], .9), 1000, 1000), "aligned");
  assert.equal(guidanceFor(detection([310,250,690,750], .9), 1000, 1000), "aligned");
  assert.equal(guidanceFor(detection([290,250,710,750], .9), 1000, 1000), "farther");
});

test("default alignment rejects significant off-centre, roll, and perspective distortion", () => {
  assert.equal(guidanceFor(detection([230,300,530,700], .9), 1000, 1000), "left");
  assert.equal(guidanceFor(detection([280,300,580,700], .9), 1000, 1000), "aligned");
  assert.equal(guidanceFor(detection([470,300,770,700], .9), 1000, 1000), "right");
  assert.equal(guidanceFor(detection([420,300,720,700], .9), 1000, 1000), "aligned");

  const rolled = detection([350,300,650,700], .9);
  rolled.keypoints[1].y = 350; rolled.keypoints[2].y = 750;
  assert.equal(guidanceFor(rolled, 1000, 1000), "rotate_right");

  const rolledOtherWay = detection([350,300,650,700], .9);
  rolledOtherWay.keypoints[1].y = 250; rolledOtherWay.keypoints[2].y = 650;
  assert.equal(guidanceFor(rolledOtherWay, 1000, 1000), "rotate_left");

  const perspective = detection([350,300,650,700], .9);
  perspective.keypoints[2].y = 590;
  assert.equal(guidanceFor(perspective, 1000, 1000), "step_right");

  const perspectiveOtherWay = detection([350,300,650,700], .9);
  perspectiveOtherWay.keypoints[3].y = 590;
  assert.equal(guidanceFor(perspectiveOtherWay, 1000, 1000), "step_left");

  const tiltDown = detection([350,300,650,700], .9);
  tiltDown.keypoints[2].x = 600; tiltDown.keypoints[3].x = 400;
  assert.equal(guidanceFor(tiltDown, 1000, 1000), "lower_phone");

  const tiltUp = detection([350,300,650,700], .9);
  tiltUp.keypoints[1].x = 550;
  assert.equal(guidanceFor(tiltUp, 1000, 1000), "tilt_up");
});

test("negative vertical perspective retains its previous tolerance", () => {
  const mildVerticalPerspective = detection([350,300,650,700], .9);
  mildVerticalPerspective.keypoints[1].x = 580;
  assert.equal(guidanceFor(mildVerticalPerspective, 1000, 1000), "aligned");
  assert.equal(DEFAULT_ALIGNMENT_REFERENCE.maximumPerspectiveDifference, .22);
  assert.equal(DEFAULT_ALIGNMENT_REFERENCE.maximumVerticalPerspectiveDifference, .28);
});

test("weak or inconsistent corners cannot produce pose guidance or alignment", () => {
  const weak = detection([350,300,650,700], .26);
  weak.keypoints[2].confidence = KEYPOINT_CONFIDENCE_THRESHOLD - .01;
  assert.equal(poseCornersReliable(weak), false);
  assert.equal(guidanceFor(weak, 1000, 1000), "steady");

  const misplacedBottomCorner = detection([350,300,650,700], .26);
  misplacedBottomCorner.keypoints = [
    { x: 420, y: 330, confidence: .8 }, { x: 480, y: 340, confidence: .8 },
    { x: 500, y: 760, confidence: .8 }, { x: 420, y: 670, confidence: .8 },
  ];
  assert.equal(poseCornersReliable(misplacedBottomCorner), false);
  assert.equal(guidanceFor(misplacedBottomCorner, 1000, 1000), "steady");
});

test("box guidance remains available while corner reliability is low", () => {
  const offCentre = detection([0,300,200,700], .26);
  offCentre.keypoints.forEach((point) => { point.confidence = .2; });
  assert.equal(guidanceFor(offCentre, 1000, 1000), "left");
});

test("pose reliability requires two consecutive trustworthy readings", () => {
  const first = updatePoseReliabilityStreak(0, true);
  const second = updatePoseReliabilityStreak(first.count, true);
  assert.deepEqual(first, { count: 1, ready: false });
  assert.deepEqual(second, { count: 2, ready: true });
  assert.deepEqual(updatePoseReliabilityStreak(second.count, false), { count: 0, ready: false });
});

test("side-angle correction takes priority over image centering", () => {
  const viewedFromLeft = detection([0,250,300,750], .9);
  viewedFromLeft.keypoints[1].y = 314; viewedFromLeft.keypoints[2].y = 686;
  assert.equal(guidanceFor(viewedFromLeft, 1000, 1000), "step_right");

  const viewedFromRight = detection([700,250,1000,750], .9);
  viewedFromRight.keypoints[0].y = 314; viewedFromRight.keypoints[3].y = 686;
  assert.equal(guidanceFor(viewedFromRight, 1000, 1000), "step_left");
});

test("saved pose calibration keeps current safety tolerances", () => {
  const reference = alignmentReferenceFromStored({
    centerTolerance: .05,
    maximumPerspectiveDifference: .12,
    maximumVerticalPerspectiveDifference: .12,
    targetArea: .1,
    targetHeightDifference: .2,
  });
  assert.equal(reference.centerTolerance, .10);
  assert.equal(reference.maximumPerspectiveDifference, .22);
  assert.equal(reference.maximumVerticalPerspectiveDifference, .28);
  assert.equal(reference.targetArea, .1);
  assert.equal(reference.targetHeightDifference, 0);
});

test("confirmed alignment has wider exit limits than entry limits", () => {
  const nearEdge = detection([460,300,760,700], .9);
  assert.equal(guidanceFor(nearEdge, 1000, 1000), "right");
  assert.equal(guidanceFor(nearEdge, 1000, 1000, alignmentExitReference()), "aligned");
  assert.equal(guidanceFor(detection([530,300,830,700], .9), 1000, 1000, alignmentExitReference()), "right");
  assert.ok(Math.abs(alignmentExitReference().centerTolerance - .12) < 1e-9);
  assert.equal(alignmentExitReference().maximumRollDegrees, 9);
  assert.equal(alignmentExitReference().maximumPerspectiveDifference, .26);
  assert.ok(Math.abs(alignmentExitReference().maximumVerticalPerspectiveDifference - .32) < 1e-9);
});

test("moderate side perspective receives angle guidance before centering", () => {
  const moderateHorizontalDifference = detection([350,300,650,700], .9);
  moderateHorizontalDifference.keypoints[2].y = 630;
  assert.equal(guidanceFor(moderateHorizontalDifference, 1000, 1000), "step_right");
});

test("automatic capture and guidance share the absolute side-perspective limit", () => {
  const mildSideAngle = detection([350,300,650,700], .9);
  mildSideAngle.keypoints[2].y = 630;
  assert.equal(guidanceFor(mildSideAngle, 1000, 1000), "step_right");
  assert.equal(CAPTURE_SIDE_PERSPECTIVE_TOLERANCE, .12);
  assert.equal(captureAlignmentReady(mildSideAngle, 1000, 1000), false);
  assert.equal(captureAlignmentReady(detection([350,300,650,700], .9), 1000, 1000), true);
});

test("classifier-assisted mode keeps absolute geometry and angle-vote capture gates", () => {
  const sideView = detection([350,300,650,700], .9);
  sideView.angleClass = "camera_left";
  sideView.keypoints[1].y = 300;
  sideView.keypoints[2].y = 600;
  assert.equal(guidanceFor(sideView, 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, false), "aligned");
  assert.equal(captureAlignmentVeto(sideView, 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, false, false), "side_perspective");
  assert.equal(captureAlignmentVeto(sideView, 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, true, false), "side_perspective");
  assert.equal(guidanceFor(sideView, 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, true), "step_right");
  assert.equal(CLASSIFIER_SIDE_PERSPECTIVE_TOLERANCE, .12);

  const frontalWithNoisyVote = detection([350,300,650,700], .9);
  frontalWithNoisyVote.angleClass = "camera_right";
  assert.equal(captureAlignmentVeto(frontalWithNoisyVote, 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, false, false), null);
});

test("side votes only veto capture when absolute perspective is borderline", () => {
  assert.equal(ANGLE_VOTE_CORROBORATION_TOLERANCE, .08);
  const borderline = detection([350,300,650,700], .9);
  borderline.keypoints[1].y = 310;
  borderline.keypoints[2].y = 674;
  const sidePerspective = Math.abs(alignmentMeasurements(borderline, 1000, 1000).heightDifference);
  assert.ok(sidePerspective > ANGLE_VOTE_CORROBORATION_TOLERANCE);
  assert.ok(sidePerspective < CAPTURE_SIDE_PERSPECTIVE_TOLERANCE);
  assert.equal(captureAlignmentVeto(borderline, 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, false), "angle_vote");
  assert.equal(captureAlignmentVeto(borderline, 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, true), null);
});

test("frontal geometry is not blocked by a noisy side label", () => {
  const fromLeft = detection([350,300,650,700], .9);
  fromLeft.angleClass = "camera_left";
  assert.equal(guidanceFor(fromLeft, 1000, 1000), "aligned");
  assert.equal(captureAlignmentVeto(fromLeft, 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, false), null);
  assert.equal(captureAlignmentReady(fromLeft, 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, false), true);
});

test("acceptability correction directions require corroborating geometry", () => {
  assert.equal(ACCEPTABILITY_SIDE_CORROBORATION_TOLERANCE, .08);
  assert.equal(ACCEPTABILITY_VERTICAL_CORROBORATION_TOLERANCE, .20);
  assert.equal(acceptabilityCorrectionFor(detection([350,300,650,700], .9), 1000, 1000), null);

  const minorVerticalNoise = detection([350,300,650,700], .9);
  minorVerticalNoise.keypoints[1].x = 590;
  assert.equal(acceptabilityCorrectionFor(minorVerticalNoise, 1000, 1000), null);

  const highFieldView = detection([350,300,650,700], .9);
  highFieldView.keypoints[2].x = 618.05;
  highFieldView.keypoints[3].x = 381.95;
  assert.ok(Math.abs(alignmentMeasurements(highFieldView, 1000, 1000).widthDifference - .213) < 1e-9);
  assert.equal(acceptabilityCorrectionFor(highFieldView, 1000, 1000), "lower_phone");

  const viewedFromLeft = detection([350,300,650,700], .9);
  viewedFromLeft.keypoints[2].y = 660;
  assert.equal(acceptabilityCorrectionFor(viewedFromLeft, 1000, 1000), "step_right");

  const viewedFromRight = detection([350,300,650,700], .9);
  viewedFromRight.keypoints[3].y = 660;
  assert.equal(acceptabilityCorrectionFor(viewedFromRight, 1000, 1000), "step_left");

  const tiltedUp = detection([350,300,650,700], .9);
  tiltedUp.keypoints[1].x = 560;
  assert.equal(acceptabilityCorrectionFor(tiltedUp, 1000, 1000), "tilt_up");
});

test("off-centre side view from field diagnostics cannot use the frontal override", () => {
  const fieldView = detection([240,300,540,700], .9);
  fieldView.angleClass = "camera_left";
  assert.ok(Math.abs(alignmentMeasurements(fieldView, 1000, 1000).centerX - .39) < 1e-9);
  assert.equal(guidanceFor(fieldView, 1000, 1000), "left");
  assert.equal(captureAlignmentVeto(fieldView, 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, false), "position");
  assert.equal(captureAlignmentReady(fieldView, 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, false), false);
});

test("minor perspective noise is not blocked by the angle vote", () => {
  const borderline = detection([350,300,650,700], .9);
  borderline.keypoints[1].y = 316;
  borderline.keypoints[2].y = 684;
  assert.equal(captureAlignmentVeto(borderline, 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, false), null);
  assert.equal(captureAlignmentReady(borderline, 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, true), true);
});

test("angle voting uses a centered majority and needs four matching side votes", () => {
  const mixed = updateAngleCaptureWindow(["camera_left", "camera_right", "centered", "camera_left"], "centered");
  assert.equal(mixed.ready, false);
  assert.equal(mixed.sideDirection, null);

  const centeredMajority = updateAngleCaptureWindow(["centered", "camera_left", "centered", "camera_right"], "centered");
  assert.equal(centeredMajority.ready, true);
  assert.equal(centeredMajority.sideDirection, null);

  const staleCenteredMajority = updateAngleCaptureWindow(["centered", "centered", "camera_left", "centered"], "camera_left");
  assert.equal(staleCenteredMajority.ready, true);

  const consistentlyRight = updateAngleCaptureWindow(["camera_right", "camera_right", "centered", "camera_right"], "camera_right");
  assert.equal(consistentlyRight.ready, false);
  assert.equal(consistentlyRight.sideDirection, "camera_right");
  const alternatingSides = updateAngleCaptureWindow(["camera_left", "camera_right", "camera_left", "camera_right"], "camera_left");
  assert.equal(alternatingSides.ready, false);
  assert.equal(alternatingSides.sideDirection, null);
  const splitSides = updateAngleCaptureWindow(["camera_left", "camera_right", "centered", "camera_right"], "camera_left");
  assert.equal(splitSides.sideDirection, null);
  assert.deepEqual(updateAngleCaptureWindow(consistentlyRight.history, null), { history: [], ready: false, sideDirection: null });
});

test("four matching side votes require supporting perspective to veto capture", () => {
  const narrowSideView = detection([350,300,650,700], .9);
  assert.equal(captureAlignmentVeto(narrowSideView, 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, true, true, true), null);
  assert.equal(captureAlignmentVeto(narrowSideView, 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, true, true, false), null);
  narrowSideView.keypoints[2].y = 660;
  narrowSideView.angleClass = "camera_right";
  assert.equal(captureAlignmentVeto(narrowSideView, 1000, 1000, DEFAULT_ALIGNMENT_REFERENCE, true, true, true, true), "angle_vote");
  assert.equal(sideAngleCorrectionFor(narrowSideView, 1000, 1000), "step_right");
  narrowSideView.keypoints[2].y = 680;
  assert.equal(sideAngleCorrectionFor(narrowSideView, 1000, 1000), null);
  assert.equal(acceptabilityCorrectionFor(narrowSideView, 1000, 1000), null);
});

test("site calibration records distance and roll without normalizing perspective", () => {
  const slot = detection([350,300,650,700], .9);
  slot.keypoints[1].y = 318; slot.keypoints[2].y = 638;
  const measured = alignmentMeasurements(slot, 1000, 1000);
  const reference = referenceAtCurrentPose(slot, 1000, 1000);
  assert.equal(reference.targetArea, measured.area);
  assert.equal(reference.targetRollDegrees, measured.rollDegrees);
  assert.equal(reference.targetWidthDifference, 0);
  assert.equal(reference.targetHeightDifference, 0);
  assert.notEqual(guidanceFor(slot, 1000, 1000, reference), "aligned");
});

test("guidance stabilization uses the majority of the latest three results", () => {
  const first = stabilizeGuidance([], "left");
  const second = stabilizeGuidance(first.history, "closer");
  const third = stabilizeGuidance(second.history, "left");
  assert.equal(third.guidance, "left");
  assert.deepEqual(third.history, ["left", "closer", "left"]);
});

test("guidance stabilization does not switch on one opposing result", () => {
  const left = stabilizeGuidance([], "left");
  const reinforcedLeft = stabilizeGuidance(left.history, "left");
  const right = stabilizeGuidance(reinforcedLeft.history, "right");
  assert.equal(right.guidance, "left");
  assert.deepEqual(right.history, ["left", "left", "right"]);

  const tiltUp = stabilizeGuidance([], "tilt_up");
  const tiltDown = stabilizeGuidance(tiltUp.history, "tilt_down");
  assert.equal(tiltDown.guidance, "tilt_down");
});

test("missing detection immediately cancels previous movement guidance", () => {
  const lost = stabilizeGuidance(["tilt_up", "tilt_up", "tilt_up"], "searching");
  assert.deepEqual(lost, { history: ["searching"], guidance: "searching" });
});

test("one unreliable reading does not replace a stable pose instruction", () => {
  const first = stabilizeGuidance(["tilt_up", "tilt_up", "tilt_up"], "steady");
  assert.deepEqual(first, { history: ["tilt_up", "tilt_up", "steady"], guidance: "tilt_up" });
  const second = stabilizeGuidance(first.history, "steady");
  assert.deepEqual(second, { history: ["tilt_up", "steady", "steady"], guidance: "steady" });
});

test("automatic capture requires 800 milliseconds of continuous alignment", () => {
  const first = updateAlignmentHold(null, 100, true);
  assert.deepEqual(first, { startedAt: 100, ready: false });
  assert.equal(ALIGNMENT_HOLD_MS, 800);
  assert.equal(updateAlignmentHold(first.startedAt, 899, true).ready, false);
  assert.equal(updateAlignmentHold(first.startedAt, 900, true).ready, true);
  assert.deepEqual(updateAlignmentHold(first.startedAt, 900, false), { startedAt: null, ready: false });
});

test("temporary loss fades the last result for one second regardless of missed frame count", () => {
  const previous = detection([100, 100, 500, 500], .9);
  let state = recoverTemporaryLoss(null, previous, 0, null, 100);
  assert.equal(state.state, "visible");
  for (const now of [130, 160, 300, 600, 900, 1099]) {
    state = recoverTemporaryLoss(state.detection, null, state.misses, state.lastSeenAt, now);
    assert.equal(state.detection, previous);
    assert.equal(state.state, "fading");
    assert.equal(state.lastSeenAt, 100);
  }
  state = recoverTemporaryLoss(state.detection, null, state.misses, state.lastSeenAt, 1100);
  assert.equal(state.detection, null); assert.equal(state.state, "lost");
  assert.equal(recoverTemporaryLoss(previous, null, 0, 100, 1500).state, "lost");
  assert.equal(recoverTemporaryLoss(null, null, 0, null, 10).state, "lost");
});

test("display box survives a short miss and smoothly resumes with fresh observations", () => {
  const previous = detection([100, 100, 500, 500], .9);
  const current = detection([120, 100, 520, 500], .8);
  const missing = recoverTemporaryLoss(previous, null, 0, 100, 400);
  assert.equal(recoveredDisplayDetection(previous, null, missing, 100, 400), previous);
  const resumed = recoverTemporaryLoss(missing.detection, current, missing.misses, missing.lastSeenAt, 600);
  assert.equal(resumed.state, "visible"); assert.equal(resumed.misses, 0); assert.equal(resumed.lastSeenAt, 600);
  assert.deepEqual(recoveredDisplayDetection(previous, current, resumed, 100, 600), smoothDetection(previous, current));
  const expired = recoverTemporaryLoss(previous, null, 0, 100, 1100);
  assert.equal(recoveredDisplayDetection(previous, null, expired, 100, 1100), null);
  const late = recoverTemporaryLoss(previous, current, 0, 100, 1500);
  assert.deepEqual(recoveredDisplayDetection(previous, current, late, 100, 1500), current);
});

test("automatic performance balancing returns bounded cooldowns", () => {
  assert.equal(balancedDelay(400), 30);
  assert.equal(balancedDelay(900), 30);
  assert.equal(balancedDelay(50), 100);
  assert.equal(balancedDelay(200), 50);
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


test("corner diagnostics identify the failure without weakening reliability checks", () => {
  const normal=detection([350,300,650,700],.9);
  assert.equal(poseCornerFailure(normal),null);
  const weak=structuredClone(normal);weak.keypoints[2].confidence=.2;
  assert.equal(poseCornerFailure(weak),"Low bottom-right confidence (20%)");
  assert.equal(poseCornersReliable(weak),false);
  const malformed=structuredClone(normal);malformed.keypoints[2].y=950;
  assert.equal(poseCornerFailure(malformed),"Top/bottom edges disagree");
  assert.equal(poseCornersReliable(malformed),false);
  const invalid=structuredClone(normal);invalid.keypoints[0].confidence=NaN;
  assert.equal(poseCornerFailure(invalid),"Invalid corner coordinates or confidence");
  assert.equal(poseCornersReliable(invalid),false);
});

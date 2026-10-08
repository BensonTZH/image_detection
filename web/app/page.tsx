"use client";

import { correctionResolved, updateActionInstruction, type ActionInstructionState } from "../lib/action-guidance";

import { evidenceGapLimit, freshInference, type CompletedInference } from "../lib/inference-timing";
import { stabilizeMeasurements, type MeasurementHistory, type Measurements } from "../lib/measurement-stability";
import { confirmedSideGuidance, emptySideAngleState, updateSideAngleState, STRONG_SIDE_THRESHOLD } from "../lib/side-angle";
import { freezeCameraFrame, encodeValidatedFrame } from "../lib/capture-frame";
import { updateTrackingEvidence, recoverPoseReliability, preserveAcceptabilityHistory, type TrackingEvidence } from "../lib/tracking-continuity";
import { highAngleInput } from "../lib/high-angle-input";
import { canPauseCapture, chooseLiveGuidance, emptyCaptureHold, emptyInstructionState, liveGeometryDecision, liveSideCorrection, waitingGuidanceCopy, updateCaptureHold } from "../lib/live-guidance";
import { ChangeEvent, useEffect, useRef, useState } from "react";
import type * as Ort from "onnxruntime-web";
import { decodeHighAngleOutput, emptyHighAngleState, ENABLE_HIGH_ANGLE_CLASSIFIER, HIGH_ANGLE_MODEL_SIZE, HIGH_ANGLE_MODEL_URL, HIGH_ANGLE_THRESHOLD, updateHighAngleState } from "../lib/high-angle";
import { ACCEPTABILITY_MODEL_SIZE, ACCEPTABILITY_MODEL_URL, ACCEPTABILITY_SUCCESS_THRESHOLD, captureAcceptabilityVeto, decodeAcceptabilityOutput, ENABLE_ACCEPTABILITY_CLASSIFIER, updateAcceptabilityWindow } from "../lib/acceptability";
import { acceptabilityCorrectionFor, alignmentMeasurements, alignmentReferenceFromStored, balancedDelay, CAPTURE_SIDE_PERSPECTIVE_TOLERANCE, decodePoseOutput, DEFAULT_ALIGNMENT_REFERENCE, DETECTION_CONFIDENCE_THRESHOLD, DETECTION_DISPLAY_GRACE_MS, guidanceFor, KEYPOINT_CONFIDENCE_THRESHOLD, letterboxGeometry, MODEL_SIZE, poseCornersReliable, POSE_RELIABILITY_REQUIRED_READINGS, recoverTemporaryLoss, recoveredDisplayDetection, referenceAtCurrentPose, summarizePerformance, updateAngleCaptureWindow, type AlignmentReference, type AngleClass, type CaptureVeto, type Detection, type Guidance, type PerformanceSample } from "../lib/pose";

const guidance: Record<Guidance, { eyebrow: string; title: string; detail: string }> = {
  searching: { eyebrow: "No confident detection", title: "Stand in front of the return slot", detail: "Point your camera at the return slot." },
  steady: { eyebrow: "Slot found", title: "Keep the slot in view", detail: "Keep the whole opening visible." },
  framing: { eyebrow: "Slot found", title: "Move back", detail: "Step back slightly so the full slot stays visible." },
  left: { eyebrow: "Slot found", title: "Move left", detail: "Move your phone slightly to the left." },
  right: { eyebrow: "Slot found", title: "Move right", detail: "Move your phone slightly to the right." },
  step_left: { eyebrow: "Slot found", title: "Move left", detail: "Move slightly to the left." },
  step_right: { eyebrow: "Slot found", title: "Move right", detail: "Move slightly to the right." },
  closer: { eyebrow: "Slot found", title: "Move closer", detail: "Walk forward until the opening fills more of the view." },
  farther: { eyebrow: "Slot found", title: "Move back", detail: "Step back until the opening is the right size." },
  rotate_left: { eyebrow: "Slot found", title: "Rotate phone left", detail: "Turn the top of your phone slightly left." },
  rotate_right: { eyebrow: "Slot found", title: "Rotate phone right", detail: "Turn the top of your phone slightly right." },
  tilt_up: { eyebrow: "Slot found", title: "Tilt phone up", detail: "Tilt your phone slightly upward." },
  tilt_down: { eyebrow: "Slot found", title: "Tilt phone down", detail: "Tilt your phone slightly downward." },
  lower_phone: { eyebrow: "Slot found", title: "Lower your phone slightly", detail: "Move your phone down to the height of the slot’s centre." },
  aligned: { eyebrow: "Angle accepted", title: "Hold steady for photo", detail: "Keep your phone still. Taking photo automatically." },
};
const demoOrder: Guidance[] = ["searching", "steady", "left", "right", "step_left", "step_right", "closer", "farther", "rotate_left", "rotate_right", "tilt_up", "tilt_down", "lower_phone", "aligned"];
const ALIGNMENT_STORAGE_KEY = "cupdetector-alignment-reference-v3";
// Temporary testing UI: change this single flag to false to hide the entire
// post-capture diagnostics panel without changing detection behavior.
const SHOW_CAPTURE_DIAGNOSTICS = true;
const captureVetoLabel: Record<CaptureVeto, string> = {
  no_detection: "Waiting for slot",
  measurement_stability: "Confirming stable measurements",
  stale_frame: "Waiting for a fresh reading",
  corners: "Waiting for stable corners",
  position: "Follow the movement guide",
  side_perspective: "Face the slot more directly",
  angle_vote: "Checking side angle",
  side_class: "Checking confident side angle",
  acceptability: "Checking viewing angle",
  high_angle: "Checking phone height",
};

type CaptureDiagnostics = {
  captureVeto: CaptureVeto | null;
  holdReady: boolean;
  centerPercent: number; centerMinimumPercent: number; centerMaximumPercent: number;
  areaPercent: number; areaMinimumPercent: number; areaMaximumPercent: number;
  sidePerspectivePercent: number; sidePerspectiveLimitPercent: number;
  verticalPerspectivePercent: number; verticalPerspectiveLimitPercent: number; positiveVerticalPerspectiveLimitPercent: number;
  rotationDegrees: number; rotationLimitDegrees: number;
  detectionConfidencePercent: number; minimumCornerConfidencePercent: number;
  cornersReliable: boolean; reliableReadings: number;
  angleHistory: AngleClass[];
  acceptabilityProbability: number | null;
  highAngleProbability: number | null;
  strongSideReadings: number;
};

function CaptureDiagnosticsPanel({ values, onHide }: { values: CaptureDiagnostics; onHide: () => void }) {
  const signed = (value: number, suffix: string) => `${value >= 0 ? "+" : ""}${value.toFixed(1)}${suffix}`;
  return <section className="capture-diagnostics" aria-label="Temporary capture diagnostics">
    <header><div className="diagnostics-heading"><strong>Capture diagnostics</strong><span>Temporary testing panel</span></div><div className="diagnostics-actions"><b>{values.captureVeto === null && values.holdReady ? "Passed" : "Check"}</b><button type="button" onClick={onHide}>Hide metrics</button></div></header>
    <div className="diagnostics-grid">
      <div><span>Centre</span><strong>{values.centerPercent.toFixed(1)}%</strong><small>{values.centerMinimumPercent.toFixed(0)}–{values.centerMaximumPercent.toFixed(0)}%</small></div>
      <div><span>Frame area</span><strong>{values.areaPercent.toFixed(1)}%</strong><small>{values.areaMinimumPercent.toFixed(0)}–{values.areaMaximumPercent.toFixed(0)}%</small></div>
      <div><span>Side perspective</span><strong>{signed(values.sidePerspectivePercent, "%")}</strong><small>limit ±{values.sidePerspectiveLimitPercent.toFixed(0)}%</small></div>
      <div><span>Vertical perspective</span><strong>{signed(values.verticalPerspectivePercent, "%")}</strong><small>limits −{values.verticalPerspectiveLimitPercent.toFixed(0)}% / +{values.positiveVerticalPerspectiveLimitPercent.toFixed(0)}%</small></div>
      <div><span>Rotation</span><strong>{signed(values.rotationDegrees, "°")}</strong><small>limit ±{values.rotationLimitDegrees.toFixed(0)}°</small></div>
      <div><span>Detection</span><strong>{values.detectionConfidencePercent.toFixed(0)}%</strong><small>minimum {(DETECTION_CONFIDENCE_THRESHOLD * 100).toFixed(0)}%</small></div>
      <div><span>Lowest corner</span><strong>{values.minimumCornerConfidencePercent.toFixed(0)}%</strong><small>minimum {(KEYPOINT_CONFIDENCE_THRESHOLD * 100).toFixed(0)}%</small></div>
      <div><span>Corner geometry</span><strong>{values.cornersReliable ? "Reliable" : "Unreliable"}</strong><small>{values.reliableReadings}/{POSE_RELIABILITY_REQUIRED_READINGS} readings</small></div>
      <div><span>Angle votes</span><strong>{values.angleHistory.map((value) => value === "centered" ? "C" : value === "camera_left" ? "L" : "R").join(" ") || "—"}</strong><small>C centre · L/R side</small></div>
      <div><span>Strong side readings</span><strong>{values.strongSideReadings}/5</strong><small>4 matching at {(STRONG_SIDE_THRESHOLD * 100).toFixed(0)}% to guide</small></div>
      <div><span>High angle</span><strong>{values.highAngleProbability === null ? "Off" : `${(values.highAngleProbability * 100).toFixed(0)}%`}</strong><small>warning at {(HIGH_ANGLE_THRESHOLD * 100).toFixed(0)}%</small></div>
      <div><span>Acceptability</span><strong>{values.acceptabilityProbability === null ? "Bypassed" : `${(values.acceptabilityProbability * 100).toFixed(0)}%`}</strong><small>minimum {(ACCEPTABILITY_SUCCESS_THRESHOLD * 100).toFixed(0)}%</small></div>
    </div>
  </section>;
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const poseSessionRef = useRef<Ort.InferenceSession | null>(null);
  const acceptabilitySessionRef = useRef<Ort.InferenceSession | null>(null);
  const highAngleSessionRef = useRef<Ort.InferenceSession | null>(null);
  const highAngleRef = useRef(emptyHighAngleState());
  const sideAngleRef = useRef(emptySideAngleState());
  const [highAngleProbability, setHighAngleProbability] = useState<number | null>(null);
  const ortRef = useRef<typeof import("onnxruntime-web") | null>(null);
  const inputCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const highAngleCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const liveActiveRef = useRef(false);
  const liveTimerRef = useRef<number | null>(null);
  const overlayExpiryRef = useRef<number | null>(null);
  const lastDetectionAtRef = useRef<number | null>(null);
  const imageUrlRef = useRef<string | null>(null);
  const samplesRef = useRef<PerformanceSample[]>([]);
  const lastDetectionRef = useRef<Detection | null>(null);
  const smoothedDetectionRef = useRef<Detection | null>(null);
  const missedDetectionsRef = useRef(0);
  const instructionRef = useRef<ActionInstructionState>(emptyInstructionState());
  const cameraRunRef = useRef(0);
  const angleClassHistoryRef = useRef<AngleClass[]>([]);
  const poseReliabilityStreakRef = useRef(0);
  const measurementHistoryRef = useRef<MeasurementHistory>([]);
  const completedInferenceRef = useRef<CompletedInference>(null);
  const trackingEvidenceRef = useRef<TrackingEvidence>(null);
  const acceptabilitySampledAtRef = useRef<number | null>(null);
  const captureInProgressRef = useRef(false);
  const acceptabilityProbabilityRef = useRef<number | null>(null);
  const acceptabilityHistoryRef = useRef<number[]>([]);
  const captureHoldRef = useRef(emptyCaptureHold());
  const alignmentReferenceRef = useRef<AlignmentReference>(DEFAULT_ALIGNMENT_REFERENCE);

  const [cameraState, setCameraState] = useState<"idle" | "starting" | "active" | "blocked">("idle");
  const [mode, setMode] = useState<"home" | "camera" | "demo" | "image">("home");
  const [guide, setGuide] = useState<Guidance>("searching");
  const [demoIndex, setDemoIndex] = useState(0);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [modelState, setModelState] = useState<"idle" | "loading" | "running" | "done" | "error">("idle");
  const [result, setResult] = useState<{ detection: Detection | null; milliseconds: number } | null>(null);
  const [error, setError] = useState("");
  const [showStats, setShowStats] = useState(false);
  const [modelLoadMs, setModelLoadMs] = useState(0);
  const [metrics, setMetrics] = useState(() => summarizePerformance([]));
  const [lossState, setLossState] = useState<"visible" | "fading" | "lost">("lost");
  const [liveGeometryStats, setLiveGeometryStats] = useState<{ vertical: number | null; rawVertical: number | null; corners: string; correction: Guidance }>({ vertical: null, rawVertical: null, corners: "No slot detected", correction: "searching" });
  const [captured, setCaptured] = useState(false);
  const [captureDiagnostics, setCaptureDiagnostics] = useState<CaptureDiagnostics | null>(null);
  const [captureDiagnosticsVisible, setCaptureDiagnosticsVisible] = useState(true);
  const [calibrationSaved, setCalibrationSaved] = useState(false);
  const [captureVeto, setCaptureVeto] = useState<CaptureVeto | null>("no_detection");
  const [angleHistory, setAngleHistory] = useState<AngleClass[]>([]);
  const [acceptabilityProbability, setAcceptabilityProbability] = useState<number | null>(null);

  function stopCamera() {
    liveActiveRef.current = false;
    completedInferenceRef.current = null;
    trackingEvidenceRef.current = null;
    acceptabilitySampledAtRef.current = null;
    cameraRunRef.current += 1;
    captureHoldRef.current = emptyCaptureHold();
    sideAngleRef.current = emptySideAngleState();
    if (liveTimerRef.current !== null) window.clearTimeout(liveTimerRef.current);
    liveTimerRef.current = null;
    if (overlayExpiryRef.current !== null) window.clearTimeout(overlayExpiryRef.current);
    overlayExpiryRef.current = null;
    lastDetectionAtRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    const overlay = overlayRef.current;
    overlay?.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
  }

  useEffect(() => {
    try { const stored = localStorage.getItem(ALIGNMENT_STORAGE_KEY); if (stored) alignmentReferenceRef.current = alignmentReferenceFromStored(JSON.parse(stored)); } catch { alignmentReferenceRef.current = DEFAULT_ALIGNMENT_REFERENCE; }
    return () => { stopCamera(); if (imageUrlRef.current?.startsWith("blob:")) URL.revokeObjectURL(imageUrlRef.current); };
  }, []);

  async function ensureSessions() {
    if (!poseSessionRef.current) {
      const wasmOrt = await import("onnxruntime-web/wasm"); wasmOrt.env.wasm.numThreads = 1;
      poseSessionRef.current = await wasmOrt.InferenceSession.create("/models/slot-pose.onnx?v=angle-pose-baseline-v3-left-angle", { executionProviders: ["wasm"] });
      ortRef.current = wasmOrt;
    }
    if (!ortRef.current) throw new Error("Detector runtime is unavailable.");
    if (ENABLE_ACCEPTABILITY_CLASSIFIER && !acceptabilitySessionRef.current) {
      acceptabilitySessionRef.current = await ortRef.current.InferenceSession.create(ACCEPTABILITY_MODEL_URL, { executionProviders: ["wasm"] });
    }
    if (ENABLE_HIGH_ANGLE_CLASSIFIER && !highAngleSessionRef.current) {
      highAngleSessionRef.current = await ortRef.current.InferenceSession.create(HIGH_ANGLE_MODEL_URL, { executionProviders: ["wasm"] });
    }
    return { highAngleSession: highAngleSessionRef.current, ort: ortRef.current, poseSession: poseSessionRef.current, acceptabilitySession: acceptabilitySessionRef.current };
  }

  async function infer(source: CanvasImageSource, width: number, height: number) {
    const canvas = inputCanvasRef.current ?? document.createElement("canvas"); inputCanvasRef.current = canvas;
    canvas.width = MODEL_SIZE; canvas.height = MODEL_SIZE;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Canvas is unavailable.");
    const geometry = letterboxGeometry(width, height);
    context.fillStyle = "rgb(114,114,114)"; context.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
    context.drawImage(source, geometry.padX, geometry.padY, geometry.width, geometry.height);
    const rgba = context.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
    const chw = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE); const plane = MODEL_SIZE * MODEL_SIZE;
    for (let pixel = 0; pixel < plane; pixel++) { chw[pixel] = rgba[pixel * 4] / 255; chw[plane + pixel] = rgba[pixel * 4 + 1] / 255; chw[plane * 2 + pixel] = rgba[pixel * 4 + 2] / 255; }
    const { ort, poseSession } = await ensureSessions();
    const started = performance.now();
    const outputs = await poseSession.run({ images: new ort.Tensor("float32", chw, [1, 3, MODEL_SIZE, MODEL_SIZE]) });
    const milliseconds = performance.now() - started;
    const tensor = outputs.output0;
    if (!tensor || tensor.dims.length !== 3 || tensor.dims[1] !== 19) throw new Error("The model returned an unexpected output shape.");
    const detection = decodePoseOutput(tensor.data as Float32Array, Number(tensor.dims[2]), geometry, width, height)[0] ?? null;
    return { detection, milliseconds, input: chw };
  }

  async function classifyAcceptability(input: Float32Array) {
    if (!ENABLE_ACCEPTABILITY_CLASSIFIER) return { successProbability: null, accepted: true, milliseconds: 0 };
    const { ort, acceptabilitySession } = await ensureSessions();
    if (!acceptabilitySession) throw new Error("Acceptability classifier is unavailable.");
    const started = performance.now();
    const outputs = await acceptabilitySession.run({ images: new ort.Tensor("float32", input, [1, 3, ACCEPTABILITY_MODEL_SIZE, ACCEPTABILITY_MODEL_SIZE]) });
    const milliseconds = performance.now() - started;
    const tensor = outputs.output0;
    if (!tensor || tensor.dims.length !== 2 || tensor.dims[0] !== 1 || tensor.dims[1] !== 2) throw new Error("The acceptability model returned an unexpected output shape.");
    return { ...decodeAcceptabilityOutput(tensor.data as Float32Array), milliseconds };
  }

  async function classifyHighAngle(source: CanvasImageSource, width: number, height: number, detection: Detection) {
    if (!ENABLE_HIGH_ANGLE_CLASSIFIER) return { probability: null, milliseconds: 0 };
    const { ort, highAngleSession } = await ensureSessions();
    if (!highAngleSession) throw new Error("High-angle classifier is unavailable.");
    const started = performance.now();
    const canvas = highAngleCanvasRef.current ?? document.createElement("canvas");
    highAngleCanvasRef.current = canvas;
    const input = highAngleInput(source, width, height, detection.box, canvas);
    const outputs = await highAngleSession.run({ images: new ort.Tensor("float32", input, [1, 3, HIGH_ANGLE_MODEL_SIZE, HIGH_ANGLE_MODEL_SIZE]) });
    const tensor = outputs.output0;
    if (!tensor || tensor.dims.length !== 2 || tensor.dims[0] !== 1 || tensor.dims[1] !== 2) throw new Error("Unexpected high-angle output shape.");
    return { probability: decodeHighAngleOutput(tensor.data as Float32Array), milliseconds: performance.now() - started };
  }

  function drawDetection(context: CanvasRenderingContext2D, detection: Detection | null, width: number, showCorners = detection ? poseCornersReliable(detection) : false) {
    if (!detection) return;
    const line = Math.max(1.5, width / 500); context.lineWidth = line; context.strokeStyle = "rgba(185,242,39,.65)"; context.fillStyle = "rgba(185,242,39,.65)";
    const [x1, y1, x2, y2] = detection.box; context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (!showCorners) return;
    context.beginPath(); detection.keypoints.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y)); context.closePath(); context.stroke();
    detection.keypoints.forEach((point) => { context.beginPath(); context.arc(point.x, point.y, line * 1.8, 0, Math.PI * 2); context.fill(); });
  }

  function drawLiveOverlay(detection: Detection | null, width: number, height: number, showCorners: boolean) {
    const canvas = overlayRef.current; if (!canvas) return;
    canvas.width = width; canvas.height = height;
    const context = canvas.getContext("2d"); if (!context) return;
    context.clearRect(0, 0, width, height); drawDetection(context, detection, width, showCorners);
  }

  async function runLiveLoop() {
    if (!liveActiveRef.current) return;
    const video = videoRef.current;
    if (!video || document.hidden || video.readyState < 2) {
      completedInferenceRef.current = null;
      captureHoldRef.current = emptyCaptureHold();
      liveTimerRef.current = window.setTimeout(runLiveLoop, 500); return;
    }
    const run = cameraRunRef.current;
    const stillCurrent = () => liveActiveRef.current && run === cameraRunRef.current;
    try {
      const sampledAt = performance.now();
      const maxEvidenceGap = evidenceGapLimit(completedInferenceRef.current, sampledAt);
      const frame = freezeCameraFrame(video);
      const current = await infer(frame, frame.width, frame.height);
      if (!stillCurrent()) return;
      const currentCornersReliable = current.detection !== null && poseCornersReliable(current.detection);
      const continuity = updateTrackingEvidence(trackingEvidenceRef.current, current.detection, currentCornersReliable, sampledAt, maxEvidenceGap);
      trackingEvidenceRef.current = continuity.state;
      if (!continuity.sameTrack) {
        poseReliabilityStreakRef.current = 0;
        sideAngleRef.current = emptySideAngleState();
        angleClassHistoryRef.current = [];
        highAngleRef.current = emptyHighAngleState();
        acceptabilityHistoryRef.current = [];
        acceptabilitySampledAtRef.current = null;
        instructionRef.current = emptyInstructionState();
        captureHoldRef.current = emptyCaptureHold();
        smoothedDetectionRef.current = null;
      }
      const observedAt = performance.now();
      const previousSeenAt = lastDetectionAtRef.current;
      const recovered = recoverTemporaryLoss(lastDetectionRef.current, current.detection, missedDetectionsRef.current, previousSeenAt, observedAt);
      const displayedDetection = recoveredDisplayDetection(smoothedDetectionRef.current, current.detection, recovered, previousSeenAt, observedAt);
      lastDetectionRef.current = recovered.detection; missedDetectionsRef.current = recovered.misses;
      lastDetectionAtRef.current = recovered.lastSeenAt; setLossState(recovered.state);
      smoothedDetectionRef.current = displayedDetection;
      if (overlayExpiryRef.current !== null) window.clearTimeout(overlayExpiryRef.current);
      overlayExpiryRef.current = null;
      if (recovered.state === "fading" && recovered.lastSeenAt !== null) {
        const seenAt = recovered.lastSeenAt;
        // Expire even when inference is slow. Repeated misses cannot extend it.
        overlayExpiryRef.current = window.setTimeout(() => {
          if (!stillCurrent() || lastDetectionAtRef.current !== seenAt) return;
          overlayExpiryRef.current = null;
          lastDetectionRef.current = null; smoothedDetectionRef.current = null; lastDetectionAtRef.current = null;
          setLossState("lost"); setGuide("searching"); instructionRef.current = emptyInstructionState();
          drawLiveOverlay(null, frame.width, frame.height, false);
        }, Math.max(0, DETECTION_DISPLAY_GRACE_MS - (observedAt - seenAt)));
      }
      const poseReliability = recoverPoseReliability(poseReliabilityStreakRef.current, currentCornersReliable, continuity.retain);
      poseReliabilityStreakRef.current = poseReliability.count;
      const stable = stabilizeMeasurements(measurementHistoryRef.current, current.detection,
        frame.width, frame.height, sampledAt, continuity.sameTrack, maxEvidenceGap);
      measurementHistoryRef.current = stable.history;
      const geometryReliable = currentCornersReliable;
      if (currentCornersReliable || !continuity.retain) {
        sideAngleRef.current = updateSideAngleState(sideAngleRef.current, current.detection, currentCornersReliable, sampledAt, undefined, maxEvidenceGap);
      }
      // Four matching side votes support the current geometry-based angle gate.
      // Mixed left/right votes do not establish a side direction.
      const angleVote = current.detection
        ? updateAngleCaptureWindow(angleClassHistoryRef.current, current.detection.angleClass)
        : { history: angleClassHistoryRef.current, ready: false, sideDirection: null };
      angleClassHistoryRef.current = angleVote.history;
      const currentGeometry = liveGeometryDecision(current.detection, frame.width, frame.height,
        alignmentReferenceRef.current, currentCornersReliable, angleVote.ready, Boolean(angleVote.sideDirection));
      // Filter angle measurements without reinstating the old independent
      // stability warmup. Current box position/size remain unsmoothed.
      const geometry = liveGeometryDecision(current.detection, frame.width, frame.height,
        alignmentReferenceRef.current, currentCornersReliable, angleVote.ready, Boolean(angleVote.sideDirection), stable.measured ?? undefined);
      const raw = current.detection ? geometry.raw : continuity.retain ? "steady" : "searching";
      // Both the filtered estimate and the actual saved frame must pass.
      let currentCaptureVeto = geometry.veto ?? currentGeometry.veto;
      if (sideAngleRef.current.blocked) currentCaptureVeto = "side_class";
      let acceptabilityCorrection: Guidance | null = null;
      if (ENABLE_ACCEPTABILITY_CLASSIFIER && currentCaptureVeto === null && recovered.state === "visible") {
        const acceptability = await classifyAcceptability(current.input);
        if (!stillCurrent()) return;
        current.milliseconds += acceptability.milliseconds;
        if (acceptabilitySampledAtRef.current === null || sampledAt - acceptabilitySampledAtRef.current > maxEvidenceGap) acceptabilityHistoryRef.current = [];
        // The shared hold confirms repeated passing classifier results; do not
        // add a second three-reading warmup before that hold can begin.
        const stableAcceptability = updateAcceptabilityWindow(acceptabilityHistoryRef.current, acceptability.successProbability ?? 0, 1, 1);
        acceptabilitySampledAtRef.current = sampledAt;
        acceptabilityHistoryRef.current = stableAcceptability.history;
        acceptabilityProbabilityRef.current = stableAcceptability.probability;
        setAcceptabilityProbability(stableAcceptability.probability);
        currentCaptureVeto = captureAcceptabilityVeto<CaptureVeto>(currentCaptureVeto, stableAcceptability);
        if (currentCaptureVeto === "acceptability") {
          acceptabilityCorrection = acceptabilityCorrectionFor(current.detection, frame.width, frame.height, instructionRef.current.guidance, stable.measured ?? undefined);
        }
      } else {
        // A short tracking gap may retain history, but an actual alignment
        // failure clears it. A current classifier result is still required.
        if (!preserveAcceptabilityHistory(continuity.retain, !current.detection || !geometryReliable, acceptabilitySampledAtRef.current, sampledAt)) {
          acceptabilityHistoryRef.current = [];
          acceptabilitySampledAtRef.current = null;
        }
        acceptabilityProbabilityRef.current = null;
        setAcceptabilityProbability(null);
      }
      // Keep evaluating height while side/distance guidance is active so its
      // capture veto remains enforced until the corrected view passes.
      if (current.detection) {
        const highAngle = await classifyHighAngle(frame, frame.width, frame.height, current.detection);
        if (!stillCurrent()) return;
        current.milliseconds += highAngle.milliseconds;
        highAngleRef.current = updateHighAngleState(highAngleRef.current, highAngle.probability);
      } else if (!continuity.sameTrack) {
        highAngleRef.current = emptyHighAngleState();
      }
      if (document.hidden) {
        completedInferenceRef.current = null;
        captureHoldRef.current = emptyCaptureHold();
        liveTimerRef.current = window.setTimeout(runLiveLoop, 500); return;
      }
      setHighAngleProbability(current.detection ? highAngleRef.current.probability : null);
      if (ENABLE_HIGH_ANGLE_CLASSIFIER && current.detection && highAngleRef.current.blocked) currentCaptureVeto = "high_angle";
      const completedAt = performance.now();
      const fresh = freshInference(sampledAt, completedAt);
      completedInferenceRef.current = fresh ? { sampledAt, completedAt } : null;
      if (!fresh) currentCaptureVeto = "stale_frame";
      const frameReady = currentCaptureVeto === null && recovered.state === "visible";
      const effectiveVeto = recovered.state === "visible" ? currentCaptureVeto : "no_detection";
      const uncertain = canPauseCapture(effectiveVeto, geometry.veto, continuity.retain, fresh);
      const alignmentHold = updateCaptureHold(captureHoldRef.current, sampledAt, frameReady, maxEvidenceGap, uncertain);
      captureHoldRef.current = alignmentHold;
      setCaptureVeto(effectiveVeto);
      setAngleHistory(angleVote.history);
      const confirmedSide = confirmedSideGuidance(sideAngleRef.current);
      const angleCorrection = liveSideCorrection(current.detection, frame.width, frame.height,
        instructionRef.current.guidance, currentCornersReliable, stable.measured ?? undefined,
        geometry.veto, currentGeometry.veto, Boolean(angleVote.sideDirection));
      const nextGuidance = chooseLiveGuidance({ raw, currentRaw: currentGeometry.raw, frameReady: alignmentHold.confirmed && (frameReady || alignmentHold.paused), veto: currentCaptureVeto,
        confirmedSide, angleCorrection, heightConfirmed: highAngleRef.current.lowerPhone, acceptabilityCorrection });
      instructionRef.current = updateActionInstruction(instructionRef.current, nextGuidance, sampledAt, {
        sameTrack: continuity.sameTrack, fresh, reliable: currentCornersReliable,
        resolved: correctionResolved(instructionRef.current.guidance, stable.raw, stable.measured,
          alignmentReferenceRef.current, sideAngleRef.current.blocked, highAngleRef.current.blocked),
        sideConfirmed: confirmedSide !== null && nextGuidance === confirmedSide,
        passingFrame: frameReady, maxGapMs: maxEvidenceGap,
      });
      const displayedGuidance = instructionRef.current.guidance;
      const captureReady = alignmentHold.ready;
      setLiveGeometryStats({ vertical: geometry.vertical, rawVertical: stable.raw?.widthDifference ?? null, correction: geometry.raw,
        corners: geometry.cornerFailure ?? "Reliable" });
      setGuide(displayedGuidance); setResult(current); setModelState("done");
      drawLiveOverlay(displayedDetection, video.videoWidth, video.videoHeight, poseReliability.ready);
      samplesRef.current = [...samplesRef.current, { inferenceMs: current.milliseconds, completedAt: performance.now(), detected: Boolean(current.detection) }].slice(-30);
      setMetrics(summarizePerformance(samplesRef.current));
      // Capture only on a fresh passing frame after enough observed valid time.
      // Paused intervals contribute no time and resumption needs two good frames.
      if (captureReady) {
        capturePhoto(frame, current.detection, alignmentReferenceRef.current, currentCaptureVeto, captureReady);
        return;
      }
      liveTimerRef.current = window.setTimeout(runLiveLoop, balancedDelay(current.milliseconds));
    } catch (reason) {
      if (!stillCurrent()) return;
      setError(reason instanceof Error ? reason.message : "Live detection failed."); setModelState("error"); stopCamera();
    }
  }

  async function startLiveDetection() {
    measurementHistoryRef.current = []; trackingEvidenceRef.current = null; acceptabilitySampledAtRef.current = null;
    highAngleRef.current = emptyHighAngleState(); setHighAngleProbability(null);
    setCameraState("starting"); setMode("camera"); setModelState("idle"); setError(""); setResult(null); setGuide("searching"); setShowStats(false); setLossState("lost"); setCaptured(false); setCaptureDiagnostics(null); setCaptureDiagnosticsVisible(true); setCalibrationSaved(false); setCaptureVeto("no_detection"); setAngleHistory([]); setAcceptabilityProbability(null); samplesRef.current = []; lastDetectionRef.current = null; smoothedDetectionRef.current = null; missedDetectionsRef.current = 0; instructionRef.current = emptyInstructionState(); angleClassHistoryRef.current = []; poseReliabilityStreakRef.current = 0; captureInProgressRef.current = false; acceptabilityProbabilityRef.current = null; acceptabilityHistoryRef.current = []; captureHoldRef.current = emptyCaptureHold(); setMetrics(summarizePerformance([]));
    try {
      stopCamera();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } } });
      streamRef.current = stream;
      if (!videoRef.current) throw new Error("Camera preview is unavailable.");
      videoRef.current.srcObject = stream; await videoRef.current.play(); setCameraState("active"); setModelState("loading");
      const loadStarted = performance.now(); await ensureSessions(); setModelLoadMs(performance.now() - loadStarted); setModelState("done"); liveActiveRef.current = true; void runLiveLoop();
    } catch (reason) {
      stopCamera(); setCameraState("blocked"); setModelState("error");
      const named = reason as { name?: string };
      setError(named?.name === "NotAllowedError" ? "Camera access is needed. Allow camera access in Chrome, then try again." : "The detector couldn’t be prepared. Check your connection and try again.");
    }
  }

  function backHome() { stopCamera(); setCameraState("idle"); setModelState("idle"); setResult(null); setCaptured(false); setCaptureDiagnostics(null); setCaptureDiagnosticsVisible(true); setMode("home"); }
  function enterDemo() { stopCamera(); setMode("demo"); setCameraState("idle"); setDemoIndex(0); setGuide(demoOrder[0]); setResult(null); }
  function nextDemoState() { const next = (demoIndex + 1) % demoOrder.length; setDemoIndex(next); setGuide(demoOrder[next]); }

  function calibrateAlignment() {
    const video = videoRef.current, detection = smoothedDetectionRef.current; if (!video || !detection || poseReliabilityStreakRef.current < POSE_RELIABILITY_REQUIRED_READINGS) return;
    const next = referenceAtCurrentPose(detection, video.videoWidth, video.videoHeight, alignmentReferenceRef.current);
    alignmentReferenceRef.current = next; localStorage.setItem(ALIGNMENT_STORAGE_KEY, JSON.stringify(next));
    instructionRef.current = emptyInstructionState(); captureHoldRef.current = emptyCaptureHold(); setCalibrationSaved(true);
  }

  function capturePhoto(frame: HTMLCanvasElement, detection: Detection | null, reference: AlignmentReference, finalVeto: CaptureVeto | null, holdReady: boolean, stableMeasurements?: Measurements) {
    if (!detection || captureInProgressRef.current || finalVeto !== null || !holdReady) return;
    captureInProgressRef.current = true;
    setCaptureDiagnosticsVisible(true);
    const measured = stableMeasurements ?? alignmentMeasurements(detection, frame.width, frame.height);
    setCaptureDiagnostics({
      captureVeto: finalVeto,
      holdReady,
      centerPercent: measured.centerX * 100,
      centerMinimumPercent: (reference.centerX - reference.centerTolerance) * 100,
      centerMaximumPercent: (reference.centerX + reference.centerTolerance) * 100,
      areaPercent: measured.area * 100,
      areaMinimumPercent: (reference.targetArea - reference.areaTolerance) * 100,
      areaMaximumPercent: (reference.targetArea + reference.areaTolerance) * 100,
      sidePerspectivePercent: measured.heightDifference * 100,
      sidePerspectiveLimitPercent: Math.min(reference.maximumPerspectiveDifference, CAPTURE_SIDE_PERSPECTIVE_TOLERANCE) * 100,
      verticalPerspectivePercent: measured.widthDifference * 100,
      verticalPerspectiveLimitPercent: reference.maximumVerticalPerspectiveDifference * 100,
      positiveVerticalPerspectiveLimitPercent: Math.min(reference.maximumPositiveVerticalPerspectiveDifference, reference.maximumVerticalPerspectiveDifference) * 100,
      rotationDegrees: measured.rollDegrees - reference.targetRollDegrees,
      rotationLimitDegrees: reference.maximumRollDegrees,
      detectionConfidencePercent: detection.confidence * 100,
      minimumCornerConfidencePercent: Math.min(...detection.keypoints.map((point) => point.confidence)) * 100,
      cornersReliable: poseReliabilityStreakRef.current >= POSE_RELIABILITY_REQUIRED_READINGS,
      reliableReadings: poseReliabilityStreakRef.current,
      angleHistory: angleClassHistoryRef.current,
      acceptabilityProbability: acceptabilityProbabilityRef.current,
      highAngleProbability: highAngleRef.current.probability,
      strongSideReadings: sideAngleRef.current.history.filter(Boolean).length,
    });
    const rendered = encodeValidatedFrame(frame); imageUrlRef.current = rendered; setImageUrl(rendered);
    stopCamera(); setCameraState("idle"); setMode("image"); setCaptured(true); setModelState("done"); setGuide("aligned");
  }

  async function loadImage(file: File) {
    stopCamera(); setMode("image"); setModelState("loading"); setError(""); setResult(null); setCaptureDiagnostics(null);
    if (imageUrlRef.current?.startsWith("blob:")) URL.revokeObjectURL(imageUrlRef.current);
    const sourceUrl = URL.createObjectURL(file); imageUrlRef.current = sourceUrl; setImageUrl(sourceUrl);
    try {
      const image = new Image(); image.src = sourceUrl; await image.decode(); setModelState("running");
      const current = await infer(image, image.naturalWidth, image.naturalHeight);
      const reliablePose = current.detection ? poseCornersReliable(current.detection) : false;
      const poseGuide = guidanceFor(current.detection, image.naturalWidth, image.naturalHeight, alignmentReferenceRef.current, reliablePose);
      let imageGuide = poseGuide;
      if (poseGuide === "aligned") {
        const acceptability = await classifyAcceptability(current.input);
        current.milliseconds += acceptability.milliseconds;
        acceptabilityProbabilityRef.current = acceptability.successProbability;
        setAcceptabilityProbability(acceptability.successProbability);
        if (!acceptability.accepted) imageGuide = acceptabilityCorrectionFor(current.detection, image.naturalWidth, image.naturalHeight) ?? "steady";
      } else {
        acceptabilityProbabilityRef.current = null;
        setAcceptabilityProbability(null);
      }
      const highAngle = current.detection ? await classifyHighAngle(image, image.naturalWidth, image.naturalHeight, current.detection) : { probability: null, milliseconds: 0 };
      current.milliseconds += highAngle.milliseconds;
      setHighAngleProbability(highAngle.probability);
      if (highAngle.probability !== null && highAngle.probability >= HIGH_ANGLE_THRESHOLD && poseGuide !== "farther") imageGuide = "lower_phone";
      setGuide(imageGuide); setResult(current); setCaptured(false); setModelState("done");
      const canvas = document.createElement("canvas"); canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
      const context = canvas.getContext("2d"); if (!context) return;
      context.drawImage(image, 0, 0); drawDetection(context, current.detection, image.naturalWidth, reliablePose);
      const rendered = canvas.toDataURL("image/jpeg", .9); imageUrlRef.current = rendered; setImageUrl(rendered);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Image analysis failed."); setModelState("error"); }
  }

  function onFile(event: ChangeEvent<HTMLInputElement>) { const file = event.target.files?.[0]; if (file) void loadImage(file); event.target.value = ""; }
  const isLive = mode === "camera" && cameraState === "active";
  const copy = guide === "steady" && isLive ? waitingGuidanceCopy(captureVeto, Boolean(result?.detection)) : guidance[guide];
  const imageBusy = mode === "image" && (modelState === "loading" || modelState === "running");

  return <main className="app-shell"><section className="camera-stage" aria-label="Cup return slot detector">
    <video ref={videoRef} className={isLive ? "camera-feed visible" : "camera-feed"} playsInline muted /><canvas ref={overlayRef} className={`${isLive ? "live-overlay visible" : "live-overlay"}${lossState === "fading" ? " fading" : ""}`} aria-hidden="true" />
    {mode === "image" && imageUrl && <img className="result-image" src={imageUrl} alt={captured ? "Captured aligned photo" : "Selected test with slot detection overlay"} />}
    <div className="ambient" aria-hidden="true"><span /><span /><span /></div>
    <div className="top-bar"><button className="brand brand-button" onClick={backHome} aria-label="Return to CupDetector home"><span className="brand-mark">C</span><span>CupDetector</span></button><div className={`status-pill ${isLive && modelState === "done" ? "live" : ""}`}><i />{isLive && modelState === "loading" ? "Preparing" : isLive ? "Live detection" : modelState === "done" ? "Result" : mode === "demo" ? "Demo" : "Ready"}</div></div>
    {mode === "home" && <div className="permission-card"><div className="camera-icon">◎</div><h1>Find the return slot</h1><p>CupDetector finds the machine’s return slot and guides your phone into position. It does not detect cups.</p><button className="primary" onClick={startLiveDetection}>Start live detection</button><input ref={inputRef} className="file-input" type="file" accept="image/*" onChange={onFile} /><button className="secondary" onClick={() => inputRef.current?.click()}>Test an image</button><button className="text-button" onClick={enterDemo}>Preview guidance only</button><small className="privacy-note">Camera images stay on this device.</small></div>}
    {cameraState === "starting" && <div className="permission-card"><div className="loader" /><h1>Requesting camera access…</h1><p>Allow camera access when Chrome asks.</p></div>}
    {isLive && modelState === "loading" && <div className="permission-card compact-card"><div className="loader" /><h1>Preparing detector…</h1><p>This may take a little longer on the first visit.</p></div>}
    {cameraState === "blocked" && mode === "camera" && <div className="permission-card"><div className="camera-icon">!</div><h1>Live detection unavailable</h1><p>{error || "Allow camera access in Chrome settings, then try again."}</p><button className="primary" onClick={startLiveDetection}>Try again</button><button className="text-button" onClick={backHome}>Choose an image instead</button></div>}
    {imageBusy && <div className="permission-card compact-card"><div className="loader" /><h1>{modelState === "loading" ? "Preparing image…" : "Running detector…"}</h1><p>The first result may take longer while the model loads.</p></div>}
    {modelState === "error" && mode === "image" && <div className="permission-card"><div className="camera-icon">!</div><h1>Couldn’t analyse image</h1><p>{error}</p><button className="primary" onClick={() => inputRef.current?.click()}>Try another image</button></div>}
    {SHOW_CAPTURE_DIAGNOSTICS && captured && captureDiagnostics && (captureDiagnosticsVisible ? <CaptureDiagnosticsPanel values={captureDiagnostics} onHide={() => setCaptureDiagnosticsVisible(false)} /> : <button type="button" className="show-diagnostics-button" onClick={() => setCaptureDiagnosticsVisible(true)}>Show metrics</button>)}
    {(mode === "demo" || (mode === "image" && modelState === "done") || (isLive && modelState === "done")) && <div className="guidance-panel" aria-live="polite"><div className={`signal signal-${guide}`}>{guide === "aligned" ? "✓" : guide === "left" || guide === "step_left" ? "←" : guide === "right" || guide === "step_right" ? "→" : guide === "closer" ? "↑" : guide === "farther" || guide === "framing" ? "↓" : guide === "rotate_left" ? "↶" : guide === "rotate_right" ? "↷" : guide === "tilt_up" ? "↑" : guide === "tilt_down" || guide === "lower_phone" ? "↓" : "◎"}</div><div className="guidance-copy"><p>{captured ? "Photo saved" : copy.eyebrow}</p><h1>{captured ? "Photo captured" : copy.title}</h1><span>{captured ? "The aligned photo is ready for the next step." : copy.detail}</span>{result && !captured && (!isLive || showStats) && <small>{result.detection ? `${Math.round(result.detection.confidence * 100)}% confidence · ` : ""}{Math.round(result.milliseconds)} ms inference</small>}</div>{mode === "demo" && <button className="next-state" onClick={nextDemoState}>Next</button>}</div>}
    {isLive && showStats && <div className="performance-panel"><div className="geometry-stat"><span>Filtered vertical perspective</span><strong>{liveGeometryStats.vertical === null ? "Waiting" : `${liveGeometryStats.vertical >= 0 ? "+" : ""}${(liveGeometryStats.vertical * 100).toFixed(1)}%`}</strong><em>Raw {liveGeometryStats.rawVertical === null ? "—" : `${(liveGeometryStats.rawVertical * 100).toFixed(1)}%`} · limit +{(alignmentReferenceRef.current.maximumPositiveVerticalPerspectiveDifference * 100).toFixed(0)}% / −{(alignmentReferenceRef.current.maximumVerticalPerspectiveDifference * 100).toFixed(0)}%</em></div><div className="geometry-stat"><span>Corner check</span><strong>{liveGeometryStats.corners}</strong></div><div className="geometry-correction"><span>Filtered geometry</span><strong>{liveGeometryStats.correction === "aligned" ? "Within limits" : liveGeometryStats.correction === "steady" ? "Corners unconfirmed" : guidance[liveGeometryStats.correction].title}</strong></div><div><span>High angle</span><strong>{!ENABLE_HIGH_ANGLE_CLASSIFIER ? "Off" : highAngleProbability === null ? "Waiting" : `${Math.round(highAngleProbability * 100)}%`}</strong></div><div className="capture-gate-stat"><span>Capture gate</span><strong>{captureVeto ? captureVetoLabel[captureVeto] : "Holding for photo"}</strong></div><div><span>Acceptability</span><strong>{!ENABLE_ACCEPTABILITY_CLASSIFIER ? "Off" : acceptabilityProbability === null ? "Waiting" : `${Math.round(acceptabilityProbability * 100)}%`}</strong></div><div className="angle-history-stat"><span>Angle history</span><strong>{angleHistory.length ? angleHistory.map((value) => value === "centered" ? "C" : value === "camera_left" ? "L" : "R").join(" ") : "—"}</strong></div><div><span>Average</span><strong>{metrics.count ? `${Math.round(metrics.averageMs)} ms` : "—"}</strong></div><div><span>p95</span><strong>{metrics.count ? `${Math.round(metrics.p95Ms)} ms` : "—"}</strong></div><div><span>Updates</span><strong>{metrics.updatesPerMinute ? `${metrics.updatesPerMinute.toFixed(1)}/min` : "—"}</strong></div><div><span>Detected</span><strong>{metrics.count ? `${Math.round(metrics.detectionRate * 100)}%` : "—"}</strong></div><small>{videoRef.current?.videoWidth || 0}×{videoRef.current?.videoHeight || 0} camera · {Math.round(modelLoadMs)} ms model load · WASM · last {metrics.count}/30 results</small>{lastDetectionRef.current && <button className="calibrate-button" onClick={calibrateAlignment}>{calibrationSaved ? "View calibrated" : "Calibrate this view"}</button>}</div>}
    {(isLive || mode === "demo" || (mode === "image" && modelState === "done")) && <div className="bottom-actions"><button onClick={isLive ? backHome : captured ? startLiveDetection : () => inputRef.current?.click()}>{isLive ? "Stop" : captured ? "Retake" : "Test another image"}</button>{isLive && <button onClick={() => setShowStats((value) => !value)}>{showStats ? "Hide stats" : "Show stats"}</button>}<span>Return-slot guidance</span></div>}
    <input ref={mode === "home" ? undefined : inputRef} className="file-input" type="file" accept="image/*" onChange={onFile} />
  </section></main>;
}

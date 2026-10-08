import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(new Request("http://localhost/", { headers: { accept: "text/html" } }), { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } }, { waitUntil() {}, passThroughOnException() {} });
}

test("server renders the Phase 8 recovery entry point", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /CupDetector/);
  assert.doesNotMatch(html, /CupGuide/);
  assert.match(html, /Find the return slot/);
  assert.match(html, /It does not detect cups/);
  assert.match(html, /Camera images stay on this device/);
  assert.doesNotMatch(html, /profile-picker|fast profile|cool profile/);
  assert.match(html, /Test an image/);
  assert.match(html, /Start live detection/);
  assert.doesNotMatch(html, /Phase 7/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/);
});

test("ships all three ONNX models and browser inference code", async () => {
  const [page, poseModel, acceptabilityModel] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    access(new URL("../public/models/slot-pose.onnx", import.meta.url)),
    access(new URL("../public/models/slot-acceptability.onnx", import.meta.url)),
    access(new URL("../public/models/slot-high-angle.onnx", import.meta.url)),
  ]);
  assert.match(page, /onnxruntime-web/);
  assert.match(page, /InferenceSession\.create/);
  assert.match(page, /angle-pose-baseline-v3-left-angle/);
  assert.match(page, /tensor\.dims\[1\] !== 19/);
  assert.match(page, /decodePoseOutput/);
  assert.match(page, /updateAcceptabilityWindow/);
  assert.match(page, /updateCaptureHold/);
  assert.match(page, /captureHoldRef/);
  assert.match(page, /freezeCameraFrame/);
  assert.match(page, /encodeValidatedFrame/);
  assert.match(page, /updateCaptureHold\(captureHoldRef.current, sampledAt, frameReady, maxEvidenceGap, uncertain\)/);
  assert.match(page, /captureInProgressRef/);
  assert.match(page, /Keep your phone still\. Taking photo automatically\./);
  assert.match(page, /Rotate phone left/);
  assert.match(page, /Rotate phone right/);
  assert.match(page, /Tilt phone up/);
  assert.match(page, /Tilt phone down/);
  assert.match(page, /Move left/);
  assert.match(page, /Move right/);
  assert.match(page, /Move back/);
  assert.doesNotMatch(page, /Move toward the middle|Aim left|Aim right|Step left|Step right/);
  assert.match(page, /Calibrate this view/);
  assert.match(page, /referenceAtCurrentPose/);
  assert.match(page, /currentCaptureVeto = geometry\.veto/);
  assert.match(page, /acceptabilityCorrectionFor/);
  assert.match(page, /classifyAcceptability/);
  assert.match(page, /classifyHighAngle/);
  assert.match(page, /Lower your phone/);
  assert.match(page, /ENABLE_ACCEPTABILITY_CLASSIFIER/);
  assert.match(page, /currentCaptureVeto = captureAcceptabilityVeto<CaptureVeto>/);
  assert.doesNotMatch(page, /if \(acceptabilityCorrection\) currentCaptureVeto/);
  assert.match(page, /values\.captureVeto === null && values\.holdReady/);
  assert.match(page, /Capture gate/);
  assert.match(page, /Angle history/);
  assert.match(page, /Angle votes/);
  assert.match(page, /setGuide\(displayedGuidance\)/);
  assert.match(page, /recoveredDisplayDetection/);
  assert.match(page, /liveGeometryDecision\(current\.detection/);
  assert.match(page, /instructionRef/);
  assert.match(page, /updateActionInstruction/);
  assert.match(page, /SHOW_CAPTURE_DIAGNOSTICS/);
  assert.match(page, /Capture diagnostics/);
  assert.match(page, /Side perspective/);
  assert.match(page, /minimumCornerConfidencePercent/);
  assert.match(page, /Hide metrics/);
  assert.match(page, /Show metrics/);
  assert.doesNotMatch(page, /Level your phone/);
  assert.doesNotMatch(page, /Face the slot directly/);
  assert.doesNotMatch(page, /Square up/);
  assert.doesNotMatch(page, /Take photo/);
  assert.equal(poseModel, undefined);
  assert.equal(acceptabilityModel, undefined);
});

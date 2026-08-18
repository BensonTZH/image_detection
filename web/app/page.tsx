"use client";

import { ChangeEvent, useEffect, useRef, useState } from "react";
import type * as Ort from "onnxruntime-web";
import { balancedDelay, decodePoseOutput, DEFAULT_ALIGNMENT_REFERENCE, guidanceFor, letterboxGeometry, MODEL_SIZE, recoverTemporaryLoss, referenceAtCurrentDistance, stabilizeGuidance, summarizePerformance, type AlignmentReference, type Detection, type Guidance, type PerformanceSample } from "../lib/pose";

const guidance: Record<Guidance, { eyebrow: string; title: string; detail: string }> = {
  searching: { eyebrow: "No confident detection", title: "Slot not found", detail: "Keep the full cup-return opening in view." },
  left: { eyebrow: "Slot found", title: "Move left", detail: "Shift your phone slowly to the left." },
  right: { eyebrow: "Slot found", title: "Move right", detail: "Shift your phone slowly to the right." },
  closer: { eyebrow: "Slot found", title: "Move closer", detail: "Walk forward until the opening fills more of the view." },
  farther: { eyebrow: "Slot found", title: "Move back", detail: "Step back until the opening is the right size." },
  straighten: { eyebrow: "Slot found", title: "Square up", detail: "Hold the phone level and face the slot straight on." },
  aligned: { eyebrow: "Position confirmed", title: "Ready for photo", detail: "The slot is centred, correctly sized and square." },
};
const demoOrder: Guidance[] = ["searching", "left", "right", "closer", "farther", "straighten", "aligned"];

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sessionRef = useRef<Ort.InferenceSession | null>(null);
  const ortRef = useRef<typeof import("onnxruntime-web") | null>(null);
  const inputCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const liveActiveRef = useRef(false);
  const liveTimerRef = useRef<number | null>(null);
  const guidanceHistoryRef = useRef<Guidance[]>([]);
  const imageUrlRef = useRef<string | null>(null);
  const samplesRef = useRef<PerformanceSample[]>([]);
  const lastDetectionRef = useRef<Detection | null>(null);
  const missedDetectionsRef = useRef(0);
  const alignedCountRef = useRef(0);
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
  const [photoReady, setPhotoReady] = useState(false);
  const [captured, setCaptured] = useState(false);

  function stopCamera() {
    liveActiveRef.current = false;
    if (liveTimerRef.current !== null) window.clearTimeout(liveTimerRef.current);
    liveTimerRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    const overlay = overlayRef.current;
    overlay?.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
  }

  useEffect(() => {
    try { const stored = localStorage.getItem("cupdetector-alignment-reference"); if (stored) alignmentReferenceRef.current = { ...DEFAULT_ALIGNMENT_REFERENCE, ...JSON.parse(stored) }; } catch { alignmentReferenceRef.current = DEFAULT_ALIGNMENT_REFERENCE; }
    return () => { stopCamera(); if (imageUrlRef.current?.startsWith("blob:")) URL.revokeObjectURL(imageUrlRef.current); };
  }, []);

  async function ensureSession() {
    if (!sessionRef.current) {
      const wasmOrt = await import("onnxruntime-web/wasm"); wasmOrt.env.wasm.numThreads = 1;
      sessionRef.current = await wasmOrt.InferenceSession.create("/models/slot-pose.onnx", { executionProviders: ["wasm"] });
      ortRef.current = wasmOrt;
    }
    if (!ortRef.current) throw new Error("Detector runtime is unavailable.");
    return { ort: ortRef.current, session: sessionRef.current };
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
    const { ort, session } = await ensureSession();
    const started = performance.now();
    const outputs = await session.run({ images: new ort.Tensor("float32", chw, [1, 3, MODEL_SIZE, MODEL_SIZE]) });
    const milliseconds = performance.now() - started;
    const tensor = outputs.output0;
    if (!tensor || tensor.dims.length !== 3 || tensor.dims[1] !== 17) throw new Error("The model returned an unexpected output shape.");
    const detection = decodePoseOutput(tensor.data as Float32Array, Number(tensor.dims[2]), geometry, width, height)[0] ?? null;
    return { detection, milliseconds };
  }

  function drawDetection(context: CanvasRenderingContext2D, detection: Detection | null, width: number) {
    if (!detection) return;
    const line = Math.max(3, width / 250); context.lineWidth = line; context.strokeStyle = "#b9f227"; context.fillStyle = "#b9f227";
    const [x1, y1, x2, y2] = detection.box; context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    context.beginPath(); detection.keypoints.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y)); context.closePath(); context.stroke();
    detection.keypoints.forEach((point) => { context.beginPath(); context.arc(point.x, point.y, line * 2.2, 0, Math.PI * 2); context.fill(); });
  }

  function drawLiveOverlay(detection: Detection | null, width: number, height: number) {
    const canvas = overlayRef.current; if (!canvas) return;
    canvas.width = width; canvas.height = height;
    const context = canvas.getContext("2d"); if (!context) return;
    context.clearRect(0, 0, width, height); drawDetection(context, detection, width);
  }

  async function runLiveLoop() {
    if (!liveActiveRef.current) return;
    const video = videoRef.current;
    if (!video || document.hidden || video.readyState < 2) { liveTimerRef.current = window.setTimeout(runLiveLoop, 500); return; }
    try {
      const current = await infer(video, video.videoWidth, video.videoHeight);
      if (!liveActiveRef.current) return;
      const recovered = recoverTemporaryLoss(lastDetectionRef.current, current.detection, missedDetectionsRef.current);
      lastDetectionRef.current = recovered.detection; missedDetectionsRef.current = recovered.misses; setLossState(recovered.state);
      const raw = recovered.state === "fading" ? guide : guidanceFor(recovered.detection, video.videoWidth, video.videoHeight, alignmentReferenceRef.current);
      const stable = stabilizeGuidance(guidanceHistoryRef.current, raw);
      guidanceHistoryRef.current = stable.history; setGuide(stable.guidance); setResult(current); setModelState("done");
      alignedCountRef.current = raw === "aligned" && recovered.state === "visible" ? alignedCountRef.current + 1 : 0;
      setPhotoReady(alignedCountRef.current >= 3);
      drawLiveOverlay(recovered.detection, video.videoWidth, video.videoHeight);
      samplesRef.current = [...samplesRef.current, { inferenceMs: current.milliseconds, completedAt: performance.now(), detected: Boolean(current.detection) }].slice(-30);
      setMetrics(summarizePerformance(samplesRef.current));
      liveTimerRef.current = window.setTimeout(runLiveLoop, balancedDelay(current.milliseconds));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Live detection failed."); setModelState("error"); stopCamera();
    }
  }

  async function startLiveDetection() {
    setCameraState("starting"); setMode("camera"); setModelState("idle"); setError(""); setResult(null); setGuide("searching"); setShowStats(false); setLossState("lost"); setPhotoReady(false); setCaptured(false); guidanceHistoryRef.current = []; samplesRef.current = []; lastDetectionRef.current = null; missedDetectionsRef.current = 0; alignedCountRef.current = 0; setMetrics(summarizePerformance([]));
    try {
      stopCamera();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } } });
      streamRef.current = stream;
      if (!videoRef.current) throw new Error("Camera preview is unavailable.");
      videoRef.current.srcObject = stream; await videoRef.current.play(); setCameraState("active"); setModelState("loading");
      const loadStarted = performance.now(); await ensureSession(); setModelLoadMs(performance.now() - loadStarted); setModelState("done"); liveActiveRef.current = true; void runLiveLoop();
    } catch (reason) {
      stopCamera(); setCameraState("blocked"); setModelState("error");
      const named = reason as { name?: string };
      setError(named?.name === "NotAllowedError" ? "Camera access is needed. Allow camera access in Chrome, then try again." : "The detector couldn’t be prepared. Check your connection and try again.");
    }
  }

  function backHome() { stopCamera(); setCameraState("idle"); setModelState("idle"); setResult(null); setMode("home"); }
  function enterDemo() { stopCamera(); setMode("demo"); setCameraState("idle"); setDemoIndex(0); setGuide(demoOrder[0]); setResult(null); }
  function nextDemoState() { const next = (demoIndex + 1) % demoOrder.length; setDemoIndex(next); setGuide(demoOrder[next]); }

  function calibrateDistance() {
    const video = videoRef.current, detection = lastDetectionRef.current; if (!video || !detection) return;
    const next = referenceAtCurrentDistance(detection, video.videoWidth, video.videoHeight, alignmentReferenceRef.current);
    alignmentReferenceRef.current = next; localStorage.setItem("cupdetector-alignment-reference", JSON.stringify(next));
    guidanceHistoryRef.current = []; alignedCountRef.current = 0; setPhotoReady(false);
  }

  function capturePhoto() {
    const video = videoRef.current; if (!video || !photoReady) return;
    const canvas = document.createElement("canvas"); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const context = canvas.getContext("2d"); if (!context) return;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const rendered = canvas.toDataURL("image/jpeg", .92); imageUrlRef.current = rendered; setImageUrl(rendered);
    stopCamera(); setCameraState("idle"); setMode("image"); setCaptured(true); setModelState("done"); setGuide("aligned");
  }

  async function loadImage(file: File) {
    stopCamera(); setMode("image"); setModelState("loading"); setError(""); setResult(null);
    if (imageUrlRef.current?.startsWith("blob:")) URL.revokeObjectURL(imageUrlRef.current);
    const sourceUrl = URL.createObjectURL(file); imageUrlRef.current = sourceUrl; setImageUrl(sourceUrl);
    try {
      const image = new Image(); image.src = sourceUrl; await image.decode(); setModelState("running");
      const current = await infer(image, image.naturalWidth, image.naturalHeight);
      setGuide(guidanceFor(current.detection, image.naturalWidth, image.naturalHeight, alignmentReferenceRef.current)); setResult(current); setCaptured(false); setModelState("done");
      const canvas = document.createElement("canvas"); canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
      const context = canvas.getContext("2d"); if (!context) return;
      context.drawImage(image, 0, 0); drawDetection(context, current.detection, image.naturalWidth);
      const rendered = canvas.toDataURL("image/jpeg", .9); imageUrlRef.current = rendered; setImageUrl(rendered);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Image analysis failed."); setModelState("error"); }
  }

  function onFile(event: ChangeEvent<HTMLInputElement>) { const file = event.target.files?.[0]; if (file) void loadImage(file); event.target.value = ""; }
  const copy = guidance[guide]; const isLive = mode === "camera" && cameraState === "active"; const imageBusy = mode === "image" && (modelState === "loading" || modelState === "running");

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
    {(mode === "demo" || (mode === "image" && modelState === "done") || (isLive && modelState === "done")) && <div className="guidance-panel" aria-live="polite"><div className={`signal signal-${guide}`}>{guide === "aligned" ? "✓" : guide === "left" ? "←" : guide === "right" ? "→" : guide === "closer" ? "↑" : guide === "farther" ? "↓" : guide === "straighten" ? "↔" : "◎"}</div><div className="guidance-copy"><p>{captured ? "Photo saved" : lossState === "fading" && isLive ? "Detection interrupted" : copy.eyebrow}</p><h1>{captured ? "Photo captured" : lossState === "fading" && isLive ? "Hold steady" : copy.title}</h1><span>{captured ? "The aligned photo is ready for the next step." : lossState === "fading" && isLive ? "Keep the return slot in view." : copy.detail}</span>{result && !captured && <small>{result.detection ? `${Math.round(result.detection.confidence * 100)}% confidence · ` : ""}{Math.round(result.milliseconds)} ms inference</small>}</div>{mode === "demo" && <button className="next-state" onClick={nextDemoState}>Next</button>}{isLive && guide === "aligned" && <button className="capture-button" disabled={!photoReady} onClick={capturePhoto}>{photoReady ? "Take photo" : "Hold…"}</button>}</div>}
    {isLive && showStats && <div className="performance-panel"><div><span>Average</span><strong>{metrics.count ? `${Math.round(metrics.averageMs)} ms` : "—"}</strong></div><div><span>p95</span><strong>{metrics.count ? `${Math.round(metrics.p95Ms)} ms` : "—"}</strong></div><div><span>Updates</span><strong>{metrics.updatesPerMinute ? `${metrics.updatesPerMinute.toFixed(1)}/min` : "—"}</strong></div><div><span>Detected</span><strong>{metrics.count ? `${Math.round(metrics.detectionRate * 100)}%` : "—"}</strong></div><small>{videoRef.current?.videoWidth || 0}×{videoRef.current?.videoHeight || 0} camera · {Math.round(modelLoadMs)} ms model load · WASM · last {metrics.count}/30 results</small>{lastDetectionRef.current && <button className="calibrate-button" onClick={calibrateDistance}>Use this distance</button>}</div>}
    {(isLive || mode === "demo" || (mode === "image" && modelState === "done")) && <div className="bottom-actions"><button onClick={isLive ? backHome : () => inputRef.current?.click()}>{isLive ? "Stop" : "Test another image"}</button>{isLive && <button onClick={() => setShowStats((value) => !value)}>{showStats ? "Hide stats" : "Show stats"}</button>}<span>Return-slot guidance</span></div>}
    <input ref={mode === "home" ? undefined : inputRef} className="file-input" type="file" accept="image/*" onChange={onFile} />
  </section></main>;
}

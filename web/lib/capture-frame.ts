// Freeze once before inference. Both inference and JPEG export use this object,
// so motion while the models run cannot substitute an unvalidated camera frame.
export function freezeCameraFrame(video: HTMLVideoElement, createCanvas = () => document.createElement("canvas")): HTMLCanvasElement {
  const frame = createCanvas();
  frame.width = video.videoWidth;
  frame.height = video.videoHeight;
  if (!frame.width || !frame.height) throw new Error("Camera frame is unavailable.");
  const context = frame.getContext("2d");
  if (!context) throw new Error("Camera frame could not be prepared.");
  context.drawImage(video, 0, 0, frame.width, frame.height);
  return frame;
}

export function encodeValidatedFrame(frame: HTMLCanvasElement): string {
  return frame.toDataURL("image/jpeg", .92);
}

// v7 training contract: unrectified 9:16 slot crop in a 320-square RGB tensor.
// Keep the perspective intact so the classifier can recognize a high view.
export function highAngleInput(
  source: CanvasImageSource,
  width: number,
  height: number,
  box: readonly number[],
  canvas: HTMLCanvasElement,
): Float32Array {
  const [x1, y1, x2, y2] = box;
  if (box.length !== 4 || ![width, height, ...box].every(Number.isFinite) || width <= 0 || height <= 0 || x2 <= x1 || y2 <= y1) {
    throw new Error("Invalid slot crop.");
  }
  const cropHeight = Math.max((y2 - y1) * 1.25, (x2 - x1) * 3 * 16 / 9);
  const cropWidth = cropHeight * 9 / 16;
  const scale = 320 / cropHeight;
  const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
  canvas.width = canvas.height = 320;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas is unavailable.");
  context.fillStyle = "rgb(114,114,114)";
  context.fillRect(0, 0, 320, 320);
  context.save();
  context.beginPath(); context.rect(70, 0, 180, 320); context.clip();
  // Transform the source rather than clipping its crop to the frame edges:
  // out-of-frame margins must retain the same gray padding used in training.
  context.drawImage(source, 70 - (cx - cropWidth / 2) * scale, -(cy - cropHeight / 2) * scale, width * scale, height * scale);
  context.restore();
  const rgba = context.getImageData(0, 0, 320, 320).data;
  const plane = 320 * 320, input = new Float32Array(3 * plane);
  for (let pixel = 0; pixel < plane; pixel++) {
    input[pixel] = rgba[pixel * 4] / 255;
    input[plane + pixel] = rgba[pixel * 4 + 1] / 255;
    input[2 * plane + pixel] = rgba[pixel * 4 + 2] / 255;
  }
  return input;
}

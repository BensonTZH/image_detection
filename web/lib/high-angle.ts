// One-switch rollback: disables loading, guidance, and capture blocking.
export const ENABLE_HIGH_ANGLE_CLASSIFIER = true;
export const HIGH_ANGLE_MODEL_URL = "/models/slot-high-angle.onnx?v=high-angle-v7-six-roi-20260923";
export const HIGH_ANGLE_MODEL_SIZE = 320;
// Preserve the user's warning threshold with the v7 slot-crop classifier.
export const HIGH_ANGLE_THRESHOLD = .40;

export type HighAngleState = {
  probability: number | null;
  readings: number;
  highStreak: number;
  clearStreak: number;
  lowerPhone: boolean;
  blocked: boolean;
};

export function emptyHighAngleState(enabled = ENABLE_HIGH_ANGLE_CLASSIFIER): HighAngleState {
  return { probability: null, readings: 0, highStreak: 0, clearStreak: 0, lowerPhone: false, blocked: enabled };
}

export function decodeHighAngleOutput(output: Float32Array): number {
  if (output.length !== 2 || !Array.from(output).every(value => Number.isFinite(value) && value >= 0 && value <= 1)) {
    throw new Error("The high-angle model returned invalid probabilities.");
  }
  // Training/export contract: acceptable, high_angle.
  return output[1];
}

export function updateHighAngleState(
  previous: HighAngleState,
  probability: number | null,
  enabled = ENABLE_HIGH_ANGLE_CLASSIFIER,
  threshold = HIGH_ANGLE_THRESHOLD,
): HighAngleState {
  if (!enabled || probability === null) return emptyHighAngleState(enabled);
  if (!Number.isFinite(probability) || probability < 0 || probability > 1) throw new Error("Invalid high-angle score.");
  const highStreak = probability >= threshold ? previous.highStreak + 1 : 0;
  const clearStreak = probability < Math.max(.01, threshold - .10) ? previous.clearStreak + 1 : 0;
  const lowerPhone = highStreak >= 2 || (previous.lowerPhone && clearStreak < 2);
  const readings = Math.min(2, previous.readings + 1);
  // A single high reading immediately resets capture, but does not flash a
  // new instruction. Two clear readings release a confirmed warning.
  return { probability, readings, highStreak, clearStreak, lowerPhone,
    blocked: readings < 2 || lowerPhone || probability >= threshold };
}

// Rollback switch: set this to false to restore pose-only capture without
// changing the pose model, alignment rules, or camera flow.
export const ENABLE_ACCEPTABILITY_CLASSIFIER = true;
export const ACCEPTABILITY_MODEL_SIZE = 320;
export const ACCEPTABILITY_SUCCESS_THRESHOLD = .75;
export const ACCEPTABILITY_MODEL_URL = "/models/slot-acceptability.onnx?v=reviewed-acceptability-v3-addon-20260917";

export type AcceptabilityResult = {
  errorProbability: number;
  successProbability: number;
  accepted: boolean;
};

export function decodeAcceptabilityOutput(
  output: Float32Array,
  threshold = ACCEPTABILITY_SUCCESS_THRESHOLD,
): AcceptabilityResult {
  if (output.length !== 2) throw new Error(`Unexpected acceptability output length: ${output.length}`);
  const [errorProbability, successProbability] = output;
  if (![errorProbability, successProbability].every(Number.isFinite)) throw new Error("The acceptability model returned invalid probabilities.");
  return { errorProbability, successProbability, accepted: successProbability >= threshold };
}

export function captureAcceptabilityVeto<T extends string>(
  poseVeto: T | null,
  result: Pick<AcceptabilityResult, "accepted"> | null,
  enabled = ENABLE_ACCEPTABILITY_CLASSIFIER,
): T | "acceptability" | null {
  if (poseVeto) return poseVeto;
  if (!enabled) return null;
  return result?.accepted ? null : "acceptability";
}

export function updateAcceptabilityWindow(
  previous: number[],
  next: number,
  windowSize = 5,
  minimumReadings = 3,
  threshold = ACCEPTABILITY_SUCCESS_THRESHOLD,
) {
  const history = [...previous, next].slice(-windowSize);
  const ordered = [...history].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  const median = ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
  return { history, probability: median, accepted: history.length >= minimumReadings && median >= threshold && next >= threshold };
}

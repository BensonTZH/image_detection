export type CompletedInference = { sampledAt: number; completedAt: number } | null;
export const MAX_PROCESSING_MS = 2500;

// Only time spent processing the immediately preceding frame earns an
// allowance. An idle tab, missing result, or arbitrary old timestamp does not.
export function evidenceGapLimit(previous: CompletedInference, now: number) {
  if (!previous) return 1200;
  const processing = previous.completedAt-previous.sampledAt;
  const idle = now-previous.completedAt;
  if (!Number.isFinite(processing) || !Number.isFinite(idle) || processing < 0
    || processing > MAX_PROCESSING_MS || idle < 0 || idle > 250) return 1200;
  return Math.max(1200, processing+250);
}
export function freshInference(sampledAt: number, completedAt: number) {
  return completedAt >= sampledAt && completedAt-sampledAt <= MAX_PROCESSING_MS;
}

import assert from "node:assert/strict";
import test from "node:test";
import { decodeHighAngleOutput, emptyHighAngleState, HIGH_ANGLE_THRESHOLD, updateHighAngleState } from "../lib/high-angle.ts";
import { updateAlignmentHold } from "../lib/pose.ts";

test("40 percent trial blocks capture at the boundary and clears below 30 percent", () => {
  assert.equal(HIGH_ANGLE_THRESHOLD, .40);
  let state = updateHighAngleState(emptyHighAngleState(), .39);
  state = updateHighAngleState(state, .39);
  assert.equal(state.blocked, false);
  const hold = updateAlignmentHold(null, 1000, !state.blocked);
  state = updateHighAngleState(state, .40);
  assert.equal(state.blocked, true);
  assert.equal(state.lowerPhone, false);
  assert.equal(updateAlignmentHold(hold.startedAt, 1300, !state.blocked).ready, false);
  state = updateHighAngleState(state, .41);
  assert.equal(state.lowerPhone, true);
  state = updateHighAngleState(state, .31);
  state = updateHighAngleState(state, .31);
  assert.equal(state.lowerPhone, true);
  state = updateHighAngleState(state, .29);
  assert.equal(state.blocked, true);
  state = updateHighAngleState(state, .29);
  assert.equal(state.blocked, false);
});

test("one high frame blocks capture immediately without flashing the height instruction", () => {
  let state = updateHighAngleState(emptyHighAngleState(), .1, true, .5);
  state = updateHighAngleState(state, .1, true, .5);
  assert.equal(state.blocked, false);
  const started = updateAlignmentHold(null, 1000, !state.blocked);
  state = updateHighAngleState(state, .9, true, .5);
  assert.equal(state.lowerPhone, false);
  assert.equal(state.blocked, true);
  assert.deepEqual(updateAlignmentHold(started.startedAt, 1400, !state.blocked), { startedAt: null, ready: false });
});

test("confirmed high view stays blocked through one noisy clear frame", () => {
  let state = updateHighAngleState(emptyHighAngleState(), .9, true, .5);
  state = updateHighAngleState(state, .9, true, .5);
  assert.equal(state.lowerPhone, true);
  state = updateHighAngleState(state, .1, true, .5);
  assert.equal(state.lowerPhone, true);
  assert.equal(state.blocked, true);
  state = updateHighAngleState(state, .45, true, .5);
  assert.equal(state.blocked, true);
  state = updateHighAngleState(state, .1, true, .5);
  state = updateHighAngleState(state, .1, true, .5);
  assert.equal(state.blocked, false);
  assert.equal(state.lowerPhone, false);
});

test("missing detection resets history and rollback bypasses height gate", () => {
  let state = updateHighAngleState(emptyHighAngleState(), .9, true, .5);
  state = updateHighAngleState(state, .9, true, .5);
  assert.deepEqual(updateHighAngleState(state, null), emptyHighAngleState());
  assert.deepEqual(updateHighAngleState(state, .99, false), emptyHighAngleState(false));
  assert.equal(emptyHighAngleState(false).blocked, false);
});

test("high-angle output validates shape and finite normalized scores", () => {
  assert.equal(decodeHighAngleOutput(new Float32Array([.25,.75])), .75);
  for (const values of [[1], [NaN, .5], [.5, Infinity], [-.1,1.1]]) {
    assert.throws(() => decodeHighAngleOutput(new Float32Array(values)));
  }
});

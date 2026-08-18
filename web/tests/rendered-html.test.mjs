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

test("ships the ONNX model and browser inference code", async () => {
  const [page, model] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    access(new URL("../public/models/slot-pose.onnx", import.meta.url)),
  ]);
  assert.match(page, /onnxruntime-web/);
  assert.match(page, /InferenceSession\.create/);
  assert.match(page, /decodePoseOutput/);
  assert.equal(model, undefined);
});

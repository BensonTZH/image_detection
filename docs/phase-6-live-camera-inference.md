# Phase 6 - Live Camera Inference

## Purpose

Phase 6 runs the ONNX slot detector repeatedly on the rear-camera preview and updates the overlay and guidance automatically. It builds on the verified Phase 5 preprocessing and decoder rather than introducing a second inference path.

## What is implemented

- Rear-camera capture over HTTPS.
- Continuous browser-side ONNX inference with only one inference in flight.
- A 650 ms pause after each result to reduce load on the OPPO Reno Z.
- Live box and four-corner polygon overlay.
- Confidence and inference-time display.
- Guidance stabilization using the majority of the latest three results.
- Automatic inference pause while the browser tab is hidden.
- Camera, timer, and overlay cleanup when detection stops or the page closes.
- Phase 5 still-image testing remains available.

## Automated test

```bash
cd /Users/bensontan/Desktop/image_detection/web
npm test
```

Tests cover Phase 5 preprocessing and decoding, all guidance branches, three-result stabilization, production build, model packaging, and the Phase 6 entry screen.

## OPPO Reno Z test

Use the private HTTPS CupDetector URL or the ADB localhost procedure in
[`localhost-phone-testing.md`](localhost-phone-testing.md). A local
`http://192.168...` address cannot access the phone camera in Chrome; with ADB
port reversal, open `http://localhost:3000` on the phone instead.

1. Open CupDetector in Chrome and press **Start live detection**.
2. Allow camera access.
3. Point the phone at a validation image displayed full-screen on the MacBook.
4. Confirm a lime box and corner polygon appear over the slot.
5. Move the phone left, right, closer, and farther away. Confirm guidance changes automatically.
6. Confirm the instruction does not rapidly flicker between states.
7. Cover the camera or point away. Confirm it changes to **Slot not found** after several results.
8. Switch to another app for 20 seconds, return, and confirm detection resumes.
9. Press **Stop detection** and confirm the camera indicator switches off.
10. Run continuously for five minutes and record the typical inference time, device temperature, freezes, and reloads.

## Pass criteria

- Rear camera opens through the private HTTPS site.
- Live overlay follows a displayed validation image.
- Only one inference runs at a time.
- Guidance reacts correctly and remains stable.
- Detection resumes after the page returns from the background.
- Stop detection releases the camera.
- Five minutes complete without a Chrome freeze or reload.
- Target update rate is approximately one result per second; slower is acceptable for the POC if guidance remains usable.

## Limitations

- Testing against a MacBook screen introduces glare, blur, and display patterns. Final accuracy requires the real office machine.
- The model and WebAssembly runtime require a substantial first download. Later visits may use Chrome's cache.
- The OPPO Reno Z may thermally throttle during extended use.
- Guidance thresholds remain the initial Phase 5 POC values and should be calibrated with real machine measurements.

## Exit gate

- [x] Continuous inference loop implemented.
- [x] Live overlay and stabilized guidance implemented.
- [x] Lifecycle cleanup and hidden-page pause implemented.
- [x] Automated tests and production build pass.
- [ ] OPPO Reno Z validation-screen test passes.
- [ ] Five-minute stability test passes.
- [ ] Real office-machine test passes when the machine is available.

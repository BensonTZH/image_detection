# Phase 4 - Mobile Camera Interface Testing

## Purpose

Phase 4 adds the phone-first camera interface and mocked alignment guidance. It deliberately does not load the ONNX model yet. The goal is to prove that the OPPO Reno Z can open the rear camera, that the screen remains usable in portrait orientation, and that every guidance state is understandable before live inference is connected.

## What is implemented

- Rear-camera request using `facingMode: environment`.
- Full-screen portrait camera preview with safe-area support.
- Camera permission, loading, active, and blocked states.
- Alignment frame and five mocked guidance states: searching, move left, move right, move closer, and aligned.
- Demo mode for testing all states without a second phone or a live detector.
- Accessible live guidance text, touch-sized controls, and reduced-motion support.

## Run on the Mac

```bash
cd /Users/bensontan/Desktop/image_detection/web
npm install
npm run dev
```

Open the local address printed by the server. Use **Preview without camera**, then press **Next** to inspect all five mocked states.

## Test on the OPPO Reno Z

Phone camera access requires HTTPS or a trusted localhost origin. Use the
published HTTPS URL, or use the verified ADB localhost procedure in
[`localhost-phone-testing.md`](localhost-phone-testing.md). A plain
`http://192.168...` Wi-Fi address can load the page but Chrome normally blocks
live camera access.

1. Connect the OPPO Reno Z to the internet and open the published URL in Google Chrome.
2. Hold the phone vertically and press **Start camera**.
3. When Chrome asks, choose **Allow**.
4. Confirm that the rear camera opens. If the front camera opens instead, record the Chrome and Android versions.
5. Point at the office machine and confirm the camera remains smooth while walking closer and moving left or right.
6. Press **Demo guidance**, then press **Next** repeatedly.
7. Confirm all states are readable: Point at the machine, Move left, Move right, Move closer, and Perfect — hold still.
8. Rotate the phone once. Portrait is the supported POC orientation; confirm returning to portrait restores the layout.
9. Deny camera permission once and confirm the recovery screen offers **Try again** and **Use demo mode**.

## Pass criteria

- Rear camera opens in Chrome after permission is granted.
- Camera fills the screen without stretching.
- No controls are hidden by the browser bars or screen edges.
- Guidance text is readable at normal arm's length.
- Every demo state changes when **Next** is pressed.
- Permission denial produces a useful recovery screen.
- The page remains responsive for five minutes with no freeze or reload.

## Known limitation

The boxes and instructions in this phase are simulated. Pointing at the real machine will not automatically change the guidance. Static ONNX inference and browser-side output decoding are the next phase; live camera inference follows after those results match Python.

## Exit gate

- [x] Mobile camera interface implemented.
- [x] Mock guidance state controls implemented.
- [x] Desktop build passes.
- [ ] OPPO Reno Z rear-camera permission test passes.
- [ ] All five demo states are accepted on the OPPO Reno Z.
- [ ] Five-minute phone stability test passes.

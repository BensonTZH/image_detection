# Phase 7 - Phone Performance Measurement and Tuning

## Purpose

Phase 7 measures how live ONNX inference behaves on the OPPO Reno Z. It does not retrain the model or claim to improve detection accuracy.

## Automatic scheduling

Balanced scheduling was accepted as good enough during POC review and is now automatic. It targets roughly one completed result per second by adapting the pause to the latest inference time. Only one inference can run at a time. The Fast and Cool selectors were removed because performance controls are not appropriate for the client-facing interface.

## Metrics

Press **Show stats** during live detection to display:

- Average inference time across the latest 30 results.
- p95 inference time, showing slower results near the top of the observed range.
- Effective completed updates per minute.
- Percentage of sampled frames with a detection.
- Camera input resolution.
- Model initialization time.

These measurements remain in the current browser session and are not uploaded or stored.

## Automated test

```bash
cd /Users/bensontan/Desktop/image_detection/web
npm test
```

Tests cover preprocessing, decoding, guidance, stabilization, bounded automatic scheduling, performance summaries, model packaging, the Phase 7 entry screen, and the production build.

## OPPO Reno Z test

Use the private HTTPS CupDetector site:

1. Press **Start live detection**.
2. Point at the same validation image displayed on the MacBook.
3. Press **Show stats** and wait two minutes.
4. Record average ms, p95 ms, updates/minute, detection percentage, model-load time, and camera resolution.
5. Note whether guidance feels responsive, whether Chrome freezes or reloads, and whether the phone becomes noticeably warm.
6. Detection percentage against a monitor is diagnostic only and must not be treated as real-machine accuracy.

## Exit gate

- [x] Rolling performance measurement implemented.
- [x] Adaptive Balanced scheduling implemented and selected as the fixed behavior.
- [x] Client-facing profile selector removed.
- [x] Automated tests and production build pass.
- [ ] Record two-minute Balanced metrics on the OPPO Reno Z.
- [ ] Complete a five-minute stability run.

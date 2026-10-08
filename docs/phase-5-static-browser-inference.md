# Phase 5 - Static Browser Inference

## Purpose

Phase 5 runs the trained slot-pose model inside Chrome on one still image at a time. This isolates browser preprocessing and ONNX output decoding before Phase 6 attempts continuous inference on live camera frames.

## What is implemented

- Browser-side ONNX Runtime using the WebAssembly backend and one worker thread for broad Android compatibility.
- Local loading of the 10.5 MB `slot-pose.onnx` model.
- Image selection or rear-camera still capture through Chrome's file picker.
- Ultralytics-compatible 640 x 640 letterbox preprocessing with RGB FP32 CHW input.
- Decoding of the `[1, 17, 8400]` pose output.
- Confidence filtering, single-class non-maximum suppression, and coordinate restoration.
- Slot box, TL/TR/BR/BL polygon, confidence, and inference-time display.
- Static guidance: slot not found, move left, move right, move closer, or aligned.

The selected image and model inference remain inside the browser. The app does not upload the image to an application server.

## Automated test

```bash
cd /Users/bensontan/Desktop/image_detection/web
npm test
```

The tests verify letterbox geometry, coordinate restoration, confidence rejection, NMS, every guidance branch, the production build, and the rendered page.

## Desktop validation-image test

1. Open the hosted CupDetector URL in Chrome.
2. Press **Test an image**.
3. Select an image from `dataset/images/val`.
4. Wait for the model result. The first image takes longer because Chrome downloads and initializes the model.
5. Confirm the lime box surrounds the slot and the polygon follows TL, TR, BR, BL.
6. Confirm confidence, inference time, and guidance are displayed.
7. Repeat with at least 20 validation images, including distant, tilted, dark, bright, and cropped examples.

## OPPO Reno Z test before the office machine is ready

Copy 15-20 images from `dataset/images/val` to the phone, or place them in a cloud folder accessible from Chrome. Then:

1. Open the private HTTPS CupDetector URL in Chrome.
2. Press **Test an image** and choose a validation image.
3. Confirm Chrome does not reload while loading the model.
4. Inspect the overlay and record confidence, guidance, and inference time.
5. Press **Test another image** and repeat.
6. Include at least one unrelated photo. It should report **Slot not found**; note that the existing validation split itself contains no negative images.

## Pass criteria

- All automated tests pass.
- All 20 fixed validation images produce a result without browser errors.
- Box and corner overlays visually match the slot on at least 18 of 20 images.
- Keypoint labels remain ordered TL, TR, BR, BL.
- An unrelated negative image does not produce a confident slot detection.
- Median OPPO Reno Z inference is below one second after the first model load.
- Chrome completes 20 sequential images without freezing or reloading.

## Limitations

- Validation images are familiar evaluation data, so this tests browser integration more than real-world accuracy.
- Guidance uses initial POC thresholds: horizontal centre below 42% means left, above 58% means right, and box area below 12% means closer.
- Still-image capture may open the camera through Android's picker, but this is not live inference.
- The model detects the machine's cup-return slot; it does not detect or track the cup itself.

## Exit gate

- [x] ONNX model loads in the web application.
- [x] Preprocessing, decoding, NMS, overlay, and guidance are implemented.
- [x] Automated tests and production build pass.
- [ ] Twenty validation images pass visual review.
- [ ] At least one unrelated negative image is rejected.
- [ ] OPPO Reno Z timing and stability criteria pass.

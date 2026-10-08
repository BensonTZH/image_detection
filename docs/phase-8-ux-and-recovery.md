# Phase 8 — User experience and recovery

Phase 8 makes CupDetector clearer and less jumpy without changing the trained model.

## What changed

- The start screen says that CupDetector detects the machine's return slot, not cups.
- Camera startup uses two plain messages: `Requesting camera access…` and `Preparing detector…`.
- Camera permission and detector preparation failures show simple recovery instructions and a **Try again** button.
- One missed live result keeps the last outline. A second missed result fades it and shows **Hold steady**. A third removes it and returns to **Slot not found**.
- The start screen states that camera images stay on the device.
- Internal phase wording was removed from the client-facing screen.

## Verify on the OPPO Reno Z

Open the private CupDetector link in Google Chrome.

1. On the start screen, confirm it says **It does not detect cups** and **Camera images stay on this device**.
2. Tap **Start live detection**. Before granting permission, confirm **Requesting camera access…** appears.
3. Allow the camera. Confirm the next message is **Preparing detector…**, followed by live detection.
4. Point the camera at a clear return slot until the green outline appears.
5. Briefly cover the camera or move the slot out of view. The outline should persist briefly, then fade with **Hold steady**, then disappear with **Slot not found** if the slot remains hidden.
6. Bring the slot back into view and confirm detection resumes automatically.
7. To check permission recovery, block camera access for the site in Chrome, reload, and tap **Start live detection**. Confirm the app tells you camera access is needed and provides **Try again**.
8. Re-enable camera permission and tap **Try again**. Confirm the camera opens.
9. Tap **Test an image** and confirm image testing still produces a return-slot outline and guidance.

The brief loss behavior depends on detection speed. On a slower phone, each of the three steps may remain visible longer.

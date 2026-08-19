# Alignment, calibration, and capture testing

## Implemented behaviour

CupDetector now checks the detected slot's four corners before allowing a photo:

- horizontal centring;
- minimum size, with **Move closer** guidance;
- maximum size, with **Move back** guidance;
- top-edge rotation and unequal opposite-edge lengths, with **Square up** guidance;
- three consecutive good results before the **Take photo** button is enabled.

## Default reference

The built-in POC reference expects the slot centre at 50% of the frame, allows an 8% horizontal tolerance, targets 18% of the frame area with a 7% tolerance, allows 7 degrees of roll, and allows a 25% opposite-edge difference.

These are starting values, not completed office calibration.

## Calibrate the expected distance at the office

1. Stand at the intended shooting position with the OPPO Reno Z.
2. Open live detection and place the slot at the desired size in the frame.
3. Tap **Show stats**.
4. Tap **Use this distance** while the slot outline is visible.
5. The reference is saved on that phone and remains after closing Chrome.

Repeat calibration if Chrome site data is cleared, the phone changes, or a materially different machine is used.

## Verify guidance and capture

1. Stand too far away and confirm **Move closer**.
2. Stand too near and confirm **Move back**.
3. Rotate the phone or view the slot from a strong side angle and confirm **Square up**.
4. Centre the slot at the calibrated distance and face it straight on.
5. Confirm **Ready for photo** appears.
6. Hold the position. The disabled **Hold…** button should become **Take photo** after three good detection results.
7. Tap **Take photo** and confirm a clean camera frame is shown without the green diagnostic overlay.

The captured photo currently remains in the page for the POC. Passing it into OR-Cup Step 3 is a separate integration task.

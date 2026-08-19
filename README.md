# CupDetector

CupDetector is a browser-based proof of concept that uses an ONNX pose model
to find the cup-return tube, draw its box and four corners, and provide live
alignment guidance. Inference runs on the device; camera frames are not sent to
an application backend.

## Prerequisites

- Node.js 22.13 or newer. Node.js 20 cannot start the current Vinext version.
- npm.
- Python 3.12 for dataset validation, training, evaluation, and ONNX export.
- Android SDK Platform Tools only when testing a phone camera through USB.

## Install the Python requirements

From the repository root on macOS or Linux:

```bash
cd /path/to/image_detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ml/requirements.txt
python -m unittest discover -s ml/tests -v
```

On Windows PowerShell, replace the activation command with:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r ml\requirements.txt
python -m unittest discover -s ml\tests -v
```

The Python environment is needed for the ML scripts. The browser application
uses the exported model in `web/public/models/slot-pose.onnx` and does not run
Python while serving the page.

## Install and test the web app

The npm project is inside the `web` directory. The repository root does not
contain a `package.json`, so do not run `npm ci`, `npm test`, or `npm run dev`
until after changing into `web`.

On macOS or Linux:

```bash
cd /path/to/image_detection
cd web
nvm use 22
npm ci
npm test
npm run dev
```

Open `http://localhost:3000` on the development computer. Keep the Terminal
window running and press `Control+C` when finished.

The current npm scripts use macOS/Linux-style inline environment variables.
Windows users should follow the Windows guide, which provides compatible
PowerShell commands.

## Test on an Android phone

Follow the guide for the operating system running the local server:

- [macOS localhost and phone testing](docs/localhost-phone-testing.md)
- [Windows localhost and phone testing](docs/windows-localhost-phone-testing.md)

For live phone-camera testing over USB, configure ADB port reversal and open
`http://localhost:3000` on the phone. A plain `http://192.168...` Wi-Fi address
can load the page and test existing images, but Chrome normally blocks live
camera access on that insecure origin.

## Project layout

```text
ml/scripts/                       Dataset, training, evaluation, and export tools
ml/tests/                         Python unit tests
ml/requirements.txt              Python ML dependencies
web/app/                          CupDetector interface
web/lib/pose.ts                   ONNX decoding and alignment guidance
web/public/models/slot-pose.onnx  Browser model
web/tests/                        Browser logic and build tests
docs/                             Setup and testing guides
```

Datasets, training runs, virtual environments, generated build folders, and
other large local outputs should remain ignored. The browser ONNX model is an
intentional application asset and must remain committed.

## Useful commands

Run these from the `web` directory on macOS or Linux:

- `npm run dev`: start local development.
- `npm run build`: create the production build.
- `npm test`: run pose tests, create a production build, and check the rendered
  app and packaged model.
- `npm run lint`: run the web lint checks.

Run the Python unit tests from the repository root with the virtual environment
activated:

```bash
python -m unittest discover -s ml/tests -v
```

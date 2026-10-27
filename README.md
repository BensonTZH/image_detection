# CupDetector

CupDetector finds a cup-return slot, estimates its viewing angle and guides the
user before automatically capturing a photo. It does **not** detect or count
cups. Camera inference runs in the browser using three included ONNX models;
camera frames are not uploaded to an application backend.

This README is the main setup guide for the **`dev` branch**. You do not need the
release ZIP or START_HERE to work from this repository. No hosting account is
required for local use.

## Required dependencies

- **Git**, with access to this repository.
- **Node.js 24 LTS and npm** (recommended). The project requires Node >=22.13.
- A browser and webcam for computer testing, or a phone for rear-camera testing.
- **Python 3.12 only for training and Python tools**; the web app does not use it.

### Install Node.js and npm first

Install Node.js 24 LTS using the installer from [nodejs.org](https://nodejs.org/en/download).
Reopen Terminal or PowerShell, then check:

```sh
node -v
npm -v
```

Both commands must print version numbers before continuing. Downloading an
installer without running it does not install Node.

On macOS, if you already use [Homebrew](https://brew.sh), you can instead run:

```sh
brew install node@24
```

Then add its directory to your shell PATH. This also works when installation
succeeds but `brew link` reports a conflicting npm file; no forced overwrite is
needed if these version checks succeed:

```sh
printf '\nexport PATH="%s/opt/node@24/bin:$PATH"\n' "$(brew --prefix)" >> ~/.zshrc
source ~/.zshrc
node -v
npm -v
```

On Windows, install Git for Windows and the Node.js installer, then reopen
PowerShell. If PowerShell blocks `npm.ps1`, use `npm.cmd` instead of `npm`.

## Clone and install

Keep the working copy outside iCloud, OneDrive, Dropbox and other synchronized
folders. On macOS, use `~/Projects`, **not Desktop or Documents** if those are
synced. Cloud file activity can interfere with the development file watcher.

macOS/Linux:

```sh
mkdir -p ~/Projects
cd ~/Projects
git clone --branch dev https://github.com/BensonTZH/image_detection.git
cd image_detection/web
npm ci
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force C:\Projects | Out-Null
Set-Location C:\Projects
git clone --branch dev https://github.com/BensonTZH/image_detection.git
Set-Location image_detection\web
npm ci
```

The npm project is in **`web`**. Run the following npm commands there. Install
from the committed lockfile; do not copy someone else's `node_modules` folder.

## Run on the computer

```sh
npm run dev -- --hostname 127.0.0.1
```

Open **http://127.0.0.1:3000** on that computer. Keep the terminal running; press
Ctrl+C to stop it. The computer's own camera works on this loopback address
without an HTTPS certificate. Allow camera access when prompted.

The server uses port 3000 and fails if it is occupied. Stop your other server,
or choose a different port with `--port 3001` and use that port in the browser.

## Run on a phone

Choose one connection method:

| Connection | Camera access | What to open on the phone |
| --- | --- | --- |
| Wi-Fi with plain HTTP | Page/image testing only | `http://COMPUTER_IP:3000` |
| Wi-Fi with trusted HTTPS | Live camera | `https://COMPUTER_IP:3000` |
| Android with USB port forwarding | Live camera, no certificate needed | `http://localhost:3000` after forwarding |

**A phone's localhost refers to the phone**, unless you explicitly set up USB
port forwarding. Binding the server to 127.0.0.1 also prevents Wi-Fi access.

For a quick Wi-Fi connectivity check, stop the existing server and run:

```sh
npm run dev:lan
```

Connect both devices to the same non-guest network. Find the computer's Wi-Fi
IPv4 address in network settings (on macOS: System Settings → Wi-Fi → Details →
TCP/IP; on Windows: `ipconfig`). If it is `192.168.1.25`, open
`http://192.168.1.25:3000` on the phone. Replace that example IP with your own.
This checks connectivity, **not camera permission**.

For live camera setup, follow the [phone connection guide](docs/localhost-phone-testing.md).
It includes certificate generation, phone trust settings, `npm run dev:https`,
and Android USB forwarding. [Windows-specific notes](docs/windows-localhost-phone-testing.md)
cover PowerShell, firewall and USB tools.

Vinext uses **`--hostname`**, not `--host`. `dev:lan` and `dev:https` set the
correct network binding for you.

## Check the app

1. Confirm the page shows **CupDetector**, **Test an image** and **Start live detection**.
2. Use Test an image with a slot photo to check model loading.
3. Start live detection, grant camera access and show the complete slot.
4. Check left/right, height and distance guidance; hold an acceptable view for
   automatic capture and try Retake.
5. Repeat on the actual target phone. A working page or passing automated tests
   cannot confirm real camera guidance quality.

Run all web unit tests, build the app and check its rendered output:

```sh
npm test
```

For only the unit tests, use `npm run test:unit`. To run a production build
without a development file watcher:

```sh
npm run build
npm run start -- --hostname 127.0.0.1 --port 3000
```

The production command above is for same-computer HTTP testing. The documented
`dev:https` certificate setup applies to the development server, not `start`.

## Models and training data

All three runtime models and their metadata are committed in `web/public/models`:

| Model | Responsibility |
| --- | --- |
| `slot-pose.onnx` | Slot box, four corners and centred/left/right view classification |
| `slot-high-angle.onnx` | High-view classification from a crop around the detected slot |
| `slot-acceptability.onnx` | Overall acceptable/unacceptable view classification |

The app combines model results with geometry, timing and guidance rules. See
[current models and training](docs/current-models-and-training.md) for versions,
thresholds, code locations, known limitations and retraining steps.

**Training images and checkpoints are not in GitHub.** Obtain
`cupdetector-training.zip` separately if you need to retrain. Extract it into the
repository root so `dataset/`, `runs/` and `artifacts/` sit beside `web/` and
`ml/`. Do not extract it inside `web` or commit it to Git.

For Python tools, from the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r ml/requirements.txt
python -m unittest discover -s ml/tests -v
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate
with `.\.venv\Scripts\Activate.ps1`. Model training can require additional GPU
setup; a working browser app does not require any Python dependencies.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `node` or `npm`: command not found | Install Node and configure PATH above; reopen the terminal. |
| npm cannot find `package.json` | Change into the repository's `web` folder. |
| Git authentication fails | Confirm repository access and use your Git credential manager or SSH setup. |
| Phone cannot reach localhost | Use the computer's IP over Wi-Fi, or configure Android USB forwarding. |
| Phone cannot reach the computer's IP | Use `npm run dev:lan` or `dev:https`; check the actual port, same network and host firewall. Guest/office Wi-Fi may isolate devices. |
| Phone page opens but camera is unavailable | Use trusted HTTPS or Android USB localhost; allow camera permissions. |
| Certificate warning | Check phone CA trust, certificate expiry and that the address matches the certificate. Reissue it if the computer's IP changes. |
| Config files repeatedly “changed”, HMR floods or page stays loading | Stop the server. Use a fresh clone outside synchronized folders, run `npm ci`, and restart there. Confirm the printed path is the new folder. Close duplicate dev servers. |
| One “scanning dependencies” message on first launch | Normal initial compilation. Repeated unsolicited restarts are not normal. |
| A model fails to load | Check all three files in `web/public/models`; inspect browser console/network errors and retry. |

When reporting a persistent loading problem, include `pwd` (PowerShell:
`Get-Location`), Node/npm versions, the server URL and the first error after a
fresh start. Do not repeatedly edit thresholds to fix server loading problems.

The lockfile currently has dependency advisories. Review upgrades separately;
do not run `npm audit fix --force` as a setup step because it can change the
working dependency stack.

## Working on the project

Pull the latest `dev`, create a feature branch, make changes and run `npm test`
before submitting them. Keep datasets, model-training runs, certificates,
private keys and dependency folders out of Git. Commit intentional browser
model updates together with matching metadata and cache-version URLs.

Before a model or guidance change, record the working commit. To return to it,
use a separate checkout of that commit, install its locked dependencies and
restart its server. Do not use a destructive reset over uncommitted work.

Older `docs/phase-*` and experiment reports describe historical behavior; this
README and the current-model guide are the starting points for the current app.

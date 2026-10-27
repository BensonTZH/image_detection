# Windows local testing

Use the [README](../README.md) for the current clone, dependency and launch
commands. Node.js 24 LTS and Git for Windows are recommended. Work in a local
folder such as `C:\Projects`, outside OneDrive.

The npm scripts no longer use Unix-only environment assignments, so run them
from PowerShell or Command Prompt. If execution policy blocks `npm.ps1`, use
`npm.cmd` instead. Native Windows execution has not been tested on this Mac.

From `C:\Projects\image_detection\web`:

```powershell
npm ci
npm run dev -- --hostname 127.0.0.1
```

For a phone over Wi-Fi, stop that server and use `npm run dev:lan`. Allow Node
through Windows Firewall on the trusted **Private** network. Run `ipconfig` and
use the active Wi-Fi adapter's IPv4 address; HTTP tests connectivity but does
not enable the phone camera.

Follow the [phone guide](localhost-phone-testing.md) for trusted HTTPS or Android
USB forwarding. It includes PowerShell certificate commands. Install Windows
Android Platform Tools from Google for USB, add its folder to PATH and install
the manufacturer's USB driver if the phone is not detected.

To inspect a port conflict:

```powershell
Get-NetTCPConnection -LocalPort 3000 -State Listen
```

Run `npm test` for the complete web test/build workflow. For local production
serving, use `npm run build`, then
`npm run start -- --hostname 127.0.0.1 --port 3000`.

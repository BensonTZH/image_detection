# Windows Localhost and Android Phone Testing

## Purpose

This guide runs CupDetector on a Windows computer and connects an Android phone
for still-image and live-camera testing. It does not require deployment or
ChatGPT authentication.

## 1. Install the prerequisites

Install:

- Git for Windows.
- Node.js 22.13 or newer.
- Android SDK Platform Tools for `adb` phone testing.
- An OPPO USB driver if Windows detects the phone but ADB does not.

Open PowerShell and confirm the tools are available:

```powershell
node --version
npm --version
adb version
```

The Node.js result must be `v22.13.0` or newer. Node.js 20 cannot start the
current Vinext version.

If PowerShell blocks an `npm.ps1` or `npx.ps1` script, use `npm.cmd` and
`npx.cmd` in the commands below, or run them from Command Prompt.

## 2. Clone and enter the project

Replace the example path with the location of the repository:

```powershell
cd C:\Projects\image_detection\web
npm ci
```

The repository must include the following runtime files:

```text
web/package.json
web/package-lock.json
web/vite.config.ts
web/.openai/hosting.json
web/public/models/slot-pose.onnx
```

The ONNX model is required for detection. The hosting configuration is required
because the current Vite configuration imports it, but it does not require a
ChatGPT sign-in when running locally.

## 3. Run the automated checks

The current npm scripts use macOS/Linux-style inline environment variables. On
Windows, run the equivalent checks directly:

```powershell
node --experimental-strip-types --test tests/pose.test.ts
npx vinext build
node --test tests/rendered-html.test.mjs
```

All tests must pass and the build must complete before phone testing.

## 4. Start the app for Windows-only testing

```powershell
npx vinext dev
```

Open this address in Chrome or Edge on the Windows computer:

```text
http://localhost:3000
```

Keep PowerShell running. Press `Control+C` when finished.

## 5. Open the app over Wi-Fi

For another device on the same Wi-Fi network, Vinext requires `--hostname`:

```powershell
npx vinext dev --hostname 0.0.0.0
```

When Windows Firewall asks, allow Node.js access on **Private networks** only.

Find the computer's IPv4 address:

```powershell
ipconfig
```

Look under the active Wi-Fi adapter for **IPv4 Address**. If it is
`192.168.1.50`, open this on the phone:

```text
http://192.168.1.50:3000
```

This Wi-Fi address can load the page and use **Test an image**. Chrome normally
blocks live-camera access on a plain HTTP network address.

If the phone reports **Connection refused**:

1. Confirm the server printed a `Network` URL.
2. Confirm both devices are on the same non-guest Wi-Fi network.
3. Allow Node.js through Windows Firewall on Private networks.
4. Disconnect VPNs temporarily if they block local-network traffic.
5. Check that port 3000 is listening:

```powershell
Get-NetTCPConnection -LocalPort 3000 -State Listen
```

## 6. Enable Android USB debugging

USB tethering is not required. Enable USB debugging instead:

1. Open **Settings > About phone** on the Android phone.
2. Tap **Build number** seven times to enable Developer options.
3. Open **Settings > Additional settings > Developer options**.
4. Enable **Developer options** and **USB debugging**.
5. Connect the phone with a USB data cable.
6. Unlock the phone and select **File Transfer** as its USB mode.
7. Accept **Allow USB debugging?** and optionally select **Always allow from
   this computer**.

Confirm the connection:

```powershell
adb kill-server
adb start-server
adb devices -l
```

The device line must end with `device`.

- If it says `unauthorized`, unlock the phone and accept the authorization
  prompt.
- If the list is empty, turn off USB tethering, use File Transfer mode, try a
  different data cable or USB port, install the phone's Windows USB driver, or
  revoke USB debugging authorizations and reconnect.

## 7. Use live detection through USB

Keep CupDetector running on port 3000, then create the USB port mapping:

```powershell
adb reverse tcp:3000 tcp:3000
adb reverse --list
```

Keep the USB cable connected. In Chrome on the phone, close CupDetector tabs
that use the computer's Wi-Fi address. Open this exact address instead:

```text
http://localhost:3000
```

Do not use `http://192.168.1.50:3000` for live detection. ADB maps the phone's
localhost address to the Windows development server, and Chrome permits the
local page to request camera access.

Press **Start live detection** and choose **Allow** when Chrome asks.

## 8. Troubleshoot "Live detection unavailable"

Check these items in order:

1. The phone address bar says exactly `http://localhost:3000`.
2. `adb devices -l` shows the phone as `device`.
3. `adb reverse --list` contains `tcp:3000 tcp:3000`.
4. The CupDetector development server is still running on port 3000.
5. Android **Settings > Apps > Chrome > Permissions > Camera** is allowed.
6. Chrome's site settings allow the camera for localhost.
7. No other application is actively using the phone camera.

## 9. Stop and disconnect

Stop the development server with `Control+C`, then remove the USB mapping:

```powershell
adb reverse --remove tcp:3000
```

## Windows repository improvement

For a permanently cross-platform repository, change the `dev`, `build`, and
`start` scripts to use `cross-env`, or remove the Unix-only inline environment
assignment after confirming the Vite configuration supplies the same setting.
Until that change is made, the direct `npx vinext` commands in this guide avoid
the incompatible npm script syntax.

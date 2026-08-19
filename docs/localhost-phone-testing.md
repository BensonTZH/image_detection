# Localhost and Android Phone Testing

## Purpose

This guide starts CupDetector locally on the MacBook and connects an Android
phone for image and live-camera testing. The verified setup uses Node.js 22,
Vinext port 3000, Android Debug Bridge (ADB), and Google Chrome.

## 1. Check Node.js

Vinext requires Node.js 22.13 or newer. This project has been verified with
Node.js 22.23.2.

```bash
node --version
```

If the result starts with `v20`, select Node.js 22:

```bash
nvm use 22
```

To make Node.js 22 the default for new Terminal windows:

```bash
nvm alias default 22
```

## 2. Install and test the web app

From a fresh clone, install the exact dependency versions and run all automated
tests:

```bash
cd /Users/bensontan/Desktop/image_detection/web
npm ci
npm test
```

`npm test` runs the pose and guidance tests, creates a production build, and
checks that the app and ONNX model are included correctly.

## 3. Start localhost on the MacBook

For MacBook-only testing:

```bash
cd /Users/bensontan/Desktop/image_detection/web
nvm use 22
npm run dev
```

Open:

```text
http://localhost:3000
```

Keep the Terminal window running. Press `Control+C` when finished.

## 4. Open the page over office Wi-Fi

To expose the server to devices on the same Wi-Fi network, Vinext requires the
`--hostname` option. The similar `--host` option is ignored by this version.

```bash
npm run dev -- --hostname 0.0.0.0
```

Find the MacBook's Wi-Fi address:

```bash
ipconfig getifaddr en0
```

If the result is `192.168.1.50`, the phone can open:

```text
http://192.168.1.50:3000
```

This address is suitable for loading the page and using **Test an image**. Live
camera access is normally blocked because Chrome does not treat a plain HTTP
Wi-Fi address as a secure camera origin.

If the phone reports **Connection refused**, confirm that Terminal printed a
`Network` URL and that macOS allows incoming connections for Node.js. Also
confirm that both devices are on the same non-guest Wi-Fi network.

## 5. Enable live-camera testing through USB

USB tethering is not required. Enable **USB debugging** instead:

1. On the phone, open **Settings > About phone**.
2. Tap **Build number** seven times to enable Developer options.
3. Open **Settings > Additional settings > Developer options**.
4. Enable **Developer options** and **USB debugging**.
5. Connect the phone with a USB data cable, unlock it, and select **File
   Transfer** as the USB mode.
6. Accept **Allow USB debugging?** and optionally select **Always allow from
   this computer**.

Install Android Platform Tools on the MacBook if `adb` is unavailable:

```bash
brew install android-platform-tools
```

Confirm that the phone is authorized:

```bash
adb kill-server
adb start-server
adb devices -l
```

The device line must end with `device`. If it says `unauthorized`, unlock the
phone and accept the authorization prompt. If no device appears, disable USB
tethering, use File Transfer mode, try another data cable or USB port, revoke
USB debugging authorizations, and reconnect.

Forward the phone's localhost port to the MacBook server:

```bash
adb reverse tcp:3000 tcp:3000
adb reverse --list
```

Keep the USB cable connected. In Chrome on the phone, close any CupDetector
tabs using the Wi-Fi address and open this exact address:

```text
http://localhost:3000
```

Do not use `http://192.168.1.50:3000` for live-camera testing. Through ADB,
Chrome sees `localhost` as a secure local origin and can request camera access.
Press **Start live detection** and choose **Allow** when Chrome asks.

## 6. Troubleshoot "Live detection unavailable"

Check these items in order:

1. The phone address bar says exactly `http://localhost:3000`.
2. `adb devices -l` shows the phone as `device`.
3. `adb reverse --list` contains `tcp:3000 tcp:3000`.
4. The MacBook development server is still running.
5. Android **Settings > Apps > Chrome > Permissions > Camera** is allowed.
6. Chrome's site settings allow the camera for localhost.
7. No other app is actively using the camera.

## 7. Disconnect cleanly

Stop the development server with `Control+C`, then remove the USB port mapping:

```bash
adb reverse --remove tcp:3000
```

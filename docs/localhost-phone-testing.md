# Local phone testing

Start with the [README](../README.md) to install Node, clone `dev` into a local
folder and run `npm ci` in `web`. Run all npm commands below from `web`.
Stop the previous server before switching connection methods.

## Wi-Fi: check connectivity first

```sh
npm run dev:lan
```

This uses Vinext's `--hostname 0.0.0.0` option. `--host` is not the correct option
for this CLI. Both devices must be on the same network without client isolation.
Allow incoming Node connections on your trusted local network if the firewall
prompts.

Find the computer's current Wi-Fi IPv4 address:

- macOS: System Settings → Wi-Fi → Details → TCP/IP. `ipconfig getifaddr en0`
  also works when Wi-Fi is `en0`; use settings if it prints nothing.
- Windows: `ipconfig`, then the IPv4 address under the active Wi-Fi adapter.
- Linux: network settings or `ip address`.

If the address is `192.168.1.25`, open `http://192.168.1.25:3000` on the phone.
Use your actual address. This should load the page and Test an image; live camera
access requires the HTTPS or Android USB method below.

## Wi-Fi: trusted HTTPS for the phone camera

### 1. Install the certificate tool

On macOS with Homebrew:

```sh
brew install mkcert
mkcert -install
```

For Windows or Linux, follow the [mkcert installation instructions](https://github.com/FiloSottile/mkcert#installation),
then run `mkcert -install` on the computer. This creates a local certificate
authority and installs its trust on that computer; it does not configure the
phone or the app automatically.

### 2. Create a certificate for your address

From `web`, replace `192.168.1.25` with the computer's current Wi-Fi IP:

macOS/Linux:

```sh
mkdir -p .certs
mkcert -cert-file .certs/local.pem -key-file .certs/local-key.pem localhost 127.0.0.1 ::1 192.168.1.25
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force .certs | Out-Null
mkcert -cert-file .certs/local.pem -key-file .certs/local-key.pem localhost 127.0.0.1 ::1 192.168.1.25
```

The `.certs` directory is ignored by Git. Reissue the certificate if the IP
changes, or reserve the computer's address in your router.

### 3. Trust the CA on your phone

Run `mkcert -CAROOT` on the computer to locate **rootCA.pem**. Transfer only that
public CA certificate to your own test phone, using a cable or trusted file
transfer. Never share `rootCA-key.pem` or `local-key.pem`. Do not place the CA
private key or the CA directory in the app's public folder.

- **Android:** in Settings, search for “Install a certificate” and choose **CA
  certificate** (often under Security → Encryption & credentials). Select the
  transferred rootCA.pem. Menu names vary by manufacturer. Restart Chrome.
  Managed devices may prohibit installing a user CA; use an approved IT setup
  or the Android USB method instead.
- **iPhone/iPad:** open the certificate, install its profile under Settings →
  General → VPN & Device Management, then enable its trust under Settings →
  General → About → Certificate Trust Settings. See [Apple's certificate trust
  instructions](https://support.apple.com/en-us/102390). Restart Safari.

### 4. Start HTTPS and open the phone URL

```sh
npm run dev:https
```

Open **`https://192.168.1.25:3000`** on the phone, substituting the IP used in the
certificate. The page should open **without a certificate warning**. Choose
Start live detection and allow camera access. A bypassed certificate warning is
not a substitute for configuring trust.

`dev:https` checks for `.certs/local.pem` and `.certs/local-key.pem`, enables
Vite's TLS server and binds to the local network. Regular `dev`/`dev:lan` remain
HTTP even when certificate files exist. Add `-- --port 3001` if you need another
port and update the URL accordingly.

When finished, stop with Ctrl+C. Remove the test CA/profile from the phone if it
is no longer needed. Keep keys private and generate a separate CA on each
maintainer's computer.

## Android USB: live camera without certificates

This alternative forwards the phone's localhost to the computer through USB.
It does not use the phone's Wi-Fi address or require USB tethering.

1. Enable Android Developer options (typically tap Build number seven times),
   then enable **USB debugging**.
2. Connect a USB data cable, unlock the phone and accept the debugging prompt.
   Select File Transfer if required by the phone.
3. Install Android Platform Tools. On macOS with Homebrew:

```sh
brew install --cask android-platform-tools
```

On other systems use Google's [SDK Platform Tools](https://developer.android.com/tools/releases/platform-tools).
Confirm `adb devices -l` lists the phone as **device**, not unauthorized.

Start the HTTP server in one terminal:

```sh
npm run dev -- --hostname 127.0.0.1
```

In another terminal:

```sh
adb reverse tcp:3000 tcp:3000
adb reverse --list
```

Keep the cable connected and open **`http://localhost:3000`** in Chrome on the
phone. Port forwarding makes this localhost address reach the computer. Allow
camera access and start detection. Change both ports if the server uses another
port. See [Chrome's local-server forwarding guide](https://developer.chrome.com/docs/devtools/remote-debugging/local-server)
for the DevTools alternative.

When finished:

```sh
adb reverse --remove tcp:3000
```

Stop the server with Ctrl+C. Reconnecting the phone may require repeating the
forwarding command. This USB method does not apply to iPhones.

## Troubleshooting

- **Can't reach the page:** verify the URL, server terminal, correct port, Wi-Fi
  IP and network binding. Check firewall and guest-network isolation. `localhost`
  on the phone only reaches the computer after USB forwarding is active.
- **Loading/restart loop:** use a fresh checkout outside cloud-synced folders;
  follow the README troubleshooting before debugging phone permissions.
- **HTTPS starts but certificate is rejected:** check CA trust and the IP in the
  certificate. Do not disable browser security or use insecure-origin flags.
- **Camera unavailable on an otherwise working page:** check browser/site and
  OS camera permissions; close other camera apps. Confirm HTTPS is trusted, or
  that USB forwarding and localhost are in use.
- **ADB unauthorized:** unlock the phone and accept the prompt. If no phone is
  listed, check the cable, USB debugging and Windows device drivers.
- **Testing limits:** desktop HTTP/HTTPS checks do not validate a phone's trust
  store, camera permissions or the model's accuracy on live views. Complete a
  physical-device test before considering the setup ready.

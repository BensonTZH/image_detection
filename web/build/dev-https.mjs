import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
for (const name of ["local.pem", "local-key.pem"]) {
  if (!existsSync(new URL(`../.certs/${name}`, import.meta.url))) {
    console.error(`Missing web/.certs/${name}. Follow docs/localhost-phone-testing.md to create your local HTTPS certificate.`);
    process.exit(1);
  }
}

const child = spawn(process.execPath, [
  fileURLToPath(new URL("../node_modules/vinext/dist/cli.js", import.meta.url)),
  "dev", "--hostname", "0.0.0.0", ...process.argv.slice(2),
], {
  cwd: root,
  env: { ...process.env, CUPDETECTOR_HTTPS: "1" },
  stdio: "inherit",
});
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => child.kill(signal));
}
child.on("error", (error) => {
  console.error(error.message);
  process.exitCode = 1;
});
child.on("exit", (code, signal) => {
  process.exitCode = code ?? (signal === "SIGINT" ? 130 : 1);
});

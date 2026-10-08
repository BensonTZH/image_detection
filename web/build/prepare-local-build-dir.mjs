import { execFile } from "node:child_process";
import { lstat, readFile, rm } from "node:fs/promises";
import { resolve } from "node:path";
import { promisify } from "node:util";

const run = promisify(execFile);

const projectDirectory = process.cwd();
const packageJson = resolve(projectDirectory, "package.json");
const hostingConfig = resolve(projectDirectory, ".openai", "hosting.json");

// Refuse to clean anything unless this is unmistakably the Sites project root.
await Promise.all([readFile(packageJson), readFile(hostingConfig)]);

if (process.platform !== "darwin") {
  process.exit(0);
}

const phase = process.argv[2];
if (phase !== "before" && phase !== "after") {
  throw new Error("Expected a build phase: before or after");
}

const buildDirectory = resolve(projectDirectory, "dist");
const legacyBuildCache = resolve(projectDirectory, ".sites-build.nosync");

async function inspect(path) {
  try {
    return await lstat(path);
  } catch (error) {
    if (error.code === "ENOENT") return undefined;
    throw error;
  }
}

const currentBuild = await inspect(buildDirectory);

if (phase === "before") {
  // dist is generated and ignored. Removing it avoids reading stale iCloud
  // placeholders while Vite cleans the output directory.
  if (currentBuild) {
    await rm(buildDirectory, { recursive: true, force: true });
  }
  await rm(legacyBuildCache, { recursive: true, force: true });
  console.log("Removed stale cloud-backed deployment output");
  process.exit(0);
}

if (!currentBuild?.isDirectory()) {
  throw new Error("The production build did not create dist");
}

// Sites requires dist to be a real directory inside the project. Marking that
// generated directory as ignored by macOS File Provider keeps it resident while
// preserving the exact path and directory type expected by the packager.
await run("xattr", ["-w", "com.apple.fileprovider.ignore#P", "1", buildDirectory]);

console.log("Deployment output is protected from cloud offloading");

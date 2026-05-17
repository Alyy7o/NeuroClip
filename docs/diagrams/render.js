#!/usr/bin/env node
/**
 * Render every .mmd file in this folder to PNG + SVG using @mermaid-js/mermaid-cli.
 *
 * Usage:
 *   npm install
 *   npm run render
 */
const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const here = __dirname;
const mmdc = path.join(
  here,
  "node_modules",
  ".bin",
  process.platform === "win32" ? "mmdc.cmd" : "mmdc"
);

if (!fs.existsSync(mmdc)) {
  console.error("mmdc not found. Run `npm install` inside docs/diagrams first.");
  process.exit(1);
}

const sources = fs
  .readdirSync(here)
  .filter((f) => f.endsWith(".mmd"))
  .sort();

if (sources.length === 0) {
  console.error("No .mmd source files found.");
  process.exit(1);
}

const cfg = path.join(here, "mermaid.config.json");
const pup = path.join(here, "puppeteer.config.json");

let okCount = 0;
let failCount = 0;
const failures = [];

for (const src of sources) {
  const base = src.replace(/\.mmd$/, "");
  const inFile = path.join(here, src);

  for (const fmt of ["png", "svg"]) {
    const outFile = path.join(here, `${base}.${fmt}`);
    const args = [
      "-i", inFile,
      "-o", outFile,
      "-c", cfg,
      "-p", pup,
      "-b", "white",
      "-t", "default",
    ];
    if (fmt === "png") {
      args.push("--scale", "2", "-w", "1800");
    }
    console.log(`> mmdc ${src} -> ${path.basename(outFile)}`);
    const r = spawnSync(mmdc, args, { stdio: "inherit", shell: true });
    if (r.status === 0) {
      okCount++;
    } else {
      failCount++;
      failures.push(`${src} -> ${fmt}`);
    }
  }
}

console.log(`\nDone. Success: ${okCount}, Failed: ${failCount}`);
if (failures.length) {
  console.log("Failed renders:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
}

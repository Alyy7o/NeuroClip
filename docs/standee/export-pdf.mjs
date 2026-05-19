/**
 * Export NeuroClip standee to print-ready PDF (24.5 × 60.5 in with bleed).
 *
 * Usage (from docs/standee/):
 *   npm install
 *   node export-pdf.mjs
 */

import puppeteer from 'puppeteer';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const htmlPath = path.join(__dirname, 'standee.html');
const outPath = path.join(__dirname, 'standee-print.pdf');

if (!fs.existsSync(htmlPath)) {
  console.error('Missing standee.html at', htmlPath);
  process.exit(1);
}

const fileUrl = 'file:///' + htmlPath.replace(/\\/g, '/');

console.log('Loading', fileUrl);
console.log('Writing', outPath);

const browser = await puppeteer.launch({ headless: true });
try {
  const page = await browser.newPage();
  await page.goto(fileUrl, { waitUntil: 'networkidle0', timeout: 120000 });
  await page.pdf({
    path: outPath,
    width: '24.5in',
    height: '60.5in',
    printBackground: true,
    margin: { top: 0, right: 0, bottom: 0, left: 0 },
    preferCSSPageSize: true,
  });
  console.log('Done:', outPath);
} finally {
  await browser.close();
}

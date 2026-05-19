# NeuroClip FYP Standee (24×60 in)

Print-ready standee poster for the NUML Faisalabad open house / FYP expo. Follows the official standee layout: NUML header → project title → four content sections → project visuals → footer.

## Files

| File | Description |
|------|-------------|
| `standee.html` | Poster structure and content |
| `standee.css` | Print dimensions, typography, colors |
| `numl-logo.png` | **You provide** — official NUML logo (min 3×3 in on poster) |
| `export-pdf.mjs` | Optional Puppeteer script to generate PDF |
| `standee-print.pdf` | Generated output (not committed; see `.gitignore`) |

## Before you export

1. Copy the official **NUML logo** to `docs/standee/numl-logo.png` (PNG, high resolution; do not stretch or recolor).
2. Open `standee.html` in Chrome or Edge. Images load via relative paths to `paper_figures/` and `docs/diagrams/`.
3. **Optional:** Replace the interface design image with a live UI screenshot:
   - Save captures as `screenshot-dashboard.png` in this folder.
   - In `standee.html`, change the second figure `src` to `screenshot-dashboard.png`.

## Preview on screen

Open `standee.html` in a browser. The page auto-scales to fit your window. **Printing always uses full 24.5×60.5 in size** (including bleed).

For local preview without `file://` restrictions, run a static server from the repo root:

```bash
npx --yes serve . -p 3456
```

Then open: `http://localhost:3456/docs/standee/standee.html`

## Export to PDF (Chrome / Edge — recommended)

1. Open `standee.html` (file or via local server above).
2. **Ctrl+P** (Print).
3. **Destination:** Save as PDF.
4. **Paper size:** Custom — **24.5 × 60.5 inches** (if unavailable, choose closest large format and set scale to 100%, or use Puppeteer below).
5. **Margins:** None.
6. **Background graphics:** On.
7. **Scale:** 100% (do not shrink to fit unless your printer vendor specifies otherwise).
8. Save as `standee-print.pdf`.

Submit **print-ready PDF** plus this **editable source** (`standee.html`, `standee.css`) per NUML checklist.

## Export to PDF (Puppeteer)

From `docs/standee/`:

```bash
npm install
node export-pdf.mjs
```

Output: `standee-print.pdf` at 24.5×60.5 in with backgrounds.

Requires Node.js 18+. First run downloads Chromium via Puppeteer.

## Print shop notes

- **Trim size:** 24 × 60 in (portrait).
- **Bleed:** 0.25 in on all sides (canvas is 24.5 × 60.5 in).
- **Resolution:** Export at 300 DPI when possible (Puppeteer uses print vector layout; confirm with your vendor).
- **Colors:** RGB PDF from browser; ask the shop if they need CMYK conversion.

## NUML checklist

- [ ] 24×60 in portrait, 0.25 in bleed
- [ ] Official NUML logo top-center, unmodified
- [ ] Sections: Project Brief, Objective, Methodology, Tools & Technology (in order)
- [ ] 2–4 project images with captions
- [ ] Footer: Supervisor, each student + roll number, department
- [ ] Body text ≥20 pt, headings ≥36 pt, footer ≥18 pt
- [ ] PDF + editable HTML/CSS submitted

## Team (footer on poster)

- **Supervisor:** Subhan Arif
- **Students:** Ali Javed (FC-200), Asad Sardar (FC-211)
- **Department:** Department of Computer Science

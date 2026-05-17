# NeuroClip Chapter 4 Diagrams

This folder holds the **source** (`.mmd`) and the **rendered images** (`.png`, `.svg`) for every figure used in Chapter 4 of the NeuroClip thesis (`../NeuroClip_Chapter4_Diagrams.md`). Compression is modeled as a separate module across the diagrams, including validation, profile selection, FFmpeg processing, output storage, preview/download, and history logging.

## Files

| Figure | Source | Raster (thesis) | Vector (scalable) |
|---|---|---|---|
| 4.2  Use Case Diagram                   | `4.2_use_case.mmd`                | `4.2_use_case.png`                | `4.2_use_case.svg`                |
| 4.4  Activity Diagram                   | `4.4_activity.mmd`                | `4.4_activity.png`                | `4.4_activity.svg`                |
| 4.5.1 DFD Level 0 (Context)             | `4.5.1_dfd_level0.mmd`            | `4.5.1_dfd_level0.png`            | `4.5.1_dfd_level0.svg`            |
| 4.5.2 DFD Level 1 (Processes)           | `4.5.2_dfd_level1.mmd`            | `4.5.2_dfd_level1.png`            | `4.5.2_dfd_level1.svg`            |
| 4.5.3 DFD Level 2 (Clip Search + Compression) | `4.5.3_dfd_level2_search.mmd`     | `4.5.3_dfd_level2_search.png`     | `4.5.3_dfd_level2_search.svg`     |
| 4.6  System Sequence Diagram (Search + Compression) | `4.6_system_sequence.mmd`         | `4.6_system_sequence.png`         | `4.6_system_sequence.svg`         |
| 4.7.1 Sequence — Email Verification     | `4.7.1_sequence_email_verify.mmd` | `4.7.1_sequence_email_verify.png` | `4.7.1_sequence_email_verify.svg` |
| 4.7.2 Sequence — Clip Search + Compression End-to-End | `4.7.2_sequence_clip_search.mmd`  | `4.7.2_sequence_clip_search.png`  | `4.7.2_sequence_clip_search.svg`  |
| 4.8  Design Class Diagram               | `4.8_class_diagram.mmd`           | `4.8_class_diagram.png`           | `4.8_class_diagram.svg`           |
| 4.9.1 Interface Design                  | `4.9.1_interface_design.mmd`      | `4.9.1_interface_design.png`      | `4.9.1_interface_design.svg`      |
| 4.9.2 Component Level Design            | `4.9.2_component_level.mmd`       | `4.9.2_component_level.png`       | `4.9.2_component_level.svg`       |
| 4.9.3 Deployment Diagram                | `4.9.3_deployment.mmd`            | `4.9.3_deployment.png`            | `4.9.3_deployment.svg`            |

## Re-rendering

After editing any `.mmd` source file, re-render with:

```powershell
cd docs/diagrams
npm install        # only the first time
npm run render
```

The renderer (`render.js`) walks every `*.mmd` in this folder and produces both a high-resolution `*.png` (scale × 2, white background) and a vector `*.svg` using Mermaid CLI's bundled headless Chromium.

### Custom theme

Visual styling (font family, colours, padding, etc.) lives in `mermaid.config.json` so all diagrams keep a consistent look. Edit that file and re-run `npm run render` to restyle every diagram at once.

## Embedding in the thesis

* **Word / DOCX:** Insert → Picture → choose the `.png`. Add the caption *Arial Narrow, size 10* beneath the figure (e.g. `Figure 4.2: NeuroClip Use Case Diagram`).
* **LaTeX:** Use the `.svg` (via `\includesvg`) or convert to PDF (`inkscape ... --export-type=pdf`) for crisper print output.

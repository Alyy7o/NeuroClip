# NeuroClip Chapter 4 — Academic-Style Diagrams

This folder contains the **thesis-ready** version of every Chapter 4 diagram.
The visuals are deliberately drawn to look like the references typically used
in HCI / SE textbooks:

| Diagram family | Style |
| -------------- | ------------------------------------------------------------------ |
| Use case        | UML actors (stick figures), system rectangle, oval use cases, dashed `«include»` / `«extend»` arrows |
| Activity        | Black initial / final nodes, rounded activity states, decision diamonds, fork/join bars, guard labels |
| DFD (0/1/2)     | Orange external entities, orange process circles/boxes (Yourdon / Gane–Sarson palette) |
| Sequence        | Actor + object lifelines, dashed lifelines, activation rectangles, dashed return arrows |
| Class           | UML 3-section boxes, multiplicities, hollow aggregation diamonds, dashed `«uses»` dependencies |
| Interface       | Page-map rectangles with grouped routes, lollipop/socket interfaces |
| Component       | Component icons (small two-rectangle stub), provided/required interfaces |
| Deployment      | 3-D node boxes containing components, labelled communication paths |

All figures embed the **figure caption directly** (Arial-fallback serif, italic),
so the rendered PNG/SVG can be dropped straight into the thesis without extra
caption typesetting.

The **compression module** is shown as a separate, self-contained module across
every relevant diagram (use case, activity, DFDs, sequence, class, component,
deployment) — exactly as required by Chapter 4.

## Files

| File | Description |
| ---- | ----------- |
| `shapes.py` | Reusable matplotlib drawing primitives (actor, oval, diamond, lifeline, class box, component, deployment node, etc.) |
| `diagrams.py` | One function per figure (`draw_use_case()`, `draw_activity()`, …) |
| `render_academic_diagrams.py` | CLI entry point that renders every diagram to PNG **and** SVG |
| `4.2_use_case.{png,svg}` … `4.9.3_deployment.{png,svg}` | Generated outputs (numbered to match Chapter 4 figure IDs) |

## How to regenerate

The renderer only needs `matplotlib` (already installed in the project venv).

```bash
python docs/diagrams_academic/render_academic_diagrams.py
```

That command rewrites every `*.png` and `*.svg` next to the script.

## How to edit a single diagram

1. Open `diagrams.py` and find the `draw_<name>()` function (e.g. `draw_class_diagram`).
2. Adjust the layout — the helpers in `shapes.py` give you ready-made primitives:
   - `actor(ax, cx, base_y, "Label")`
   - `usecase(ax, cx, cy, "Search Clips by Query")`
   - `dfd_process_circle(ax, cx, cy, r, "P4", "Clip Search & Assembly")`
   - `class_box(ax, x, y_top, w, "ClassName", attrs=[…], methods=[…])`
   - `component_box`, `deploy_node`, `lollipop`, `socket`, `seq_message`, …
3. Re-run the renderer.

The layouts use plain matplotlib coordinates (no DSL), so placing a new shape is
just a matter of picking an `(x, y)` position in the figure.

## Numbering

Figure numbers follow the existing chapter numbering used in
`../NeuroClip_Chapter4_Diagrams.md`:

```
4.2  Use Case
4.4  Activity
4.5.1  DFD Level 0 (context)
4.5.2  DFD Level 1
4.5.3  DFD Level 2 (Search + Compression explosion)
4.6    System Sequence
4.7.1  Sequence — Email Verification + opening Compression
4.7.2  Sequence — Clip Search end-to-end + Compression end-to-end
4.8    Design Class Diagram
4.9.1  Interface Design (page map + Compression UI flow)
4.9.2  Component Level Design (Compression Service highlighted)
4.9.3  Deployment Diagram (Compression Worker highlighted)
```

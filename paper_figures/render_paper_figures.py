"""Render figures used in the NeuroClip IEEE conference paper (paper.html).

This script is intentionally self-contained (it does not depend on the
``docs/diagrams_academic`` package) because the conference-paper figures use a
slightly different visual style (light-blue rounded blocks with bold dark
labels, similar to the framework-architecture figures commonly seen in IEEE
conference papers).

Outputs (PNG + SVG) are written next to this script:

    fig1_system_architecture.png / .svg
    fig2_dual_encoder_training.png / .svg

Run from anywhere:

    python paper_figures/render_paper_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Visual style (mirrors common IEEE conference framework figures)
# ---------------------------------------------------------------------------
BLOCK_FILL = "#E7F0FA"        # very light blue interior
BLOCK_STROKE = "#1F3F66"      # deep navy border
BLOCK_TEXT = "#0E1F36"        # near-black navy text
ACCENT_FILL = "#FFF4E0"       # warm cream for accent / output blocks
ACCENT_STROKE = "#A65A1B"     # warm brown border
ACCENT_TEXT = "#5A2D00"
GROUP_STROKE = "#1F3F66"
ARROW_COLOR = "#1F3F66"
DASH = (0, (5, 4))


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def new_canvas(width: float, height: float, dpi: int = 200) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def save(fig: plt.Figure, basename: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(
            HERE / f"{basename}.{ext}",
            bbox_inches="tight",
            pad_inches=0.18,
            facecolor="white",
        )
    plt.close(fig)


def block(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str | None = None,
    *,
    fill: str = BLOCK_FILL,
    stroke: str = BLOCK_STROKE,
    text_color: str = BLOCK_TEXT,
    title_size: int = 10,
    subtitle_size: int = 8,
    radius: float = 0.10,
) -> None:
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=fill, edgecolor=stroke, linewidth=1.4,
    )
    ax.add_patch(p)
    if subtitle:
        ax.text(
            x + w / 2, y + h / 2 + 0.18, title,
            ha="center", va="center",
            fontsize=title_size, color=text_color, weight="bold",
            family="serif",
        )
        ax.text(
            x + w / 2, y + h / 2 - 0.22, subtitle,
            ha="center", va="center",
            fontsize=subtitle_size, color=text_color,
            family="serif",
        )
    else:
        ax.text(
            x + w / 2, y + h / 2, title,
            ha="center", va="center",
            fontsize=title_size, color=text_color, weight="bold",
            family="serif",
        )


def group_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    stroke: str = GROUP_STROKE,
) -> None:
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor="none", edgecolor=stroke, linewidth=1.0, linestyle=DASH,
    )
    ax.add_patch(p)
    ax.text(
        x + 0.20, y + h - 0.05, label,
        ha="left", va="top",
        fontsize=9, color=stroke, weight="bold", style="italic",
        family="serif",
    )


def arrow(
    ax: plt.Axes,
    x1: float, y1: float,
    x2: float, y2: float,
    *,
    label: str | None = None,
    label_offset: tuple[float, float] = (0, 0.18),
    color: str = ARROW_COLOR,
    lw: float = 1.5,
    label_size: int = 8,
) -> None:
    p = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        color=color, linewidth=lw,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(p)
    if label:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=label_size, color=color, family="serif", style="italic")


def line(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float,
         *, color: str = ARROW_COLOR, lw: float = 1.5) -> None:
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, solid_capstyle="round")


def caption(ax: plt.Axes, text: str) -> None:
    xlim = ax.get_xlim()
    ax.text(
        (xlim[0] + xlim[1]) / 2,
        ax.get_ylim()[0] + 0.18,
        text,
        ha="center", va="bottom",
        fontsize=10, family="serif", style="italic",
        color="#000",
    )


# ---------------------------------------------------------------------------
# Decorative input/output icons
# ---------------------------------------------------------------------------

def video_icon(ax: plt.Axes, cx: float, cy: float, *, scale: float = 1.0) -> None:
    """Simple film-strip / play icon."""
    w, h = 1.2 * scale, 0.85 * scale
    rect = Rectangle((cx - w / 2, cy - h / 2), w, h,
                     facecolor="#FFFFFF", edgecolor=BLOCK_STROKE, linewidth=1.2)
    ax.add_patch(rect)
    # film strip holes
    hole_w = 0.10 * scale
    hole_h = 0.10 * scale
    for i in range(4):
        hx = cx - w / 2 + 0.10 + i * 0.30 * scale
        ax.add_patch(Rectangle((hx, cy + h / 2 - 0.16), hole_w, hole_h,
                               facecolor=BLOCK_STROKE, edgecolor=BLOCK_STROKE))
        ax.add_patch(Rectangle((hx, cy - h / 2 + 0.06), hole_w, hole_h,
                               facecolor=BLOCK_STROKE, edgecolor=BLOCK_STROKE))
    # play triangle
    play = Polygon(
        [(cx - 0.12 * scale, cy + 0.18 * scale),
         (cx - 0.12 * scale, cy - 0.18 * scale),
         (cx + 0.20 * scale, cy)],
        facecolor=BLOCK_STROKE, edgecolor=BLOCK_STROKE,
    )
    ax.add_patch(play)


def query_icon(ax: plt.Axes, cx: float, cy: float, *, scale: float = 1.0) -> None:
    """Magnifying-glass query icon."""
    r = 0.30 * scale
    ax.add_patch(Circle((cx - 0.10 * scale, cy + 0.05 * scale), r,
                        facecolor="#FFFFFF", edgecolor=BLOCK_STROKE, linewidth=1.4))
    line(ax,
         cx + 0.12 * scale, cy - 0.18 * scale,
         cx + 0.40 * scale, cy - 0.45 * scale,
         color=BLOCK_STROKE, lw=2.5)


def clip_stack_icon(ax: plt.Axes, cx: float, cy: float, *, scale: float = 1.0) -> None:
    """Three offset rectangles to represent a stack of returned clips."""
    w, h = 0.95 * scale, 0.55 * scale
    for i, off in enumerate([0.18, 0.09, 0.0]):
        ax.add_patch(Rectangle(
            (cx - w / 2 - off, cy - h / 2 + off), w, h,
            facecolor="#FFFFFF", edgecolor=ACCENT_STROKE, linewidth=1.2,
        ))
    # play triangle on the front clip
    ax.add_patch(Polygon(
        [(cx - 0.08 * scale, cy + 0.12 * scale),
         (cx - 0.08 * scale, cy - 0.12 * scale),
         (cx + 0.14 * scale, cy)],
        facecolor=ACCENT_STROKE, edgecolor=ACCENT_STROKE,
    ))


# ===========================================================================
# Figure 1 — NeuroClip System Architecture (Summarization + Compression)
# ===========================================================================

def draw_fig1_system_architecture() -> None:
    fig, ax = new_canvas(16.0, 9.4)

    # ---- Inputs -----------------------------------------------------------
    video_icon(ax, 1.05, 7.30, scale=1.10)
    ax.text(1.05, 6.40, "Input Video",
            ha="center", va="center", fontsize=10, weight="bold",
            family="serif", color=BLOCK_TEXT)

    query_icon(ax, 1.05, 2.40, scale=1.10)
    ax.text(1.05, 1.60, "User Query",
            ha="center", va="center", fontsize=10, weight="bold",
            family="serif", color=BLOCK_TEXT)

    # ---- INDEXING / SUMMARIZATION group ----------------------------------
    group_box(ax, 2.40, 4.55, 9.30, 4.45,
              "Module A : Multimodal Indexing & Summarization Pipeline")

    # Top row : ASR branch
    block(ax, 2.70, 7.55, 2.30, 1.05,
          "ASR Transcription", "AssemblyAI (SRT, word-level ts)")
    block(ax, 5.45, 7.55, 2.40, 1.05,
          "Sentence Units", "(text, start_time, end_time)")

    # Middle row : OCR branch
    block(ax, 2.70, 5.95, 2.30, 1.05,
          "Frame Sampling", "OpenCV @ 3 s interval")
    block(ax, 5.45, 5.95, 2.40, 1.05,
          "OCR Recognition", "EasyOCR + custom OCR model")

    # Fusion / Embedding column
    block(ax, 8.30, 6.55, 3.10, 1.45,
          "Multimodal Fusion +",
          "Sentence Embedding\n(all-MiniLM-L6-v2,  d = 384)")

    # Vector store
    block(ax, 5.45, 4.85, 5.95, 0.80,
          "Vector Store : Supabase (PostgreSQL + pgvector)",
          fill="#F4F8FD", title_size=10)

    # ---- RETRIEVAL group --------------------------------------------------
    group_box(ax, 2.40, 1.55, 9.30, 2.65,
              "Module B : Two-Stage Retrieval & Clip Construction")

    block(ax, 2.70, 2.50, 2.30, 1.05,
          "Query Embedding", "all-MiniLM-L6-v2 (d = 384)")
    block(ax, 5.45, 2.50, 2.40, 1.05,
          "Bi-Encoder Search", "cosine top-N over windows")
    block(ax, 8.40, 2.50, 3.00, 1.05,
          "Cross-Encoder Re-rank",
          "ms-marco-MiniLM-L-6-v2")

    # ---- POST-PROCESS column (clip + compression) -------------------------
    group_box(ax, 12.15, 1.55, 3.55, 7.45,
              "Module C : Clip + Compression")

    block(ax, 12.40, 7.55, 3.10, 1.05,
          "Neighbour Merging",
          "gap < 2 s  →  unified segment")
    block(ax, 12.40, 5.95, 3.10, 1.05,
          "Clip Extraction", "FFmpeg (start / end + pad)")
    block(ax, 12.40, 4.40, 3.10, 1.05,
          "Codec-Aware Compression",
          "H.265 / HEVC  (NVENC | libx265)")

    # ---- OUTPUT -----------------------------------------------------------
    block(ax, 12.40, 2.65, 3.10, 1.10,
          "Ranked Playable Clips",
          "+ summaries  (frontend)",
          fill=ACCENT_FILL, stroke=ACCENT_STROKE, text_color=ACCENT_TEXT)
    clip_stack_icon(ax, 13.95, 1.95, scale=1.05)

    # ---- ARROWS : indexing branch ----------------------------------------
    # Input video → ASR & Frame sampling
    arrow(ax, 1.85, 7.30, 2.70, 8.10)
    arrow(ax, 1.85, 7.30, 2.70, 6.50)

    # ASR → Sentence Units
    arrow(ax, 5.00, 8.10, 5.45, 8.10)
    # Frame Sampling → OCR
    arrow(ax, 5.00, 6.50, 5.45, 6.50)

    # Sentence Units & OCR → Fusion / Embedding
    arrow(ax, 7.85, 8.10, 8.30, 7.55, label="speech text",
          label_offset=(0.05, 0.20), label_size=7)
    arrow(ax, 7.85, 6.50, 8.30, 7.05, label="visual text",
          label_offset=(0.10, -0.20), label_size=7)

    # Embedding → Vector Store
    arrow(ax, 9.85, 6.55, 9.85, 5.65, label="384-dim vectors",
          label_offset=(0.95, 0), label_size=7)

    # ---- ARROWS : retrieval branch ---------------------------------------
    # Query → Query Embedding
    arrow(ax, 1.85, 2.40, 2.70, 3.02)
    arrow(ax, 5.00, 3.02, 5.45, 3.02)
    arrow(ax, 7.85, 3.02, 8.40, 3.02)

    # Vector store → Bi-Encoder (top-N candidates)
    arrow(ax, 6.65, 4.85, 6.65, 3.55, label="top-N candidates",
          label_offset=(0.95, 0), label_size=7)

    # Vector store → Cross-Encoder (window text)
    arrow(ax, 9.20, 4.85, 9.20, 3.55, label="window text",
          label_offset=(0.65, 0), label_size=7)

    # Cross-Encoder Re-rank → enter Module C from the right side, climb up
    # to the top of the column and drop into "Neighbour Merging" so the
    # whole right column reads cleanly from top to bottom.
    line(ax, 11.40, 3.02, 11.85, 3.02, color=ARROW_COLOR, lw=1.5)
    line(ax, 11.85, 3.02, 11.85, 8.30, color=ARROW_COLOR, lw=1.5)
    arrow(ax, 11.85, 8.30, 12.40, 8.10, label="ranked windows",
          label_offset=(-0.15, 0.20), label_size=7)

    # Vertical chain (top → bottom) inside the Clip + Compression module
    arrow(ax, 13.95, 7.55, 13.95, 7.00)   # Neighbour Merging → Clip Extraction
    arrow(ax, 13.95, 5.95, 13.95, 5.45)   # Clip Extraction → Codec-Aware Compression
    arrow(ax, 13.95, 4.40, 13.95, 3.75)   # Compression → Ranked Playable Clips

    save(fig, "fig1_system_architecture")


# ===========================================================================
# Figure 2 — Custom dual-encoder OCR / multimodal training pipeline
# ===========================================================================

def draw_fig2_dual_encoder_training() -> None:
    fig, ax = new_canvas(15.5, 8.6)

    # Dataset
    block(ax, 0.40, 6.60, 3.00, 1.30,
          "Dataset : MSR-VTT",
          "199,994 captions  /  20 k val")

    # Pre-processing
    block(ax, 4.10, 6.60, 3.00, 1.30,
          "Pre-processing",
          "video → frame sampling\n+ caption tokenisation")

    # Image encoder
    block(ax, 7.80, 7.40, 3.30, 1.10,
          "Image Encoder",
          "ResNet-50  →  W_img · v")

    # Text encoder
    block(ax, 7.80, 6.05, 3.30, 1.10,
          "Text Encoder",
          "DistilBERT [CLS]  →  W_txt · t")

    # Projection / L2-norm
    block(ax, 11.80, 6.60, 3.20, 1.30,
          "L2-Normalised Embeddings",
          "f̂(v),  ĝ(t)  ∈  ℝ^256")

    # Contrastive loss
    block(ax, 4.10, 4.30, 3.00, 1.40,
          "Symmetric InfoNCE Loss",
          "L = (L_img + L_txt) / 2\nτ = 0.07")

    # Optimiser
    block(ax, 7.80, 4.30, 3.30, 1.40,
          "AdamW Optimiser",
          "lr = 1e-4,  λ = 1e-4\nβ₁ = 0.9,  β₂ = 0.999")

    # Scheduler
    block(ax, 11.80, 4.30, 3.20, 1.40,
          "Cosine LR Schedule",
          "η_min = 1e-6,  T = 5 epochs")

    # Training platform
    block(ax, 0.40, 4.30, 3.00, 1.40,
          "Training Platform",
          "Kaggle  Tesla P100 (16 GB)\nbatch = 32,  epochs = 5",
          fill="#F4F8FD")

    # Evaluation block
    block(ax, 0.40, 1.95, 5.20, 1.55,
          "Evaluation Metrics  (image → text)",
          "Recall@1 / @5 / @10\nMeanRank,  MedianRank",
          fill=ACCENT_FILL, stroke=ACCENT_STROKE, text_color=ACCENT_TEXT)

    # Final results
    block(ax, 6.30, 1.95, 4.40, 1.55,
          "Final Epoch Results",
          "R@1 = 0.527   R@5 = 0.831\n"
          "R@10 = 0.918   MeanRank = 5.34",
          fill=ACCENT_FILL, stroke=ACCENT_STROKE, text_color=ACCENT_TEXT)

    # Deployment back to retrieval
    block(ax, 11.40, 1.95, 3.60, 1.55,
          "Deployed in NeuroClip",
          "frame-text re-ranking\n+ OCR cross-modal scoring",
          fill=ACCENT_FILL, stroke=ACCENT_STROKE, text_color=ACCENT_TEXT)

    # ---- Arrows ----------------------------------------------------------
    arrow(ax, 3.40, 7.25, 4.10, 7.25)
    arrow(ax, 7.10, 7.25, 7.80, 7.95)
    arrow(ax, 7.10, 7.25, 7.80, 6.60)
    arrow(ax, 11.10, 7.95, 11.80, 7.45)
    arrow(ax, 11.10, 6.60, 11.80, 7.05)

    # Embeddings → Loss
    arrow(ax, 13.40, 6.60, 13.40, 5.70)
    line(ax, 13.40, 5.70, 5.60, 5.70)
    arrow(ax, 5.60, 5.70, 5.60, 5.70)  # tiny end
    line(ax, 5.60, 5.70, 5.60, 5.70)
    arrow(ax, 5.60, 5.70, 5.60, 5.70)
    # cleaner: explicit two-segment routing
    line(ax, 13.40, 6.60, 13.40, 5.95, color=ARROW_COLOR, lw=1.5)
    line(ax, 13.40, 5.95, 5.60, 5.95, color=ARROW_COLOR, lw=1.5)
    arrow(ax, 5.60, 5.95, 5.60, 5.70)

    # Loss ↔ Optimiser ↔ Scheduler
    arrow(ax, 7.10, 5.00, 7.80, 5.00)
    arrow(ax, 11.10, 5.00, 11.80, 5.00)

    # Training platform → loss / optimiser
    arrow(ax, 3.40, 5.00, 4.10, 5.00)

    # Optimiser → final metrics
    line(ax, 9.45, 4.30, 9.45, 3.85, color=ARROW_COLOR, lw=1.5)
    arrow(ax, 9.45, 3.85, 8.50, 3.50)

    # Eval → Final results → Deployed
    arrow(ax, 5.60, 2.72, 6.30, 2.72)
    arrow(ax, 10.70, 2.72, 11.40, 2.72)

    # Training platform → Eval (down-left link)
    line(ax, 1.90, 4.30, 1.90, 3.85, color=ARROW_COLOR, lw=1.5)
    arrow(ax, 1.90, 3.85, 1.90, 3.50)

    save(fig, "fig2_dual_encoder_training")


# ===========================================================================
# Driver
# ===========================================================================

ALL_FIGURES = [
    ("fig1  system architecture       ", draw_fig1_system_architecture),
    ("fig2  dual-encoder training     ", draw_fig2_dual_encoder_training),
]


def render_all() -> None:
    print("Rendering NeuroClip paper figures ...")
    for name, fn in ALL_FIGURES:
        print(f"  drawing {name} ...", flush=True)
        fn()
    print(f"Done. Outputs in: {HERE}")


if __name__ == "__main__":
    render_all()

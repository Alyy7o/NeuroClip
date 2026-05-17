"""Shared drawing primitives for the NeuroClip academic-style Chapter 4 diagrams.

All helpers operate on a matplotlib ``Axes`` and use only black/grey strokes by
default so that the resulting figures match the look of typical UML / DFD
illustrations seen in academic theses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

LINE = "#1a1a1a"
DFD_ORANGE_FILL = "#F4B183"
DFD_ORANGE_STROKE = "#C0571E"
ACTIVITY_FILL = "#FFF8DC"
ACTIVITY_STROKE = "#7B5E1A"
CLASS_FILL = "#FFFFFF"
COMPONENT_FILL = "#FFFFFF"
DEPLOY_FILL = "#FFFFFF"
DASH = (0, (4, 3))


def new_canvas(width: float, height: float, dpi: int = 150) -> tuple[plt.Figure, Axes]:
    """Create a new figure/axes with a clean academic look."""
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def add_caption(ax: Axes, text: str) -> None:
    """Captions are added in the thesis text, not inside generated images."""
    return


def save(fig: plt.Figure, path_no_ext: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(
            f"{path_no_ext}.{ext}",
            bbox_inches="tight",
            pad_inches=0.25,
            facecolor="white",
        )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generic primitives
# ---------------------------------------------------------------------------

def text(ax: Axes, x: float, y: float, label: str, *, size: int = 9, weight: str = "normal", style: str = "normal", ha: str = "center", va: str = "center", color: str = LINE, family: str = "sans-serif") -> None:
    ax.text(x, y, label, ha=ha, va=va, fontsize=size, color=color, weight=weight, style=style, family=family)


def line(ax: Axes, x1: float, y1: float, x2: float, y2: float, *, color: str = LINE, lw: float = 1.0, dashed: bool = False) -> None:
    ls = "--" if dashed else "-"
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=ls, solid_capstyle="round")


def arrow(ax: Axes, x1: float, y1: float, x2: float, y2: float, *, label: str | None = None, dashed: bool = False, color: str = LINE, lw: float = 1.0, label_offset: tuple[float, float] = (0, 0.15), label_size: int = 8, double: bool = False, mutation: int = 12) -> None:
    style = "<->" if double else "->"
    ls = "--" if dashed else "-"
    p = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=mutation, color=color, linewidth=lw, linestyle=ls, shrinkA=2, shrinkB=2)
    ax.add_patch(p)
    if label:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        text(ax, mx, my, label, size=label_size)


def rect(ax: Axes, x: float, y: float, w: float, h: float, *, fill: str = "white", stroke: str = LINE, lw: float = 1.0, dashed: bool = False, label: str | None = None, label_pos: str = "top", label_size: int = 9, label_weight: str = "normal", padding: float = 0.15) -> Rectangle:
    ls = "--" if dashed else "-"
    r = Rectangle((x, y), w, h, facecolor=fill, edgecolor=stroke, linewidth=lw, linestyle=ls)
    ax.add_patch(r)
    if label:
        if label_pos == "top":
            text(ax, x + w / 2, y + h - padding, label, size=label_size, weight=label_weight, va="top")
        elif label_pos == "center":
            text(ax, x + w / 2, y + h / 2, label, size=label_size, weight=label_weight)
        elif label_pos == "above":
            text(ax, x + w / 2, y + h + 0.15, label, size=label_size, weight=label_weight, va="bottom")
    return r


def rounded_rect(ax: Axes, x: float, y: float, w: float, h: float, *, fill: str = "white", stroke: str = LINE, lw: float = 1.0, label: str | None = None, label_size: int = 9, weight: str = "normal", radius: float = 0.18) -> FancyBboxPatch:
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=fill, edgecolor=stroke, linewidth=lw,
    )
    ax.add_patch(p)
    if label:
        text(ax, x + w / 2, y + h / 2, label, size=label_size, weight=weight)
    return p


def oval(ax: Axes, cx: float, cy: float, w: float, h: float, *, label: str = "", fill: str = "white", stroke: str = LINE, lw: float = 1.0, label_size: int = 9) -> Ellipse:
    e = Ellipse((cx, cy), w, h, facecolor=fill, edgecolor=stroke, linewidth=lw)
    ax.add_patch(e)
    if label:
        text(ax, cx, cy, label, size=label_size)
    return e


def diamond(ax: Axes, cx: float, cy: float, w: float, h: float, *, label: str = "", fill: str = ACTIVITY_FILL, stroke: str = ACTIVITY_STROKE, lw: float = 1.0, label_size: int = 8) -> None:
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    p = Polygon(pts, facecolor=fill, edgecolor=stroke, linewidth=lw, closed=True)
    ax.add_patch(p)
    if label:
        text(ax, cx, cy, label, size=label_size)


def filled_circle(ax: Axes, cx: float, cy: float, r: float, *, fill: str = LINE, stroke: str = LINE) -> None:
    ax.add_patch(Circle((cx, cy), r, facecolor=fill, edgecolor=stroke, linewidth=1.0))


def end_node(ax: Axes, cx: float, cy: float, r: float = 0.16) -> None:
    ax.add_patch(Circle((cx, cy), r * 1.45, facecolor="white", edgecolor=LINE, linewidth=1.2))
    ax.add_patch(Circle((cx, cy), r * 0.85, facecolor=LINE, edgecolor=LINE))


def fork_bar(ax: Axes, cx: float, cy: float, w: float, h: float = 0.08) -> None:
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=LINE, edgecolor=LINE))


# ---------------------------------------------------------------------------
# Use-case helpers (UML)
# ---------------------------------------------------------------------------

def actor(ax: Axes, cx: float, base_y: float, label: str = "", *, scale: float = 1.0, label_below: bool = True, label_size: int = 9) -> None:
    head_r = 0.18 * scale
    body_h = 0.55 * scale
    arm_w = 0.45 * scale
    leg_w = 0.40 * scale
    leg_h = 0.55 * scale
    head_cy = base_y + leg_h + body_h + head_r
    ax.add_patch(Circle((cx, head_cy), head_r, facecolor="white", edgecolor=LINE, linewidth=1.2))
    line(ax, cx, head_cy - head_r, cx, base_y + leg_h, lw=1.2)
    body_top = head_cy - head_r - 0.05
    arm_y = body_top - 0.25 * scale
    line(ax, cx - arm_w / 2, arm_y, cx + arm_w / 2, arm_y, lw=1.2)
    line(ax, cx, base_y + leg_h, cx - leg_w / 2, base_y, lw=1.2)
    line(ax, cx, base_y + leg_h, cx + leg_w / 2, base_y, lw=1.2)
    if label:
        if label_below:
            text(ax, cx, base_y - 0.18, label, size=label_size, weight="bold")
        else:
            text(ax, cx, head_cy + head_r + 0.18, label, size=label_size, weight="bold")


def usecase(ax: Axes, cx: float, cy: float, label: str, *, w: float = 1.85, h: float = 0.7, label_size: int = 8) -> tuple[float, float, float, float]:
    """Draw a UML use-case ellipse and return its bounding box (x, y, w, h)."""
    oval(ax, cx, cy, w, h, label=label, fill="white", stroke=LINE, label_size=label_size)
    return (cx - w / 2, cy - h / 2, w, h)


def uc_assoc(ax: Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    """Solid line, no arrow, used between actor and use case."""
    line(ax, x1, y1, x2, y2, lw=1.0)


def uc_dependency(ax: Axes, x1: float, y1: float, x2: float, y2: float, label: str) -> None:
    """Dashed open-arrow include/extend dependency with stereotype label."""
    p = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12, color=LINE, linewidth=1.0, linestyle=(0, (4, 3)), shrinkA=4, shrinkB=4)
    ax.add_patch(p)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    text(ax, mx, my + 0.13, label, size=8, style="italic")


def system_box(ax: Axes, x: float, y: float, w: float, h: float, label: str, stereotype: str | None = None) -> None:
    rect(ax, x, y, w, h, fill="white", stroke=LINE, lw=1.4)
    if stereotype:
        text(ax, x + w - 0.2, y + h - 0.25, stereotype, size=8, style="italic", ha="right", va="top")
        text(ax, x + w - 0.2, y + h - 0.55, label, size=10, weight="bold", ha="right", va="top")
    else:
        text(ax, x + w - 0.2, y + h - 0.25, label, size=10, weight="bold", ha="right", va="top")


# ---------------------------------------------------------------------------
# DFD helpers (orange academic style)
# ---------------------------------------------------------------------------

def dfd_external(ax: Axes, x: float, y: float, w: float, h: float, label: str) -> None:
    rect(ax, x, y, w, h, fill=DFD_ORANGE_FILL, stroke=DFD_ORANGE_STROKE, lw=1.2)
    text(ax, x + w / 2, y + h / 2, label, size=10, weight="bold")


def dfd_process_circle(ax: Axes, cx: float, cy: float, r: float, number: str, name: str) -> None:
    ax.add_patch(Circle((cx, cy), r, facecolor=DFD_ORANGE_FILL, edgecolor=DFD_ORANGE_STROKE, linewidth=1.2))
    text(ax, cx, cy + 0.18, number, size=11, weight="bold")
    text(ax, cx, cy - 0.22, name, size=10)


def dfd_process_box(ax: Axes, x: float, y: float, w: float, h: float, number: str, name: str) -> None:
    rect(ax, x, y, w, h, fill=DFD_ORANGE_FILL, stroke=DFD_ORANGE_STROKE, lw=1.2)
    line(ax, x, y + h - 0.45, x + w, y + h - 0.45, lw=1.0, color=DFD_ORANGE_STROKE)
    text(ax, x + 0.2, y + h - 0.22, number, size=10, weight="bold", ha="left")
    text(ax, x + w / 2, y + (h - 0.45) / 2, name, size=9)


def dfd_store(ax: Axes, x: float, y: float, w: float, h: float, number: str, label: str) -> None:
    """Open rectangle data store (Gane-Sarson / Yourdon style)."""
    rect(ax, x, y, w, h, fill="white", stroke=LINE, lw=1.0)
    line(ax, x, y, x, y + h, lw=2.0)
    line(ax, x + 0.45, y, x + 0.45, y + h, lw=1.0)
    text(ax, x + 0.22, y + h / 2, number, size=9, weight="bold")
    text(ax, x + 0.45 + (w - 0.45) / 2, y + h / 2, label, size=9)


def dfd_arrow(ax: Axes, x1: float, y1: float, x2: float, y2: float, label: str, *, label_offset: tuple[float, float] = (0, 0.18), label_size: int = 8) -> None:
    arrow(ax, x1, y1, x2, y2, label=label, label_offset=label_offset, label_size=label_size, mutation=14)


# ---------------------------------------------------------------------------
# Sequence-diagram helpers
# ---------------------------------------------------------------------------

@dataclass
class Lifeline:
    cx: float
    label: str
    is_actor: bool = False
    width: float = 1.7


def seq_setup(ax: Axes, lifelines: Sequence[Lifeline], top: float, bottom: float, header_h: float = 0.55) -> dict[str, float]:
    centres: dict[str, float] = {}
    for ll in lifelines:
        centres[ll.label] = ll.cx
        if ll.is_actor:
            actor(ax, ll.cx, top - 0.05, ll.label, scale=0.75)
        else:
            rect(ax, ll.cx - ll.width / 2, top - header_h, ll.width, header_h, fill="white", stroke=LINE, lw=1.0, label=ll.label, label_pos="center", label_size=9, label_weight="bold")
        line(ax, ll.cx, top - header_h - 0.02 if not ll.is_actor else top - 0.55, ll.cx, bottom, dashed=True, lw=0.9)
    return centres


def seq_message(ax: Axes, x1: float, x2: float, y: float, label: str, *, dashed: bool = False, self_call: bool = False, label_size: int = 8) -> None:
    if self_call:
        loop_w = 0.55
        x_top = x1
        ax.add_patch(FancyArrowPatch((x_top, y), (x_top + loop_w, y), arrowstyle="-", color=LINE, linewidth=1.0))
        ax.add_patch(FancyArrowPatch((x_top + loop_w, y), (x_top + loop_w, y - 0.25), arrowstyle="-", color=LINE, linewidth=1.0))
        ax.add_patch(FancyArrowPatch((x_top + loop_w, y - 0.25), (x_top, y - 0.25), arrowstyle="->", color=LINE, linewidth=1.0, mutation_scale=12))
        text(ax, x_top + loop_w + 0.1, y - 0.12, label, size=label_size, ha="left")
        return
    p = FancyArrowPatch((x1, y), (x2, y), arrowstyle="->", mutation_scale=12, color=LINE, linewidth=1.0, linestyle=("--" if dashed else "-"))
    ax.add_patch(p)
    text(ax, (x1 + x2) / 2, y + 0.12, label, size=label_size)


def seq_activation(ax: Axes, cx: float, y_top: float, y_bottom: float, *, w: float = 0.18) -> None:
    rect(ax, cx - w / 2, y_bottom, w, y_top - y_bottom, fill="white", stroke=LINE, lw=1.0)


# ---------------------------------------------------------------------------
# Class / interface diagram helpers (UML)
# ---------------------------------------------------------------------------

def class_box(ax: Axes, x: float, y: float, w: float, name: str, attrs: Sequence[str], methods: Sequence[str], *, abstract: bool = False, stereotype: str | None = None, line_height: float = 0.32, header_h: float = 0.55) -> tuple[float, float, float, float]:
    attrs_h = max(len(attrs), 1) * line_height + 0.18
    methods_h = max(len(methods), 1) * line_height + 0.18
    h = header_h + attrs_h + methods_h
    y0 = y - h
    rect(ax, x, y0, w, h, fill=CLASS_FILL, stroke=LINE, lw=1.0)
    line(ax, x, y0 + attrs_h + methods_h, x + w, y0 + attrs_h + methods_h, lw=1.0)
    line(ax, x, y0 + methods_h, x + w, y0 + methods_h, lw=1.0)
    name_y = y0 + attrs_h + methods_h + header_h / 2
    if stereotype:
        text(ax, x + w / 2, name_y + 0.13, stereotype, size=8, style="italic")
        text(ax, x + w / 2, name_y - 0.08, name, size=10, weight="bold", style=("italic" if abstract else "normal"))
    else:
        text(ax, x + w / 2, name_y, name, size=10, weight="bold", style=("italic" if abstract else "normal"))
    cy = y0 + methods_h + attrs_h - 0.18
    for a in attrs or [" "]:
        text(ax, x + 0.15, cy, a, size=8, ha="left")
        cy -= line_height
    cy = y0 + methods_h - 0.18
    for m in methods or [" "]:
        text(ax, x + 0.15, cy, m, size=8, ha="left")
        cy -= line_height
    return x, y0, w, h


def assoc_line(ax: Axes, x1: float, y1: float, x2: float, y2: float, *, m1: str = "", m2: str = "", role: str = "", lw: float = 1.0) -> None:
    line(ax, x1, y1, x2, y2, lw=lw)
    if m1:
        text(ax, x1 + (0.1 if x2 > x1 else -0.1), y1 + 0.18, m1, size=8, ha=("left" if x2 > x1 else "right"))
    if m2:
        text(ax, x2 - (0.1 if x2 > x1 else -0.1), y2 + 0.18, m2, size=8, ha=("right" if x2 > x1 else "left"))
    if role:
        text(ax, (x1 + x2) / 2, (y1 + y2) / 2 + 0.18, role, size=8, style="italic")


def dependency_line(ax: Axes, x1: float, y1: float, x2: float, y2: float, *, label: str = "") -> None:
    p = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=10, color=LINE, linewidth=1.0, linestyle=(0, (4, 3)), shrinkA=2, shrinkB=2)
    ax.add_patch(p)
    if label:
        text(ax, (x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, size=8, style="italic")


def aggregation_diamond(ax: Axes, cx: float, cy: float, *, size: float = 0.18, filled: bool = False) -> None:
    pts = [(cx, cy + size), (cx + size, cy), (cx, cy - size), (cx - size, cy)]
    fill = LINE if filled else "white"
    ax.add_patch(Polygon(pts, facecolor=fill, edgecolor=LINE, linewidth=1.0))


def inheritance_triangle(ax: Axes, cx: float, cy: float, direction: str = "up", size: float = 0.22) -> None:
    if direction == "up":
        pts = [(cx, cy + size), (cx - size, cy - size), (cx + size, cy - size)]
    elif direction == "down":
        pts = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
    elif direction == "left":
        pts = [(cx - size, cy), (cx + size, cy + size), (cx + size, cy - size)]
    else:
        pts = [(cx + size, cy), (cx - size, cy + size), (cx - size, cy - size)]
    ax.add_patch(Polygon(pts, facecolor="white", edgecolor=LINE, linewidth=1.0))


# ---------------------------------------------------------------------------
# Component / deployment helpers
# ---------------------------------------------------------------------------

def component_box(ax: Axes, x: float, y: float, w: float, h: float, name: str) -> None:
    rect(ax, x, y, w, h, fill=COMPONENT_FILL, stroke=LINE, lw=1.0)
    icon_x, icon_y = x + w - 0.55, y + h - 0.45
    rect(ax, icon_x, icon_y, 0.4, 0.3, fill="white", stroke=LINE, lw=1.0)
    rect(ax, icon_x - 0.08, icon_y + 0.06, 0.16, 0.07, fill="white", stroke=LINE, lw=1.0)
    rect(ax, icon_x - 0.08, icon_y + 0.18, 0.16, 0.07, fill="white", stroke=LINE, lw=1.0)
    text(ax, x + w / 2 - 0.15, y + h / 2, name, size=10, weight="bold")


def lollipop(ax: Axes, x_start: float, y: float, length: float, label: str, *, side: str = "right") -> None:
    if side == "right":
        x_end = x_start + length
        line(ax, x_start, y, x_end, y, lw=1.0)
        ax.add_patch(Circle((x_end, y), 0.08, facecolor="white", edgecolor=LINE, linewidth=1.0))
        text(ax, x_end + 0.12, y, label, size=8, ha="left")
    else:
        x_end = x_start - length
        line(ax, x_start, y, x_end, y, lw=1.0)
        ax.add_patch(Circle((x_end, y), 0.08, facecolor="white", edgecolor=LINE, linewidth=1.0))
        text(ax, x_end - 0.12, y, label, size=8, ha="right")


def socket(ax: Axes, x_start: float, y: float, length: float, label: str, *, side: str = "right") -> None:
    if side == "right":
        x_end = x_start + length
        line(ax, x_start, y, x_end, y, lw=1.0)
        arc = mpatches.Arc((x_end, y), 0.22, 0.22, angle=0, theta1=90, theta2=270, color=LINE, linewidth=1.2)
        ax.add_patch(arc)
        text(ax, x_end - 0.15, y - 0.18, label, size=8, ha="right")
    else:
        x_end = x_start - length
        line(ax, x_start, y, x_end, y, lw=1.0)
        arc = mpatches.Arc((x_end, y), 0.22, 0.22, angle=0, theta1=270, theta2=90, color=LINE, linewidth=1.2)
        ax.add_patch(arc)
        text(ax, x_end + 0.15, y - 0.18, label, size=8, ha="left")


def deploy_node(ax: Axes, x: float, y: float, w: float, h: float, name: str, depth: float = 0.3) -> None:
    rect(ax, x, y, w, h, fill=DEPLOY_FILL, stroke=LINE, lw=1.2)
    top_pts = [(x, y + h), (x + depth, y + h + depth), (x + w + depth, y + h + depth), (x + w, y + h)]
    ax.add_patch(Polygon(top_pts, facecolor="#F5F5F5", edgecolor=LINE, linewidth=1.0, closed=True))
    side_pts = [(x + w, y), (x + w + depth, y + depth), (x + w + depth, y + h + depth), (x + w, y + h)]
    ax.add_patch(Polygon(side_pts, facecolor="#FAFAFA", edgecolor=LINE, linewidth=1.0, closed=True))
    text(ax, x + w / 2, y + h - 0.22, name, size=10, weight="bold", ha="center")

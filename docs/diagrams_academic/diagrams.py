"""All NeuroClip Chapter 4 diagrams, drawn in academic UML / DFD style.

Each public function ``draw_*(out_basename)`` builds a single diagram and saves
it as both ``<out_basename>.png`` and ``<out_basename>.svg`` next to this file.
"""
from __future__ import annotations

from pathlib import Path

from shapes import (
    actor,
    add_caption,
    aggregation_diamond,
    arrow,
    assoc_line,
    class_box,
    component_box,
    dependency_line,
    deploy_node,
    dfd_arrow,
    dfd_external,
    dfd_process_box,
    dfd_process_circle,
    dfd_store,
    diamond,
    end_node,
    fork_bar,
    filled_circle,
    inheritance_triangle,
    Lifeline,
    line,
    lollipop,
    new_canvas,
    oval,
    rect,
    rounded_rect,
    save,
    seq_activation,
    seq_message,
    seq_setup,
    socket,
    system_box,
    text,
    uc_assoc,
    uc_dependency,
    usecase,
    LINE,
    DFD_ORANGE_FILL,
    DFD_ORANGE_STROKE,
    ACTIVITY_FILL,
    ACTIVITY_STROKE,
)

HERE = Path(__file__).resolve().parent


def _out(basename: str) -> str:
    return str(HERE / basename)


# ===========================================================================
# 4.2  Use-Case Diagram
# ===========================================================================

def draw_use_case() -> None:
    fig, ax = new_canvas(22, 14.6)

    # Actors (left/right outside the system boundary)
    actor(ax, 1.4, 8.5, "End User", scale=1.2)
    actor(ax, 1.4, 1.6, "Admin", scale=1.2)
    actor(ax, 21.0, 11.0, "Backend\n(FastAPI)", scale=1.0, label_size=8)
    actor(ax, 21.0, 7.6, "Supabase\n(Auth/DB/Storage)", scale=1.0, label_size=8)
    actor(ax, 21.0, 4.4, "AssemblyAI", scale=1.0, label_size=8)
    actor(ax, 21.0, 1.0, "Blurring Engine\n(YOLO + BoT-SORT +\nCaffe + OpenFace)", scale=1.0, label_size=7)

    # System boundary - title placed at top-left so it doesn't collide with the
    # Compression / Blurring module titles which sit at the top of the right half.
    sys_x, sys_y, sys_w, sys_h = 3.4, 0.6, 16.6, 12.7
    rect(ax, sys_x, sys_y, sys_w, sys_h, fill="white", stroke=LINE, lw=1.4)
    text(ax, sys_x + 0.2, sys_y + sys_h - 0.25, "«module»", size=8, style="italic", ha="left", va="top")
    text(ax, sys_x + 0.2, sys_y + sys_h - 0.55, "NeuroClip System", size=10, weight="bold", ha="left", va="top")

    # Core use cases on the left half of the system box (UC9 moves into Blurring module)
    uc_data = [
        ("UC1", 5.6, 12.2, "Register / Login"),
        ("UC2", 5.6, 11.0, "Upload Video"),
        ("UC3", 5.6, 9.8, "Generate Transcript"),
        ("UC4", 5.6, 8.6, "Compute & Store\nEmbeddings"),
        ("UC5", 5.6, 7.4, "Search Clips by Query"),
        ("UC6", 5.6, 6.2, "Play Retrieved Clips"),
        ("UC7", 5.6, 5.0, "View History"),
        ("UC8", 5.6, 3.8, "Summarize Video"),
        ("UC11", 8.6, 12.2, "Manage Profile"),
        ("UC12", 8.6, 1.5, "Save Processing\nHistory"),
        ("UC13", 8.6, 2.7, "Manage Content / Logs"),
    ]
    coords: dict[str, tuple[float, float, float, float]] = {}
    for uid, cx, cy, label in uc_data:
        coords[uid] = usecase(ax, cx, cy, label, w=2.7, h=0.85, label_size=8)

    # Compression module: dashed nested boundary (centre column of the right half)
    cm_x, cm_y, cm_w, cm_h = 10.6, 1.0, 4.0, 11.6
    rect(ax, cm_x, cm_y, cm_w, cm_h, fill="white", stroke=LINE, lw=1.2, dashed=True)
    text(ax, cm_x + cm_w / 2, cm_y + cm_h - 0.18, "«module» Compression", size=9, weight="bold", style="italic", ha="center", va="top")
    cu_data = [
        ("UC10", cm_x + cm_w / 2, cm_y + cm_h - 0.95, "Compress Video"),
        ("UC14", cm_x + cm_w / 2, cm_y + cm_h - 2.95, "Select Source Video"),
        ("UC15", cm_x + cm_w / 2, cm_y + cm_h - 4.95, "Choose Profile"),
        ("UC16", cm_x + cm_w / 2, cm_y + cm_h - 6.95, "Run FFmpeg Compression"),
        ("UC17", cm_x + cm_w / 2, cm_y + cm_h - 8.95, "Preview / Download\nCompressed File"),
    ]
    for uid, cx, cy, label in cu_data:
        coords[uid] = usecase(ax, cx, cy, label, w=3.4, h=0.85, label_size=8)

    # Blurring module: dashed nested boundary (rightmost column inside the system)
    bm_x, bm_y, bm_w, bm_h = 15.0, 1.0, 4.6, 11.6
    rect(ax, bm_x, bm_y, bm_w, bm_h, fill="white", stroke=LINE, lw=1.2, dashed=True)
    text(ax, bm_x + bm_w / 2, bm_y + bm_h - 0.18, "«module» Blurring", size=9, weight="bold", style="italic", ha="center", va="top")
    bu_data = [
        ("UC9",  bm_x + bm_w / 2, bm_y + bm_h - 0.95, "Anonymize Video /\nBlur Targets"),
        ("UC18", bm_x + bm_w / 2, bm_y + bm_h - 2.50, "Upload Reference\nImages"),
        ("UC19", bm_x + bm_w / 2, bm_y + bm_h - 4.05, "Generate Master\nBiometric Signature"),
        ("UC20", bm_x + bm_w / 2, bm_y + bm_h - 5.55, "Detect & Track Persons\n(YOLO + BoT-SORT)"),
        ("UC21", bm_x + bm_w / 2, bm_y + bm_h - 7.05, "Re-Identify Faces\n(Caffe + OpenFace)"),
        ("UC22", bm_x + bm_w / 2, bm_y + bm_h - 8.55, "Apply Gaussian Blur\n(face / body fallback)"),
        ("UC23", bm_x + bm_w / 2, bm_y + bm_h - 10.05, "Save Anonymized Video"),
    ]
    for uid, cx, cy, label in bu_data:
        coords[uid] = usecase(ax, cx, cy, label, w=4.0, h=0.95, label_size=7)

    # === Actor-to-use-case associations ==================================
    # All cross-system links are routed with right-angle (L-shape) segments
    # so they never pass THROUGH the ovals.  Each supporting actor owns a
    # short vertical "trunk" on the right edge of the canvas (with a unique
    # x so trunks don't pile up) and a unique horizontal rail above the
    # system box that drops down into the target use case from outside.

    # --- End User -------------------------------------------------------
    user_anchor = (1.95, 9.6)
    # Direct straight associations to the left-column use cases.  These
    # never cross another oval because UC1..UC8 share the same column.
    for uid in ("UC1", "UC2", "UC5", "UC6", "UC7", "UC8"):
        x, y, w, h = coords[uid]
        uc_assoc(ax, user_anchor[0], user_anchor[1], x, y + h / 2)
    # User rail (inside the system box, just below the box title and above
    # every use-case oval) carries the long-reach links to UC11, UC10 and
    # UC9 so they never slice across the left-column ovals.
    user_rail_y = 12.95
    line(ax, user_anchor[0], user_anchor[1], user_anchor[0], user_rail_y, lw=1.0)
    user_rail_targets = [
        ("UC11", coords["UC11"][0] + coords["UC11"][2] / 2),
        ("UC10", coords["UC10"][0] + coords["UC10"][2] / 2),
        ("UC9",  coords["UC9"][0]  + coords["UC9"][2]  / 2),
    ]
    farthest_user_x = max(cx for _, cx in user_rail_targets)
    line(ax, user_anchor[0], user_rail_y, farthest_user_x, user_rail_y, lw=1.0)
    for uid, cx in user_rail_targets:
        x, y, w, h = coords[uid]
        line(ax, cx, user_rail_y, cx, y + h, lw=1.0)

    # --- Admin ----------------------------------------------------------
    admin_anchor = (1.95, 2.7)
    for uid in ("UC1", "UC13"):
        x, y, w, h = coords[uid]
        uc_assoc(ax, admin_anchor[0], admin_anchor[1], x, y + h / 2)

    # --- Helper: route a supporting actor through a top rail ------------
    def route_top(anchor, trunk_x, rail_y, drop_x, target_uid):
        x, y, w, h = coords[target_uid]
        cy = y + h / 2
        # actor arm -> trunk -> top rail -> drop -> right edge of UC
        line(ax, anchor[0], anchor[1], trunk_x, anchor[1], lw=1.0)
        line(ax, trunk_x, anchor[1], trunk_x, rail_y, lw=1.0)
        line(ax, trunk_x, rail_y, drop_x, rail_y, lw=1.0)
        line(ax, drop_x, rail_y, drop_x, cy, lw=1.0)
        line(ax, drop_x, cy, x + w, cy, lw=1.0)

    # --- Backend (FastAPI) ---------------------------------------------
    back_anchor = (20.45, 11.85)
    back_trunk_x = 20.20
    back_rail_y  = 13.55
    route_top(back_anchor, back_trunk_x, back_rail_y, 7.05, "UC4")
    # Backend also drives FFmpeg compression: drop into the gap between
    # the Compression and Blurring modules and enter UC16 from the right.
    x16, y16, w16, h16 = coords["UC16"]
    line(ax, back_trunk_x, back_rail_y, 14.75, back_rail_y, lw=1.0)
    line(ax, 14.75, back_rail_y, 14.75, y16 + h16 / 2, lw=1.0)
    line(ax, 14.75, y16 + h16 / 2, x16 + w16, y16 + h16 / 2, lw=1.0)

    # --- Supabase -------------------------------------------------------
    sup_anchor = (20.45, 8.5)
    sup_trunk_x = 20.30
    sup_rail_y  = 13.75
    # Up to UC1 (authentication)
    route_top(sup_anchor, sup_trunk_x, sup_rail_y, 7.15, "UC1")
    # Down to UC12 (save history) using a bottom rail below all ovals
    bottom_rail_y = 0.85
    x12, y12, w12, h12 = coords["UC12"]
    target_x_12 = x12 + w12 / 2
    line(ax, sup_anchor[0], sup_anchor[1], sup_trunk_x, sup_anchor[1], lw=1.0)
    line(ax, sup_trunk_x, sup_anchor[1], sup_trunk_x, bottom_rail_y, lw=1.0)
    line(ax, sup_trunk_x, bottom_rail_y, target_x_12, bottom_rail_y, lw=1.0)
    line(ax, target_x_12, bottom_rail_y, target_x_12, y12, lw=1.0)

    # --- AssemblyAI -----------------------------------------------------
    asm_anchor = (20.45, 5.3)
    asm_trunk_x = 20.40
    asm_rail_y  = 13.95
    route_top(asm_anchor, asm_trunk_x, asm_rail_y, 7.25, "UC3")

    # --- Blurring Engine ------------------------------------------------
    # UC20 and UC21 sit on the right edge of the Blurring module, so the
    # short diagonals from the actor never cross other ovals.
    blurring_anchor = (20.45, 1.85)
    for uid in ("UC20", "UC21"):
        x, y, w, h = coords[uid]
        uc_assoc(ax, blurring_anchor[0], blurring_anchor[1], x + w, y + h / 2)

    # Includes / extends helpers
    def edge(uid: str, side: str) -> tuple[float, float]:
        x, y, w, h = coords[uid]
        if side == "right":
            return x + w, y + h / 2
        if side == "left":
            return x, y + h / 2
        if side == "top":
            return x + w / 2, y + h
        return x + w / 2, y

    # Includes from upload pipeline
    uc_dependency(ax, *edge("UC2", "bottom"), *edge("UC3", "top"), "«include»")
    uc_dependency(ax, *edge("UC3", "bottom"), *edge("UC4", "top"), "«include»")
    uc_dependency(ax, *edge("UC8", "top"), *edge("UC4", "bottom"), "«include»")
    # Search saves history (cleaner: arrow goes right then down to UC12)
    uc_dependency(ax, *edge("UC5", "right"), *edge("UC12", "left"), "«include»")
    # Play extends Search
    uc_dependency(ax, *edge("UC6", "top"), *edge("UC5", "bottom"), "«extend»")

    # Compression module includes (vertical chain inside its dashed box)
    uc_dependency(ax, *edge("UC10", "bottom"), *edge("UC14", "top"), "«include»")
    uc_dependency(ax, *edge("UC14", "bottom"), *edge("UC15", "top"), "«include»")
    uc_dependency(ax, *edge("UC15", "bottom"), *edge("UC16", "top"), "«include»")
    uc_dependency(ax, *edge("UC16", "bottom"), *edge("UC17", "top"), "«include»")
    # Compression also writes history (left-going arrow)
    uc_dependency(ax, *edge("UC17", "left"), *edge("UC12", "right"), "«include»")

    # Blurring module includes (vertical chain inside its dashed box)
    uc_dependency(ax, *edge("UC9", "bottom"), *edge("UC18", "top"), "«include»")
    uc_dependency(ax, *edge("UC18", "bottom"), *edge("UC19", "top"), "«include»")
    uc_dependency(ax, *edge("UC19", "bottom"), *edge("UC20", "top"), "«include»")
    uc_dependency(ax, *edge("UC20", "bottom"), *edge("UC21", "top"), "«include»")
    uc_dependency(ax, *edge("UC21", "bottom"), *edge("UC22", "top"), "«include»")
    uc_dependency(ax, *edge("UC22", "bottom"), *edge("UC23", "top"), "«include»")
    # Blurring also writes history - route left then down via UC12
    uc_dependency(ax, *edge("UC23", "left"), *edge("UC12", "right"), "«include»")

    add_caption(ax, "Figure 4.2: NeuroClip Use Case Diagram (Compression and Blurring modeled as separate modules).")
    save(fig, _out("4.2_use_case"))


# ===========================================================================
# 4.4  Activity Diagram
# ===========================================================================

def draw_activity() -> None:
    fig, ax = new_canvas(22, 26)

    cx_main = 5.0
    cx_comp = 11.0
    cx_blur = 17.5

    def state(cx: float, cy: float, label: str, *, w: float = 3.4, h: float = 0.75, label_size: int = 9) -> None:
        rounded_rect(ax, cx - w / 2, cy - h / 2, w, h, fill=ACTIVITY_FILL, stroke=ACTIVITY_STROKE, lw=1.0, label=label, label_size=label_size)

    # ----- Phase 1: Authentication & Upload -----
    filled_circle(ax, cx_main, 25.5, 0.18)
    arrow(ax, cx_main, 25.30, cx_main, 24.85)
    state(cx_main, 24.55, "Register Account")
    arrow(ax, cx_main, 24.15, cx_main, 23.70)
    state(cx_main, 23.40, "Send Verification Email")
    arrow(ax, cx_main, 23.00, cx_main, 22.55)

    diamond(ax, cx_main, 22.15, 1.8, 1.0, label="Email\nVerified?")
    text(ax, cx_main - 1.05, 22.15, "[no]", size=8, ha="right", style="italic")
    text(ax, cx_main + 1.05, 21.55, "[yes]", size=8, ha="left", style="italic")
    arrow(ax, cx_main - 0.9, 22.15, cx_main - 2.10, 22.15)
    rounded_rect(ax, cx_main - 4.30, 21.85, 2.2, 0.6, fill=ACTIVITY_FILL, stroke=ACTIVITY_STROKE, lw=1.0, label="Resend Link", label_size=8)
    line(ax, cx_main - 3.20, 22.45, cx_main - 3.20, 23.40, lw=1.0)
    arrow(ax, cx_main - 3.20, 23.40, cx_main - 1.70, 23.40)

    arrow(ax, cx_main, 21.65, cx_main, 21.20)
    state(cx_main, 20.90, "Login")
    arrow(ax, cx_main, 20.50, cx_main, 20.05)
    state(cx_main, 19.75, "Onboarding")
    arrow(ax, cx_main, 19.35, cx_main, 18.90)
    state(cx_main, 18.60, "Open Dashboard")
    arrow(ax, cx_main, 18.20, cx_main, 17.75)
    state(cx_main, 17.45, "Upload Video / Paste URL")
    arrow(ax, cx_main, 17.05, cx_main, 16.60)

    diamond(ax, cx_main, 16.20, 1.8, 1.0, label="Valid File\n& Quota?")
    text(ax, cx_main + 1.05, 15.60, "[yes]", size=8, ha="left", style="italic")
    text(ax, cx_main - 1.05, 16.20, "[no]", size=8, ha="right", style="italic")
    arrow(ax, cx_main - 0.9, 16.20, cx_main - 2.30, 16.20)
    rounded_rect(ax, cx_main - 4.50, 15.90, 2.2, 0.6, fill=ACTIVITY_FILL, stroke=ACTIVITY_STROKE, lw=1.0, label="Show Error", label_size=8)
    line(ax, cx_main - 3.40, 16.50, cx_main - 3.40, 17.45, lw=1.0)
    arrow(ax, cx_main - 3.40, 17.45, cx_main - 1.70, 17.45)

    arrow(ax, cx_main, 15.70, cx_main, 15.25)
    state(cx_main, 14.95, "Save to Supabase Storage")
    arrow(ax, cx_main, 14.55, cx_main, 14.10)
    state(cx_main, 13.80, "Generate Transcript (AssemblyAI)")
    arrow(ax, cx_main, 13.40, cx_main, 12.95)

    diamond(ax, cx_main, 12.55, 1.8, 1.0, label="Transcript\nReady?")
    text(ax, cx_main - 1.05, 12.55, "[no]", size=8, ha="right", style="italic")
    text(ax, cx_main + 1.05, 11.95, "[yes]", size=8, ha="left", style="italic")
    arrow(ax, cx_main - 0.9, 12.55, cx_main - 2.30, 12.55)
    rounded_rect(ax, cx_main - 4.45, 12.25, 2.1, 0.6, fill=ACTIVITY_FILL, stroke=ACTIVITY_STROKE, lw=1.0, label="Re-queue Job", label_size=8)
    line(ax, cx_main - 3.40, 12.85, cx_main - 3.40, 13.80, lw=1.0)
    arrow(ax, cx_main - 3.40, 13.80, cx_main - 1.70, 13.80)

    arrow(ax, cx_main, 12.05, cx_main, 11.60)
    state(cx_main, 11.30, "Compute Embeddings")
    arrow(ax, cx_main, 10.90, cx_main, 10.45)
    state(cx_main, 10.15, "Store Vectors in pgvector")
    arrow(ax, cx_main, 9.75, cx_main, 9.30)
    state(cx_main, 9.00, "Video Ready Notification")
    arrow(ax, cx_main, 8.60, cx_main, 8.15)

    # ----- Choose Module Decision (3-way: search / compress / blur) -----
    diamond(ax, cx_main, 7.75, 2.4, 1.1, label="Choose Module")
    text(ax, cx_main - 1.25, 7.45, "[search]", size=8, ha="right", style="italic")
    text(ax, cx_main + 1.30, 8.10, "[compress]", size=8, ha="left", style="italic")
    text(ax, cx_main + 1.30, 7.45, "[blur]", size=8, ha="left", style="italic")

    # [blur] arrow: leave the right vertex, go up a touch, then far right and down
    line(ax, cx_main + 1.20, 7.75, cx_main + 1.20, 8.50, lw=1.0)
    line(ax, cx_main + 1.20, 8.50, cx_blur, 8.50, lw=1.0)
    arrow(ax, cx_blur, 8.50, cx_blur, 7.95)

    # ----- Search Branch (left column) -----
    arrow(ax, cx_main, 7.20, cx_main, 6.75)
    state(cx_main, 6.45, "Enter Search Query")
    arrow(ax, cx_main, 6.05, cx_main, 5.60)
    state(cx_main, 5.30, "Semantic Search + Rerank + Merge", w=4.4)
    arrow(ax, cx_main, 4.90, cx_main, 4.45)
    state(cx_main, 4.15, "Render Playable Clips")
    arrow(ax, cx_main, 3.75, cx_main, 3.30)
    state(cx_main, 3.00, "Save Search History")

    # ----- Compression Branch (right column) -----
    arrow(ax, cx_main + 1.2, 7.75, cx_comp - 1.7, 7.75)
    state(cx_comp, 7.75, "Select Source Video", w=3.4)
    arrow(ax, cx_comp, 7.35, cx_comp, 6.90)
    state(cx_comp, 6.60, "Choose Compression Profile", w=3.8)
    arrow(ax, cx_comp, 6.20, cx_comp, 5.75)
    state(cx_comp, 5.45, "Submit Compression Request", w=3.8)
    arrow(ax, cx_comp, 5.05, cx_comp, 4.60)
    state(cx_comp, 4.30, "FFmpeg Compresses Video", w=3.8)
    arrow(ax, cx_comp, 3.90, cx_comp, 3.45)

    diamond(ax, cx_comp, 3.05, 2.0, 1.0, label="Compression\nSuccessful?")
    text(ax, cx_comp + 1.05, 2.45, "[yes]", size=8, ha="left", style="italic")
    text(ax, cx_comp - 1.05, 3.05, "[no]", size=8, ha="right", style="italic")
    # No -> Retry (lower) -> Choose Profile
    arrow(ax, cx_comp - 1.0, 3.05, cx_comp - 2.5, 3.05)
    rounded_rect(ax, cx_comp - 4.5, 2.75, 2.0, 0.6, fill=ACTIVITY_FILL, stroke=ACTIVITY_STROKE, lw=1.0, label="Retry (lower)", label_size=8)
    line(ax, cx_comp - 3.5, 3.35, cx_comp - 3.5, 6.60, lw=1.0)
    arrow(ax, cx_comp - 3.5, 6.60, cx_comp - 1.9, 6.60)

    # Yes -> Store / Preview / History
    arrow(ax, cx_comp, 2.55, cx_comp, 2.15)
    state(cx_comp, 1.85, "Store Compressed File", w=3.4)
    arrow(ax, cx_comp, 1.45, cx_comp, 1.00)
    state(cx_comp, 0.70, "Preview / Download", w=3.4)
    arrow(ax, cx_comp, 0.30, cx_comp, -0.15)
    # Compression history flow merges into the search column for the join bar
    line(ax, cx_main, 2.65, cx_main, 1.30, lw=1.0)
    line(ax, cx_comp, 0.30, cx_comp, -0.10, lw=1.0)
    line(ax, cx_comp, -0.10, cx_main, -0.10, lw=1.0)
    state(cx_main, -0.40, "Save Compression History", w=3.8)

    # ----- Blurring Branch (right column, full Auto-Anonymizer pipeline) -----
    state(cx_blur, 7.65, "Upload Reference Images", w=3.6)
    arrow(ax, cx_blur, 7.27, cx_blur, 6.90)
    state(cx_blur, 6.60, "Generate Master Biometric\nSignature (OpenFace 128-d)", w=3.8, h=0.95)
    arrow(ax, cx_blur, 6.12, cx_blur, 5.75)

    # === per-frame loop wrapper (dashed) ===
    loop_x = cx_blur - 2.20
    loop_top = 5.65
    loop_bottom = -1.30
    rect(ax, loop_x, loop_bottom, 4.40, loop_top - loop_bottom, fill="white", stroke=LINE, lw=0.8, dashed=True)
    text(ax, loop_x + 0.12, loop_top - 0.15, "loop [for each frame]", size=7, style="italic", ha="left", va="top")

    # YOLO + BoT-SORT detect/track
    state(cx_blur, 5.10, "YOLO-World + BoT-SORT\nDetect & Track Persons", w=3.8, h=0.85)
    arrow(ax, cx_blur, 4.67, cx_blur, 4.30)

    # Engine throttle decision
    diamond(ax, cx_blur, 3.90, 2.0, 1.0, label="Engine Throttle\n(frame % 3 == 0)?")
    text(ax, cx_blur + 1.05, 3.30, "[yes]", size=8, ha="left", style="italic")
    text(ax, cx_blur - 1.05, 3.90, "[no]", size=8, ha="right", style="italic")
    # [no] -> skip face flow, go straight to "Apply Blur" via left rail
    arrow(ax, cx_blur - 1.0, 3.90, cx_blur - 2.10, 3.90)
    line(ax, cx_blur - 2.10, 3.90, cx_blur - 2.10, -0.55, lw=1.0)
    arrow(ax, cx_blur - 2.10, -0.55, cx_blur - 1.7, -0.55)

    # Yes -> Caffe + OpenFace
    arrow(ax, cx_blur, 3.40, cx_blur, 3.05)
    state(cx_blur, 2.75, "Caffe SSD Face Crop +\nOpenFace 128-d Embed", w=3.8, h=0.85)
    arrow(ax, cx_blur, 2.32, cx_blur, 1.95)

    # Cosine similarity decision
    diamond(ax, cx_blur, 1.55, 2.0, 0.95, label="Cosine \u2265 0.65?")
    text(ax, cx_blur + 1.05, 0.95, "[yes]", size=8, ha="left", style="italic")
    text(ax, cx_blur - 1.05, 1.55, "[no]", size=8, ha="right", style="italic")

    # [no] -> fail counter -> consecutive >=30 -> REJECTED_IDS (right rail back into loop)
    arrow(ax, cx_blur - 1.0, 1.55, cx_blur - 1.95, 1.55)
    rounded_rect(ax, cx_blur - 4.05, 1.25, 2.10, 0.6, fill=ACTIVITY_FILL, stroke=ACTIVITY_STROKE, lw=1.0, label="fail++ \u2192 if \u226530\nadd to REJECTED_IDS", label_size=7)
    line(ax, cx_blur - 3.00, 1.25, cx_blur - 3.00, -0.10, lw=1.0)
    arrow(ax, cx_blur - 3.00, -0.10, cx_blur - 1.7, -0.10)

    # [yes] -> ACTIVE_TARGET_IDS
    arrow(ax, cx_blur, 1.07, cx_blur, 0.70)
    state(cx_blur, 0.40, "Add ID to ACTIVE_TARGET_IDS", w=3.8)
    arrow(ax, cx_blur, 0.02, cx_blur, -0.30)

    # Apply Blur (face or body fallback)
    state(cx_blur, -0.55, "Apply 99\u00d799 Gaussian Blur\n(face or top 22% body fallback)", w=4.0, h=0.95)
    arrow(ax, cx_blur, -1.02, cx_blur, -1.30)
    state(cx_blur, -1.65, "Write Anonymized Frame", w=3.6)
    # === end of loop body ===
    arrow(ax, cx_blur, -2.05, cx_blur, -2.40)
    state(cx_blur, -2.70, "Save Anonymization History", w=3.8)

    # Route the blur branch's tail back to the join bar (left along bottom)
    line(ax, cx_blur, -3.07, cx_blur, -3.50, lw=1.0)
    line(ax, cx_blur, -3.50, (cx_main + cx_comp) / 2, -3.50, lw=1.0)

    # Join bar - now spans search/compress/blur
    join_cx = (cx_main + cx_comp) / 2
    fork_bar(ax, join_cx, -3.85, w=(cx_blur - cx_main) + 2.4)
    line(ax, cx_main, -0.78, cx_main, -3.81, lw=1.0)
    line(ax, join_cx, -3.50, join_cx, -3.81, lw=1.0)

    arrow(ax, join_cx, -3.89, join_cx, -4.30)
    end_node(ax, join_cx, -4.60)

    # Adjust limits so nothing is clipped
    ax.set_ylim(-5.4, 26)
    ax.set_xlim(0, 22)

    add_caption(ax, "Figure 4.4: NeuroClip Activity Diagram (Search, Compression and Blurring branches join at one end node).")
    save(fig, _out("4.4_activity"))


# ===========================================================================
# 4.5.1  DFD Level 0 (Context)
# ===========================================================================

def draw_dfd_level0() -> None:
    fig, ax = new_canvas(14, 8)

    # External entities
    dfd_external(ax, 0.6, 5.6, 2.0, 0.9, "End User")
    dfd_external(ax, 0.6, 1.6, 2.0, 0.9, "Admin")
    dfd_external(ax, 11.4, 5.6, 2.0, 0.9, "AssemblyAI")
    dfd_external(ax, 11.4, 1.6, 2.0, 0.9, "Supabase")

    # Process circle (NeuroClip System)
    cx, cy, r = 7.0, 4.0, 1.45
    dfd_process_circle(ax, cx, cy, r, "0.0", "NeuroClip System")

    # Flows
    # End User -> NC and back
    dfd_arrow(ax, 2.6, 6.0, cx - 1.05, cy + 0.95, "Video upload, query,\ncompression settings,\nreference images,\nblur threshold,\nprofile updates", label_offset=(-0.2, 0.75))
    dfd_arrow(ax, cx - 1.05, cy + 0.55, 2.6, 5.65, "Clips, compressed videos,\nanonymized videos,\ntranscripts, history,\nnotifications", label_offset=(0.0, -0.65))

    # Admin
    dfd_arrow(ax, 2.6, 2.0, cx - 1.05, cy - 0.95, "Policies, moderation\nactions", label_offset=(-0.2, -0.45))
    dfd_arrow(ax, cx - 1.05, cy - 0.55, 2.6, 1.85, "Logs, analytics,\ncontent reports", label_offset=(0.0, 0.40))

    # AssemblyAI
    dfd_arrow(ax, cx + 1.05, cy + 0.95, 11.4, 6.0, "Audio stream", label_offset=(0.2, 0.45))
    dfd_arrow(ax, 11.4, 5.7, cx + 1.05, cy + 0.55, "Transcript JSON", label_offset=(0.1, -0.35))

    # Supabase
    dfd_arrow(ax, cx + 1.05, cy - 0.95, 11.4, 2.0, "Auth, source files,\ncompressed outputs,\nvectors", label_offset=(0.2, -0.55))
    dfd_arrow(ax, 11.4, 1.85, cx + 1.05, cy - 0.55, "Sessions, signed URLs,\nrows, stored media", label_offset=(0.0, 0.45))

    add_caption(ax, "Figure 4.5.1: DFD Level 0 - Context Diagram (covers Search, Compression and Blurring).")
    save(fig, _out("4.5.1_dfd_level0"))


# ===========================================================================
# 4.5.2  DFD Level 1 (Major Processes)
# ===========================================================================

def draw_dfd_level1() -> None:
    fig, ax = new_canvas(22, 15)

    # External entities along the borders
    dfd_external(ax, 0.4, 13.4, 2.2, 0.9, "End User")
    dfd_external(ax, 0.4, 0.6, 2.2, 0.9, "Admin")
    dfd_external(ax, 19.4, 13.4, 2.2, 0.9, "AssemblyAI")

    # Process circles - 3 top, 3 mid (P4, P6, P7), 1 bottom (P5)
    r = 0.95
    procs = [
        ("P1", 5.0, 13.5, "Auth &\nProfile"),
        ("P2", 9.5, 13.5, "Video\nIngestion"),
        ("P3", 14.0, 13.5, "Embedding\nComputation"),
        ("P4", 5.0, 6.5, "Clip Search &\nAssembly"),
        ("P6", 12.0, 6.5, "Compression\nModule"),
        ("P7", 18.0, 6.5, "Blurring\nModule"),
        ("P5", 9.5, 2.2, "Admin /\nContent Ops"),
    ]
    coords: dict[str, tuple[float, float, float]] = {}
    for pid, cx, cy, name in procs:
        dfd_process_circle(ax, cx, cy, r, pid, name)
        coords[pid] = (cx, cy, r)

    # Data stores - top row (D1-D4) and bottom row (D5-D11)
    stores = [
        ("D1", 2.5, 10.6, 2.6, 0.6, "Profiles"),
        ("D2", 7.5, 10.6, 2.6, 0.6, "Videos"),
        ("D3", 11.5, 10.6, 2.8, 0.6, "Transcripts"),
        ("D4", 15.5, 10.6, 3.4, 0.6, "Embeddings (pgvector)"),
        ("D7", 0.4, 4.4, 2.6, 0.6, "Knowledge / Vectors"),
        ("D5", 4.0, 4.4, 2.4, 0.6, "Clip Assets"),
        ("D6", 7.0, 4.4, 2.8, 0.6, "Processing History"),
        ("D8", 10.6, 4.4, 2.8, 0.6, "Compressed Videos"),
        ("D9", 14.0, 4.4, 2.6, 0.6, "Reference Images"),
        ("D10", 17.2, 4.4, 2.6, 0.6, "Master Signatures"),
        ("D11", 20.0, 4.4, 1.7, 0.6, "Anonymized"),
    ]
    s_coords: dict[str, tuple[float, float, float, float]] = {}
    for sid, x, y, w, h, name in stores:
        dfd_store(ax, x, y, w, h, sid, name)
        s_coords[sid] = (x, y, w, h)

    # ---- Flows ----
    # User <-> P1 (credentials / session)
    dfd_arrow(ax, 2.6, 14.0, 4.05, 13.65, "credentials", label_offset=(0.0, 0.30))
    dfd_arrow(ax, 4.05, 13.35, 2.6, 13.55, "session", label_offset=(0.0, -0.30))
    # P1 -> D1
    dfd_arrow(ax, 4.7, 12.6, 3.8, 11.2, "")

    # User -> P2 (video / URL)
    dfd_arrow(ax, 2.6, 13.7, 8.55, 13.5, "video / URL", label_offset=(0.0, 0.4))
    # P2 -> D2
    dfd_arrow(ax, 9.5, 12.55, 8.8, 11.2, "file", label_offset=(-0.3, 0.3))
    # P2 <-> AssemblyAI
    dfd_arrow(ax, 10.45, 13.65, 19.4, 13.95, "audio")
    dfd_arrow(ax, 19.4, 13.65, 10.45, 13.35, "transcript", label_offset=(-2.0, -0.25))
    # P2 -> D3
    dfd_arrow(ax, 9.9, 12.65, 12.5, 11.2, "segments", label_offset=(0.6, 0.2))
    # P2 -> P3 (job_id)
    dfd_arrow(ax, 10.45, 13.5, 13.05, 13.5, "job_id")
    # P3 -> D3 / D4
    dfd_arrow(ax, 13.6, 12.65, 13.0, 11.2, "")
    dfd_arrow(ax, 14.0, 12.55, 17.0, 11.2, "vectors", label_offset=(0.6, 0.2))

    # User -> P4 (query) and P4 -> User (clips) along the left edge
    dfd_arrow(ax, 1.5, 13.4, 4.6, 7.4, "query", label_offset=(-0.6, 1.8))
    dfd_arrow(ax, 4.6, 7.0, 1.5, 13.4, "clips", label_offset=(-0.7, -1.8))
    # P4 -> D3 / D4 / D5 / D6
    dfd_arrow(ax, 5.4, 7.35, 12.0, 10.6, "read", label_offset=(0.5, 0.4))
    dfd_arrow(ax, 5.7, 7.20, 16.5, 10.6, "search", label_offset=(0.6, 0.4))
    dfd_arrow(ax, 5.0, 5.65, 5.0, 5.0, "write")
    dfd_arrow(ax, 5.7, 5.80, 8.2, 5.0, "log", label_offset=(0.4, 0.4))

    # User -> P6 (compression settings) - route along the top, then down to P6
    line(ax, 2.6, 14.05, 12.0, 14.05, lw=1.0)
    line(ax, 12.0, 14.05, 12.0, 7.50, lw=1.0)
    arrow(ax, 12.0, 7.55, 12.0, 7.50, label="compression settings", label_offset=(-3.4, 0.20))
    # Compressed file URL back to user - route along the BOTTOM rail (below the stores), out of P4's way
    line(ax, 11.05, 6.50, 11.05, 6.10, lw=1.0)
    line(ax, 11.05, 6.10, 1.50, 6.10, lw=1.0)
    arrow(ax, 1.50, 6.10, 1.50, 13.40, label="compressed file URL", label_offset=(0.20, -3.5))
    # P6 <-> D2 (read source)
    dfd_arrow(ax, 11.6, 7.35, 9.0, 10.6, "read source", label_offset=(-0.5, 0.4))
    # P6 -> D8
    dfd_arrow(ax, 12.0, 5.55, 12.0, 5.0, "ffmpeg output", label_offset=(0.7, 0.2))
    # P6 -> D6
    dfd_arrow(ax, 11.3, 5.80, 9.5, 5.0, "compression event", label_offset=(-0.3, 0.35))

    # ---- Blurring Module (P7) flows ----
    # User -> P7 (reference images + threshold) - route along the very top, then down to P7
    line(ax, 2.6, 14.50, 18.0, 14.50, lw=1.0)
    line(ax, 18.0, 14.50, 18.0, 7.50, lw=1.0)
    arrow(ax, 18.0, 7.55, 18.0, 7.50, label="reference images + threshold", label_offset=(-2.0, 0.20))
    # P7 -> User (anonymized video URL) - route up the right side then along the very top back to user
    line(ax, 18.95, 6.50, 21.5, 6.50, lw=1.0)
    line(ax, 21.5, 6.50, 21.5, 14.85, lw=1.0)
    line(ax, 21.5, 14.85, 1.50, 14.85, lw=1.0)
    arrow(ax, 1.50, 14.85, 1.50, 13.85, label="anonymized video URL", label_offset=(7.0, 0.20))
    # P7 -> D2 (read source) - route up directly between D3 and D4 to avoid D2 overlap with P6 line
    line(ax, 17.5, 7.35, 16.0, 10.20, lw=1.0)
    arrow(ax, 16.0, 10.20, 16.0, 10.20, label="read source", label_offset=(0.4, -0.5))
    # P7 -> D9 (write refs)
    dfd_arrow(ax, 17.4, 5.65, 15.2, 5.0, "write refs", label_offset=(-0.4, 0.25))
    # P7 -> D10 (master sig)
    dfd_arrow(ax, 18.0, 5.55, 18.4, 5.0, "master sig (vec)", label_offset=(0.6, 0.20))
    # P7 -> D11 (anonymized output)
    dfd_arrow(ax, 18.6, 5.65, 20.4, 5.0, "ffmpeg + cv2", label_offset=(0.5, 0.20))
    # P7 -> D6 (anonymization event) - short L-route via the bottom rail to D6
    line(ax, 17.05, 6.50, 17.05, 5.30, lw=1.0)
    line(ax, 17.05, 5.30, 9.50, 5.30, lw=1.0)
    arrow(ax, 9.50, 5.30, 9.50, 5.05, label="anonymization event", label_offset=(-2.5, 0.20))

    # Admin -> P5 -> D7 / D6
    dfd_arrow(ax, 2.6, 1.05, 8.55, 2.2, "ingest / policies")
    dfd_arrow(ax, 8.8, 2.9, 1.7, 4.4, "")
    dfd_arrow(ax, 10.0, 3.05, 8.4, 4.4, "review")

    add_caption(ax, "Figure 4.5.2: DFD Level 1 with the Compression (P6) and Blurring (P7) modules.")
    save(fig, _out("4.5.2_dfd_level1"))


# ===========================================================================
# 4.5.3  DFD Level 2 (Search + Compression module explosions)
# ===========================================================================

def draw_dfd_level2() -> None:
    fig, ax = new_canvas(28, 16)

    dfd_external(ax, 13.4, 14.6, 2.2, 0.9, "End User")

    # Three dashed module columns, side-by-side with breathing room between P6 and P7
    rect(ax, 3.4, 1.2, 5.6, 12.6, fill="white", stroke=LINE, lw=1.2, dashed=True)
    text(ax, 8.85, 13.65, "P4: Clip Search & Assembly", size=10, weight="bold", ha="right", va="top")

    rect(ax, 9.4, 1.2, 5.6, 12.6, fill="white", stroke=LINE, lw=1.2, dashed=True)
    text(ax, 14.85, 13.65, "P6: Video Compression Module", size=10, weight="bold", ha="right", va="top")

    rect(ax, 17.4, 1.2, 5.6, 12.6, fill="white", stroke=LINE, lw=1.2, dashed=True)
    text(ax, 22.85, 13.65, "P7: Blurring Module", size=10, weight="bold", ha="right", va="top")

    # P4 processes
    cx_p4 = 6.2
    p4 = [
        ("P4.1", 12.6, "Intent Classification"),
        ("P4.2", 10.8, "Retrieval (semantic + keyword)"),
        ("P4.3", 9.0, "Reranking & Neighbor Merging"),
        ("P4.4", 7.2, "Clip Boundary Assembly"),
        ("P4.5", 5.4, "History Logging"),
    ]
    for pid, cy, name in p4:
        dfd_process_box(ax, cx_p4 - 1.85, cy - 0.55, 3.7, 1.1, pid, name)

    # P6 processes
    cx_p6 = 12.2
    p6 = [
        ("P6.1", 12.6, "Validate Source Video"),
        ("P6.2", 10.8, "Select Compression Profile"),
        ("P6.3", 9.0, "Run FFmpeg Compression"),
        ("P6.4", 7.2, "Store Output & Signed URL"),
        ("P6.5", 5.4, "Compression History Logging"),
    ]
    for pid, cy, name in p6:
        dfd_process_box(ax, cx_p6 - 1.85, cy - 0.55, 3.7, 1.1, pid, name)

    # P7 processes (NEW)
    cx_p7 = 20.2
    p7 = [
        ("P7.1", 12.6, "Validate Source +\nReferences"),
        ("P7.2", 11.1, "Generate Master\nBiometric Signature"),
        ("P7.3", 9.6, "YOLO-World Detect +\nBoT-SORT Track"),
        ("P7.4", 8.1, "Engine Throttle\n(frame % 3 == 0)"),
        ("P7.5", 6.6, "Caffe Face Crop +\nOpenFace Embed"),
        ("P7.6", 5.1, "Cosine Similarity\nvs Master"),
        ("P7.7", 3.6, "Apply Gaussian Blur\n/ Body Fallback"),
        ("P7.8", 2.1, "Write Anonymized Video\n+ History"),
    ]
    for pid, cy, name in p7:
        dfd_process_box(ax, cx_p7 - 1.85, cy - 0.65, 3.7, 1.3, pid, name)

    # Stores - left wall (D2-D5, D7), bottom (D6), right wall for P6/P7 outputs
    dfd_store(ax, 0.4, 11.4, 2.6, 0.6, "D2", "Videos")
    dfd_store(ax, 0.4, 10.2, 2.6, 0.6, "D3", "Transcripts")
    dfd_store(ax, 0.4, 9.0, 2.6, 0.6, "D4", "Embeddings")
    dfd_store(ax, 0.4, 7.8, 2.6, 0.6, "D5", "Clip Assets")
    dfd_store(ax, 11.9, 0.3, 2.6, 0.6, "D6", "Processing History")
    # D8 sits in the dedicated gap between P6 and P7 columns
    dfd_store(ax, 15.2, 6.6, 2.0, 0.6, "D8", "Compressed")
    # New stores for blurring on the far right wall
    dfd_store(ax, 23.4, 11.4, 4.0, 0.6, "D9", "Reference Images")
    dfd_store(ax, 23.4, 10.2, 4.0, 0.6, "D10", "Master Signatures (vector)")
    dfd_store(ax, 23.4, 9.0, 4.0, 0.6, "D11", "Anonymized Videos")

    # ---- P4 internal flow ----
    dfd_arrow(ax, 13.4, 14.6, 7.0, 13.2, "query", label_offset=(-0.4, 0.3))
    dfd_arrow(ax, cx_p4, 12.05, cx_p4, 11.35, "normalized intent")
    dfd_arrow(ax, cx_p4 - 1.85, 10.8, 3.0, 10.4, "read")
    dfd_arrow(ax, cx_p4 - 1.85, 10.5, 3.0, 9.3, "search vectors", label_offset=(-0.5, 0.3))
    dfd_arrow(ax, cx_p4, 10.25, cx_p4, 9.55, "candidate spans")
    dfd_arrow(ax, cx_p4, 8.45, cx_p4, 7.75, "merged spans")
    dfd_arrow(ax, cx_p4 - 1.85, 7.2, 3.0, 8.1, "write clips", label_offset=(-0.3, 0.4))
    dfd_arrow(ax, cx_p4, 6.65, cx_p4, 5.95, "record")
    dfd_arrow(ax, cx_p4, 4.85, cx_p4, 0.9, "log")
    # Clip URLs back to the user (route up the left side, outside the dashed P4 box)
    line(ax, cx_p4 - 1.85, 7.2, 3.0, 7.2, lw=1.0)
    line(ax, 3.0, 7.2, 3.0, 13.5, lw=1.0)
    arrow(ax, 3.0, 13.5, 13.4, 14.6, label="clip URLs", label_offset=(0.0, 0.2))

    # ---- P6 internal flow ----
    dfd_arrow(ax, 13.6, 14.6, 11.4, 13.2, "video_id + quality settings", label_offset=(-1.6, 0.4))
    # P6.1 reads from D2 - route ABOVE the P4 column to avoid crossing process boxes
    line(ax, cx_p6 - 1.85, 13.30, 3.0, 13.30, lw=1.0)
    line(ax, 3.0, 13.30, 3.0, 12.05, lw=1.0)
    arrow(ax, 3.0, 12.05, 3.0, 12.00)
    text(ax, 6.5, 13.45, "validate against D2", size=8, ha="center", style="italic")
    dfd_arrow(ax, cx_p6, 12.05, cx_p6, 11.35, "valid source")
    dfd_arrow(ax, cx_p6, 10.25, cx_p6, 9.55, "bitrate / resolution / codec")
    dfd_arrow(ax, cx_p6, 8.45, cx_p6, 7.75, "compressed file")
    dfd_arrow(ax, cx_p6 + 1.85, 7.2, 15.2, 6.9, "store output", label_offset=(0.4, 0.30))
    dfd_arrow(ax, cx_p6, 6.65, cx_p6, 5.95, "compression record")
    dfd_arrow(ax, cx_p6, 4.85, cx_p6, 0.9, "log")
    # Download/preview URL back to user (route through the gap between P6 and P7 columns)
    line(ax, cx_p6 + 1.85, 7.2, 15.2, 7.2, lw=1.0)
    line(ax, 15.2, 7.2, 15.2, 14.20, lw=1.0)
    line(ax, 15.2, 14.20, 13.6, 14.20, lw=1.0)
    arrow(ax, 13.6, 14.20, 13.6, 14.60, label="download / preview URL", label_offset=(-2.5, -0.20))

    # ---- P7 internal flow (Blurring) ----
    # User -> P7 (request anonymization)
    dfd_arrow(ax, 15.6, 14.6, 19.4, 13.2, "video_id + reference images + threshold", label_offset=(2.0, 0.45))
    # P7.1 reads source from D2 - route ABOVE the P4 + P6 columns
    line(ax, cx_p7 - 1.85, 13.10, 2.6, 13.10, lw=1.0)
    text(ax, 11.0, 13.25, "validate source against D2", size=8, ha="center", style="italic")
    line(ax, 2.6, 13.10, 2.6, 12.10, lw=1.0)
    arrow(ax, 2.6, 12.10, 2.6, 12.05)
    # Vertical chain inside P7
    dfd_arrow(ax, cx_p7, 11.95, cx_p7, 11.75, "valid src + refs")
    dfd_arrow(ax, cx_p7, 10.45, cx_p7, 10.25, "128-d master vec")
    dfd_arrow(ax, cx_p7, 8.95, cx_p7, 8.75, "frame + tracker IDs")
    dfd_arrow(ax, cx_p7, 7.45, cx_p7, 7.25, "[yes] every 3rd frame")
    dfd_arrow(ax, cx_p7, 5.95, cx_p7, 5.75, "candidate embedding")
    dfd_arrow(ax, cx_p7, 4.45, cx_p7, 4.25, "ACTIVE_TARGET_IDS")
    dfd_arrow(ax, cx_p7, 2.95, cx_p7, 2.75, "blurred frame")
    # P7.2 writes refs to D9, master sig to D10
    dfd_arrow(ax, cx_p7 + 1.85, 11.10, 23.4, 11.65, "write refs", label_offset=(0.4, 0.30))
    dfd_arrow(ax, cx_p7 + 1.85, 10.85, 23.4, 10.45, "store vec", label_offset=(0.4, -0.30))
    # P7.8 writes anonymized output to D11
    dfd_arrow(ax, cx_p7 + 1.85, 2.10, 23.4, 9.0, "ffmpeg + cv2 output", label_offset=(0.7, 1.2))
    # P7.8 logs to D6 - route along the bottom rail
    line(ax, cx_p7, 1.45, cx_p7, 0.6, lw=1.0)
    line(ax, cx_p7, 0.6, 14.5, 0.6, lw=1.0)
    arrow(ax, 14.5, 0.6, 14.5, 0.6, label="log", label_offset=(-2.6, 0.20))
    # Anonymized video URL back to user (route up the far right wall)
    line(ax, cx_p7 + 1.85, 2.10, 27.4, 2.10, lw=1.0)
    line(ax, 27.4, 2.10, 27.4, 14.50, lw=1.0)
    line(ax, 27.4, 14.50, 15.6, 14.50, lw=1.0)
    arrow(ax, 15.6, 14.50, 15.6, 14.60, label="anonymized video URL", label_offset=(-4.5, 0.20))

    add_caption(ax, "Figure 4.5.3: DFD Level 2 - explosion of P4 (Search), P6 (Compression) and P7 (Blurring) modules.")
    save(fig, _out("4.5.3_dfd_level2_search"))


# ===========================================================================
# 4.6  System Sequence Diagram (Search + Compression flows)
# ===========================================================================

def draw_system_sequence() -> None:
    fig, ax = new_canvas(11, 14)
    top, bottom = 13.4, 0.3
    lifelines = [Lifeline(1.5, "End User", is_actor=True), Lifeline(7.0, ":NeuroClip System")]
    seq_setup(ax, lifelines, top, bottom)

    sys = 7.0
    user = 1.5

    # ----- Search flow -----
    seq_message(ax, user, sys, 12.7, "enterQuery(job_id, queryText, params)")
    seq_message(ax, sys, user, 12.3, "ack (loading state)", dashed=True)
    seq_activation(ax, sys, 12.7, 11.6)
    seq_message(ax, sys, sys, 11.7, "process query (search + assemble)", self_call=True)
    seq_message(ax, sys, user, 11.0, "clipResults[ {start, end, url, score} ]", dashed=True)

    seq_message(ax, user, sys, 10.4, "playClip(url)")
    seq_message(ax, sys, user, 10.0, "video stream", dashed=True)

    seq_message(ax, user, sys, 9.3, "saveToHistory()")
    seq_message(ax, sys, user, 8.9, "historyEntryCreated(job_id)", dashed=True)

    # Compression Flow separator
    line(ax, 0.6, 8.4, 10.4, 8.4, dashed=True, lw=0.8)
    text(ax, 5.5, 8.5, "Compression Flow", size=9, style="italic")

    seq_message(ax, user, sys, 8.0, "requestCompression(video_id, quality, resolution, bitrate)")
    seq_message(ax, sys, user, 7.6, "ack (compression queued)", dashed=True)
    seq_activation(ax, sys, 8.0, 5.7)
    seq_message(ax, sys, sys, 7.0, "validate source + run FFmpeg compression", self_call=True)
    seq_message(ax, sys, sys, 6.3, "store compressed file + log history", self_call=True)
    seq_message(ax, sys, user, 5.6, "compressedVideoUrl(size_before, size_after, url)", dashed=True)

    # Blurring Flow separator (NEW)
    line(ax, 0.6, 5.0, 10.4, 5.0, dashed=True, lw=0.8)
    text(ax, 5.5, 5.10, "Blurring Flow", size=9, style="italic")

    seq_message(ax, user, sys, 4.6, "requestAnonymization(video_id, reference_images[], threshold=0.65)")
    seq_message(ax, sys, user, 4.2, "ack (anonymization queued)", dashed=True)
    seq_activation(ax, sys, 4.6, 1.0)
    seq_message(ax, sys, sys, 3.6, "generate master biometric signature from references", self_call=True)
    seq_message(ax, sys, sys, 2.9, "loop frames: YOLO+BoT-SORT detect, throttle %3,\nCaffe+OpenFace re-identify, apply blur", self_call=True)
    seq_message(ax, sys, sys, 2.0, "store anonymized video + log history", self_call=True)
    seq_message(ax, sys, user, 1.2, "anonymizedVideoUrl(target_ids_blurred, total_frames, processing_time)", dashed=True)

    add_caption(ax, "Figure 4.6: System Sequence Diagram - Clip Search, Compression and Blurring scenarios.")
    save(fig, _out("4.6_system_sequence"))


# ===========================================================================
# 4.7.1  Sequence - Email Verification + opening compression page
# ===========================================================================

def draw_sequence_auth() -> None:
    fig, ax = new_canvas(13, 11)
    top, bottom = 10.4, 0.3
    lifelines = [
        Lifeline(1.2, "End User", is_actor=True),
        Lifeline(4.0, ":Frontend"),
        Lifeline(6.5, ":Supabase Auth"),
        Lifeline(9.0, ":Email Service"),
        Lifeline(11.6, ":Supabase DB"),
    ]
    seq_setup(ax, lifelines, top, bottom)

    seq_message(ax, 1.2, 4.0, 9.7, "1: signUp(email, password)")
    seq_message(ax, 4.0, 6.5, 9.3, "2: auth.signUp()")
    seq_message(ax, 6.5, 9.0, 8.9, "3: send verification link")
    seq_message(ax, 9.0, 1.2, 8.5, "4: email with link", dashed=True)

    seq_message(ax, 1.2, 9.0, 8.0, "5: click verification link")
    seq_message(ax, 9.0, 6.5, 7.6, "6: verify(token)")
    seq_message(ax, 6.5, 4.0, 7.2, "7: redirect to PROD_URL/callback", dashed=True)
    seq_message(ax, 4.0, 6.5, 6.7, "8: exchangeCodeForSession()")
    seq_message(ax, 6.5, 4.0, 6.3, "9: session + user", dashed=True)
    seq_message(ax, 4.0, 11.6, 5.9, "10: upsert profile row")
    seq_message(ax, 11.6, 4.0, 5.5, "11: ok", dashed=True)
    seq_message(ax, 4.0, 1.2, 5.0, "12: route to /dashboard", dashed=True)

    # Open Compression Module separator
    line(ax, 0.6, 4.7, 12.4, 4.7, dashed=True, lw=0.8)
    text(ax, 6.5, 4.85, "Open Compression Module", size=9, style="italic")

    seq_message(ax, 1.2, 4.0, 4.4, "13: open Compression module")
    seq_message(ax, 4.0, 6.5, 4.0, "14: getSession()")
    seq_message(ax, 6.5, 4.0, 3.6, "15: valid session", dashed=True)
    seq_message(ax, 4.0, 1.2, 3.1, "16: render /compression page", dashed=True)

    # Open Blurring Module separator (NEW)
    line(ax, 0.6, 2.7, 12.4, 2.7, dashed=True, lw=0.8)
    text(ax, 6.5, 2.85, "Open Blurring Module", size=9, style="italic")

    seq_message(ax, 1.2, 4.0, 2.4, "17: open Blurring module")
    seq_message(ax, 4.0, 6.5, 2.0, "18: getSession()")
    seq_message(ax, 6.5, 4.0, 1.6, "19: valid session", dashed=True)
    seq_message(ax, 4.0, 1.2, 1.1, "20: render /blurring page", dashed=True)

    add_caption(ax, "Figure 4.7.1: Sequence Diagram - Email Verification and Opening Compression / Blurring Modules.")
    save(fig, _out("4.7.1_sequence_email_verify"))


# ===========================================================================
# 4.7.2  Sequence - Clip search + Compression
# ===========================================================================

def draw_sequence_search_compress() -> None:
    fig, ax = new_canvas(30, 26)
    top, bottom = 25.0, 0.4
    lifelines = [
        Lifeline(1.2, "End User", is_actor=True),
        Lifeline(3.8, ":Frontend"),
        Lifeline(6.4, ":FastAPI"),
        Lifeline(9.0, ":SearchService"),
        Lifeline(11.6, ":EmbeddingService"),
        Lifeline(14.2, ":VectorStore"),
        Lifeline(16.8, ":StorageService"),
        Lifeline(19.4, ":CompressionService"),
        Lifeline(22.0, ":BlurringService"),
        Lifeline(24.6, ":BiometricService"),
        Lifeline(27.0, ":FrameDetector"),
        Lifeline(29.4, ":HistoryService"),
    ]
    seq_setup(ax, lifelines, top, bottom, header_h=0.65)

    # ----- Clip Search Flow -----
    line(ax, 0.5, 24.30, 29.9, 24.30, dashed=True, lw=0.8)
    text(ax, 15.0, 24.45, "Clip Search Flow", size=11, style="italic", weight="bold")

    seq_message(ax, 1.2, 3.8, 23.85, "1: submit(query, top_k)")
    seq_message(ax, 3.8, 6.4, 23.40, "2: POST /clips/search-db")
    seq_message(ax, 6.4, 9.0, 22.95, "3: process_query(job_id, query, params)")
    seq_message(ax, 9.0, 11.6, 22.50, "4: encode(query)")
    seq_message(ax, 11.6, 9.0, 22.05, "5: query_vector", dashed=True)
    seq_message(ax, 9.0, 14.2, 21.60, "6: search(query_vector, top_k)")
    seq_message(ax, 14.2, 9.0, 21.15, "7: candidate_rows[]", dashed=True)
    seq_message(ax, 9.0, 9.0, 20.60, "8: rerank() + merge_neighbors()", self_call=True)
    seq_message(ax, 9.0, 16.8, 19.85, "9: get_clip_url(video_id, start, end)")
    seq_message(ax, 16.8, 9.0, 19.40, "10: signed_urls[]", dashed=True)
    seq_message(ax, 9.0, 29.4, 18.95, "11: record_event(...)")
    seq_message(ax, 29.4, 9.0, 18.50, "12: history_id", dashed=True)
    seq_message(ax, 9.0, 6.4, 18.05, "13: ClipResult[]", dashed=True)
    seq_message(ax, 6.4, 3.8, 17.60, "14: 200 JSON {clips, history_id}", dashed=True)
    seq_message(ax, 3.8, 1.2, 17.15, "15: render player + clip list", dashed=True)

    # ----- Compression Flow -----
    line(ax, 0.5, 16.50, 29.9, 16.50, dashed=True, lw=0.8)
    text(ax, 15.0, 16.65, "Compression Flow", size=11, style="italic", weight="bold")

    seq_message(ax, 1.2, 3.8, 16.05, "16: requestCompression(video_id, preset)")
    seq_message(ax, 3.8, 6.4, 15.60, "17: POST /compress-video")
    seq_message(ax, 6.4, 19.4, 15.15, "18: compress_video(video_id, preset)")
    seq_message(ax, 19.4, 16.8, 14.70, "19: fetch source video")
    seq_message(ax, 16.8, 19.4, 14.25, "20: source path / signed URL", dashed=True)
    seq_message(ax, 19.4, 19.4, 13.65, "21: ffmpeg -i source -b:v / -crf preset output", self_call=True)
    seq_message(ax, 19.4, 16.8, 12.95, "22: upload compressed output")
    seq_message(ax, 16.8, 19.4, 12.50, "23: compressed signed URL", dashed=True)
    seq_message(ax, 19.4, 29.4, 12.05, "24: record_event(module='compression', status='ok')")
    seq_message(ax, 29.4, 19.4, 11.60, "25: compression_history_id", dashed=True)
    seq_message(ax, 19.4, 6.4, 11.15, "26: CompressionResult", dashed=True)
    seq_message(ax, 6.4, 1.2, 10.70, "27: 200 JSON {compressed_url, metrics}", dashed=True)

    # ----- Blurring Flow -----
    line(ax, 0.5, 10.00, 29.9, 10.00, dashed=True, lw=0.8)
    text(ax, 15.0, 10.15, "Blurring Flow", size=11, style="italic", weight="bold")

    seq_message(ax, 1.2, 3.8, 9.55, "28: requestAnonymization(video_id, refs[], threshold=0.65)")
    seq_message(ax, 3.8, 6.4, 9.10, "29: POST /anonymize-video")
    seq_message(ax, 6.4, 22.0, 8.65, "30: anonymize_video(...)")
    seq_message(ax, 22.0, 16.8, 8.20, "31: fetch source + reference images")
    seq_message(ax, 16.8, 22.0, 7.75, "32: source path + refs[]", dashed=True)
    seq_message(ax, 22.0, 24.6, 7.30, "33: generate_master_signature(refs[])")
    seq_message(ax, 24.6, 22.0, 6.85, "34: master_vector(128-d)", dashed=True)

    # loop fragment around messages 35-40
    rect(ax, 7.6, 2.85, 22.0, 3.55, fill="white", stroke=LINE, lw=1.0, dashed=True)
    text(ax, 7.75, 6.25, "loop [for each frame]", size=8, weight="bold", style="italic", ha="left", va="top")

    seq_message(ax, 22.0, 27.0, 5.85, "35: detect_persons() + track_ids()")
    seq_message(ax, 27.0, 22.0, 5.40, "36: tracker_ids[]", dashed=True)
    seq_message(ax, 22.0, 22.0, 4.90, "37: throttle check (frame % 3 == 0)", self_call=True)
    seq_message(ax, 22.0, 24.6, 4.20, "38: crop_face() + encode_face()")
    seq_message(ax, 24.6, 22.0, 3.75, "39: face_embedding(128-d)", dashed=True)
    seq_message(ax, 22.0, 22.0, 3.20, "40: cosine_similarity \u2265 0.65 + apply_blur(face | body_top_22%)", self_call=True)

    seq_message(ax, 22.0, 16.8, 2.45, "41: upload anonymized output")
    seq_message(ax, 16.8, 22.0, 2.00, "42: anonymized signed URL", dashed=True)
    seq_message(ax, 22.0, 29.4, 1.55, "43: record_event(module='blurring', target_ids_blurred=N)")
    seq_message(ax, 29.4, 22.0, 1.10, "44: blurring_history_id", dashed=True)
    seq_message(ax, 22.0, 6.4, 0.65, "45: BlurringResult", dashed=True)

    add_caption(ax, "Figure 4.7.2: Sequence Diagram - Clip Search, Compression and Blurring End-to-End.")
    save(fig, _out("4.7.2_sequence_clip_search"))


# ===========================================================================
# 4.8  Design Class Diagram
# ===========================================================================

def draw_class_diagram() -> None:
    fig, ax = new_canvas(28, 26)

    # ===== Top row: domain entities =====
    class_box(ax, 0.5, 25.5, 3.4, "User", ["+id : UUID", "+email : string", "+display_name : string", "+created_at : datetime"], ["+login()", "+logout()"])
    class_box(ax, 4.8, 25.5, 3.8, "Video", ["+id : UUID", "+user_id : UUID", "+title : string", "+video_url : string", "+duration : float", "+created_at : datetime"], [])
    class_box(ax, 9.5, 25.5, 3.8, "TranscriptSegment", ["+id : UUID", "+video_id : UUID", "+start : float", "+end : float", "+text : string"], [])
    class_box(ax, 14.2, 25.5, 3.8, "EmbeddingRow", ["+id : UUID", "+video_id : UUID", "+job_id : UUID", "+type : string", "+vector : vector", "+created_at : datetime"], [])
    class_box(ax, 18.9, 25.5, 3.0, "ProcessingHistory", ["+id : UUID", "+user_id : UUID", "+job_id : UUID", "+module : string", "+query : string", "+status : string"], [])

    # ===== Middle: search request / result =====
    class_box(ax, 0.5, 19.0, 3.4, "ClipRequest", ["+id : UUID", "+job_id : UUID", "+query : string", "+top_k : int", "+params : dict"], [])
    class_box(ax, 4.8, 19.0, 3.4, "ClipResult", ["+id : UUID", "+job_id : UUID", "+start : float", "+end : float", "+url : string", "+score : float"], [])

    # ===== Compression module cluster =====
    rect(ax, 9.3, 12.2, 12.6, 7.0, fill="white", stroke=LINE, lw=1.2, dashed=True)
    text(ax, 21.8, 19.05, "«module» Compression", size=10, weight="bold", style="italic", ha="right", va="top")
    class_box(ax, 9.7, 19.0, 3.8, "CompressionRequest", ["+id : UUID", "+video_id : UUID", "+preset : string", "+target_resolution : int", "+target_bitrate : int", "+crf : float"], [])
    class_box(ax, 14.4, 19.0, 4.0, "CompressionResult", ["+id : UUID", "+request_id : UUID", "+output_url : string", "+original_size : int", "+compressed_size : int", "+compression_ratio : float", "+created_at : datetime"], [])
    class_box(ax, 9.7, 14.4, 8.8, "CompressionService", [], [
        "+validate_source(video_id) : bool",
        "+build_ffmpeg_args(preset) : list",
        "+compress_video(video_id, preset) : CompressionResult",
        "+calculate_metrics(input, output) : dict",
    ])

    # ===== Service classes =====
    class_box(ax, 0.5, 14.4, 3.8, "Transcriber", [], ["+generate_transcript(audio) : Transcript"])
    class_box(ax, 4.8, 14.4, 4.0, "EmbeddingService", [], ["+encode_sentences(list) : list", "+encode_query(string) : vector", "+store_vectors(rows)"])
    class_box(ax, 0.5, 11.5, 3.8, "SearchService", [], ["+process_query(job_id, query, params)", "+rerank(candidates)", "+merge_neighbors(candidates)"])
    class_box(ax, 4.8, 11.5, 4.0, "StorageService", [], ["+save_video(file)", "+get_clip_url(video_id, start, end)"])
    class_box(ax, 18.9, 14.4, 3.0, "HistoryService", [], ["+record_event(...)", "+list_for_user(user_id)"])

    # ===== Blurring module cluster (NEW, full-width bottom band) =====
    # Cluster sits above caption (caption at y=0.25); we leave y=0..1.0 clear.
    rect(ax, 0.4, 1.0, 27.2, 8.5, fill="white", stroke=LINE, lw=1.2, dashed=True)
    text(ax, 27.5, 9.35, "«module» Blurring", size=10, weight="bold", style="italic", ha="right", va="top")

    # Top row of cluster: data classes (compact attr lists)
    class_box(ax, 0.7, 9.0, 4.0, "BlurringRequest", [
        "+id : UUID",
        "+video_id : UUID",
        "+reference_images : list[str]",
        "+similarity_threshold = 0.65",
        "+engine_throttle = 3",
        "+grace_period / max_missing = 30 / 60",
    ], [])
    class_box(ax, 5.4, 9.0, 4.0, "BlurringResult", [
        "+id : UUID", "+request_id : UUID", "+output_url : string",
        "+total_frames : int", "+target_ids_blurred : list[int]",
        "+processing_time : float",
    ], [])
    class_box(ax, 10.1, 9.0, 4.0, "BiometricSignature", [
        "+id : UUID", "+video_id : UUID", "+source_image : string",
        "+vector : vector(128)",
    ], [])
    class_box(ax, 14.8, 9.0, 4.4, "TrackerState", [
        "+active_target_ids : set[int]", "+rejected_ids : set[int]",
        "+frames_since_seen : dict[int,int]",
    ], [])

    # Middle of cluster: BlurringService (wide)
    class_box(ax, 5.4, 5.0, 13.8, "BlurringService", [], [
        "+validate_inputs(video_id, refs[]) : bool",
        "+generate_master_signature(refs[]) : vector(128)",
        "+process_frame(frame, throttle, state) : Frame",
        "+blur_video(video_id, request : BlurringRequest) : BlurringResult",
    ])

    # Bottom row of cluster: 4 ML wrappers
    class_box(ax, 0.7, 2.6, 4.4, "YoloWorldDetector", [], ["+detect_persons(frame) : Box[]"])
    class_box(ax, 5.6, 2.6, 4.4, "BotSortTracker", [], ["+update(detections) : TrackedBox[]"])
    class_box(ax, 10.5, 2.6, 4.6, "CaffeFaceDetector", [], ["+detect_face(person_box) : Box | None"])
    class_box(ax, 15.6, 2.6, 4.6, "OpenFaceEmbedder", [], ["+encode_face(face_crop) : vector(128)"])

    # ===== Associations (carefully routed to avoid crossings) =====
    # User 1 -> 0..* Video (aggregation)
    aggregation_diamond(ax, 4.0, 24.5, size=0.18)
    line(ax, 4.18, 24.5, 4.8, 24.5, lw=1.0)
    text(ax, 4.0, 24.7, "1", size=8, ha="left")
    text(ax, 4.7, 24.7, "0..*", size=8, ha="right")

    # Video 1 -> 0..* TranscriptSegment
    line(ax, 8.6, 24.5, 9.5, 24.5, lw=1.0)
    text(ax, 8.7, 24.7, "1", size=8, ha="left")
    text(ax, 9.4, 24.7, "0..*", size=8, ha="right")

    # Video 1 -> 0..* EmbeddingRow
    line(ax, 6.7, 22.4, 6.7, 21.95, lw=1.0)
    line(ax, 6.7, 21.95, 16.1, 21.95, lw=1.0)
    line(ax, 16.1, 21.95, 16.1, 22.4, lw=1.0)
    text(ax, 6.85, 22.05, "1", size=8, ha="left")
    text(ax, 15.95, 22.05, "0..*", size=8, ha="right")

    # User 1 -> 0..* ProcessingHistory
    line(ax, 2.2, 22.4, 2.2, 21.55, lw=1.0)
    line(ax, 2.2, 21.55, 20.4, 21.55, lw=1.0)
    line(ax, 20.4, 21.55, 20.4, 22.4, lw=1.0)
    text(ax, 2.4, 21.65, "1", size=8, ha="left")
    text(ax, 20.25, 21.65, "0..*", size=8, ha="right")

    # ClipRequest 1 -> 0..* ClipResult
    line(ax, 3.9, 17.5, 4.8, 17.5, lw=1.0)
    text(ax, 4.0, 17.65, "1", size=8, ha="left")
    text(ax, 4.7, 17.65, "0..*", size=8, ha="right")

    # Video 1 -> 0..* CompressionRequest
    line(ax, 6.7, 22.4, 6.7, 20.4, lw=1.0)
    line(ax, 6.7, 20.4, 11.6, 20.4, lw=1.0)
    line(ax, 11.6, 20.4, 11.6, 19.5, lw=1.0)
    text(ax, 8.8, 20.55, "compresses", size=8, style="italic")
    text(ax, 11.45, 19.7, "0..*", size=8, ha="right")

    # CompressionRequest 1 -> 1 CompressionResult
    line(ax, 13.5, 17.5, 14.4, 17.5, lw=1.0)
    text(ax, 13.6, 17.65, "1", size=8, ha="left")
    text(ax, 14.3, 17.65, "1", size=8, ha="right")

    # CompressionRequest -> CompressionService (produces)
    dependency_line(ax, 11.6, 15.6, 11.6, 16.6, label="produces")
    # CompressionService -> HistoryService
    dependency_line(ax, 18.5, 14.0, 18.9, 14.0, label="logs")

    # Service-layer dependencies
    dependency_line(ax, 6.7, 11.5, 6.7, 12.0, label="uses")
    dependency_line(ax, 5.0, 11.5, 5.4, 12.0, label="uses")

    # ===== Blurring relationships =====
    # Video 1 -> 0..* BlurringRequest - route along the FAR left margin to avoid passing through service classes
    line(ax, 4.8, 22.4, 4.8, 21.0, lw=1.0)
    line(ax, 4.8, 21.0, 0.15, 21.0, lw=1.0)
    line(ax, 0.15, 21.0, 0.15, 9.05, lw=1.0)
    line(ax, 0.15, 9.05, 0.7, 9.05, lw=1.0)
    text(ax, 0.30, 15.5, "anonymizes", size=8, style="italic", ha="left")
    text(ax, 0.85, 9.20, "0..*", size=8, ha="left")

    # BlurringRequest 1 -> 1 BlurringResult
    line(ax, 4.7, 7.0, 5.4, 7.0, lw=1.0)
    text(ax, 4.8, 7.15, "1", size=8, ha="left")
    text(ax, 5.3, 7.15, "1", size=8, ha="right")

    # BlurringRequest -> BlurringService (produces) - down-right
    line(ax, 4.5, 6.4, 4.5, 5.7, lw=1.0)
    line(ax, 4.5, 5.7, 5.4, 5.7, lw=1.0)
    text(ax, 4.6, 5.85, "produces", size=8, style="italic", ha="left")

    # BlurringService -- BiometricSignature (produces)
    dependency_line(ax, 12.1, 7.4, 12.1, 7.0, label="produces")
    # BlurringService -- TrackerState (manages)
    dependency_line(ax, 16.0, 7.4, 16.0, 7.0, label="manages")

    # BlurringService ..> 4 ML wrappers (short verticals)
    dependency_line(ax, 7.0, 4.4, 3.0, 3.6, label="uses")
    dependency_line(ax, 9.5, 4.4, 7.8, 3.6, label="uses")
    dependency_line(ax, 14.0, 4.4, 12.8, 3.6, label="uses")
    dependency_line(ax, 17.0, 4.4, 17.9, 3.6, label="uses")

    # BlurringService ..> StorageService and HistoryService (right-side rail outside cluster)
    line(ax, 19.2, 6.5, 22.5, 6.5, lw=1.0, dashed=True)
    line(ax, 22.5, 6.5, 22.5, 14.0, lw=1.0, dashed=True)
    line(ax, 22.5, 14.0, 21.9, 14.0, lw=1.0, dashed=True)
    text(ax, 22.7, 10.0, "logs / stores", size=8, style="italic", ha="left")

    add_caption(ax, "Figure 4.8: Design Class Diagram (Compression and Blurring as separate clusters).")
    save(fig, _out("4.8_class_diagram"))


# ===========================================================================
# 4.9.1  Interface / Page Map
# ===========================================================================

def draw_interface_design() -> None:
    fig, ax = new_canvas(20, 11)

    # Public routes
    rect(ax, 0.5, 8.6, 4.0, 1.8, fill="white", stroke=LINE, lw=1.0)
    text(ax, 0.7, 10.20, "Public Routes", size=9, weight="bold", style="italic", ha="left", va="top")
    rounded_rect(ax, 0.8, 9.30, 1.5, 0.55, label="/", label_size=9)
    rounded_rect(ax, 2.5, 9.30, 1.8, 0.55, label="/auth", label_size=9)

    # Protected routes
    rect(ax, 0.5, 0.5, 4.0, 7.6, fill="white", stroke=LINE, lw=1.0)
    text(ax, 0.7, 7.95, "Protected Routes", size=9, weight="bold", style="italic", ha="left", va="top")
    routes = [
        ("/dashboard", 7.4),
        ("/summarization", 6.7),
        ("/history", 6.0),
        ("/video/:id", 5.3),
        ("/download-clip", 4.6),
        ("/profile", 3.9),
    ]
    for label, y in routes:
        rounded_rect(ax, 0.8, y, 3.4, 0.55, label=label, label_size=9)
    # Compression route - dashed module wrapper
    rect(ax, 0.7, 2.7, 3.6, 0.7, fill="white", stroke=LINE, lw=1.2, dashed=True)
    rounded_rect(ax, 0.8, 2.8, 3.4, 0.55, label="/compression", label_size=9)
    # Blurring route - dashed module wrapper (NEW)
    rect(ax, 0.7, 1.4, 3.6, 0.7, fill="white", stroke=LINE, lw=1.2, dashed=True)
    rounded_rect(ax, 0.8, 1.5, 3.4, 0.55, label="/blurring", label_size=9)

    # Compression UI flow
    rect(ax, 5.0, 0.5, 5.0, 9.9, fill="white", stroke=LINE, lw=1.2, dashed=True)
    text(ax, 5.2, 10.20, "«module» Compression UI Flow", size=9, weight="bold", style="italic", ha="left", va="top")
    comp_steps = [
        ("C1: Select Video / Upload Source", 9.1),
        ("C2: Choose Profile\n(quality/resolution/bitrate)", 7.85),
        ("C3: Start Compression", 6.5),
        ("C4: Progress + Size Metrics", 5.35),
        ("C5: Preview / Download Output", 4.2),
        ("Save to History", 2.7),
    ]
    prev_y = None
    for label, y in comp_steps:
        rounded_rect(ax, 5.4, y, 4.2, 0.85, label=label, label_size=9)
        if prev_y is not None:
            arrow(ax, 7.5, prev_y, 7.5, y + 0.85)
        prev_y = y

    # Blurring UI flow (NEW)
    rect(ax, 10.5, 0.5, 5.0, 9.9, fill="white", stroke=LINE, lw=1.2, dashed=True)
    text(ax, 10.7, 10.20, "«module» Blurring UI Flow", size=9, weight="bold", style="italic", ha="left", va="top")
    blur_steps = [
        ("B1: Select Source Video", 9.1),
        ("B2: Upload Reference\nImage Folder", 7.85),
        ("B3: Configure Threshold (0.65)\n/ Throttle (3) / Grace (30)", 6.55),
        ("B4: Start Anonymization\n(Generate Master Signature)", 5.20),
        ("B5: Live Progress\n(frames / target IDs locked)", 3.85),
        ("B6: Preview / Download\nAnonymized Video", 2.50),
        ("Save to History", 1.0),
    ]
    prev_y = None
    for label, y in blur_steps:
        rounded_rect(ax, 10.9, y, 4.2, 1.0, label=label, label_size=8)
        if prev_y is not None:
            arrow(ax, 13.0, prev_y, 13.0, y + 1.0)
        prev_y = y

    # Shared UI components
    rect(ax, 16.0, 0.5, 3.8, 9.9, fill="white", stroke=LINE, lw=1.0)
    text(ax, 16.2, 10.20, "Shared UI Components", size=9, weight="bold", style="italic", ha="left", va="top")
    shared = ["DashboardLayout", "Header", "BottomNav", "VideoPlayer", "ProcessingLoader", "Toaster / Sonner", "ThemeToggle"]
    for i, name in enumerate(shared):
        rounded_rect(ax, 16.3, 9.2 - i * 1.20, 3.4, 0.8, label=name, label_size=9)

    # Cross arrows
    arrow(ax, 4.3, 3.05, 5.4, 9.50, label="opens", label_offset=(0.4, 0))
    arrow(ax, 4.3, 1.75, 10.9, 9.50, label="opens", label_offset=(2.0, 0))

    add_caption(ax, "Figure 4.9.1: Interface Design - Page Map with explicit Compression and Blurring UI flows.")
    save(fig, _out("4.9.1_interface_design"))


# ===========================================================================
# 4.9.2  Component Level Design
# ===========================================================================

def draw_component_level() -> None:
    fig, ax = new_canvas(22, 13)

    # Frontend
    rect(ax, 0.4, 8.5, 5.0, 4.0, fill="white", stroke=LINE, lw=1.2)
    text(ax, 0.6, 12.30, "Frontend (React + Vite + TS)", size=9, weight="bold", style="italic", ha="left", va="top")
    component_box(ax, 0.7, 10.6, 4.4, 1.2, "Pages")
    component_box(ax, 0.7, 9.0, 2.0, 1.2, "Contexts")
    component_box(ax, 3.1, 9.0, 2.0, 1.2, "React Query")

    # Backend
    rect(ax, 6.0, 7.5, 5.0, 5.0, fill="white", stroke=LINE, lw=1.2)
    text(ax, 6.2, 12.30, "Backend (FastAPI / Python)", size=9, weight="bold", style="italic", ha="left", va="top")
    component_box(ax, 6.3, 10.6, 4.4, 1.2, "API Routes")
    component_box(ax, 6.3, 9.0, 4.4, 1.2, "Pydantic Models")
    component_box(ax, 6.3, 7.7, 4.4, 1.2, "Service Layer")

    # Compression Service group (upper right)
    rect(ax, 11.6, 9.5, 5.2, 3.0, fill="white", stroke=LINE, lw=1.2, dashed=True)
    text(ax, 11.8, 12.30, "«module» Compression Service", size=9, weight="bold", style="italic", ha="left", va="top")
    component_box(ax, 11.9, 10.7, 4.6, 1.5, "CompressionService")
    text(ax, 14.2, 10.55, "validate · map preset · invoke FFmpeg · return metrics", size=7, ha="center")

    # Blurring Service group (NEW, far right upper)
    rect(ax, 17.0, 7.5, 4.8, 5.0, fill="white", stroke=LINE, lw=1.2, dashed=True)
    text(ax, 17.2, 12.30, "«module» Blurring Service", size=9, weight="bold", style="italic", ha="left", va="top")
    component_box(ax, 17.3, 10.9, 4.2, 1.2, "BlurringService")
    component_box(ax, 17.3, 9.5, 4.2, 1.2, "Tracker State Manager\n(ACTIVE / REJECTED IDs)")
    component_box(ax, 17.3, 8.1, 4.2, 1.2, "Blur Processor\n(Gaussian + Anatomy Fallback)")

    # AI/ML Pipeline (Search/Ingest)
    rect(ax, 6.0, 2.5, 5.0, 4.6, fill="white", stroke=LINE, lw=1.2)
    text(ax, 6.2, 6.95, "AI / ML Pipeline", size=9, weight="bold", style="italic", ha="left", va="top")
    component_box(ax, 6.3, 5.3, 4.4, 1.2, "AssemblyAI (Trans.)")
    component_box(ax, 6.3, 3.9, 4.4, 1.2, "sentence-transformers")
    component_box(ax, 6.3, 2.6, 4.4, 1.2, "yt-dlp + FFmpeg")

    # Compression Pipeline (lower middle)
    rect(ax, 11.6, 2.5, 5.2, 4.6, fill="white", stroke=LINE, lw=1.2)
    text(ax, 11.8, 6.95, "Compression Pipeline", size=9, weight="bold", style="italic", ha="left", va="top")
    component_box(ax, 11.9, 5.3, 4.6, 1.2, "FFmpeg Worker")
    component_box(ax, 11.9, 3.9, 4.6, 1.2, "Preset Mapper")
    component_box(ax, 11.9, 2.6, 4.6, 1.2, "Metrics Calculator + Uploader")

    # Blurring ML Pipeline (NEW, far right lower)
    rect(ax, 17.0, 0.5, 4.8, 6.6, fill="white", stroke=LINE, lw=1.2)
    text(ax, 17.2, 6.95, "Blurring ML Pipeline", size=9, weight="bold", style="italic", ha="left", va="top")
    component_box(ax, 17.3, 5.5, 4.2, 1.0, "YOLO-World Engine")
    component_box(ax, 17.3, 4.4, 4.2, 1.0, "BoT-SORT Tracker")
    component_box(ax, 17.3, 3.3, 4.2, 1.0, "Caffe Face Detector\n(Res10 SSD)")
    component_box(ax, 17.3, 2.2, 4.2, 1.0, "OpenFace Embedder\n(nn4.small2)")
    component_box(ax, 17.3, 0.9, 4.2, 1.1, "Storage Uploader\n(anonymized mp4 + sig)")

    # Lollipop / socket interfaces
    lollipop(ax, 5.4, 11.2, 0.6, "IRoutes", side="right")
    socket(ax, 6.0, 11.2, 0.6, "IRoutes", side="left")

    lollipop(ax, 11.0, 10.0, 0.6, "ICompress", side="right")
    socket(ax, 11.6, 10.0, 0.6, "ICompress", side="left")

    lollipop(ax, 11.0, 8.0, 0.6, "IPipeline", side="right")
    socket(ax, 11.6, 8.0, 0.6, "IPipeline", side="left")

    # Blurring interfaces (NEW)
    lollipop(ax, 16.4, 11.2, 0.6, "IBlur", side="right")
    socket(ax, 17.0, 11.2, 0.6, "IBlur", side="left")

    lollipop(ax, 16.4, 4.0, 0.6, "IModel", side="right")
    socket(ax, 17.0, 4.0, 0.6, "IModel", side="left")

    add_caption(ax, "Figure 4.9.2: Component Level Design with Compression and Blurring as separate service modules.")
    save(fig, _out("4.9.2_component_level"))


# ===========================================================================
# 4.9.3  Deployment Diagram
# ===========================================================================

def draw_deployment() -> None:
    fig, ax = new_canvas(20, 14)

    # Client device node (top-left)
    deploy_node(ax, 0.5, 11.0, 3.6, 1.7, "Client Device")
    component_box(ax, 0.8, 11.3, 3.0, 1.0, "Web Browser")

    # Vercel / CDN
    deploy_node(ax, 5.5, 11.0, 3.6, 1.7, "Vercel Edge / CDN")
    component_box(ax, 5.8, 11.3, 3.0, 1.0, "NeuroClip SPA")

    # Backend host (centre) - now taller for the new Blurring Worker
    deploy_node(ax, 10.5, 0.5, 5.0, 12.2, "Backend Host (Render / Railway)")
    component_box(ax, 10.8, 11.4, 4.4, 1.0, "FastAPI (Uvicorn :8000)")
    component_box(ax, 10.8, 10.0, 4.4, 1.0, "Background Worker")
    component_box(ax, 10.8, 7.8, 4.4, 1.6, "Compression Worker")
    text(ax, 13.0, 7.65, "FFmpeg presets \u00b7 CRF \u00b7 bitrate", size=7, ha="center")
    component_box(ax, 10.8, 5.6, 4.4, 1.6, "Blurring Worker (CPU/GPU)")
    text(ax, 13.0, 5.45, "PyTorch \u00b7 OpenCV \u00b7 ONNX", size=7, ha="center")
    rect(ax, 10.8, 4.0, 4.4, 1.0, fill="white", stroke=LINE, lw=1.0, label="Vector Cache (hot reload)", label_pos="center", label_size=8)
    rect(ax, 10.8, 2.5, 4.4, 1.0, fill="white", stroke=LINE, lw=1.0, label="Service Layer (Pydantic)", label_pos="center", label_size=8)

    # Supabase Cloud (taller too, to mirror the backend column)
    deploy_node(ax, 16.5, 0.5, 3.3, 12.2, "Supabase Cloud")
    component_box(ax, 16.8, 11.4, 2.7, 1.0, "Auth (JWT, RLS)")
    rect(ax, 16.8, 9.6, 2.7, 1.0, fill="white", stroke=LINE, lw=1.0, label="Postgres + pgvector", label_pos="center", label_size=8)
    text(ax, 18.15, 9.40, "(profiles, history,\nmaster signatures vec(128))", size=6, ha="center", va="top")
    rect(ax, 16.8, 6.5, 2.7, 2.4, fill="white", stroke=LINE, lw=1.0, label="Object Storage\n(videos, clips,\ncompressed outputs,\nanonymized videos,\nreference images,\nmodel weights)", label_pos="center", label_size=7)

    # External services (bottom-left)
    deploy_node(ax, 0.5, 3.5, 3.6, 3.0, "External Services")
    component_box(ax, 0.8, 5.0, 3.0, 1.0, "AssemblyAI API")
    component_box(ax, 0.8, 3.7, 3.0, 1.0, "YouTube / URLs")

    # ----- Connections (clean horizontal/vertical only where possible) -----
    arrow(ax, 4.1, 11.85, 5.5, 11.85, label="HTTPS")
    arrow(ax, 9.1, 11.85, 10.8, 11.85, label="HTTPS / JSON")

    # SPA -> Compression Worker (POST /compress-video) - route along the top with right-angle corner
    line(ax, 9.1, 12.45, 10.55, 12.45, lw=1.0)
    line(ax, 10.55, 12.45, 10.55, 8.6, lw=1.0)
    arrow(ax, 10.55, 8.6, 10.8, 8.6, label="POST /compress-video", label_offset=(-1.4, 0.25))

    # SPA -> Blurring Worker (POST /anonymize-video) - similar right-angle route, lower
    line(ax, 9.1, 12.65, 10.40, 12.65, lw=1.0)
    line(ax, 10.40, 12.65, 10.40, 6.4, lw=1.0)
    arrow(ax, 10.40, 6.4, 10.8, 6.4, label="POST /anonymize-video", label_offset=(-1.4, 0.25))

    # FastAPI -> Postgres
    arrow(ax, 15.2, 11.85, 16.8, 10.1, label="SQL", label_offset=(0.5, 0.5))
    # FastAPI -> Object Storage
    arrow(ax, 15.2, 11.6, 16.8, 7.5, label="HTTPS", label_offset=(0.4, 0.4))
    # SPA -> Auth via JS SDK (route along the top, well above the node header)
    line(ax, 9.1, 13.05, 18.15, 13.05, lw=1.0)
    text(ax, 13.6, 13.20, "JS SDK / HTTPS", size=8, ha="center")
    arrow(ax, 18.15, 13.05, 18.15, 12.7)

    # Background Worker -> Postgres / Storage
    arrow(ax, 15.2, 10.5, 16.8, 10.1, label="vectors", label_offset=(0.4, 0.3))

    # Compression Worker -> Object Storage (compressed outputs)
    arrow(ax, 15.2, 8.6, 16.8, 8.0, label="compressed mp4 / webm", label_offset=(0.7, 0.3))

    # Blurring Worker -> Object Storage (anonymized + master signatures + model weights)
    arrow(ax, 15.2, 6.4, 16.8, 7.0, label="anonymized mp4 + refs + weights", label_offset=(0.6, -0.4))
    # Blurring Worker -> Postgres + pgvector (master signature vector)
    line(ax, 15.2, 6.6, 16.20, 6.6, lw=1.0)
    line(ax, 16.20, 6.6, 16.20, 9.6, lw=1.0)
    arrow(ax, 16.20, 9.6, 16.80, 9.6, label="master signature vector(128)", label_offset=(0.4, -0.55))

    # FastAPI -> AssemblyAI (REST) - corner route along left
    line(ax, 10.8, 11.6, 4.7, 11.6, lw=1.0)
    line(ax, 4.7, 11.6, 4.7, 5.5, lw=1.0)
    arrow(ax, 4.7, 5.5, 3.8, 5.5, label="REST", label_offset=(-0.3, 0.25))

    # Background worker -> YouTube (yt-dlp) - similar L route
    line(ax, 10.8, 10.4, 4.4, 10.4, lw=1.0)
    line(ax, 4.4, 10.4, 4.4, 4.2, lw=1.0)
    arrow(ax, 4.4, 4.2, 3.8, 4.2, label="yt-dlp", label_offset=(-0.3, 0.25))

    add_caption(ax, "Figure 4.9.3: Deployment Diagram with Compression and Blurring Workers as dedicated runtimes.")
    save(fig, _out("4.9.3_deployment"))


# ===========================================================================
# Driver
# ===========================================================================

ALL_DIAGRAMS = [
    ("4.2  use_case        ", draw_use_case),
    ("4.4  activity        ", draw_activity),
    ("4.5.1 dfd_level0     ", draw_dfd_level0),
    ("4.5.2 dfd_level1     ", draw_dfd_level1),
    ("4.5.3 dfd_level2     ", draw_dfd_level2),
    ("4.6  system_sequence ", draw_system_sequence),
    ("4.7.1 sequence_auth  ", draw_sequence_auth),
    ("4.7.2 sequence_search", draw_sequence_search_compress),
    ("4.8  class_diagram   ", draw_class_diagram),
    ("4.9.1 interface      ", draw_interface_design),
    ("4.9.2 component      ", draw_component_level),
    ("4.9.3 deployment     ", draw_deployment),
]


def render_all() -> None:
    for name, fn in ALL_DIAGRAMS:
        print(f"  drawing {name} ...", flush=True)
        fn()
    print("Done.")


if __name__ == "__main__":
    render_all()

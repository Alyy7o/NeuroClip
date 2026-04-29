import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import csv
import os
import sys

# ── Paths ──
RESULTS_CSV = "kaggle_eval_dataset/evaluation_results.csv"
OUTPUT_DIR = "test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Global styling ──
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'axes.facecolor': '#f8f9fa',
    'figure.facecolor': '#ffffff',
})

# ── Color palettes ──
COLORS = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'accent': '#e74c3c',
    'purple': '#9b59b6',
    'orange': '#f39c12',
    'teal': '#1abc9c',
    'dark': '#2c3e50',
    'grey': '#95a5a6',
}

DOMAIN_COLORS = [
    '#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6',
    '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b',
]


def load_results():
    """Load evaluation results from CSV. Returns list of dicts."""
    if not os.path.exists(RESULTS_CSV):
        print(f"Error: {RESULTS_CSV} not found. Run kaggle_automated_eval.py first.")
        sys.exit(1)

    results = []
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["iou"] = float(row["iou"])
            row["query_latency"] = float(row["query_latency"])
            row["relevant"] = int(row["relevant"])
            row["pred_start"] = float(row.get("pred_start", 0))
            row["pred_end"] = float(row.get("pred_end", 0))
            row["gt_start"] = float(row.get("gt_start", 0))
            row["gt_end"] = float(row.get("gt_end", 0))
            row["num_clips"] = int(row.get("num_clips", 0))
            results.append(row)

    print(f"Loaded {len(results)} evaluation results from {RESULTS_CSV}")
    return results


# ═══════════════════════════════════════════════════════════════════
# GRAPH 1: Per-Domain Retrieval Performance (Precision@1 + Avg IoU)
# ═══════════════════════════════════════════════════════════════════
def plot_domain_performance(results):
    domains = sorted(set(r["domain"] for r in results))
    precision_vals = []
    iou_vals = []

    for d in domains:
        dr = [r for r in results if r["domain"] == d]
        precision_vals.append(sum(r["relevant"] for r in dr) / len(dr))
        iou_vals.append(sum(r["iou"] for r in dr) / len(dr))

    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, precision_vals, width, label='Precision@1', color=COLORS['primary'], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, iou_vals, width, label='Average IoU', color=COLORS['secondary'], edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Score')
    ax.set_title('Retrieval Performance by Academic Domain')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=30, ha='right')
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper right')
    ax.axhline(y=0.3, color=COLORS['accent'], linestyle='--', alpha=0.5, label='IoU Threshold (0.3)')

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'retrieval_performance.png')
    plt.savefig(path)
    plt.close()
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════
# GRAPH 2: Per-Difficulty Breakdown (Grouped Bar)
# ═══════════════════════════════════════════════════════════════════
def plot_difficulty_breakdown(results):
    difficulties = ["Easy", "Medium", "Hard"]
    precision_vals = []
    iou_vals = []
    counts = []

    for diff in difficulties:
        dr = [r for r in results if r["difficulty"] == diff]
        if dr:
            precision_vals.append(sum(r["relevant"] for r in dr) / len(dr))
            iou_vals.append(sum(r["iou"] for r in dr) / len(dr))
            counts.append(len(dr))
        else:
            precision_vals.append(0)
            iou_vals.append(0)
            counts.append(0)

    x = np.arange(len(difficulties))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width / 2, precision_vals, width, label='Precision@1', color=COLORS['purple'], edgecolor='white')
    bars2 = ax.bar(x + width / 2, iou_vals, width, label='Average IoU', color=COLORS['teal'], edgecolor='white')

    ax.set_ylabel('Score')
    ax.set_title('Retrieval Performance by Query Difficulty')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={c})" for d, c in zip(difficulties, counts)])
    ax.set_ylim([0, 1.1])
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'difficulty_breakdown.png')
    plt.savefig(path)
    plt.close()
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════
# GRAPH 3: IoU Distribution Histogram
# ═══════════════════════════════════════════════════════════════════
def plot_iou_distribution(results):
    ious = [r["iou"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax.hist(ious, bins=20, range=(0, 1), color=COLORS['primary'],
                                edgecolor='white', linewidth=1, alpha=0.85)

    # Color bars below threshold red
    for patch, left_edge in zip(patches, bins):
        if left_edge + (bins[1] - bins[0]) / 2 < 0.3:
            patch.set_facecolor(COLORS['accent'])
            patch.set_alpha(0.6)

    ax.axvline(x=0.3, color=COLORS['accent'], linestyle='--', linewidth=2, label='Relevance Threshold (0.3)')
    ax.axvline(x=np.mean(ious), color=COLORS['secondary'], linestyle='-', linewidth=2, label=f'Mean IoU ({np.mean(ious):.3f})')

    ax.set_xlabel('Temporal IoU')
    ax.set_ylabel('Number of Queries')
    ax.set_title('Distribution of Temporal IoU Across All Queries')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'iou_distribution.png')
    plt.savefig(path)
    plt.close()
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════
# GRAPH 4: Query Latency Box Plot by Domain
# ═══════════════════════════════════════════════════════════════════
def plot_query_latency(results):
    domains = sorted(set(r["domain"] for r in results))
    latency_data = []
    for d in domains:
        latency_data.append([r["query_latency"] for r in results if r["domain"] == d])

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(latency_data, labels=domains, patch_artist=True, notch=True,
                     medianprops=dict(color='white', linewidth=2))

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(DOMAIN_COLORS[i % len(DOMAIN_COLORS)])
        patch.set_alpha(0.8)

    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Query Latency Distribution by Domain')
    ax.set_xticklabels(domains, rotation=30, ha='right')

    avg_lat = np.mean([r["query_latency"] for r in results])
    ax.axhline(y=avg_lat, color=COLORS['accent'], linestyle='--', alpha=0.6,
               label=f'Overall Mean ({avg_lat:.2f}s)')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'query_latency.png')
    plt.savefig(path)
    plt.close()
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════
# GRAPH 5: Summarization Coverage (Pie Chart)
# ═══════════════════════════════════════════════════════════════════
def plot_summarization_coverage(results):
    has_summary = sum(1 for r in results if r.get("topic_explanation", "").strip())
    no_summary = len(results) - has_summary

    has_llm = sum(1 for r in results if r.get("llm_summary", "").strip())
    no_llm = len(results) - has_llm

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Topic Explanation coverage
    axes[0].pie([has_summary, no_summary],
                labels=[f'Has Summary ({has_summary})', f'No Summary ({no_summary})'],
                colors=[COLORS['secondary'], COLORS['grey']],
                autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 12},
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[0].set_title('Topic Explanation Coverage')

    # LLM Summary coverage
    axes[1].pie([has_llm, no_llm],
                labels=[f'Has LLM Summary ({has_llm})', f'No LLM Summary ({no_llm})'],
                colors=[COLORS['purple'], COLORS['grey']],
                autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 12},
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Per-Clip LLM Summary Coverage')

    plt.suptitle('Summarization Pipeline Coverage', fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'summarization_coverage.png')
    plt.savefig(path)
    plt.close()
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════
# GRAPH 6: Temporal Accuracy Scatter (Predicted vs Ground Truth)
# ═══════════════════════════════════════════════════════════════════
def plot_temporal_accuracy(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Start time comparison
    gt_starts = [r["gt_start"] for r in results]
    pred_starts = [r["pred_start"] for r in results]
    colors_scatter = [COLORS['secondary'] if r["relevant"] else COLORS['accent'] for r in results]

    axes[0].scatter(gt_starts, pred_starts, c=colors_scatter, alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    max_val = max(max(gt_starts, default=1), max(pred_starts, default=1)) * 1.1
    axes[0].plot([0, max_val], [0, max_val], 'k--', alpha=0.4, label='Perfect prediction')
    axes[0].set_xlabel('Ground Truth Start (s)')
    axes[0].set_ylabel('Predicted Start (s)')
    axes[0].set_title('Start Time: Predicted vs Ground Truth')
    axes[0].legend()

    # End time comparison
    gt_ends = [r["gt_end"] for r in results]
    pred_ends = [r["pred_end"] for r in results]

    axes[1].scatter(gt_ends, pred_ends, c=colors_scatter, alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    max_val = max(max(gt_ends, default=1), max(pred_ends, default=1)) * 1.1
    axes[1].plot([0, max_val], [0, max_val], 'k--', alpha=0.4, label='Perfect prediction')
    axes[1].set_xlabel('Ground Truth End (s)')
    axes[1].set_ylabel('Predicted End (s)')
    axes[1].set_title('End Time: Predicted vs Ground Truth')
    axes[1].legend()

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['secondary'], label='Relevant (IoU > 0.3)'),
        Patch(facecolor=COLORS['accent'], label='Not Relevant (IoU ≤ 0.3)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Temporal Alignment: Predicted vs Ground Truth Timestamps', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(OUTPUT_DIR, 'temporal_accuracy.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════
# GRAPH 7: Overall Summary Dashboard
# ═══════════════════════════════════════════════════════════════════
def plot_summary_dashboard(results):
    total = len(results)
    precision = sum(r["relevant"] for r in results) / total * 100
    avg_iou = sum(r["iou"] for r in results) / total
    avg_latency = sum(r["query_latency"] for r in results) / total
    summary_rate = sum(1 for r in results if r.get("topic_explanation", "").strip()) / total * 100

    metrics = ['Precision@1\n(%)', 'Avg IoU\n(0-1)', 'Avg Latency\n(seconds)', 'Summary Rate\n(%)']
    values = [precision, avg_iou, avg_latency, summary_rate]
    display_values = [f'{precision:.1f}%', f'{avg_iou:.3f}', f'{avg_latency:.1f}s', f'{summary_rate:.0f}%']
    bar_colors = [COLORS['primary'], COLORS['secondary'], COLORS['orange'], COLORS['purple']]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for i, ax in enumerate(axes):
        ax.barh([0], [values[i]], color=bar_colors[i], height=0.5, edgecolor='white')
        ax.set_xlim(0, max(values[i] * 1.3, 1))
        ax.set_yticks([])
        ax.set_xlabel(metrics[i], fontsize=12, fontweight='bold')
        ax.text(values[i] * 0.5, 0, display_values[i], ha='center', va='center',
                fontsize=18, fontweight='bold', color='white')

    plt.suptitle(f'NeuroClip Evaluation Dashboard (n={total} queries)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'evaluation_dashboard.png')
    plt.savefig(path)
    plt.close()
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  NeuroClip — Generating Evaluation Graphs")
    print("=" * 60)

    results = load_results()

    print(f"\nGenerating graphs from {len(results)} results...\n")

    plot_domain_performance(results)
    plot_difficulty_breakdown(results)
    plot_iou_distribution(results)
    plot_query_latency(results)
    plot_summarization_coverage(results)
    plot_temporal_accuracy(results)
    plot_summary_dashboard(results)

    print(f"\n✓ All 7 graphs saved to '{OUTPUT_DIR}/' directory!")
    print(f"  Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
            print(f"    • {f} ({size_kb:.0f} KB)")

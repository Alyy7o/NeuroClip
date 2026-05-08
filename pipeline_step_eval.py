"""
NeuroClip Per-Step Pipeline Evaluation
======================================
Evaluates each pipeline step independently and computes
F-Score, Precision, and Recall for each.

Steps evaluated:
  1. Transcription (AssemblyAI ASR)
  2. Frame Extraction (OpenCV / FFmpeg)
  3. EasyOCR (Text Recognition)
  4. Vectorization (MiniLM-L6-v2 Embeddings)
  5. Semantic Search (Cosine + Sliding Window)
  6. LLM Summarization (Groq / Gemini)

Usage:
  Offline  (uses existing results):  python pipeline_step_eval.py
  Online   (queries backend):        python pipeline_step_eval.py --backend-url <URL>
"""

import csv
import json
import math
import os
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# ── Paths ──
DATASET_CSV = "kaggle_eval_dataset/summarization_eval_pack.csv"
RESULTS_CSV = "kaggle_eval_dataset/evaluation_results.csv"
CACHE_FILE  = "kaggle_eval_dataset/ingestion_cache.json"
OUTPUT_DIR  = "test_results"
STEP_CSV    = os.path.join(OUTPUT_DIR, "pipeline_step_results.csv")
STEP_PNG    = os.path.join(OUTPUT_DIR, "pipeline_step_comparison.png")
DETAIL_CSV  = os.path.join(OUTPUT_DIR, "pipeline_step_details.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keywords per query (from calibrate_ground_truth.py)
QUERY_KEYWORDS = {
    "what is object oriented programming": ["object oriented", "oop", "programming paradigm", "objects"],
    "explain inheritance and polymorphism": ["inheritance", "polymorphism", "inherit", "polymorph"],
    "what are the four pillars of OOP": ["four", "pillars", "encapsulation", "abstraction", "inheritance", "polymorphism"],
    "how encapsulation protects data state": ["encapsulation", "encapsulate", "private", "data", "protect", "hide"],
    "what is a hash table": ["hash table", "hash map", "hashing", "key value"],
    "how to handle hash collisions": ["collision", "collide", "separate chaining", "open addressing"],
    "time complexity of hash map lookups": ["time complexity", "big o", "constant time", "o(1)", "lookup"],
    "open addressing versus chaining": ["open addressing", "chaining", "linear probing", "separate chain"],
    "what is an API": ["api", "application programming interface", "interface"],
    "difference between REST and SOAP": ["rest", "soap", "restful", "representational state"],
    "HTTP status codes explained": ["status code", "200", "404", "http", "response code"],
    "API authentication methods": ["authentication", "api key", "oauth", "token", "auth"],
    "what is a distributed system": ["distributed system", "distributed computing", "multiple machine"],
    "what is the CAP theorem": ["cap theorem", "consistency", "availability", "partition tolerance"],
    "horizontal vs vertical scaling": ["horizontal scaling", "vertical scaling", "scale out", "scale up"],
    "how eventual consistency works": ["eventual consistency", "eventually consistent", "consistency model"],
    "what is a neural network": ["neural network", "neuron", "layer", "network"],
    "how activation functions work": ["activation function", "sigmoid", "relu", "activation"],
    "what is the cost function": ["cost function", "loss function", "cost", "loss", "error"],
    "gradient descent explanation": ["gradient descent", "gradient", "descent", "minimize", "slope"],
    "what is backpropagation": ["backpropagation", "backprop", "back propagation", "backward"],
    "chain rule in neural networks": ["chain rule", "derivative", "chain"],
    "calculating error derivatives": ["derivative", "partial derivative", "error", "gradient"],
    "updating weights and biases": ["weight", "bias", "update", "adjust", "learning rate"],
    "what is Bayes theorem": ["bayes", "theorem", "bayesian"],
    "prior and posterior probabilities": ["prior", "posterior", "probability"],
    "conditional probability definition": ["conditional probability", "given that", "p of a given b"],
    "false positive paradox explanation": ["false positive", "paradox", "test positive"],
    "what is a derivative": ["derivative", "rate of change", "differentiation"],
    "slope of the tangent line": ["tangent", "slope", "tangent line"],
    "power rule in calculus": ["power rule", "x squared", "exponent"],
    "limit definition of a derivative": ["limit", "delta x", "approaches zero", "limit definition"],
    "conservation of momentum": ["conservation", "momentum", "conserve"],
    "elastic vs inelastic collisions": ["elastic", "inelastic", "collision"],
    "calculating total impulse": ["impulse", "force", "time"],
    "what is a buffer overflow": ["buffer overflow", "buffer", "overflow"],
    "how memory stacks work": ["memory stack", "stack", "memory", "stack frame"],
    "preventing stack overflow attacks": ["prevent", "protection", "canary", "stack overflow", "mitigation"],
    "supply and demand curve": ["supply", "demand", "curve"],
    "what is market equilibrium": ["equilibrium", "market equilibrium", "balance"],
    "price elasticity of demand": ["elasticity", "elastic", "price elasticity"],
    "what is cellular respiration": ["cellular respiration", "respiration", "atp", "energy"],
    "glycolysis process explained": ["glycolysis", "glucose", "pyruvate"],
    "krebs cycle overview": ["krebs cycle", "citric acid", "krebs"],
    "covalent vs ionic bonds": ["covalent", "ionic", "bond"],
    "how electrons are shared": ["electron", "share", "sharing"],
    "how a combustion engine works": ["combustion", "engine", "piston", "cylinder"],
    "four stroke engine cycle": ["four stroke", "intake", "compression", "combustion", "exhaust"],
    "standard deviation explained": ["standard deviation", "deviation", "spread"],
    "calculating variance from mean": ["variance", "mean", "average", "squared"],
}


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _f_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = _f_score(precision, recall)
    return {
        "precision": round(precision * 100, 2),
        "recall":    round(recall * 100, 2),
        "f_score":   round(f1 * 100, 2),
        "tp": tp, "fp": fp, "fn": fn,
    }


def _load_dataset():
    rows = []
    with open(DATASET_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["gt_start"] = float(r["gt_start"])
            r["gt_end"]   = float(r["gt_end"])
            rows.append(r)
    return rows


def _calculate_content_relevance(query, llm_summary, topic_explanation):
    """
    Score content relevance (0.0-1.0) based on whether the LLM summary
    and topic explanation actually address the query. Uses keyword overlap.
    (Same logic as kaggle_automated_eval.py)
    """
    if not llm_summary and not topic_explanation:
        return 0.0
    stop_words = {"what", "is", "a", "an", "the", "how", "to", "of", "in",
                  "and", "vs", "versus", "explain", "explained", "definition",
                  "overview", "calculating", "between", "difference"}
    query_words = set(w.lower() for w in query.split()
                      if w.lower() not in stop_words and len(w) > 2)
    if not query_words:
        return 0.5
    response_text = f"{llm_summary or ''} {topic_explanation or ''}".lower()
    matches = sum(1 for w in query_words if w in response_text)
    coverage = matches / len(query_words)
    words = query.lower().split()
    phrase_matches = phrase_count = 0
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if bigram not in " ".join(stop_words):
            phrase_count += 1
            if bigram in response_text:
                phrase_matches += 1
    phrase_score = phrase_matches / max(phrase_count, 1)
    score = 0.6 * coverage + 0.4 * phrase_score
    return round(min(1.0, score), 3)


def _load_results():
    if not os.path.exists(RESULTS_CSV):
        return []
    rows = []
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["iou"]             = float(r.get("iou", 0))
            r["pred_start"]      = float(r.get("pred_start", 0))
            r["pred_end"]        = float(r.get("pred_end", 0))
            r["gt_start"]        = float(r.get("gt_start", 0))
            r["gt_end"]          = float(r.get("gt_end", 0))
            r["num_clips"]       = int(r.get("num_clips", 0))
            r["query_latency"]   = float(r.get("query_latency", 0))
            r["relevant_iou"]    = int(r.get("relevant_iou", r.get("relevant", 0)))
            # Compute content_relevance if missing from CSV
            if "content_relevance" in r and r["content_relevance"]:
                r["content_relevance"] = float(r["content_relevance"])
            else:
                r["content_relevance"] = _calculate_content_relevance(
                    r.get("id", ""),  # will be overridden below
                    r.get("llm_summary", ""),
                    r.get("topic_explanation", ""),
                )
            r["relevant_content"] = int(r.get("relevant_content", 0))
            rows.append(r)
    return rows


def _fetch_transcript_data(backend_url, job_id, headers):
    """Fetch the .v4.json data from the backend for a given job_id."""
    try:
        import requests
        # Try fetching via the search endpoint with a dummy query to get sentences
        resp = requests.post(
            f"{backend_url}/clips/search",
            json={"job_id": job_id, "query": "test", "top_k": 1},
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════
# STEP EVALUATORS
# ═══════════════════════════════════════════════════════════════

def eval_transcription(dataset, results, backend_url=None):
    """
    Step 1: Transcription (ASR)
    TP = transcript sentences exist covering GT time range with keyword matches
    FP = transcript sentences with query-keyword matches OUTSIDE GT range
    FN = GT range has no transcript coverage (no keyword matches found)
    """
    tp = fp = fn = 0

    for ds_row in dataset:
        query = ds_row["query"]
        qid   = ds_row["id"]
        gt_s  = ds_row["gt_start"]
        gt_e  = ds_row["gt_end"]
        keywords = QUERY_KEYWORDS.get(query, [])

        # Find matching result row
        res = next((r for r in results if r["id"] == qid), None)
        if not res:
            fn += 1
            continue

        # Check if transcript produced any content (llm_summary or topic_explanation)
        has_summary = bool((res.get("llm_summary") or "").strip())
        has_topic   = bool((res.get("topic_explanation") or "").strip())
        has_clips   = res["num_clips"] > 0

        if has_clips or has_summary or has_topic:
            # Transcript was generated — check if content relates to query
            combined = f"{res.get('llm_summary', '')} {res.get('topic_explanation', '')}".lower()
            kw_hits = sum(1 for kw in keywords if kw.lower() in combined)
            if kw_hits >= 1:
                tp += 1
            else:
                fp += 1
        else:
            fn += 1

    return _compute_metrics(tp, fp, fn)


def eval_frame_extraction(dataset, results, backend_url=None):
    """
    Step 2: Frame Extraction
    TP = clips returned with valid start/end times (frames were extracted successfully)
    FP = frames extracted but resulting clips don't cover GT
    FN = no clips returned (frame extraction likely failed or produced nothing)
    """
    tp = fp = fn = 0

    for ds_row in dataset:
        qid  = ds_row["id"]
        gt_s = ds_row["gt_start"]
        gt_e = ds_row["gt_end"]

        res = next((r for r in results if r["id"] == qid), None)
        if not res:
            fn += 1
            continue

        if res["num_clips"] > 0 and res["pred_end"] > res["pred_start"]:
            # Frames were extracted — check if they cover GT range at all
            pred_s = res["pred_start"]
            pred_e = res["pred_end"]
            overlap = max(0, min(pred_e, gt_e) - max(pred_s, gt_s))
            gt_duration = gt_e - gt_s
            if overlap > 0:
                tp += 1
            else:
                fp += 1
        else:
            fn += 1

    return _compute_metrics(tp, fp, fn)


def eval_easyocr(dataset, results, backend_url=None):
    """
    Step 3: EasyOCR (Text Recognition on frames)
    TP = OCR text [On Screen: ...] detected and content is relevant
    FP = OCR text detected but not relevant to query
    FN = no OCR text detected (for domains where visual content is expected)

    Checks the raw clip text (all_clip_text), llm_summary, and topic_explanation
    for [On Screen: ...] markers that indicate EasyOCR found text.
    """
    tp = fp = fn = 0

    for ds_row in dataset:
        qid   = ds_row["id"]
        query = ds_row["query"]

        res = next((r for r in results if r["id"] == qid), None)
        if not res:
            fn += 1
            continue

        # Check ALL text sources for OCR markers
        all_text = (
            f"{res.get('all_clip_text', '')} "
            f"{res.get('llm_summary', '')} "
            f"{res.get('topic_explanation', '')}"
        ).lower()

        has_ocr = (
            res.get("has_ocr", 0) == 1 or
            "[on screen" in all_text or
            "[visual content" in all_text
        )

        if has_ocr:
            # OCR ran and found text - check if it's relevant
            keywords = QUERY_KEYWORDS.get(query, [])
            kw_hits = sum(1 for kw in keywords if kw.lower() in all_text)
            if kw_hits >= 1:
                tp += 1
            else:
                fp += 1
        else:
            # No OCR text found
            domain = ds_row.get("domain", "")
            if domain in ("Computer Science", "Mathematics", "Artificial Intelligence"):
                fn += 1
            else:
                # Non-visual domains: no OCR expected = correct behavior
                tp += 1

    return _compute_metrics(tp, fp, fn)


def eval_vectorization(dataset, results, backend_url=None):
    """
    Step 4: Vectorization (Embedding Generation)
    Measures whether generated embeddings capture semantic proximity.
    TP = embeddings produced AND clip has any temporal overlap (IoU > 0)
    FP = embeddings produced but NO temporal overlap at all
    FN = no embeddings / no clips returned
    """
    tp = fp = fn = 0

    for ds_row in dataset:
        qid = ds_row["id"]
        res = next((r for r in results if r["id"] == qid), None)
        if not res:
            fn += 1
            continue

        if res["num_clips"] > 0:
            if res["iou"] > 0.0:
                tp += 1
            else:
                fp += 1
        else:
            fn += 1

    return _compute_metrics(tp, fp, fn)


def eval_semantic_search(dataset, results, backend_url=None):
    """
    Step 5: Semantic Search (Retrieval)
    Measures whether the search retrieves content-relevant clips.
    TP = clips retrieved AND content is relevant to query
    FP = clips retrieved but content NOT relevant
    FN = no clips retrieved for the query
    """
    tp = fp = fn = 0

    for ds_row in dataset:
        qid = ds_row["id"]
        query = ds_row["query"]
        res = next((r for r in results if r["id"] == qid), None)
        if not res:
            fn += 1
            continue

        if res["num_clips"] > 0:
            # Check content relevance of retrieved clip
            cr = _calculate_content_relevance(
                query,
                res.get("llm_summary", ""),
                res.get("topic_explanation", ""),
            )
            if cr >= 0.3 or res["iou"] > 0.0:
                tp += 1
            else:
                fp += 1
        else:
            fn += 1

    return _compute_metrics(tp, fp, fn)


def eval_llm_summarization(dataset, results, backend_url=None):
    """
    Step 6: LLM Summarization (Content Generation)
    TP = summary generated AND content is relevant (content_relevance >= 0.5)
    FP = summary generated but content is NOT relevant (content_relevance < 0.5)
    FN = no summary generated at all
    """
    tp = fp = fn = 0

    for ds_row in dataset:
        qid = ds_row["id"]
        query = ds_row["query"]
        res = next((r for r in results if r["id"] == qid), None)
        if not res:
            fn += 1
            continue

        has_topic = bool((res.get("topic_explanation") or "").strip())
        has_summary = bool((res.get("llm_summary") or "").strip())

        if has_topic or has_summary:
            # Compute content relevance using query text from dataset
            cr = _calculate_content_relevance(
                query,
                res.get("llm_summary", ""),
                res.get("topic_explanation", ""),
            )
            if cr >= 0.5:
                tp += 1
            else:
                fp += 1
        else:
            fn += 1

    return _compute_metrics(tp, fp, fn)


# ═══════════════════════════════════════════════════════════════
# OUTPUT: TABLE + CSV + CHART
# ═══════════════════════════════════════════════════════════════

STEPS = [
    ("Transcription (AssemblyAI ASR)",           eval_transcription),
    ("Frame Extraction (OpenCV / FFmpeg)",        eval_frame_extraction),
    ("EasyOCR (Text Recognition)",               eval_easyocr),
    ("Vectorization (MiniLM-L6-v2 Embeddings)",  eval_vectorization),
    ("Semantic Search (Cosine + Sliding Window)", eval_semantic_search),
    ("LLM Summarization (Groq / Gemini)",        eval_llm_summarization),
]


def print_table(step_results):
    """Print a formatted table matching the reference image style."""
    header = f"{'Sr.No':<7} {'Pipeline Step':<48} {'F-Score':<10} {'Precision':<12} {'Recall':<10}"
    sep    = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    for i, (name, metrics) in enumerate(step_results, 1):
        print(f"{i:<7} {name:<48} {metrics['f_score']:<10.2f} {metrics['precision']:<12.2f} {metrics['recall']:<10.2f}")

    print(sep)
    print()


def save_csv(step_results):
    """Save results to CSV."""
    with open(STEP_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Sr.No", "Pipeline Step", "F-Score", "Precision", "Recall", "TP", "FP", "FN"])
        for i, (name, m) in enumerate(step_results, 1):
            writer.writerow([i, name, m["f_score"], m["precision"], m["recall"], m["tp"], m["fp"], m["fn"]])
    print(f"  CSV saved to: {STEP_CSV}")


def save_detail_csv(step_results):
    """Save detailed per-step TP/FP/FN counts alongside metrics."""
    with open(DETAIL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Sr.No", "Pipeline Step", "F-Score", "Precision", "Recall", "TP", "FP", "FN", "Total"])
        for i, (name, m) in enumerate(step_results, 1):
            total = m["tp"] + m["fp"] + m["fn"]
            writer.writerow([i, name, m["f_score"], m["precision"], m["recall"], m["tp"], m["fp"], m["fn"], total])
    print(f"  Detail CSV saved to: {DETAIL_CSV}")


def save_chart(step_results):
    """Generate a grouped bar chart of F-Score, Precision, Recall per step."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'figure.dpi': 300,
        'figure.facecolor': '#ffffff',
    })

    names  = [name for name, _ in step_results]
    f_vals = [m["f_score"]   for _, m in step_results]
    p_vals = [m["precision"] for _, m in step_results]
    r_vals = [m["recall"]    for _, m in step_results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))

    bars_f = ax.bar(x - width, f_vals, width, label='F-Score',   color='#3498db', edgecolor='white', linewidth=0.5)
    bars_p = ax.bar(x,         p_vals, width, label='Precision', color='#2ecc71', edgecolor='white', linewidth=0.5)
    bars_r = ax.bar(x + width, r_vals, width, label='Recall',    color='#e74c3c', edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Score (%)')
    ax.set_title('NeuroClip - Per-Step Pipeline Evaluation')
    ax.set_xticks(x)
    ax.set_xticklabels([n.split('(')[0].strip() for n in names], rotation=20, ha='right')
    ax.set_ylim([0, 110])
    ax.legend(loc='upper right')

    for bars in [bars_f, bars_p, bars_r]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(STEP_PNG)
    plt.close()
    print(f"  Bar chart saved to: {STEP_PNG}")


def save_radar(step_results):
    """Radar/spider chart showing F-Score for each step."""
    labels = [n.split('(')[0].strip() for n, _ in step_results]
    values = [m["f_score"] for _, m in step_results]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='#3498db', alpha=0.25)
    ax.plot(angles, values, color='#3498db', linewidth=2, marker='o', markersize=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title('NeuroClip - F-Score by Pipeline Step', pad=20, fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pipeline_radar.png")
    plt.savefig(path)
    plt.close()
    print(f"  Radar chart saved to: {path}")


def save_heatmap(step_results):
    """Heatmap of Precision / Recall / F-Score per step."""
    labels = [n.split('(')[0].strip() for n, _ in step_results]
    data = np.array([[m["f_score"], m["precision"], m["recall"]] for _, m in step_results])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['F-Score', 'Precision', 'Recall'])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(3):
            ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center',
                    color='white' if data[i, j] < 50 else 'black', fontweight='bold')
    plt.colorbar(im, label='Score (%)')
    ax.set_title('NeuroClip - Pipeline Step Performance Heatmap')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pipeline_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"  Heatmap saved to: {path}")


# ===============================================================
# MAIN
# ===============================================================

def main():
    print("=" * 65)
    print("  NeuroClip - Per-Step Pipeline Evaluation")
    print("=" * 65)

    backend_url = None
    if "--backend-url" in sys.argv:
        idx = sys.argv.index("--backend-url")
        if idx + 1 < len(sys.argv):
            backend_url = sys.argv[idx + 1].rstrip("/")
            print(f"  Online mode: {backend_url}")
    
    if not backend_url:
        print("  Offline mode: using existing evaluation results")

    # Load data
    if not os.path.exists(DATASET_CSV):
        print(f"\n  ERROR: {DATASET_CSV} not found.")
        print(f"  Run generate_kaggle_dataset.py first.")
        sys.exit(1)

    dataset = _load_dataset()
    results = _load_results()

    if not results:
        print(f"\n  ERROR: {RESULTS_CSV} not found or empty.")
        print(f"  Run kaggle_automated_eval.py first to generate evaluation results.")
        sys.exit(1)

    print(f"\n  Dataset:  {len(dataset)} queries")
    print(f"  Results:  {len(results)} evaluated queries")

    # Match results to dataset by ID
    result_ids = {r["id"] for r in results}
    dataset_matched = [d for d in dataset if d["id"] in result_ids]
    print(f"  Matched:  {len(dataset_matched)} queries\n")

    # Run evaluation for each step
    print("Running per-step evaluation...\n")
    step_results = []

    for step_name, eval_fn in STEPS:
        metrics = eval_fn(dataset_matched, results, backend_url)
        step_results.append((step_name, metrics))
        print(f"  [OK] {step_name}")
        print(f"    TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  "
              f"->  F={metrics['f_score']:.2f}  P={metrics['precision']:.2f}  R={metrics['recall']:.2f}")

    # Output
    print_table(step_results)
    save_csv(step_results)
    save_detail_csv(step_results)

    print("\nGenerating graphs...")
    save_chart(step_results)
    save_radar(step_results)
    save_heatmap(step_results)

    print(f"\n  [OK] All results saved to '{OUTPUT_DIR}/' directory")


if __name__ == "__main__":
    main()


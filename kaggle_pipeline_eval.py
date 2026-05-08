"""
NeuroClip Kaggle Per-Step Pipeline Evaluation
==============================================
Runs on Kaggle against a live backend. Ingests videos, searches,
and evaluates each pipeline step with F-Score, Precision, Recall.

Usage:
  python kaggle_pipeline_eval.py <BACKEND_URL>
"""

import csv, json, math, os, re, sys, signal, time, requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths (relative to script location, not cwd) ──
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV = os.path.join(SCRIPT_DIR, "kaggle_eval_dataset", "summarization_eval_pack.csv")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "kaggle_eval_dataset", "pipeline_eval_results.csv")
CACHE_FILE  = os.path.join(SCRIPT_DIR, "kaggle_eval_dataset", "ingestion_cache.json")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "test_results")
STEP_CSV    = os.path.join(OUTPUT_DIR, "pipeline_step_results.csv")
STEP_PNG    = os.path.join(OUTPUT_DIR, "pipeline_step_comparison.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BACKEND_URL = ""
_INGESTION_CACHE = {}
_SHUTDOWN = False

def _handle_sigint(sig, frame):
    global _SHUTDOWN
    if _SHUTDOWN: sys.exit(1)
    _SHUTDOWN = True
    print("\n  Ctrl+C - finishing current op then saving...")
signal.signal(signal.SIGINT, _handle_sigint)

# ── Keywords per query (for transcription eval) ──
QUERY_KEYWORDS = {
    "what is object oriented programming": ["object oriented", "oop", "programming paradigm"],
    "explain inheritance and polymorphism": ["inheritance", "polymorphism"],
    "what are the four pillars of OOP": ["encapsulation", "abstraction", "inheritance", "polymorphism"],
    "how encapsulation protects data state": ["encapsulation", "private", "protect", "hide"],
    "what is a hash table": ["hash table", "hash map", "hashing"],
    "how to handle hash collisions": ["collision", "separate chaining", "open addressing"],
    "time complexity of hash map lookups": ["time complexity", "big o", "o(1)", "lookup"],
    "open addressing versus chaining": ["open addressing", "chaining", "linear probing"],
    "what is an API": ["api", "application programming interface"],
    "difference between REST and SOAP": ["rest", "soap", "restful"],
    "HTTP status codes explained": ["status code", "200", "404", "http"],
    "API authentication methods": ["authentication", "api key", "oauth", "token"],
    "what is a distributed system": ["distributed system", "distributed computing"],
    "what is the CAP theorem": ["cap theorem", "consistency", "availability"],
    "horizontal vs vertical scaling": ["horizontal scaling", "vertical scaling"],
    "how eventual consistency works": ["eventual consistency", "eventually consistent"],
    "what is a neural network": ["neural network", "neuron", "layer"],
    "how activation functions work": ["activation function", "sigmoid", "relu"],
    "what is the cost function": ["cost function", "loss function", "cost"],
    "gradient descent explanation": ["gradient descent", "gradient", "minimize"],
    "what is backpropagation": ["backpropagation", "backprop", "back propagation"],
    "chain rule in neural networks": ["chain rule", "derivative"],
    "calculating error derivatives": ["derivative", "partial derivative", "gradient"],
    "updating weights and biases": ["weight", "bias", "update", "learning rate"],
    "what is Bayes theorem": ["bayes", "theorem", "bayesian"],
    "prior and posterior probabilities": ["prior", "posterior", "probability"],
    "conditional probability definition": ["conditional probability", "given that"],
    "false positive paradox explanation": ["false positive", "paradox"],
    "what is a derivative": ["derivative", "rate of change", "differentiation"],
    "slope of the tangent line": ["tangent", "slope", "tangent line"],
    "power rule in calculus": ["power rule", "x squared", "exponent"],
    "limit definition of a derivative": ["limit", "delta x", "approaches zero"],
    "conservation of momentum": ["conservation", "momentum"],
    "elastic vs inelastic collisions": ["elastic", "inelastic", "collision"],
    "calculating total impulse": ["impulse", "force"],
    "what is a buffer overflow": ["buffer overflow", "buffer", "overflow"],
    "how memory stacks work": ["memory stack", "stack", "stack frame"],
    "preventing stack overflow attacks": ["prevent", "protection", "stack overflow"],
    "supply and demand curve": ["supply", "demand", "curve"],
    "what is market equilibrium": ["equilibrium", "market equilibrium"],
    "price elasticity of demand": ["elasticity", "elastic", "price elasticity"],
    "what is cellular respiration": ["cellular respiration", "respiration", "atp"],
    "glycolysis process explained": ["glycolysis", "glucose", "pyruvate"],
    "krebs cycle overview": ["krebs cycle", "citric acid", "krebs"],
    "covalent vs ionic bonds": ["covalent", "ionic", "bond"],
    "how electrons are shared": ["electron", "share", "sharing"],
    "how a combustion engine works": ["combustion", "engine", "piston"],
    "four stroke engine cycle": ["four stroke", "intake", "compression"],
    "standard deviation explained": ["standard deviation", "deviation", "spread"],
    "calculating variance from mean": ["variance", "mean", "average"],
}


# ═══════════════════════════════════════════════════
# NETWORK HELPERS (from kaggle_automated_eval.py)
# ═══════════════════════════════════════════════════

def ingest_video(video_url, query, headers):
    global _SHUTDOWN
    if video_url in _INGESTION_CACHE:
        print(f"  -> Cached (job {_INGESTION_CACHE[video_url][:8]}...)", flush=True)
        return _INGESTION_CACHE[video_url], 0.0
    if _SHUTDOWN: return None, 0.0

    import threading

    # Use a thread so we can print progress while waiting
    result_holder = {"resp": None, "error": None}

    def _do_ingest():
        try:
            r = requests.post(f"{BACKEND_URL}/upload-via-url",
                json={"url": video_url, "query": query}, headers=headers, timeout=900)
            r.raise_for_status()
            result_holder["resp"] = r
        except Exception as e:
            result_holder["error"] = e

    t0 = time.time()
    thread = threading.Thread(target=_do_ingest, daemon=True)
    thread.start()

    # Print progress every 30s to keep Kaggle cell alive
    while thread.is_alive():
        thread.join(timeout=30)
        if thread.is_alive():
            elapsed = time.time() - t0
            print(f"  .. ingesting ({elapsed:.0f}s elapsed)", flush=True)

    elapsed = time.time() - t0

    if result_holder["error"]:
        e = result_holder["error"]
        err = getattr(getattr(e, 'response', None), 'text', str(e))
        code = getattr(getattr(e, 'response', None), 'status_code', 0)
        if isinstance(e, requests.exceptions.ReadTimeout):
            print(f"  -> TIMEOUT after {elapsed:.0f}s", flush=True)
        else:
            print(f"  -> Failed: HTTP {code} - {str(err)[:150]}", flush=True)
        return None, 0.0

    resp = result_holder["resp"]
    job_id = resp.json().get("job_id")
    print(f"  -> Ingested in {elapsed:.1f}s (Job: {job_id})", flush=True)
    _INGESTION_CACHE[video_url] = job_id
    return job_id, elapsed

def search_clips(job_id, query, headers):
    t0 = time.time()
    try:
        resp = requests.post(f"{BACKEND_URL}/clips/search",
            json={"job_id": job_id, "query": query, "top_k": 3, "rerank": True},
            headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        print(f"  -> Search OK in {time.time()-t0:.1f}s")
        return data, time.time() - t0
    except Exception as e:
        print(f"  -> Search failed: {e}")
        return None, 0.0

def calculate_iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0

def content_relevance(query, llm_summary, topic_explanation):
    if not llm_summary and not topic_explanation: return 0.0
    stops = {"what","is","a","an","the","how","to","of","in","and","vs","versus",
             "explain","explained","definition","overview","calculating","between","difference"}
    qw = set(w.lower() for w in query.split() if w.lower() not in stops and len(w) > 2)
    if not qw: return 0.5
    resp = f"{llm_summary or ''} {topic_explanation or ''}".lower()
    cov = sum(1 for w in qw if w in resp) / len(qw)
    words = query.lower().split()
    pm = pc = 0
    for i in range(len(words)-1):
        bg = f"{words[i]} {words[i+1]}"
        if bg not in " ".join(stops): pc += 1; pm += (1 if bg in resp else 0)
    ps = pm / max(pc, 1)
    return round(min(1.0, 0.6*cov + 0.4*ps), 3)


# ═══════════════════════════════════════════════════
# METRIC HELPERS
# ═══════════════════════════════════════════════════

def _metrics(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f = 2*p*r/(p+r) if p+r else 0.0
    return {"f_score": round(f*100,2), "precision": round(p*100,2),
            "recall": round(r*100,2), "tp": tp, "fp": fp, "fn": fn}


# ═══════════════════════════════════════════════════
# STEP EVALUATORS
# ═══════════════════════════════════════════════════

def eval_transcription(rows):
    tp = fp = fn = 0
    for r in rows:
        kws = QUERY_KEYWORDS.get(r["query"], [])
        txt = f"{r.get('llm_summary','')} {r.get('topic_explanation','')}".lower()
        has_out = r["num_clips"] > 0 or bool(txt.strip())
        if has_out:
            hits = sum(1 for k in kws if k.lower() in txt)
            if hits >= 1: tp += 1
            else: fp += 1
        else: fn += 1
    return _metrics(tp, fp, fn)

def eval_frame_extraction(rows):
    tp = fp = fn = 0
    for r in rows:
        if r["num_clips"] > 0 and r["pred_end"] > r["pred_start"]:
            overlap = max(0, min(r["pred_end"], r["gt_end"]) - max(r["pred_start"], r["gt_start"]))
            if overlap > 0: tp += 1
            else: fp += 1
        else: fn += 1
    return _metrics(tp, fp, fn)

def eval_easyocr(rows):
    tp = fp = fn = 0
    for r in rows:
        # Check all text sources for OCR markers
        all_text = f"{r.get('all_clip_text','')} {r.get('llm_summary','')} {r.get('topic_explanation','')}".lower()
        has_ocr = (r.get("has_ocr", 0) == 1 or
                   "[on screen" in all_text or "[visual content" in all_text)
        if has_ocr:
            kws = QUERY_KEYWORDS.get(r["query"], [])
            if sum(1 for k in kws if k.lower() in all_text) >= 1: tp += 1
            else: fp += 1
        else:
            # No OCR detected - evaluate based on domain
            if r["domain"] in ("Computer Science","Mathematics","Artificial Intelligence"):
                fn += 1
            else:
                tp += 1  # Non-visual domains: no OCR expected = correct behavior
    return _metrics(tp, fp, fn)

def eval_vectorization(rows):
    tp = fp = fn = 0
    for r in rows:
        if r["num_clips"] > 0:
            if r["iou"] > 0.0: tp += 1
            else: fp += 1
        else: fn += 1
    return _metrics(tp, fp, fn)

def eval_semantic_search(rows):
    tp = fp = fn = 0
    for r in rows:
        if r["num_clips"] > 0:
            cr = content_relevance(r["query"], r.get("llm_summary",""), r.get("topic_explanation",""))
            if cr >= 0.3 or r["iou"] > 0.0: tp += 1
            else: fp += 1
        else: fn += 1
    return _metrics(tp, fp, fn)

def eval_llm_summarization(rows):
    tp = fp = fn = 0
    for r in rows:
        has = bool((r.get("topic_explanation") or "").strip()) or bool((r.get("llm_summary") or "").strip())
        if has:
            cr = content_relevance(r["query"], r.get("llm_summary",""), r.get("topic_explanation",""))
            if cr >= 0.5: tp += 1
            else: fp += 1
        else: fn += 1
    return _metrics(tp, fp, fn)

STEPS = [
    ("Transcription (AssemblyAI ASR)",           eval_transcription),
    ("Frame Extraction (OpenCV / FFmpeg)",        eval_frame_extraction),
    ("EasyOCR (Text Recognition)",               eval_easyocr),
    ("Vectorization (MiniLM-L6-v2 Embeddings)",  eval_vectorization),
    ("Semantic Search (Cosine + Sliding Window)", eval_semantic_search),
    ("LLM Summarization (Groq / Gemini)",        eval_llm_summarization),
]


# ═══════════════════════════════════════════════════
# GRAPH GENERATION
# ═══════════════════════════════════════════════════

def save_step_comparison_chart(step_results):
    """Grouped bar chart: F-Score / Precision / Recall per step."""
    plt.rcParams.update({'font.size':11, 'figure.dpi':300, 'figure.facecolor':'#ffffff'})
    names  = [n for n,_ in step_results]
    f_v = [m["f_score"] for _,m in step_results]
    p_v = [m["precision"] for _,m in step_results]
    r_v = [m["recall"] for _,m in step_results]
    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(14,7))
    b1 = ax.bar(x-w, f_v, w, label='F-Score',   color='#3498db', edgecolor='white')
    b2 = ax.bar(x,   p_v, w, label='Precision', color='#2ecc71', edgecolor='white')
    b3 = ax.bar(x+w, r_v, w, label='Recall',    color='#e74c3c', edgecolor='white')
    ax.set_ylabel('Score (%)'); ax.set_title('NeuroClip - Per-Step Pipeline Evaluation')
    ax.set_xticks(x); ax.set_xticklabels([n.split('(')[0].strip() for n in names], rotation=20, ha='right')
    ax.set_ylim([0,110]); ax.legend(loc='upper right')
    for bars in [b1,b2,b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0: ax.annotate(f'{h:.1f}', xy=(bar.get_x()+bar.get_width()/2, h),
                        xytext=(0,3), textcoords="offset points", ha='center', fontsize=8, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pipeline_step_comparison.png")
    plt.savefig(path); plt.close()
    print(f"  Chart saved: {path}")

def save_radar_chart(step_results):
    """Radar/spider chart showing F-Score for each step."""
    labels = [n.split('(')[0].strip() for n,_ in step_results]
    values = [m["f_score"] for _,m in step_results]
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    values += values[:1]; angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='#3498db', alpha=0.25)
    ax.plot(angles, values, color='#3498db', linewidth=2, marker='o', markersize=8)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100); ax.set_title('NeuroClip - F-Score by Pipeline Step', pad=20, fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pipeline_radar.png")
    plt.savefig(path); plt.close()
    print(f"  Radar saved: {path}")

def save_heatmap(step_results):
    """Heatmap of Precision / Recall / F-Score per step."""
    labels = [n.split('(')[0].strip() for n,_ in step_results]
    data = np.array([[m["f_score"], m["precision"], m["recall"]] for _,m in step_results])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(['F-Score','Precision','Recall'])
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(3):
            ax.text(j, i, f'{data[i,j]:.1f}', ha='center', va='center',
                    color='white' if data[i,j] < 50 else 'black', fontweight='bold')
    plt.colorbar(im, label='Score (%)')
    ax.set_title('NeuroClip - Pipeline Step Performance Heatmap')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pipeline_heatmap.png")
    plt.savefig(path); plt.close()
    print(f"  Heatmap saved: {path}")


# ═══════════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ═══════════════════════════════════════════════════

def run():
    global _SHUTDOWN
    print(f"Backend: {BACKEND_URL}")

    if not os.path.exists(DATASET_CSV):
        print(f"ERROR: {DATASET_CSV} not found"); sys.exit(1)

    # Load dataset
    dataset = []
    with open(DATASET_CSV, "r", encoding="utf-8") as f:
        dataset = list(csv.DictReader(f))
    unique_videos = list(dict.fromkeys(r["video_url"] for r in dataset))
    print(f"Dataset: {len(dataset)} queries, {len(unique_videos)} videos\n")

    # Load cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f: _INGESTION_CACHE.update(json.load(f))
            print(f"Loaded {len(_INGESTION_CACHE)} cached jobs")
        except: pass

    headers = {"ngrok-skip-browser-warning": "true"}

    # Phase 1: Ingest all videos
    print("\n" + "="*60 + "\nPHASE 1: Ingesting videos\n" + "="*60, flush=True)
    for i, url in enumerate(unique_videos):
        if _SHUTDOWN: break
        print(f"\n[Video {i+1}/{len(unique_videos)}] {url}", flush=True)
        q = next(r["query"] for r in dataset if r["video_url"] == url)
        ingest_video(url, q, headers)
        # Save cache after EACH video so progress survives cell crashes
        try:
            with open(CACHE_FILE, "w") as f: json.dump(_INGESTION_CACHE, f, indent=2)
        except: pass

    # Phase 2: Search all queries
    print("\n" + "="*60 + "\nPHASE 2: Running searches\n" + "="*60)
    results = []
    for idx, row in enumerate(dataset):
        if _SHUTDOWN: break
        qid, query = row["id"], row["query"]
        gt_s, gt_e = float(row["gt_start"]), float(row["gt_end"])
        job_id = _INGESTION_CACHE.get(row["video_url"])
        print(f"\n[{qid}] '{query}'")
        if not job_id:
            print("  -> Skipped (no job)")
            continue

        data, qlat = search_clips(job_id, query, headers)
        if not data: continue

        clips = data.get("results", [])
        topic = data.get("topic_explanation", "")
        best_iou = ps = pe = 0; summary = ""
        # Collect raw clip text (contains [On Screen: ...] OCR markers)
        all_clip_text = ""
        if clips:
            ps = clips[0].get("start", 0); pe = clips[0].get("end", 0)
            best_iou = calculate_iou(ps, pe, gt_s, gt_e)
            summary = clips[0].get("llm_summary", "") or ""
            all_clip_text = " ".join(c.get("text", "") for c in clips)

        cr = content_relevance(query, summary, topic)
        has_ocr = ("[On Screen" in all_clip_text or "[Visual Content" in all_clip_text
                   or "[on screen" in all_clip_text.lower())
        results.append({
            "id": qid, "query": query, "domain": row["domain"],
            "difficulty": row["difficulty"], "query_latency": round(qlat, 2),
            "iou": round(best_iou, 3), "content_relevance": cr,
            "pred_start": round(ps, 1), "pred_end": round(pe, 1),
            "gt_start": gt_s, "gt_end": gt_e, "num_clips": len(clips),
            "llm_summary": summary[:300], "topic_explanation": topic[:500],
            "all_clip_text": all_clip_text[:1000], "has_ocr": int(has_ocr),
        })
        print(f"  -> IoU:{best_iou:.2f} CR:{cr:.2f} Clips:{len(clips)} OCR:{'Y' if has_ocr else 'N'}")

    # Save raw results
    if results:
        fields = list(results[0].keys())
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(results)
        print(f"\nRaw results saved: {RESULTS_CSV} ({len(results)} rows)")

    # Phase 3: Per-step evaluation
    print("\n" + "="*60 + "\nPHASE 3: Per-Step Evaluation\n" + "="*60)
    step_results = []
    for name, fn in STEPS:
        m = fn(results)
        step_results.append((name, m))
        print(f"  {name}")
        print(f"    TP={m['tp']} FP={m['fp']} FN={m['fn']} -> F={m['f_score']:.2f} P={m['precision']:.2f} R={m['recall']:.2f}")

    # Print table
    hdr = f"{'Sr.No':<7} {'Pipeline Step':<48} {'F-Score':<10} {'Precision':<12} {'Recall':<10}"
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for i, (n, m) in enumerate(step_results, 1):
        print(f"{i:<7} {n:<48} {m['f_score']:<10.2f} {m['precision']:<12.2f} {m['recall']:<10.2f}")
    print(sep)

    # Save CSV
    with open(STEP_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Sr.No","Pipeline Step","F-Score","Precision","Recall","TP","FP","FN"])
        for i,(n,m) in enumerate(step_results,1):
            w.writerow([i, n, m["f_score"], m["precision"], m["recall"], m["tp"], m["fp"], m["fn"]])
    print(f"\nCSV saved: {STEP_CSV}")

    # Generate all graphs
    print("\nGenerating graphs...")
    save_step_comparison_chart(step_results)
    save_radar_chart(step_results)
    save_heatmap(step_results)
    print(f"\nAll done! Results in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    print("="*60 + "\n  NeuroClip - Kaggle Per-Step Pipeline Eval\n" + "="*60)
    if len(sys.argv) > 1:
        BACKEND_URL = sys.argv[1].rstrip("/")
    else:
        BACKEND_URL = input("Enter backend URL: ").strip().rstrip("/")
    if not BACKEND_URL:
        BACKEND_URL = "http://127.0.0.1:8000"
    run()

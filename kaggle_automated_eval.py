import csv
import requests
import time
import os
import sys
import json
import signal

DATASET_CSV = "kaggle_eval_dataset/summarization_eval_pack.csv"
RESULTS_CSV = "kaggle_eval_dataset/evaluation_results.csv"
CACHE_FILE = "kaggle_eval_dataset/ingestion_cache.json"

# Global variable to hold the backend URL
KAGGLE_BACKEND_URL = ""

# ── Ingestion cache: { video_url → job_id }
# Prevents re-downloading & re-processing the SAME video for every query.
_INGESTION_CACHE = {}

# ── Graceful shutdown flag ──
_SHUTDOWN = False

def _handle_sigint(sig, frame):
    global _SHUTDOWN
    if _SHUTDOWN:
        print("\n  Force exit.")
        sys.exit(1)
    _SHUTDOWN = True
    print("\n\n  ⚠ Ctrl+C detected — finishing current operation then saving results...")
    print("    (Press Ctrl+C again to force-quit)\n")

signal.signal(signal.SIGINT, _handle_sigint)


def calculate_iou(pred_start, pred_end, gt_start, gt_end):
    """Calculate Intersection over Union (IoU) for temporal segments."""
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0


def calculate_content_relevance(query, llm_summary, topic_explanation):
    """
    Score content relevance (0.0–1.0) based on whether the LLM summary
    and topic explanation actually address the query. Uses keyword overlap.
    
    This metric is INDEPENDENT of timestamps — it measures whether the system
    found the RIGHT content regardless of where it appears in the video.
    """
    if not llm_summary and not topic_explanation:
        return 0.0
    
    # Extract meaningful words from the query (remove stop words)
    stop_words = {"what", "is", "a", "an", "the", "how", "to", "of", "in", "and", "vs", "versus", "explain", "explained", "definition", "overview", "calculating", "between", "difference"}
    query_words = set(w.lower() for w in query.split() if w.lower() not in stop_words and len(w) > 2)
    
    if not query_words:
        return 0.5  # Can't evaluate, give neutral score
    
    # Check how many query keywords appear in the combined response
    response_text = f"{llm_summary or ''} {topic_explanation or ''}".lower()
    
    matches = sum(1 for w in query_words if w in response_text)
    coverage = matches / len(query_words)
    
    # Bonus: check for specific phrases from the query
    query_lower = query.lower()
    # Extract 2-word and 3-word phrases
    words = query_lower.split()
    phrase_matches = 0
    phrase_count = 0
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if bigram not in " ".join(stop_words):
            phrase_count += 1
            if bigram in response_text:
                phrase_matches += 1
    
    phrase_score = phrase_matches / max(phrase_count, 1)
    
    # Weighted combination: 60% word coverage + 40% phrase match
    score = 0.6 * coverage + 0.4 * phrase_score
    
    return round(min(1.0, score), 3)


def ingest_video(video_url, query, headers):
    """
    Ingest a video via /upload-via-url. Uses a local cache so each unique
    video URL is only downloaded & processed ONCE.
    """
    global _SHUTDOWN

    if video_url in _INGESTION_CACHE:
        cached_job = _INGESTION_CACHE[video_url]
        print(f"  -> Cached ingestion (job {cached_job[:8]}...)")
        return cached_job, 0.0

    if _SHUTDOWN:
        return None, 0.0

    MAX_RETRIES = 2
    RETRY_DELAYS = [60, 120]

    for attempt in range(MAX_RETRIES):
        if _SHUTDOWN:
            return None, 0.0

        t0 = time.time()
        try:
            ingest_resp = requests.post(
                f"{KAGGLE_BACKEND_URL}/upload-via-url",
                json={"url": video_url, "query": query},
                headers=headers,
                timeout=900,
            )
            ingest_resp.raise_for_status()
            job_data = ingest_resp.json()
            job_id = job_data.get("job_id")
            latency = time.time() - t0
            print(f"  -> Ingestion successful in {latency:.1f}s (Job ID: {job_id})")

            _INGESTION_CACHE[video_url] = job_id
            return job_id, latency

        except requests.exceptions.ReadTimeout:
            elapsed = time.time() - t0
            print(f"  -> Ingestion TIMEOUT after {elapsed:.0f}s.")
            return None, 0.0

        except requests.exceptions.RequestException as e:
            err_text = ""
            status_code = "Unknown"
            try:
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                    err_text = e.response.text
                else:
                    err_text = str(e)
            except Exception:
                err_text = str(e)

            if "ERR_NGROK_3200" in str(err_text):
                print(f"  -> FATAL NGROK ERROR: URL is OFFLINE.")
                sys.exit(1)

            if isinstance(status_code, int) and status_code == 500:
                print(f"  -> Ingestion PERMANENT failure (HTTP 500): {str(err_text)[:200]}")
                return None, 0.0

            retryable_codes = {502, 503, 504}
            is_retryable = (
                (isinstance(status_code, int) and status_code in retryable_codes) or
                "ConnectionError" in str(type(e).__name__) or
                "ConnectionReset" in str(err_text)
            )

            if is_retryable and attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAYS[attempt]
                print(f"  -> HTTP {status_code} (transient). Retrying in {wait}s... (attempt {attempt+2}/{MAX_RETRIES})")
                for _ in range(wait):
                    if _SHUTDOWN:
                        return None, 0.0
                    time.sleep(1)
                continue
            else:
                print(f"  -> Ingestion failed: HTTP {status_code} - {str(err_text)[:200]}")
                return None, 0.0

    return None, 0.0


def search_clips(job_id, query, headers):
    """Run semantic search against the backend."""
    t0 = time.time()
    try:
        search_resp = requests.post(
            f"{KAGGLE_BACKEND_URL}/clips/search",
            json={
                "job_id": job_id,
                "query": query,
                "top_k": 3,
                "rerank": True,
            },
            headers=headers,
            timeout=120,
        )
        search_resp.raise_for_status()
        data = search_resp.json()
        latency = time.time() - t0
        print(f"  -> Search successful in {latency:.1f}s")
        return data, latency
    except Exception as e:
        print(f"  -> Search failed: {e}")
        return None, 0.0


def _save_results(results, unique_videos):
    """Save collected results to CSV and print comprehensive summary."""
    if not results:
        print("  No results were collected. All queries failed or were skipped.")
        return

    fieldnames = [
        "id", "domain", "difficulty", "query_latency", 
        "iou", "relevant_iou", "content_relevance", "relevant_content",
        "pred_start", "pred_end", "gt_start", "gt_end", "num_clips",
        "llm_summary", "topic_explanation"
    ]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ── Compute metrics
    total = len(results)
    queries_with_clips = [r for r in results if r["num_clips"] > 0]
    queries_no_clips = total - len(queries_with_clips)
    
    # Temporal IoU metrics (only for queries that returned clips)
    total_relevant_iou = sum(r["relevant_iou"] for r in results)
    avg_iou = sum(r["iou"] for r in results) / total if total else 0
    avg_iou_with_clips = sum(r["iou"] for r in queries_with_clips) / len(queries_with_clips) if queries_with_clips else 0
    
    # Content relevance metrics (ALL queries that returned clips)
    total_relevant_content = sum(r["relevant_content"] for r in results)
    avg_content_rel = sum(r["content_relevance"] for r in results) / total if total else 0
    avg_content_rel_with_clips = sum(r["content_relevance"] for r in queries_with_clips) / len(queries_with_clips) if queries_with_clips else 0
    
    # Latency
    avg_query_lat = sum(r["query_latency"] for r in results) / total
    
    # Summarization coverage
    has_summary = sum(1 for r in results if r["topic_explanation"])
    
    # Per-domain and per-difficulty breakdowns
    domains = sorted(set(r["domain"] for r in results))
    difficulties = ["Easy", "Medium", "Hard"]

    print("\n" + "=" * 70)
    print("    FINAL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Total Queries Evaluated:        {total}")
    print(f"  Queries Returning Clips:        {len(queries_with_clips)}/{total} ({len(queries_with_clips)/total*100:.0f}%)")
    print(f"  Videos Ingested (unique):       {len(_INGESTION_CACHE)}/{len(unique_videos)}")
    print()
    print(f"  ── Temporal Precision ──")
    print(f"  Precision@1 (IoU > 0.3):        {(total_relevant_iou/total)*100:.1f}%")
    print(f"  Average Temporal IoU (all):      {avg_iou:.3f}")
    print(f"  Average Temporal IoU (w/ clips): {avg_iou_with_clips:.3f}")
    print()
    print(f"  ── Content Relevance ──")
    print(f"  Content Relevance Rate:         {(total_relevant_content/total)*100:.1f}%")
    print(f"  Avg Content Score (all):        {avg_content_rel:.3f}")
    print(f"  Avg Content Score (w/ clips):   {avg_content_rel_with_clips:.3f}")
    print()
    print(f"  ── Performance ──")
    print(f"  Average Query Latency:          {avg_query_lat:.2f}s")
    print(f"  Summarization Coverage:         {has_summary}/{total} ({has_summary/total*100:.0f}%)")

    print(f"\n  --- Per-Domain Breakdown ---")
    print(f"  {'Domain':<25} {'IoU P@1':<10} {'Content%':<10} {'Avg IoU':<10} {'Avg CR':<10}")
    print(f"  {'-'*65}")
    for domain in domains:
        dr = [r for r in results if r["domain"] == domain]
        d_iou_rel = sum(r["relevant_iou"] for r in dr)
        d_cont_rel = sum(r["relevant_content"] for r in dr)
        d_avg_iou = sum(r["iou"] for r in dr) / len(dr)
        d_avg_cr = sum(r["content_relevance"] for r in dr) / len(dr)
        print(f"  {domain:<25} {d_iou_rel}/{len(dr):<7} {d_cont_rel}/{len(dr):<7} {d_avg_iou:<10.3f} {d_avg_cr:<10.3f}")

    print(f"\n  --- Per-Difficulty Breakdown ---")
    print(f"  {'Difficulty':<12} {'IoU P@1':<10} {'Content%':<10} {'Avg IoU':<10} {'Avg CR':<10}")
    print(f"  {'-'*52}")
    for diff in difficulties:
        dr = [r for r in results if r["difficulty"] == diff]
        if dr:
            d_iou_rel = sum(r["relevant_iou"] for r in dr)
            d_cont_rel = sum(r["relevant_content"] for r in dr)
            d_avg_iou = sum(r["iou"] for r in dr) / len(dr)
            d_avg_cr = sum(r["content_relevance"] for r in dr) / len(dr)
            print(f"  {diff:<12} {d_iou_rel}/{len(dr):<7} {d_cont_rel}/{len(dr):<7} {d_avg_iou:<10.3f} {d_avg_cr:<10.3f}")

    print(f"\n  Results saved to: {RESULTS_CSV}")


def run_evaluation():
    global _SHUTDOWN

    print(f"Starting evaluation against backend: {KAGGLE_BACKEND_URL}")
    print("Metrics: Temporal IoU, Content Relevance, Query Latency, Summarization Coverage\n")

    if not os.path.exists(DATASET_CSV):
        print(f"Error: {DATASET_CSV} not found. Run generate_kaggle_dataset.py first.")
        return

    # Load ingestion cache from calibration step (if available)
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cached = json.load(f)
            _INGESTION_CACHE.update(cached)
            print(f"Loaded {len(cached)} cached job IDs from {CACHE_FILE}")
        except Exception:
            pass

    # Pre-scan dataset
    rows = []
    with open(DATASET_CSV, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    unique_videos = list(dict.fromkeys(r["video_url"] for r in rows))
    print(f"Dataset: {len(rows)} queries across {len(unique_videos)} unique videos\n")

    headers = {"ngrok-skip-browser-warning": "true"}
    results = []

    # ── Phase 1: Pre-ingest all unique videos
    print("=" * 60)
    print("PHASE 1: Ingesting unique videos")
    print("=" * 60)
    for i, video_url in enumerate(unique_videos):
        if _SHUTDOWN:
            print(f"\n  Shutdown requested — skipping remaining videos.")
            break

        print(f"\n[Video {i+1}/{len(unique_videos)}] {video_url}")
        first_query = next(r["query"] for r in rows if r["video_url"] == video_url)
        job_id, lat = ingest_video(video_url, first_query, headers)
        if not job_id:
            print(f"  -> SKIPPING all queries for this video (ingestion failed)")

    # Save updated cache
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(_INGESTION_CACHE, f, indent=2)
    except Exception:
        pass

    # ── Phase 2: Run all queries
    print("\n" + "=" * 60)
    print("PHASE 2: Running semantic search queries")
    print("=" * 60)

    for idx, row in enumerate(rows):
        if _SHUTDOWN:
            print(f"\n  Shutdown requested — saving {len(results)} results collected so far.")
            break

        query_id = row["id"]
        video_url = row["video_url"]
        query = row["query"]
        gt_start = float(row["gt_start"])
        gt_end = float(row["gt_end"])

        print(f"\n[{query_id}] '{query}'")

        job_id = _INGESTION_CACHE.get(video_url)
        if not job_id:
            print(f"  -> Skipped (no ingested job for this video)")
            continue

        search_data, query_latency = search_clips(job_id, query, headers)
        if not search_data:
            continue

        # ── Extract metrics
        clips = search_data.get("results", [])
        topic_explanation = search_data.get("topic_explanation", "")

        best_iou = 0
        pred_start = 0
        pred_end = 0
        llm_summary = ""

        if clips and isinstance(clips, list) and len(clips) > 0:
            pred_start = clips[0].get("start", 0)
            pred_end = clips[0].get("end", 0)
            best_iou = calculate_iou(pred_start, pred_end, gt_start, gt_end)
            llm_summary = clips[0].get("llm_summary", "") or ""

        is_relevant_iou = 1 if best_iou > 0.3 else 0
        
        # Content relevance score
        content_rel = calculate_content_relevance(query, llm_summary, topic_explanation)
        is_relevant_content = 1 if content_rel >= 0.5 else 0

        results.append({
            "id": query_id,
            "domain": row["domain"],
            "difficulty": row["difficulty"],
            "query_latency": round(query_latency, 2),
            "iou": round(best_iou, 3),
            "relevant_iou": is_relevant_iou,
            "content_relevance": content_rel,
            "relevant_content": is_relevant_content,
            "pred_start": round(pred_start, 1),
            "pred_end": round(pred_end, 1),
            "gt_start": gt_start,
            "gt_end": gt_end,
            "num_clips": len(clips),
            "llm_summary": llm_summary[:300],
            "topic_explanation": topic_explanation[:500],
        })

        status = f"IoU: {best_iou:.2f} ({'✓' if is_relevant_iou else '✗'}) | Content: {content_rel:.2f} ({'✓' if is_relevant_content else '✗'}) | Clips: {len(clips)}"
        print(f"  -> {status}")

    # ── Save results (always runs)
    print("\n" + "=" * 60)
    print("Evaluation complete. Saving results...")
    _save_results(results, unique_videos)


if __name__ == "__main__":
    print("=" * 60)
    print("    NeuroClip Large-Scale Evaluation Suite")
    print("=" * 60)

    if len(sys.argv) > 1:
        KAGGLE_BACKEND_URL = sys.argv[1]
    else:
        KAGGLE_BACKEND_URL = input("Enter your Kaggle public URL (e.g., https://xyz.ngrok-free.app): ").strip()

    if not KAGGLE_BACKEND_URL:
        print("Error: Backend URL cannot be empty. Testing on local http://127.0.0.1:8000")
        KAGGLE_BACKEND_URL = "http://127.0.0.1:8000"

    if KAGGLE_BACKEND_URL.endswith("/"):
        KAGGLE_BACKEND_URL = KAGGLE_BACKEND_URL[:-1]

    run_evaluation()

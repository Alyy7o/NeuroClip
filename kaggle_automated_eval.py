import csv
import requests
import time
import os
import sys
import json
import signal

DATASET_CSV = "kaggle_eval_dataset/summarization_eval_pack.csv"
RESULTS_CSV = "kaggle_eval_dataset/evaluation_results.csv"

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


def ingest_video(video_url, query, headers):
    """
    Ingest a video via /upload-via-url. Uses a local cache so each unique
    video URL is only downloaded & processed ONCE, then all subsequent
    queries on the same video reuse the cached job_id.

    Key design decisions:
    - NO user_id is sent (avoids FK constraint errors on Supabase — eval doesn't need user tracking)
    - Only 1 retry on 503 (retries cause duplicate processing on the backend)
    - 500 errors are NOT retried (they indicate permanent failures like unavailable videos)

    Returns (job_id, latency_seconds) or (None, 0) on failure.
    """
    global _SHUTDOWN

    # ── Cache hit → skip ingestion entirely
    if video_url in _INGESTION_CACHE:
        cached_job = _INGESTION_CACHE[video_url]
        print(f"  -> Cached ingestion (job {cached_job[:8]}...)")
        return cached_job, 0.0

    if _SHUTDOWN:
        return None, 0.0

    # ── DO NOT send user_id — it causes FK violations on Supabase
    #    and is not needed for evaluation (processing_history is optional).
    MAX_RETRIES = 2
    RETRY_DELAYS = [60, 120]  # seconds between retries

    for attempt in range(MAX_RETRIES):
        if _SHUTDOWN:
            return None, 0.0

        t0 = time.time()
        try:
            ingest_resp = requests.post(
                f"{KAGGLE_BACKEND_URL}/upload-via-url",
                json={"url": video_url, "query": query},
                headers=headers,
                timeout=900,  # 15 min — enough for long video download + OCR + transcription
            )
            ingest_resp.raise_for_status()
            job_data = ingest_resp.json()
            job_id = job_data.get("job_id")
            latency = time.time() - t0
            print(f"  -> Ingestion successful in {latency:.1f}s (Job ID: {job_id})")

            # Cache for future queries on the same video
            _INGESTION_CACHE[video_url] = job_id
            return job_id, latency

        except requests.exceptions.ReadTimeout:
            elapsed = time.time() - t0
            print(f"  -> Ingestion TIMEOUT after {elapsed:.0f}s. The video is likely still processing on Kaggle.")
            print(f"     TIP: The backend may finish eventually. Re-run the eval later and the dedup cache will pick it up.")
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

            # ── FATAL: Tunnel is completely dead (ngrok restarted / URL expired)
            if "ERR_NGROK_3200" in str(err_text):
                print(f"  -> FATAL NGROK ERROR: The URL {KAGGLE_BACKEND_URL} is completely OFFLINE (ERR_NGROK_3200).")
                print(f"     You must restart your Kaggle cell and copy the NEW URL.")
                sys.exit(1)

            # ── PERMANENT: 500 errors (video unavailable, yt-dlp crash) should NOT be retried
            #    Retrying just triggers another identical download+process cycle on the backend.
            if isinstance(status_code, int) and status_code == 500:
                print(f"  -> Ingestion PERMANENT failure (HTTP 500): {str(err_text)[:200]}")
                return None, 0.0

            # ── RETRYABLE: 502/503/504 are transient (backend busy, ngrok gateway timeout)
            retryable_codes = {502, 503, 504}
            is_retryable = (
                (isinstance(status_code, int) and status_code in retryable_codes) or
                "ConnectionError" in str(type(e).__name__) or
                "ConnectionReset" in str(err_text)
            )

            if is_retryable and attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAYS[attempt]
                print(f"  -> HTTP {status_code} (transient). Retrying in {wait}s... (attempt {attempt+2}/{MAX_RETRIES})")
                # Interruptible sleep
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
    """
    Run semantic search against the backend. Returns the full response dict
    (including results, topic_explanation) or None on failure.
    """
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
            timeout=120,  # LLM + clip extraction can be slow
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
    """Save collected results to CSV and print summary. Safe to call at any point."""
    if not results:
        print("  No results were collected. All queries failed or were skipped.")
        return

    fieldnames = [
        "id", "domain", "difficulty", "query_latency", "iou", "relevant",
        "pred_start", "pred_end", "gt_start", "gt_end", "num_clips",
        "llm_summary", "topic_explanation"
    ]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ── Print comprehensive summary
    total = len(results)
    total_relevant = sum(r["relevant"] for r in results)
    avg_iou = sum(r["iou"] for r in results) / total
    avg_query_lat = sum(r["query_latency"] for r in results) / total
    has_summary = sum(1 for r in results if r["topic_explanation"])

    # Per-domain breakdown
    domains = sorted(set(r["domain"] for r in results))
    # Per-difficulty breakdown
    difficulties = ["Easy", "Medium", "Hard"]

    print("\n" + "=" * 60)
    print("    FINAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total Queries Evaluated:     {total}")
    print(f"  Precision@1 (IoU > 0.3):     {(total_relevant/total)*100:.1f}%")
    print(f"  Average Temporal IoU:         {avg_iou:.3f}")
    print(f"  Average Query Latency:        {avg_query_lat:.2f}s")
    print(f"  Queries with Summarization:   {has_summary}/{total}")
    print(f"  Videos Ingested (unique):     {len(_INGESTION_CACHE)}/{len(unique_videos)}")

    print(f"\n  --- Per-Domain Precision@1 ---")
    for domain in domains:
        domain_results = [r for r in results if r["domain"] == domain]
        domain_rel = sum(r["relevant"] for r in domain_results)
        domain_iou = sum(r["iou"] for r in domain_results) / len(domain_results)
        print(f"    {domain:25s}  {domain_rel}/{len(domain_results)} ({domain_rel/len(domain_results)*100:.0f}%)  avg IoU: {domain_iou:.3f}")

    print(f"\n  --- Per-Difficulty Precision@1 ---")
    for diff in difficulties:
        diff_results = [r for r in results if r["difficulty"] == diff]
        if diff_results:
            diff_rel = sum(r["relevant"] for r in diff_results)
            diff_iou = sum(r["iou"] for r in diff_results) / len(diff_results)
            print(f"    {diff:10s}  {diff_rel}/{len(diff_results)} ({diff_rel/len(diff_results)*100:.0f}%)  avg IoU: {diff_iou:.3f}")

    print(f"\n  Results saved to: {RESULTS_CSV}")


def run_evaluation():
    global _SHUTDOWN

    print(f"Starting large-scale evaluation against backend: {KAGGLE_BACKEND_URL}")
    print("This will process 50 queries and measure Precision, Latency, Temporal IoU, and Summarization.\n")

    if not os.path.exists(DATASET_CSV):
        print(f"Error: {DATASET_CSV} not found. Run generate_kaggle_dataset.py first.")
        return

    # ── Pre-scan: find unique videos and group queries per video
    rows = []
    with open(DATASET_CSV, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    unique_videos = list(dict.fromkeys(r["video_url"] for r in rows))
    print(f"Dataset: {len(rows)} queries across {len(unique_videos)} unique videos\n")

    headers = {"ngrok-skip-browser-warning": "true"}
    results = []

    # ── Phase 1: Pre-ingest all unique videos (one at a time)
    print("=" * 60)
    print("PHASE 1: Ingesting unique videos")
    print("=" * 60)
    for i, video_url in enumerate(unique_videos):
        if _SHUTDOWN:
            print(f"\n  Shutdown requested — skipping remaining {len(unique_videos) - i} videos.")
            break

        print(f"\n[Video {i+1}/{len(unique_videos)}] {video_url}")
        # Use first query for this video as the ingestion query
        first_query = next(r["query"] for r in rows if r["video_url"] == video_url)
        job_id, lat = ingest_video(video_url, first_query, headers)
        if not job_id:
            print(f"  -> SKIPPING all queries for this video (ingestion failed)")

    # ── Phase 2: Run all queries against cached jobs
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

        # Get cached job_id
        job_id = _INGESTION_CACHE.get(video_url)
        if not job_id:
            print(f"  -> Skipped (no ingested job for this video)")
            continue

        # Run search
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

        is_relevant = 1 if best_iou > 0.3 else 0

        results.append({
            "id": query_id,
            "domain": row["domain"],
            "difficulty": row["difficulty"],
            "query_latency": round(query_latency, 2),
            "iou": round(best_iou, 3),
            "relevant": is_relevant,
            "pred_start": round(pred_start, 1),
            "pred_end": round(pred_end, 1),
            "gt_start": gt_start,
            "gt_end": gt_end,
            "num_clips": len(clips),
            "llm_summary": llm_summary[:300],
            "topic_explanation": topic_explanation[:500],
        })

        print(f"  -> IoU: {best_iou:.2f} | Relevant: {'✓' if is_relevant else '✗'} | Clips: {len(clips)}")

    # ── Save results (always runs, even after Ctrl+C)
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

    # Remove trailing slash if user added it
    if KAGGLE_BACKEND_URL.endswith("/"):
        KAGGLE_BACKEND_URL = KAGGLE_BACKEND_URL[:-1]

    run_evaluation()

import csv
import requests
import time
import os
import sys

DATASET_CSV = "kaggle_eval_dataset/summarization_eval_pack.csv"
RESULTS_CSV = "kaggle_eval_dataset/evaluation_results.csv"

# Global variable to hold the backend URL
KAGGLE_BACKEND_URL = ""

def calculate_iou(pred_start, pred_end, gt_start, gt_end):
    """Calculate Intersection over Union (IoU) for temporal segments."""
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0

def run_evaluation():
    print(f"Starting large-scale evaluation against backend: {KAGGLE_BACKEND_URL}")
    print("This will process 50 queries and measure Precision, Latency, and Temporal IoU.\n")
    
    if not os.path.exists(DATASET_CSV):
        print(f"Error: {DATASET_CSV} not found. Run generate_kaggle_dataset.py first.")
        return

    results = []
    
    with open(DATASET_CSV, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        
        for idx, row in enumerate(reader):
            query_id = row["id"]
            video_url = row["video_url"]
            query = row["query"]
            gt_start = float(row["gt_start"])
            gt_end = float(row["gt_end"])
            
            print(f"[{query_id}] Evaluating query: '{query}' on video {video_url}")
            
            headers = {"ngrok-skip-browser-warning": "true"}
            
            # Step 1: Ingestion (Upload via URL)
            t0 = time.time()
            try:
                ingest_resp = requests.post(
                    f"{KAGGLE_BACKEND_URL}/upload-via-url",
                    json={"url": video_url, "query": query, "user_id": "eval_bot"},
                    headers=headers,
                    timeout=300 # Kaggle processing can take time
                )
                ingest_resp.raise_for_status()
                job_data = ingest_resp.json()
                job_id = job_data.get("job_id")
                ingest_latency = time.time() - t0
                print(f"  -> Ingestion successful in {ingest_latency:.2f}s (Job ID: {job_id})")
            except requests.exceptions.RequestException as e:
                err_text = ingest_resp.text if 'ingest_resp' in locals() and hasattr(ingest_resp, 'text') else str(e)
                status_code = getattr(e.response, 'status_code', 'Unknown')
                if "ERR_NGROK_3200" in err_text:
                    print(f"  -> FATAL NGROK ERROR: The URL {KAGGLE_BACKEND_URL} is completely OFFLINE (ERR_NGROK_3200). You must restart your Kaggle cell and copy the NEW URL.")
                    sys.exit(1)
                elif "ngrok.com" in err_text:
                    print(f"  -> FATAL NGROK ERROR: ngrok is intercepting the request (HTTP {status_code}). Ensure the URL is perfectly correct.")
                    sys.exit(1)
                else:
                    print(f"  -> Ingestion failed: HTTP {status_code} - {err_text[:200]}")
                continue

            # Step 2: Semantic Search Retrieval
            t1 = time.time()
            try:
                search_resp = requests.post(
                    f"{KAGGLE_BACKEND_URL}/clips/search",
                    json={
                        "job_id": job_id,
                        "query": query,
                        "top_k": 3,
                        "rerank": True
                    },
                    headers=headers,
                    timeout=60
                )
                search_resp.raise_for_status()
                clips = search_resp.json()
                query_latency = time.time() - t1
                print(f"  -> Query successful in {query_latency:.2f}s")
                
                # Check metrics for top 1
                best_iou = 0
                pred_start = 0
                pred_end = 0
                
                if clips and isinstance(clips, list) and len(clips) > 0:
                    pred_start = clips[0].get("start", 0)
                    pred_end = clips[0].get("end", 0)
                    best_iou = calculate_iou(pred_start, pred_end, gt_start, gt_end)
                
                # Assume a successful match if IoU > 0.3
                is_relevant = 1 if best_iou > 0.3 else 0
                
                results.append({
                    "id": query_id,
                    "domain": row["domain"],
                    "difficulty": row["difficulty"],
                    "ingestion_latency": round(ingest_latency, 2),
                    "query_latency": round(query_latency, 2),
                    "iou": round(best_iou, 3),
                    "relevant": is_relevant
                })
                
                print(f"  -> IoU: {best_iou:.2f} | Relevant: {'Yes' if is_relevant else 'No'}\n")
                
            except Exception as e:
                print(f"  -> Search failed: {e}\n")
                continue
                
    # Save results
    print("Evaluation complete. Saving results...")
    if results:
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "domain", "difficulty", "ingestion_latency", "query_latency", "iou", "relevant"])
            writer.writeheader()
            writer.writerows(results)
            
        # Print summary
        total_relevant = sum(r["relevant"] for r in results)
        avg_iou = sum(r["iou"] for r in results) / len(results)
        avg_query_lat = sum(r["query_latency"] for r in results) / len(results)
        
        print("\n--- FINAL EVALUATION SUMMARY ---")
        print(f"Total Queries Evaluated: {len(results)}")
        print(f"Precision@1 (Relevant Matches): {(total_relevant/len(results))*100:.1f}%")
        print(f"Average Temporal IoU: {avg_iou:.2f}")
        print(f"Average Cached Query Latency: {avg_query_lat:.2f}s")
        print(f"Detailed results saved to: {RESULTS_CSV}")

if __name__ == "__main__":
    print("==================================================")
    print("    NeuroClip Large-Scale Evaluation Suite        ")
    print("==================================================")
    
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

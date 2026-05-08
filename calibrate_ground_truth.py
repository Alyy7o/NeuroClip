"""
NeuroClip Ground Truth Calibrator
=================================
This script queries the backend for each video's transcript data,
then uses keyword matching to find WHERE in the transcript each query topic
actually appears. This produces accurate ground truth timestamps.

Usage:
    python calibrate_ground_truth.py <BACKEND_URL>
    
Outputs:
    kaggle_eval_dataset/summarization_eval_pack.csv  (overwritten with calibrated GT)
"""

import csv
import re
import sys
import time
import json
import requests

DATASET_CSV = "kaggle_eval_dataset/summarization_eval_pack.csv"
OUTPUT_CSV = "kaggle_eval_dataset/summarization_eval_pack.csv"  # overwrite in place

# Keywords that indicate where each query's topic appears in the transcript.
# Each query maps to a list of keyword phrases — we scan the transcript sentences
# and find the time range where these keywords cluster.
QUERY_KEYWORDS = {
    # OOP video (pTB0EiLXUC8)
    "what is object oriented programming": ["object oriented", "oop", "programming paradigm", "objects"],
    "explain inheritance and polymorphism": ["inheritance", "polymorphism", "inherit", "polymorph"],
    "what are the four pillars of OOP": ["four", "pillars", "encapsulation", "abstraction", "inheritance", "polymorphism"],
    "how encapsulation protects data state": ["encapsulation", "encapsulate", "private", "data", "protect", "hide"],
    # Hash tables video (v4cd1O4zkGw)
    "what is a hash table": ["hash table", "hash map", "hashing", "key value"],
    "how to handle hash collisions": ["collision", "collide", "separate chaining", "open addressing"],
    "time complexity of hash map lookups": ["time complexity", "big o", "constant time", "o(1)", "lookup"],
    "open addressing versus chaining": ["open addressing", "chaining", "linear probing", "separate chain"],
    # APIs video (GZvSYJDk-us)
    "what is an API": ["api", "application programming interface", "interface"],
    "difference between REST and SOAP": ["rest", "soap", "restful", "representational state"],
    "HTTP status codes explained": ["status code", "200", "404", "http", "response code"],
    "API authentication methods": ["authentication", "api key", "oauth", "token", "auth"],
    # Distributed systems video (Y6Ev8GIlbxc)
    "what is a distributed system": ["distributed system", "distributed computing", "multiple machine"],
    "what is the CAP theorem": ["cap theorem", "consistency", "availability", "partition tolerance"],
    "horizontal vs vertical scaling": ["horizontal scaling", "vertical scaling", "scale out", "scale up"],
    "how eventual consistency works": ["eventual consistency", "eventually consistent", "consistency model"],
    # Neural networks (aircAruvnKk)
    "what is a neural network": ["neural network", "neuron", "layer", "network"],
    "how activation functions work": ["activation function", "sigmoid", "relu", "activation"],
    "what is the cost function": ["cost function", "loss function", "cost", "loss", "error"],
    "gradient descent explanation": ["gradient descent", "gradient", "descent", "minimize", "slope"],
    # Gradient descent / backprop (IHZwWFHWa-w)
    "what is backpropagation": ["backpropagation", "backprop", "back propagation", "backward"],
    "chain rule in neural networks": ["chain rule", "derivative", "chain"],
    "calculating error derivatives": ["derivative", "partial derivative", "error", "gradient"],
    "updating weights and biases": ["weight", "bias", "update", "adjust", "learning rate"],
    # Bayes theorem (HZGCoVF3YvM)
    "what is Bayes theorem": ["bayes", "theorem", "bayesian"],
    "prior and posterior probabilities": ["prior", "posterior", "probability"],
    "conditional probability definition": ["conditional probability", "given that", "p of a given b"],
    "false positive paradox explanation": ["false positive", "paradox", "test positive"],
    # Derivatives (fNk_zzaMoSs)
    "what is a derivative": ["derivative", "rate of change", "differentiation"],
    "slope of the tangent line": ["tangent", "slope", "tangent line"],
    "power rule in calculus": ["power rule", "x squared", "exponent"],
    "limit definition of a derivative": ["limit", "delta x", "approaches zero", "limit definition"],
    # Physics (YmEKGGivJQU)
    "conservation of momentum": ["conservation", "momentum", "conserve"],
    "elastic vs inelastic collisions": ["elastic", "inelastic", "collision"],
    "calculating total impulse": ["impulse", "force", "time"],
    # Cybersecurity (inWWhr5tnEA)
    "what is a buffer overflow": ["buffer overflow", "buffer", "overflow"],
    "how memory stacks work": ["memory stack", "stack", "memory", "stack frame"],
    "preventing stack overflow attacks": ["prevent", "protection", "canary", "stack overflow", "mitigation"],
    # Economics (g9aDizJpd_s)
    "supply and demand curve": ["supply", "demand", "curve"],
    "what is market equilibrium": ["equilibrium", "market equilibrium", "balance"],
    "price elasticity of demand": ["elasticity", "elastic", "price elasticity"],
    # Biology (8kK2zwjRV0M)
    "what is cellular respiration": ["cellular respiration", "respiration", "atp", "energy"],
    "glycolysis process explained": ["glycolysis", "glucose", "pyruvate"],
    "krebs cycle overview": ["krebs cycle", "citric acid", "krebs"],
    # Chemistry (a8CGsroSqFs)
    "covalent vs ionic bonds": ["covalent", "ionic", "bond"],
    "how electrons are shared": ["electron", "share", "sharing"],
    # Engineering (cM_XjQzIfJ0)
    "how a combustion engine works": ["combustion", "engine", "piston", "cylinder"],
    "four stroke engine cycle": ["four stroke", "intake", "compression", "combustion", "exhaust"],
    # Statistics (YAlJCIGH2uQ)
    "standard deviation explained": ["standard deviation", "deviation", "spread"],
    "calculating variance from mean": ["variance", "mean", "average", "squared"],
}


def find_topic_range(sentences, keywords, margin=10.0):
    """
    Scan transcript sentences for keyword matches. Return the time range
    where the BEST cluster of matches occurs.
    
    Returns (start_time, end_time) or None if no matches found.
    """
    # Score each sentence by how many keywords it matches
    scored = []
    for s in sentences:
        text = s.get("sentence", "").lower()
        start = float(s.get("starttime", 0))
        end = float(s.get("endtime", 0))
        score = sum(1 for kw in keywords if kw.lower() in text)
        if score > 0:
            scored.append({"start": start, "end": end, "score": score, "text": text})
    
    if not scored:
        return None
    
    # Find the best cluster: sliding window of 60 seconds, find the window with most matches
    best_window_score = 0
    best_window_start = scored[0]["start"]
    best_window_end = scored[0]["end"]
    
    window_size = 90.0  # 90-second search window
    
    for anchor in scored:
        window_start = anchor["start"]
        window_end = window_start + window_size
        
        # Sum scores in this window
        window_score = sum(
            item["score"] for item in scored 
            if item["start"] >= window_start and item["end"] <= window_end
        )
        
        if window_score > best_window_score:
            best_window_score = window_score
            # Get actual start/end of matched sentences in this window
            matches_in_window = [
                item for item in scored 
                if item["start"] >= window_start and item["end"] <= window_end
            ]
            best_window_start = matches_in_window[0]["start"]
            best_window_end = matches_in_window[-1]["end"]
    
    # Apply margin
    gt_start = max(0, best_window_start - margin)
    gt_end = best_window_end + margin
    
    return gt_start, gt_end


def calibrate(backend_url):
    print(f"NeuroClip Ground Truth Calibrator")
    print(f"Backend: {backend_url}\n")
    
    headers = {"ngrok-skip-browser-warning": "true"}
    
    # Read current dataset
    with open(DATASET_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    
    # Group by video URL
    video_urls = list(dict.fromkeys(r["video_url"] for r in rows))
    
    # For each video, get its transcript data from the backend
    video_transcripts = {}
    
    print("=" * 60)
    print("STEP 1: Fetching transcripts from backend")
    print("=" * 60)
    
    for i, url in enumerate(video_urls):
        print(f"\n[{i+1}/{len(video_urls)}] {url}")
        
        # Try to ingest (or reuse cached) the video
        try:
            resp = requests.post(
                f"{backend_url}/upload-via-url",
                json={"url": url, "query": "general content"},
                headers=headers,
                timeout=900,
            )
            if resp.status_code == 200:
                data = resp.json()
                job_id = data.get("job_id")
                sentences = data.get("data", {}).get("sentences", [])
                
                if sentences:
                    video_transcripts[url] = {
                        "job_id": job_id,
                        "sentences": sentences,
                    }
                    duration = float(sentences[-1].get("endtime", 0)) if sentences else 0
                    print(f"  ✓ Got {len(sentences)} sentences, duration: {duration:.0f}s")
                else:
                    print(f"  ✗ No sentences in response")
            else:
                print(f"  ✗ HTTP {resp.status_code}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"STEP 2: Calibrating ground truth timestamps")
    print(f"{'=' * 60}")
    
    calibrated_rows = []
    calibration_stats = {"calibrated": 0, "fallback": 0, "no_transcript": 0}
    
    for row in rows:
        query = row["query"]
        url = row["video_url"]
        
        if url not in video_transcripts:
            # Keep original timestamps (video wasn't ingested)
            calibrated_rows.append(row)
            calibration_stats["no_transcript"] += 1
            continue
        
        sentences = video_transcripts[url]["sentences"]
        keywords = QUERY_KEYWORDS.get(query)
        
        if not keywords:
            # No keywords defined — keep original
            calibrated_rows.append(row)
            calibration_stats["fallback"] += 1
            continue
        
        result = find_topic_range(sentences, keywords)
        
        if result:
            gt_start, gt_end = result
            old_start = float(row["gt_start"])
            old_end = float(row["gt_end"])
            
            row["gt_start"] = round(gt_start, 1)
            row["gt_end"] = round(gt_end, 1)
            
            print(f"  [{row['id']}] '{query}'")
            print(f"         Old: {old_start:.0f}s - {old_end:.0f}s  →  New: {gt_start:.1f}s - {gt_end:.1f}s")
            calibration_stats["calibrated"] += 1
        else:
            print(f"  [{row['id']}] '{query}' — no keyword matches found, keeping original")
            calibration_stats["fallback"] += 1
        
        calibrated_rows.append(row)
    
    # Write calibrated CSV
    print(f"\n{'=' * 60}")
    print(f"STEP 3: Saving calibrated dataset")
    print(f"{'=' * 60}")
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "video_url", "domain", "query", "difficulty", "gt_start", "gt_end"])
        for row in calibrated_rows:
            writer.writerow([
                row["id"], row["video_url"], row["domain"],
                row["query"], row["difficulty"],
                row["gt_start"], row["gt_end"]
            ])
    
    print(f"  Calibrated:     {calibration_stats['calibrated']}")
    print(f"  Fallback (orig): {calibration_stats['fallback']}")
    print(f"  No transcript:  {calibration_stats['no_transcript']}")
    print(f"  Saved to: {OUTPUT_CSV}")
    
    # Also save the job_id cache for the eval script
    cache_file = "kaggle_eval_dataset/ingestion_cache.json"
    cache = {url: data["job_id"] for url, data in video_transcripts.items()}
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"  Job ID cache saved to: {cache_file} ({len(cache)} videos)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1].rstrip("/")
    else:
        url = input("Enter backend URL: ").strip().rstrip("/")
    
    if not url:
        print("Error: URL required")
        sys.exit(1)
    
    calibrate(url)

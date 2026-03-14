import numpy as np
from typing import List, Dict, Any

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def smooth_signal(signal: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Apply a simple boxcar smoothing to the signal."""
    if len(signal) < window_size:
        return signal
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode='same')

def get_adaptive_threshold(similarities: np.ndarray, sensitivity: float = 1.0) -> float:
    """Calculate a threshold for detecting dips based on signal statistics."""
    if len(similarities) == 0:
        return 0.0
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    # We look for value < mean - (sensitivity * std)
    return mean_sim - (sensitivity * std_sim)

def segment_transcript(sentences: List[Dict[str, Any]], embeddings: List[List[float]], window_size: int = 3, sensitivity: float = 1.0) -> List[Dict[str, Any]]:
    """
    Segment a transcript into semantic topics based on embedding coherence.
    
    Args:
        sentences: List of sentence objects (must have 'sentence', 'starttime', 'endtime').
        embeddings: List of embedding vectors (floats) corresponding to sentences.
        window_size: Smoothing window size.
        sensitivity: Standard deviations below mean to consider a 'cut' (higher = fewer cuts).
        
    Returns:
        List of segments, where each segment has:
        - start_time
        - end_time
        - text (combined)
        - title (heuristic)
        - sentence_indices (list of ints)
    """
    if not sentences or not embeddings or len(sentences) != len(embeddings):
        return []
    
    n = len(embeddings)
    if n == 1:
        return [{
            "id": 0,
            "start_time": float(sentences[0].get("starttime", 0)),
            "end_time": float(sentences[0].get("endtime", 0)),
            "text": sentences[0].get("sentence", ""),
            "title": sentences[0].get("sentence", "")[:50] + "...",
            "sentence_indices": [0]
        }]

    # Convert to numpy array for speed
    vecs = np.array(embeddings)
    
    # 1. Compute coherence (similarity between S_i and S_{i+1})
    # coherence[i] is sim(vecs[i], vecs[i+1])
    # Length will be n-1
    coherence = []
    for i in range(n - 1):
        sim = cosine_similarity(vecs[i], vecs[i+1])
        coherence.append(sim)
    
    coherence = np.array(coherence)
    
    # 2. Smooth the signal to reduce noise
    smoothed = smooth_signal(coherence, window_size=window_size)
    
    # 3. Find valleys (local minima that are deep enough)
    threshold = get_adaptive_threshold(smoothed, sensitivity)
    
    cut_indices = []
    
    # Analyze the smoothed signal for dips
    for i in range(1, len(smoothed) - 1):
        is_local_min = smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]
        is_deep_enough = smoothed[i] < threshold
        
        if is_local_min and is_deep_enough:
            # i corresponds to the gap between sentence i and i+1
            # so the CUT is AFTER sentence i.
            # The next segment starts at sentence i + 1.
            cut_indices.append(i + 1)
            
    # Add start and end boundaries
    boundaries = [0] + cut_indices + [n]
    
    segments = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1] # Exclusive
        
        segment_sentences = sentences[start_idx:end_idx]
        if not segment_sentences:
            continue
            
        # Combine text
        combined_text = " ".join([s.get("sentence", "") for s in segment_sentences])
        
        # Determine times
        start_time = float(segment_sentences[0].get("starttime", 0))
        end_time = float(segment_sentences[-1].get("endtime", 0))
        
        # Simple title heuristic: First sentence or most central sentence? 
        # For speed, let's use the first sentence as a rough title.
        title = segment_sentences[0].get("sentence", "")
        if len(title) > 60:
            title = title[:60] + "..."
            
        segments.append({
            "id": i,
            "start_time": start_time,
            "end_time": end_time,
            "text": combined_text,
            "title": title,
            "sentence_indices": list(range(start_idx, end_idx))
        })
        
    return segments

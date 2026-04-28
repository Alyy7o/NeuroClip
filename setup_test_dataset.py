import os
import json
import subprocess
import urllib.request
import yt_dlp

# Dataset configuration
DATASET_DIR = "dataset"
VIDEOS_DIR = os.path.join(DATASET_DIR, "videos")
GROUND_TRUTH_FILE = os.path.join(DATASET_DIR, "test_queries.json")

# 5 short educational creative commons videos
# These represent a mini-version of the 'summarization_eval_pack'
SAMPLE_VIDEOS = [
    {
        "id": "vid_01",
        "url": "https://www.youtube.com/watch?v=FjHNvXjBwuk", # Short educational clip
        "title": "What is Machine Learning?",
        "domain": "Computer Science"
    },
    {
        "id": "vid_02",
        "url": "https://www.youtube.com/watch?v=aircAruvnKk", # 3Blue1Brown backprop
        "title": "What is backpropagation really doing?",
        "domain": "Artificial Intelligence"
    },
    {
        "id": "vid_03",
        "url": "https://www.youtube.com/watch?v=r6sGWTCMz2k",
        "title": "What is a Neural Network?",
        "domain": "Artificial Intelligence"
    }
]

TEST_QUERIES = {
    "vid_01": [
        {"query": "supervised learning definition", "difficulty": "Easy"},
        {"query": "difference between AI and ML", "difficulty": "Medium"}
    ],
    "vid_02": [
        {"query": "how the cost function changes", "difficulty": "Hard"},
        {"query": "activation function explanation", "difficulty": "Medium"}
    ]
}

def setup_dataset():
    print("Setting up NeuroClip test dataset...")
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    
    ydl_opts = {
        'format': 'best[ext=mp4][height<=720]',
        'outtmpl': os.path.join(VIDEOS_DIR, '%(id)s.%(ext)s'),
        'quiet': False,
        'noplaylist': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for video in SAMPLE_VIDEOS:
            print(f"Downloading: {video['title']}")
            try:
                ydl.download([video['url']])
            except Exception as e:
                print(f"Failed to download {video['url']}: {e}")
                
    # Save ground truth queries
    with open(GROUND_TRUTH_FILE, 'w') as f:
        json.dump(TEST_QUERIES, f, indent=4)
        
    print(f"Dataset successfully created in '{DATASET_DIR}'")

if __name__ == "__main__":
    setup_dataset()

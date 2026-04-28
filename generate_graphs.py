import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory for test results
os.makedirs("test_results", exist_ok=True)

# Set global styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.figsize': (10, 6),
    'figure.dpi': 300
})

def plot_retrieval_performance():
    labels = ['Embedding-only (Cosine)', 'Embedding + Cross-Encoder', 'Embedding + LLM Routing']
    prec_3 = [0.71, 0.82, 0.78]
    nDCG_3 = [0.73, 0.84, 0.80]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, prec_3, width, label='Precision@3', color='#3498db')
    rects2 = ax.bar(x + width/2, nDCG_3, width, label='nDCG@3', color='#2ecc71')

    ax.set_ylabel('Score')
    ax.set_title('Semantic Retrieval Performance by Pipeline Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1.0])
    ax.legend(loc='upper left')

    # Add text labels
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('test_results/retrieval_performance.png')
    plt.close()

def plot_latency_scaling():
    video_lengths = ['5 minutes', '10 minutes', '20 minutes']
    ingestion_latency = [85, 126, 258]  # Seconds (2.1m=126s, 4.3m=258s)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Ingestion Latency (Line chart)
    ax.plot(video_lengths, ingestion_latency, marker='o', markersize=10, linewidth=3, color='#e74c3c', label='Ingestion Time (s)')
    
    for i, txt in enumerate(ingestion_latency):
        ax.annotate(f"{txt}s", (video_lengths[i], ingestion_latency[i] + 10), ha='center')

    ax.set_ylabel('Time (Seconds)')
    ax.set_title('Pipeline Ingestion Latency vs. Video Length')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('test_results/ingestion_latency.png')
    plt.close()

def plot_compression_performance():
    profiles = ['30s/360p', '60s/480p', '180s/720p', '600s/720p', '300s/720p']
    original_sizes = [12, 28, 95, 380, 175]
    compressed_sizes = [3.1, 7.8, 23.4, 84.2, 38.9]

    x = np.arange(len(profiles))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, original_sizes, width, label='Original Size (MB)', color='#95a5a6')
    rects2 = ax.bar(x + width/2, compressed_sizes, width, label='Compressed Size (MB)', color='#f39c12')

    ax.set_ylabel('File Size (MB)')
    ax.set_title('H.265/HEVC Compression Effectiveness')
    ax.set_xticks(x)
    ax.set_xticklabels(profiles)
    ax.legend()
    
    # Add reduction % labels
    for i in range(len(profiles)):
        reduction = (original_sizes[i] - compressed_sizes[i]) / original_sizes[i] * 100
        ax.annotate(f'-{reduction:.1f}%',
                    xy=(x[i] + width/2, compressed_sizes[i]),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', color='#d35400')

    plt.tight_layout()
    plt.savefig('test_results/compression_performance.png')
    plt.close()

def plot_query_latency():
    # Comparing cold vs cached query latency
    lengths = ['5 min', '10 min', '20 min']
    cold_latency = [1.8, 2.1, 2.4]
    cached_latency = [0.6, 0.7, 0.8]

    x = np.arange(len(lengths))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    rects1 = ax.bar(x - width/2, cold_latency, width, label='Cold Query (s)', color='#8e44ad')
    rects2 = ax.bar(x + width/2, cached_latency, width, label='Cached Query (s)', color='#1abc9c')

    ax.set_ylabel('Latency (Seconds)')
    ax.set_title('Query Latency: Cold vs. Cached Retrieval')
    ax.set_xticks(x)
    ax.set_xticklabels(lengths)
    ax.set_ylim([0, 3.0])
    ax.legend()
    
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}s',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('test_results/query_latency.png')
    plt.close()

if __name__ == "__main__":
    print("Generating evaluation graphs based on NeuroClip Research Paper data...")
    plot_retrieval_performance()
    plot_latency_scaling()
    plot_compression_performance()
    plot_query_latency()
    print("Graphs successfully generated in 'test_results' folder!")

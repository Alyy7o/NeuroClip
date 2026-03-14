# Training Resources & Recommendations

This document outlines the best data strategies and compute environments for training your custom NeuroClip Vision model.

## 1. Recommended Datasets

Since you are building a custom model from scratch, you should start with a specific, manageable dataset before trying massive ones.

### Option A: Flickr30k (Good for Basics, Bad for Code/Slides)
*   **Content:** ~31,000 images, each with 5 descriptive captions.
*   **Domain:** **Natural Images** (People, dogs, parks, scenes).
*   **Pros:** Cleanest "Hello World" dataset.
*   **Cons:** **Poor fit for your specific use case.** It lacks "screens", "slides", or "text-heavy" images. A model trained on Flickr might think a "code editor" looks like nothing it has ever seen.

### Option B: Flickr8k (Fastest / Debugging)
*   **Content:** ~8,000 images.
*   **Why:** Perfect for testing your code pipeline. Use this *only* to make sure your code runs without crashing.

### Option C: MSR-VTT (Best for Educational / Video)
*   **Content:** 10,000 video clips (YouTube).
*   **Why:** Since you are targeting **Educational Videos (Slides, Code, Boards)**, this is your best bet among standard datasets.
    *   **Source:** YouTube clips (unlike Flickr's vacation photos).
    *   **Variety:** Includes categories like "Technology", "News", and "How-to", which are visually closer to your content than "dogs in a park".
    *   **Motion Blur:** It teaches the model to handle the slight blur or compression artifacts found in videos.

### 💡 The "Gold Standard" for you: HowTo100M (Concept)
*   *Note:* The perfect dataset for you is actually **HowTo100M** (Instructional Videos), but it is **too massive** (100M clips) for a single student laptop/Kaggle to handle easily.
*   **Strategy:** Stick to **MSR-VTT** for now. It is the "mini-version" of general video knowledge.

---

## 2. Recommendation for "Educational Content" (Slides & Code)

**Winner: Option C (MSR-VTT)**

**Why?**
1.  **Domain Match:** Flickr30k is strictly "Natural Scenes". If you show it a PowerPoint slide or a VS Code window, it will be confused. MSR-VTT has a broader visual vocabulary.
2.  **Screen Text:** Video datasets often contain occasional text-on-screen, making them slightly better adapted to your need for "boards and slides".
3.  **Frame Quality:** You are building a system for *Video Frames*. Training on *Video Frames* (from MSR-VTT) mitigates the domain shift compared to training on high-res photography (Flickr).

**Action Plan:**
1.  Use **MSR-VTT**.
2.  Extract 1 frame per second (or 1 per clip) to create your image-text pairs.
3.  *Bonus:* If you can find a small "Slide-Text" dataset (like **DocVQA** or **TextVQA**), mixing 1000 of those images in would massively help with the "Code/Board" recognition.

---

## 2. Training Time Estimate (HP i5 10th Gen Laptop)

**⚠️ Warning:** Training Deep Learning models (like ResNet + BERT) on a CPU is **extremely inefficient**.

*   **Architecture:** ResNet50 (Image) + DistilBERT (Text)
*   **Compute Intensity:** These models perform millions of matrix multiplications *per image*. CPUs process these sequentially; GPUs process thousands in parallel.

### The Estimate
If you train on **Flickr30k** (31,000 images) with a batch size of 16:
*   **Time per Step (CPU):** ~2-5 seconds.
*   **Steps per Epoch:** ~2,000 steps.
*   **Time per Epoch:** ~1.5 to 3 hours.
*   **epochs needed:** Typically 20-40 for decent convergence.
*   **Total Time:** **3 - 5 Days** of 100% CPU usage.

**Outcome for Laptop:**
*   Your laptop will likely overheat and throttle performance.
*   You cannot use the laptop for anything else during this time.
*   **Verdict:** Do **NOT** train the full model on your laptop. Use it only for writing code and debugging (e.g., training on just 100 images to see if it runs).

---

## 3. Platform Recommendations (Free GPU)

Since you need a GPU, here are the best free options:

| Feature | Google Colab (Free) | Kaggle Kernels (Recommended) |
| :--- | :--- | :--- |
| **GPU** | NVIDIA T4 (usually) | **NVIDIA P100** (often available) |
| **Time Limit** | ~12 hours / session | **30+ hours / week** |
| **Persistence** | Files deleted on disconnect | Persistent working directory |
| **Data Access** | Upload via Drive (slow) | **Instant access** to Datasets (Fast) |

### My Recommendation: **Kaggle**
1.  **Data:** Flickr30k is already hosted on Kaggle. You can "Add Data" and attach it instantly without downloading/uploading 8GB of files.
2.  **Hardware:** The P100 GPU is faster than Colab's standard T4.
3.  **Environment:** It's a Jupyter Notebook environment just like Colab.

### Workflow
1.  **Develop Locally:** Write your `model.py` and `dataset.py` on your VS Code. Test it with "Mock Data" (random numbers) or just 5 images.
2.  **Upload to Kaggle:** specific create a "New Notebook" on Kaggle.
3.  **Train:** Copy-paste your code, attach the Flickr30k dataset, and hit "Run".
4.  **Download Weights:** Once finished, download the `.pth` model file and put it in your NeuroClip backend folder.

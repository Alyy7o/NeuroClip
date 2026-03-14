# Custom NeuroClip Training Pipeline

This directory contains the code to train your own Vision-Language Model (CLIP style) for NeuroClip.

## Folder Structure

*   `model.py`: **The Architecture.** Defines the `CustomNeuroClip` class (Two-Tower: ResNet + DistilBERT).
*   `dataset.py`: **Data Loader.** Handles loading images and captions from CSV.
*   `train.py`: **Main Script.** Runs the training loop, calculates Contrastive Loss, and saves weights.
*   `requirements_training.txt`: Python libraries needed for training.

## 1. Setup

### Install Dependencies
```bash
pip install -r requirements_training.txt
```

## 2. Prepare Your Data

You need a CSV file (e.g., `train_data.csv`) with at least two columns:
*   `image_path`: Path to the image file (e.g., `images/slide_01.jpg`)
*   `caption`: The text description (e.g., "A slide showing the Python syntax")

**Using MSR-VTT:**
1.  Download MSR-VTT frames and captions.
2.  Convert their JSON format into a simple CSV matching the structure above.

## 3. How to Train (Locally for debugging)

**WARNING:** Full training on a laptop CPU is extremely slow. Use this only to verify code works.

1.  Place your images in a folder (e.g., `./images`).
2.  Update `train.py`:
    *   Set `CSV_PATH = "path/to/your.csv"`
    *   Set `IMG_ROOT = "path/to/images/"`
3.  Run:
    ```bash
    python train.py
    ```

## 4. How to Train (on Kaggle - Recommended)

1.  Create a **New Notebook** on Kaggle.
2.  **Add Data:** Search for "Flickr30k" or upload your own MSR-VTT zip.
3.  **Copy-Paste Code:**
    *   Paste content of `model.py` into a cell.
    *   Paste content of `dataset.py` into a cell.
    *   Paste content of `train.py` into a cell.
4.  **Update Paths:** Change `CSV_PATH` and `IMG_ROOT` to point to the input dataset in Kaggle (usually under `/kaggle/input/...`).
5.  **Enable GPU:** Go to Settings -> Accelerator -> GPU.
6.  **Run All.**
7.  **Download:** Download the resulting `.pth` file from the Output section.

## 5. Integrating the Trained Model

Once you have `neuroclip_v1.pth`, move it to your main backend folder.
Update `backend/main.py` to load this custom model instead of `sentence-transformers`.

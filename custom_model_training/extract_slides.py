import torch
import cv2
from PIL import Image
from torchvision import transforms
from transformers import DistilBertTokenizer
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm

# Import our model definition
try:
    from model import CustomNeuroClip
except ImportError:
    print("Error: 'model.py' not found. Please make sure it is in the same directory.")
    exit(1)

# ====================
# CONFIGURATION
# ====================
DEFAULT_MODEL_PATH = "neuroclip_v1.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Labels to classify "Educational Content"
# If a frame matches these more than "person", "background", etc., we keep it.
TARGET_QUERIES = [
    "slide presentation",
    "powerpoint slide",
    "whiteboard with writing",
    "computer code on screen",
    "text document screenshot",
    "diagram and chart"
]

DISTRACTOR_QUERIES = [
    "person speaking facing camera",
    "blur background",
    "crowd of people",
    "outdoor scenery",
    "advertisement / logo"
]

def load_model(path):
    print(f"Loading model from {path}...")
    model = CustomNeuroClip(embed_dim=256)
    try:
        # Load weights (handle CPU/GPU mapping)
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return None
    model.to(DEVICE)
    model.eval()
    return model

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_keyframes(video_path, output_dir, model, CONFIDENCE_THRESHOLD=0.35):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"Video Processing: {video_path}")
    print(f"Duration: {duration:.2f}s | FPS: {fps}")

    # Prepare Text Embeddings Once
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    all_queries = TARGET_QUERIES + DISTRACTOR_QUERIES
    encoded_text = tokenizer(all_queries, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    
    transform = get_transforms()
    
    # Process 1 frame every second
    frame_interval = int(fps) 
    current_frame = 0
    saved_count = 0
    
    pbar = tqdm(total=total_frames // frame_interval)
    
    with torch.no_grad():
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
            
            # --- MODEL INFERENCE ---
            # 1. Get Embeddings
            img_emb, txt_emb = model(img_tensor, encoded_text['input_ids'], encoded_text['attention_mask'])
            img_emb = F.normalize(img_emb, p=2, dim=1)
            txt_emb = F.normalize(txt_emb, p=2, dim=1)
            
            # 2. Similarity
            similarity = torch.matmul(img_emb, txt_emb.T).squeeze(0) # Shape: [num_queries]
            
            # 3. Check if any TARGET query has high score
            # Indices: 0 to len(TARGET_QUERIES)-1 are Targets
            # Indices: len(TARGET_QUERIES) to end are Distractors
            target_scores = similarity[:len(TARGET_QUERIES)]
            distractor_scores = similarity[len(TARGET_QUERIES):]
            
            max_target_score = target_scores.max().item()
            max_distractor_score = distractor_scores.max().item()
            
            # LOGIC:
            # - Must be somewhat confident it's a target (> THRESHOLD)
            # - Must be MORE like a target than a distractor (Target > Distractor)
            if max_target_score > CONFIDENCE_THRESHOLD and max_target_score > max_distractor_score:
                timestamp = current_frame / fps
                filename = f"slide_{int(timestamp)}s.jpg"
                save_path = os.path.join(output_dir, filename)
                
                # Save original quality frame (not the resized tensor)
                cv2.imwrite(save_path, frame) # Save as BGR for OpenCV
                saved_count += 1
            
            current_frame += frame_interval
            pbar.update(1)
            
    cap.release()
    pbar.close()
    print(f"\nExtraction Complete.")
    print(f"Saved {saved_count} keyframes to '{output_dir}/'")

def main():
    parser = argparse.ArgumentParser(description="Extract educational slides/code from video using NeuroClip")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to trained .pth model")
    parser.add_argument("--output", default="extracted_slides", help="Output directory for images")
    parser.add_argument("--threshold", type=float, default=0.35, help="Confidence threshold (0.0 - 1.0)")
    
    args = parser.parse_args()
    
    model = load_model(args.model)
    if not model:
        print("Exiting...")
        return
        
    extract_keyframes(args.video, args.output, model, args.threshold)

if __name__ == "__main__":
    main()

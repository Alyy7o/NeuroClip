import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import DistilBertModel, DistilBertTokenizer
from PIL import Image
import torch.nn.functional as F
import os
import cv2
from tqdm.auto import tqdm

# ====================
# KAGGLE CONFIGURATION
# ====================
# 1. Point this to a FOLDER containing multiple videos (e.g., MSR-VTT or your upload)
INPUT_FOLDER = "/kaggle/input/test-video" 
MODEL_PATH = "/kaggle/input/my-neuroclip-model/neuroclip_v1.pth"
BASE_OUTPUT_DIR = "/kaggle/working/filtered_output"

# Confidence Threshold (Higher = Stricter "Is this a slide?", Lower = More tolerant)
# 0.35 was default. Try 0.40 or 0.45 to reduce junk.
CONFIDENCE_THRESHOLD = 0.40

# Deduplication Threshold (Lower = More aggressive "Duplicate" detection)
# If similarity > 0.85, we skip it.
DUPLICATE_THRESHOLD = 0.85

# Minimum time (seconds) to wait before saving a "similar" slide again.
# This prevents saving 10 frames of the same slide just because the mouse moved.
MIN_SLIDE_INTERVAL = 5.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEBUG: Using device: {DEVICE}")

# ====================
# LABELS (The "Intelligence")
# ====================
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

# ====================
# MODEL DEFINITION (Inline to avoid import errors)
# ====================
class CustomNeuroClip(nn.Module):
    def __init__(self, embed_dim=256, frozen_backbones=False):
        super(CustomNeuroClip, self).__init__()
        
        # Image Encoder
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            resnet = models.resnet50(weights=weights)
        except:
            resnet = models.resnet50(pretrained=True)
            
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_projection = nn.Linear(2048, embed_dim)
        
        # Text Encoder
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.text_projection = nn.Linear(768, embed_dim)

    def forward(self, images, input_ids, attention_mask):
        img_feat = self.image_encoder(images).flatten(1)
        image_embeddings = self.image_projection(img_feat)
        
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.last_hidden_state[:, 0, :]
        text_embeddings = self.text_projection(text_feat)
        
        return image_embeddings, text_embeddings

# ====================
# LOAD MODEL
# ====================
def load_model(path):
    print(f"Loading model from {path}...")
    model = CustomNeuroClip(embed_dim=256)
    try:
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Model weights loaded.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None
    model.to(DEVICE)
    model.eval()
    return model

# ====================
# EXTRACTION LOGIC
# ====================
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_keyframes(video_path, output_dir, model, threshold=0.35):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        print("Make sure the VIDEO_PATH at the top of the script is correct!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0: fps = 24.0
    duration = total_frames / fps
    print(f"Processing: {video_path}")
    print(f"Duration: {duration:.2f}s | FPS: {fps}")

    # Prepare Query Embeddings
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    all_queries = TARGET_QUERIES + DISTRACTOR_QUERIES
    encoded_text = tokenizer(all_queries, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    
    transform = get_transforms()
    
    # Check 1 frame per second
    frame_interval = int(fps) 
    current_frame = 0
    saved_count = 0
    
    # Stats counters
    stats = {
        "scanned": 0,
        "low_conf": 0,
        "distractor": 0,
        "duplicate": 0, 
        "saved": 0
    }
    
    last_saved_emb = None
    last_saved_time = -999.0 
    
    print("Scanning video for Educational Content...")
    pbar = tqdm(total=total_frames // frame_interval)
    
    with torch.no_grad():
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break
            
            stats["scanned"] += 1
            
            # Prepare Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
            
            # Inference
            img_emb, txt_emb = model(img_tensor, encoded_text['input_ids'], encoded_text['attention_mask'])
            img_emb = F.normalize(img_emb, p=2, dim=1)
            txt_emb = F.normalize(txt_emb, p=2, dim=1)
            
            # Scores
            similarity = torch.matmul(img_emb, txt_emb.T).squeeze(0)
            target_scores = similarity[:len(TARGET_QUERIES)]
            distractor_scores = similarity[len(TARGET_QUERIES):]
            
            max_target = target_scores.max().item()
            max_distractor = distractor_scores.max().item()
            
            # LOGIC 1: Is it confident enough?
            if max_target < threshold:
                stats["low_conf"] += 1
                current_frame += frame_interval
                pbar.update(1)
                continue
                
            # LOGIC 2: Is it better than distractor?
            if max_target < max_distractor:
                stats["distractor"] += 1
                current_frame += frame_interval
                pbar.update(1)
                continue

            # LOGIC 3: Deduplication
            timestamp_s = current_frame / fps
            should_save = True
            
            if last_saved_emb is not None:
                sim_to_last = torch.cosine_similarity(img_emb, last_saved_emb).item()
                time_diff = timestamp_s - last_saved_time
                
                if sim_to_last > DUPLICATE_THRESHOLD: 
                    should_save = False
                
                if time_diff < MIN_SLIDE_INTERVAL and sim_to_last > 0.60:
                    should_save = False

            if should_save:
                filename = f"slide_{int(timestamp_s):04d}s.jpg"
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, frame)
                saved_count += 1
                stats["saved"] += 1
                last_saved_emb = img_emb.clone()
                last_saved_time = timestamp_s
            else:
                stats["duplicate"] += 1
            
            current_frame += frame_interval
            pbar.update(1)
            
    cap.release()
    pbar.close()
    
    print(f"\nDONE!")
    print(f"--- Statistics ---")
    print(f"Total Scanned: {stats['scanned']}")
    print(f"Ignored (Low Confidence): {stats['low_conf']}")
    print(f"Ignored (Not a slide/Distractor): {stats['distractor']}")
    print(f"Ignored (Duplicate/Too Soon): {stats['duplicate']}")
    print(f"SAVED (Unique Slides): {stats['saved']}")
    print(f"Output Folder: {output_dir}")
    
    # List a few files to verify
    if saved_count > 0:
        print("First 5 files:")
        print(os.listdir(output_dir)[:5])

# ====================
# OCR LOGIC & ACCURACY CHECK
# ====================
def perform_ocr(output_dir):
    print(f"\n[OCR] Starting Text Extraction for Validation on {output_dir}...")
    
    try:
        import easyocr
    except ImportError:
        print("ERROR: EasyOCR not installed! validation skipped.")
        return 0, 0

    # Initialize Reader (English)
    reader = easyocr.Reader(['en'], gpu=(DEVICE=="cuda"))
    
    files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
    
    if not files:
        return 0, 0

    notes_path = os.path.join(output_dir, "lecture_notes.txt")
    
    # Validation Counters
    total_slides = len(files)
    verified_slides = 0 # Slides with significant text
    
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("=== GENERATED LECTURE NOTES (OCR) ===\n")
        f.write(f"Source: {output_dir}\n")
        f.write("=====================================\n\n")
        
        for filename in tqdm(files, desc="Reading Text"):
            try:
                second_str = filename.replace("slide_", "").replace("s.jpg", "")
                seconds = int(second_str)
                timestamp = f"{seconds//60:02d}:{seconds%60:02d}"
            except:
                timestamp = "??:??"

            path = os.path.join(output_dir, filename)
            
            # Read Text
            result = reader.readtext(path, detail=0) 
            
            # Filter short/junk text
            clean_text = [text for text in result if len(text.strip()) > 3]
            full_text_content = " ".join(clean_text)
            
            # === ACCURACY CHECK ===
            # If a slide has > 50 characters, it's almost certainly a valid slide/board.
            # If it has < 10, it might be a false positive (just a person).
            if len(full_text_content) > 50:
                verified_slides += 1
            
            if clean_text:
                block = f"[{timestamp}] SLIDE CONTENT:\n"
                for line in clean_text:
                    block += f"- {line}\n"
                block += "\n"
                f.write(block)
    
    return total_slides, verified_slides

# ====================
# MAIN
# ====================
def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"WARNING: Input folder not found at {INPUT_FOLDER}")
        return
        
    model = load_model(MODEL_PATH)
    if not model: return

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(video_extensions)]
    
    print(f"Found {len(video_files)} videos in {INPUT_FOLDER}")
    
    # Store Report Data
    accuracy_report = []

    for i, video_file in enumerate(video_files):
        video_path = os.path.join(INPUT_FOLDER, video_file)
        subdir_name = os.path.splitext(video_file)[0] + "_slides"
        video_output_dir = os.path.join(BASE_OUTPUT_DIR, subdir_name)
        
        print(f"\n[{i+1}/{len(video_files)}] Processing: {video_file}")
        
        extract_keyframes(video_path, video_output_dir, model, CONFIDENCE_THRESHOLD)
        
        # Run OCR and get verification stats
        total, verified = perform_ocr(video_output_dir)
        
        # Calc Precision
        precision = (verified / total * 100) if total > 0 else 0.0
        accuracy_report.append({
            "video": video_file,
            "saved": total,
            "verified": verified,
            "precision": precision
        })
        
    print(f"\n" + "="*40)
    print(f"FINAL ACCURACY REPORT (Self-Graded)")
    print(f"="*40)
    for item in accuracy_report:
        print(f"Video: {item['video']}")
        print(f"  - Extracted Slides: {item['saved']}")
        print(f"  - Verified (Text-Rich): {item['verified']}")
        print(f"  - ESTIMATED PRECISION: {item['precision']:.1f}%")
        if item['precision'] < 50 and item['saved'] > 0:
            print(f"    [!] Low precision. Try increasing CONFIDENCE_THRESHOLD.")
        print("-" * 20)

    print(f"\nALL VIDEOS PROCESSED.")
    print(f"Check directories in: {BASE_OUTPUT_DIR}")

if __name__ == "__main__":
    main()

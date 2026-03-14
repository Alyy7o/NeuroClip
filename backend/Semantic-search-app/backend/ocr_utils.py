import torch
try:
    import spaces
except ImportError:
    # Fallback for local execution
    class spaces:
        @staticmethod
        def GPU(func):
            return func
import torch.nn as nn
from torchvision import models, transforms
from transformers import DistilBertModel, DistilBertTokenizer
from PIL import Image
import torch.nn.functional as F
import os
import cv2
from tqdm.auto import tqdm
from pathlib import Path

# ====================
# CONFIGURATION
# ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.40
DUPLICATE_THRESHOLD = 0.85
MIN_SLIDE_INTERVAL = 5.0

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
# MODEL DEFINITION
# ====================
class CustomNeuroClip(nn.Module):
    def __init__(self, embed_dim=256):
        super(CustomNeuroClip, self).__init__()
        
        # Image Encoder - Use ResNet18 for faster CPU inference
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
        except:
            resnet = models.resnet18(pretrained=True)
            
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_projection = nn.Linear(512, embed_dim)
        
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

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_ocr_model(path):
    model = CustomNeuroClip(embed_dim=256)
    try:
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"OCR Model weights loaded from {path}.")
        else:
            print(f"Warning: OCR Model file not found at {path}. Extraction will be skipped.")
            return None
    except Exception as e:
        print(f"Error loading OCR weights: {e}")
        return None
    model.to(DEVICE)
    model.eval()
    return model

@spaces.GPU
def extract_high_value_frames(video_path, output_dir, model, threshold=CONFIDENCE_THRESHOLD):
    if model is None:
        return 0
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0: fps = 24.0
    
    # Prepare Query Embeddings
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    all_queries = TARGET_QUERIES + DISTRACTOR_QUERIES
    encoded_text = tokenizer(all_queries, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    
    transform = get_transforms()
    
    # Process 1 frame every 2 seconds (faster on CPU)
    frame_interval = int(fps * 2.0) 
    current_frame = 0
    saved_count = 0
    
    last_saved_emb = None
    last_saved_time = -999.0 
    
    with torch.no_grad():
        while current_frame < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break
            
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
            
            # LOGIC: Confident enough AND better than distractor
            if max_target >= threshold and max_target > max_distractor:
                timestamp_s = current_frame / fps
                should_save = True
                
                # Deduplication
                if last_saved_emb is not None:
                    sim_to_last = torch.cosine_similarity(img_emb, last_saved_emb).item()
                    time_diff = timestamp_s - last_saved_time
                    
                    if sim_to_last > DUPLICATE_THRESHOLD: 
                        should_save = False
                    if time_diff < MIN_SLIDE_INTERVAL and sim_to_last > 0.60:
                        should_save = False

                if should_save:
                    filename = f"slide_{int(timestamp_s):04d}s.jpg"
                    save_path = output_dir / filename
                    cv2.imwrite(str(save_path), frame)
                    saved_count += 1
                    last_saved_emb = img_emb.clone()
                    last_saved_time = timestamp_s
            
            current_frame += frame_interval
            
    cap.release()
    return saved_count

def run_ocr_on_frames(image_folder):
    """
    Runs EasyOCR on images in image_folder and returns a list of dictionaries:
    {'timestamp': float, 'text': str}
    """
    image_folder = Path(image_folder)
    if not image_folder.exists():
        return []

    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=(DEVICE=="cuda"))
    except ImportError:
        print("EasyOCR not installed. Skipping OCR.")
        return []

    files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
    results = []
    
    for filename in files:
        try:
            # Extract timestamp from filename: slide_0045s.jpg -> 45
            seconds = int(filename.replace("slide_", "").replace("s.jpg", ""))
        except:
            seconds = 0

        path = image_folder / filename
        ocr_result = reader.readtext(str(path), detail=0) 
        
        # Filter short junk text
        clean_text = [text for text in ocr_result if len(text.strip()) > 3]
        if clean_text:
            full_content = " ".join(clean_text)
            results.append({
                "timestamp": float(seconds),
                "text": full_content
            })
            
    return results

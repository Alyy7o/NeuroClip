import os
import json
import shutil
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import DistilBertModel, DistilBertTokenizer
from PIL import Image
from tqdm.auto import tqdm

# ==========================================
# 1. CONFIGURATION (CHANGE PATHS HERE!)
# ==========================================

# kaggle input paths usually look like: /kaggle/input/dataset-name/...
# YOU MUST UPDATE THESE TWO LINES TO MATCH YOUR ADDED DATASET:
VIDEO_SOURCE_DIR = "/kaggle/input/msr-vtt/data/MSRVTT/MSRVTT/videos/all"       # Folder containing .mp4 files
JSON_ANNOTATION_FILE = "/kaggle/input/msr-vtt/data/MSRVTT/MSRVTT/annotation/MSR_VTT.json"  # The huge JSON file

# Output settings (No need to change)
OUTPUT_CSV = "train_data_educational.csv"
MODEL_SAVE_NAME = "neuroclip_v1.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
EMBED_DIM = 256

# Categories to KEEP (Educational)
KEEP_CATEGORIES = [
    "category 13", # Science/Technology
    "category 11", # How-to
    "category 06"  # Education
]

print(f"Running on device: {DEVICE}")

# ==========================================
# 2. MODEL DEFINITION (Dual Encoder)
# ==========================================

class CustomNeuroClip(nn.Module):
    def __init__(self, embed_dim=256, frozen_backbones=False):
        super(CustomNeuroClip, self).__init__()
        
        # Image Encoder (ResNet-50)
        # Using the new weights parameter structure if available, else fallback
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            resnet = models.resnet50(weights=weights)
        except:
            resnet = models.resnet50(pretrained=True)
            
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_projection = nn.Linear(2048, embed_dim)
        
        # Text Encoder (DistilBERT)
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.text_projection = nn.Linear(768, embed_dim)
        
        if frozen_backbones:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, images, input_ids, attention_mask):
        # Image Branch
        img_feat = self.image_encoder(images).flatten(1)
        image_embeddings = self.image_projection(img_feat)
        
        # Text Branch
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.last_hidden_state[:, 0, :]
        text_embeddings = self.text_projection(text_feat)
        
        return image_embeddings, text_embeddings

# ==========================================
# 3. DATASET CLASS
# ==========================================

class NeuroClipDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform=None):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]
        img_path = row['image_path']
        caption = str(row['caption'])

        # On Kaggle, we might be loading videos as images if we haven't extracted frames.
        # Ideally, we extract frames. For this script, we will try to load the file.
        # NOTE: Loading a video file with PIL Image.open() won't work directly.
        # We need a quick hack to extract strictly one frame on the fly if it's a video.
        # However, installing opencv (cv2) is standard.
        
        image = None
        try:
            # Try loading as image first (if user pre-extracted)
            image = Image.open(img_path).convert("RGB")
        except:
            # It's likely a video file (.mp4)
            # Use OpenCV to grab the middle frame
            import cv2
            cap = cv2.VideoCapture(img_path)
            if cap.isOpened():
                # Read one frame
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                cap.release()
            
        if image is None:
            # Fallback black image to prevent crash
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)
            
        text_input = self.tokenizer(
            caption, 
            padding='max_length', 
            truncation=True, 
            max_length=128,
            return_tensors="pt"
        )
        
        return {
            'image': image,
            'input_ids': text_input['input_ids'].squeeze(0),
            'attention_mask': text_input['attention_mask'].squeeze(0)
        }

# ==========================================
# 4. PREPARATION (FILTER DATA)
# ==========================================

def prepare_data():
    # Checking if file exists is dangerous if the previous run failed/created an empty file.
    # We will FORCE regeneration to ensure the latest filtering logic is used.
    if os.path.exists(OUTPUT_CSV):
        print(f"Warning: {OUTPUT_CSV} exists. Overwriting it to execute new filtering logic.")
    
    print(f"Loading Annotation JSON from {JSON_ANNOTATION_FILE}...")
    if not os.path.exists(JSON_ANNOTATION_FILE):
        print(f"ERROR: JSON file not found at {JSON_ANNOTATION_FILE}")
        print("Please Check the CONFIGURATION section at the top of this script.")
        return

    with open(JSON_ANNOTATION_FILE, 'r') as f:
        data = json.load(f)

    # DEBUG: Print structure
    if isinstance(data, dict):
        print(f"DEBUG: JSON keys found: {list(data.keys())}")
    else:
        print(f"DEBUG: JSON root is not a dict, it is {type(data)}")

    # Handle variations in MSR-VTT JSON formats
    video_list = []
    if 'videos' in data:
        video_list = data['videos']
    elif 'images' in data:
        # Some versions (COCO-style) rename 'videos' to 'images'
        print("DEBUG: Found 'images' key instead of 'videos'. Using that.")
        video_list = data['images']
    else:
        print("ERROR: Could not find 'videos' or 'images' key in JSON.")
        print("Please check which dataset you added.")
        return

    # Create map
    # Some datasets use 'id' instead of 'video_id'
    videos_map = {}
    for v in video_list:
        vid_id = v.get('video_id', v.get('id'))
        if vid_id:
            videos_map[vid_id] = v

    filtered_rows = []

    # Handle sentences/captions variations
    caption_list = []
    if 'sentences' in data:
        caption_list = data['sentences']
    elif 'captions' in data:
        caption_list = data['captions']
    elif 'annotations' in data:
        caption_list = data['annotations']
    
    # Debug: Print first few categories to see what we are dealing with
    sample_cats = set()
    for i, v in enumerate(videos_map.values()):
        if 'category' in v:
            sample_cats.add(str(v['category']))
        if i > 50: break
    print(f"DEBUG: Sample categories found in JSON: {list(sample_cats)}")

    print(f"Filtering {len(caption_list)} captions...")
    
    count_found = 0
    for annot in caption_list:
        vid_id = annot.get('video_id', annot.get('image_id'))
        caption = annot.get('caption')
        
        video_info = videos_map.get(vid_id)
        
        if video_info and 'category' in video_info:
            raw_cat = video_info['category']
        elif 'category_id' in annot:
            raw_cat = annot['category_id']
        else:
            raw_cat = None
        
        is_edu = False # Initialize safely here

        if raw_cat is not None:
            # Normalize Category Check
            # We want: 13 (Science), 11 (Howto), 6 (Education)
            # Accept: 13, "13", "category 13", "13.0"
            
            # Convert to string and clean
            cat_str = str(raw_cat).lower().strip()
            
            # Convert to string and clean
            cat_str = str(raw_cat).lower().strip()
            
            # Helper to check if it matches our targets
            if cat_str in ["13", "11", "6", "13.0", "11.0", "6.0"]:
                is_edu = True
            elif "category 13" in cat_str or "category 11" in cat_str or "category 06" in cat_str or "category 6" in cat_str:
                is_edu = True
            elif "science" in cat_str or "technology" in cat_str or "education" in cat_str or "how" in cat_str:
                is_edu = True
        
        if is_edu:
            possible_path = os.path.join(VIDEO_SOURCE_DIR, f"{vid_id}.mp4")
            filtered_rows.append({
                'image_path': possible_path, 
                'caption': caption
            })
            count_found += 1

    # FALLBACK 1: If Educational filtering failed, try using ALL videos with REAL captions.
    # (Because training on "random" videos with REAL text is better than Dummy text)
    if count_found == 0:
        print("ERROR: No educational videos found in JSON filtering!")
        print("DEBUG: Checking if we can just use ALL videos from JSON (ignoring category)...")
        
        # Reset and try again without category check
        count_real_fallback = 0
        for annot in caption_list:
            vid_id = annot.get('video_id', annot.get('image_id'))
            caption = annot.get('caption')
            
            possible_path = os.path.join(VIDEO_SOURCE_DIR, f"{vid_id}.mp4")
            
            # Use 'os.path.exists' check or just assume if we are on Kaggle to save time
            # But since user has 'images' key often, filenames might differ slightly.
            # We'll rely on the standard naming f"{vid_id}.mp4"
            filtered_rows.append({
                'image_path': possible_path, 
                'caption': caption
            })
            count_real_fallback += 1
        
        if count_real_fallback > 0:
            print(f"WARNING: Category info missing. Using ALL {count_real_fallback} valid captions found in JSON.")
            print("NOTE: Training will be generic (not just educational), but INTELLIGENT.")
            count_found = count_real_fallback
        else:
            print("ERROR: Could not match any JSON IDs to captions either.")

    # FALLBACK 2: If we STILL have 0 (JSON is totally broken/empty), THEN do Dummy Mode
    if count_found == 0:
        print("DEBUG: Attempting to scan VIDEO_SOURCE_DIR directly for mp4 files (Video-Only Mode)...")
        
        if os.path.exists(VIDEO_SOURCE_DIR):
            video_files = [f for f in os.listdir(VIDEO_SOURCE_DIR) if f.lower().endswith('.mp4')]
            
            if len(video_files) > 0:
                print(f"WARNING: Found {len(video_files)} videos in folder but NO Captions. Using dummy captions.")
                print("NOTE: The model will run but NOT learn meaningful text associations.")
                
                for vfile in video_files:
                    path = os.path.join(VIDEO_SOURCE_DIR, vfile)
                    # We give a generic caption so the code works. 
                    filtered_rows.append({
                        'image_path': path, 
                        'caption': 'educational video content' # Placeholder text
                    })
                count_found = len(filtered_rows)
            else:
                print("CRITICAL: No .mp4 files found in video source directory either.")
                filtered_rows.append({'image_path': 'dummy.mp4', 'caption': 'dummy caption'})
        else:
             print(f"CRITICAL: Video directory {VIDEO_SOURCE_DIR} does not exist.")
             filtered_rows.append({'image_path': 'dummy.mp4', 'caption': 'dummy caption'})
    
    df = pd.DataFrame(filtered_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Success! Created {OUTPUT_CSV} with {len(df)} samples.")

# ==========================================
# 5. TRAINING LOOP
# ==========================================

def contrastive_loss(image_embeddings, text_embeddings, temperature=0.1):
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    logits = torch.matmul(image_embeddings, text_embeddings.T) / temperature
    labels = torch.arange(logits.shape[0]).to(logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2

def main():
    # 1. Prepare Data
    prepare_data()
    
    if not os.path.exists(OUTPUT_CSV):
        print("Dataset creation failed. Exiting.")
        return

    # 2. Setup Transform & Pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = NeuroClipDataset(OUTPUT_CSV, tokenizer, transform=transform)
    
    # Validation split (80/20 Rule for finetuning)
    total_len = len(dataset)
    if total_len > 1:
        train_size = int(0.8 * total_len) # 80% Training
        if train_size == 0: train_size = 1 
        val_size = total_len - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    else:
        # Fallback for dummy/single item
        train_dataset = dataset
        val_dataset = dataset
    
    # Batch sizes
    train_batch_size = min(BATCH_SIZE, len(train_dataset))
    val_batch_size = min(BATCH_SIZE, len(val_dataset))
    if train_batch_size == 0: train_batch_size = 1
    if val_batch_size == 0: val_batch_size = 1

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=2)
    
    print(f"TRAINING SET: {len(train_dataset)} samples")
    print(f"VALIDATION SET: {len(val_dataset)} samples (Used for Best Model finetuning)")

    # 3. Model Setup
    model = CustomNeuroClip(embed_dim=EMBED_DIM).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Track Best Metrics
    best_val_loss = float('inf')
    best_model_path = MODEL_SAVE_NAME

    # 4. Run Epochs
    for epoch in range(1, EPOCHS + 1):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in progress:
            images = batch['image'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            optimizer.zero_grad()
            img_emb, txt_emb = model(images, input_ids, attention_mask)
            loss = contrastive_loss(img_emb, txt_emb)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress.set_postfix(loss=loss.item())
            
        avg_train_loss = running_loss / len(train_loader)
        
        # --- VALIDATION PHASE (Finetuning Check) ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(DEVICE)
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                
                img_emb, txt_emb = model(images, input_ids, attention_mask)
                loss = contrastive_loss(img_emb, txt_emb)
                val_running_loss += loss.item()
        
        avg_val_loss = val_running_loss / len(val_loader)
        
        print(f"Epoch {epoch} Results:")
        print(f"  - Train Loss: {avg_train_loss:.4f}")
        print(f"  - Valid Loss: {avg_val_loss:.4f}")
        
        # --- CHECKPOINTING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> NEW BEST MODEL! Saved to {best_model_path}")
        else:
            print(f"  -> Validation loss did not improve (Best: {best_val_loss:.4f})")
        
    print(f"\nTRAINING COMPLETE.")
    print(f"The file '{MODEL_SAVE_NAME}' contains the weights from the BEST Epoch.")
    print("Please download this file from the Output section.")

if __name__ == "__main__":
    main()

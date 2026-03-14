import os
import json
import shutil
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import DistilBertModel, DistilBertTokenizer
from PIL import Image

# ====================
# TPU IMPORTS
# ====================
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    print("TPU Libraries imported successfully!")
except ImportError:
    print("WARNING: TPUs not detected. This script requires a TPU environment (like Kaggle/Colab).")

# ====================
# CONFIGURATION
# ====================
VIDEO_SOURCE_DIR = "/kaggle/input/msr-vtt/data/MSRVTT/MSRVTT/videos/all"
JSON_ANNOTATION_FILE = "/kaggle/input/msr-vtt/data/MSRVTT/MSRVTT/annotation/MSR_VTT.json"
OUTPUT_CSV = "train_data_educational.csv"
MODEL_SAVE_NAME = "neuroclip_v1.pth"

# Hyperparameters
# NOTE: Batch size is PER CORE. Total batch = 32 * 8 = 256
BATCH_SIZE = 32  
EPOCHS = 5
LEARNING_RATE = 1e-4
EMBED_DIM = 256

# ====================
# MODEL
# ====================
class CustomNeuroClip(nn.Module):
    def __init__(self, embed_dim=256):
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
# DATASET
# ====================
class NeuroClipDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform=None):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        row = self.data.iloc[idx]
        
        # Image Load
        image = None
        try:
            image = Image.open(row['image_path']).convert("RGB")
        except:
            # Fallback for videos if paths are raw .mp4
            try:
                import cv2
                cap = cv2.VideoCapture(row['image_path'])
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                cap.release()
            except: pass
            
        if image is None:
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)
            
        # Text Load
        text_input = self.tokenizer(
            str(row['caption']), 
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

# ====================
# PREPARATION
# ====================
def prepare_csv():
    # Only Rank 0 prepares the CSV to avoid race conditions
    if not os.path.exists(OUTPUT_CSV):
        print("Rank 0: Preparing CSV...")
        # ... [Simplified Filtering Logic Same as Before] ...
        # For brevity, reusing the core logic. 
        # In a real run, COPY your filtering logic here.
        # Assuming prepare_data() logic exists or file is pre-uploaded for this example.
        
        # Quick fallback if file missing for this example:
        if not os.path.exists(JSON_ANNOTATION_FILE):
            print("Rank 0: JSON not found. Skipping CSV gen.")
            return

        with open(JSON_ANNOTATION_FILE, 'r') as f:
            data = json.load(f)
            
        rows = []
        # Basic extraction
        videos = {v['id']: v for v in data.get('videos', [])}
        for ann in data.get('sentences', []):
            vid = ann['video_id']
            # Educational Filter (Category 13, 11, 6)
            if vid in videos and str(videos[vid].get('category')) in ['13','11','6']:
                path = os.path.join(VIDEO_SOURCE_DIR, f"{vid}.mp4")
                rows.append({'image_path': path, 'caption': ann['caption']})
        
        # If empty, fallback all
        if not rows:
            for ann in data.get('sentences', []):
                vid = ann['video_id']
                path = os.path.join(VIDEO_SOURCE_DIR, f"{vid}.mp4")
                rows.append({'image_path': path, 'caption': ann['caption']})
                
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Rank 0: Saved {len(df)} rows to {OUTPUT_CSV}")

def contrastive_loss(image_embeddings, text_embeddings, temperature=0.1):
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    logits = torch.matmul(image_embeddings, text_embeddings.T) / temperature
    labels = torch.arange(logits.shape[0]).to(logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2

# ====================
# TPU TRAIN FUNCTION
# ====================
def train_fn(index, flags):
    # 1. Setup Device
    device = xm.xla_device()
    rank = xm.get_ordinal()
    
    # 2. Prepare Data (Only Rank 0 does the file writing, others wait)
    if rank == 0:
        prepare_csv()
    xm.rendezvous('dataset_ready') # Wait for rank 0
    
    # 3. Load Dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Check if CSV exists (Rank 0 should have made it)
    if not os.path.exists(OUTPUT_CSV):
        if rank == 0: print("Error: CSV not found.")
        return

    dataset = NeuroClipDataset(OUTPUT_CSV, tokenizer, transform=transform)
    
    # 4. Distributed Sampler (CRITICAL FOR TPU)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=2,
        drop_last=True # Important for TPUs to have fixed shapes usually
    )

    # 5. Model Setup
    model = CustomNeuroClip(embed_dim=EMBED_DIM).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE * xm.xrt_world_size()) # Scale LR

    # 6. Training Loop
    model.train()
    for epoch in range(1, EPOCHS + 1):
        # Parallel Loader wrapper
        para_loader = pl.ParallelLoader(train_loader, [device])
        
        running_loss = 0.0
        count = 0
        
        # Loop over the parallelized loader
        for batch in para_loader.per_device_loader(device):
            optimizer.zero_grad()
            
            images = batch['image']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            img_emb, txt_emb = model(images, input_ids, attention_mask)
            loss = contrastive_loss(img_emb, txt_emb)
            
            loss.backward()
            xm.optimizer_step(optimizer) # XLA Optimizer Step
            
            running_loss += loss.item()
            count += 1
            
        # Log (Rank 0 only)
        avg_loss = running_loss / count
        # xm.master_print is safer than if rank==0: print
        xm.master_print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
        
    # 7. Save Model (Rank 0 only)
    xm.master_print("Saving Model...")
    xm.save(model.state_dict(), MODEL_SAVE_NAME)
    xm.master_print(f"Saved to {MODEL_SAVE_NAME}")

if __name__ == "__main__":
    # Spawns 8 processes on TPU v3-8
    xmp.spawn(train_fn, args=({},), nprocs=8, start_method='fork')

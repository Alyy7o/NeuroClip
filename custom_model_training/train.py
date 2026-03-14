import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import DistilBertTokenizer
import torch.nn.functional as F
import os
from tqdm import tqdm

from model import CustomNeuroClip
from dataset import NeuroClipDataset

# --- Configuration ---
BATCH_SIZE = 16   # Reduce if OOM
EPOCHS = 10       # Adjust based on dataset size
LEARNING_RATE = 1e-4
EMBED_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_PATH = "train_data.csv" # Placeholder
IMG_ROOT = "./images"       # Placeholder

def contrastive_loss(image_embeddings, text_embeddings, temperature=0.1):
    """
    Symmetric Contrastive Loss (InfoNCE).
    """
    # Normalize embeddings to lie on the hypersphere
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    
    # Cosine similarity matrix: (Batch, Batch)
    logits = torch.matmul(image_embeddings, text_embeddings.T) / temperature
    
    # Labels: The diagonal elements are the matches (0->0, 1->1, etc.)
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size).to(logits.device)
    
    # Calculate loss in both directions
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    return (loss_i2t + loss_t2i) / 2

def train(model, dataloader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    progress = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress:
        images = batch['image'].to(DEVICE)
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass
        img_emb, txt_emb = model(images, input_ids, attention_mask)
        
        # Compute loss
        loss = contrastive_loss(img_emb, txt_emb)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        progress.set_postfix(loss=loss.item())
        
    return running_loss / len(dataloader)

def main():
    print(f"Training on: {DEVICE}")
    
    # 1. Prepare Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Check if CSV exists, if not create dummy for user demo
    if not os.path.exists(CSV_PATH):
        print(f"⚠️  No dataset found at {CSV_PATH}. Creating a dummy csv for demonstration...")
        import pandas as pd
        dummy_df = pd.DataFrame({
            'image_path': ['example.jpg'] * 5, 
            'caption': ['This is a test caption' for _ in range(5)]
        })
        dummy_df.to_csv(CSV_PATH, index=False)
        # Create a dummy image too
        from PIL import Image
        Image.new('RGB', (224, 224)).save('example.jpg')
    
    dataset = NeuroClipDataset(CSV_PATH, tokenizer, transform=transform, root_dir=IMG_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Prepare Model
    model = CustomNeuroClip(embed_dim=EMBED_DIM).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training Loop
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, dataloader, optimizer, epoch)
        print(f"Epoch {epoch} complete. Avg Loss: {loss:.4f}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), f"neuroclip_v1_epoch{epoch}.pth")

    print("Training Complete. Model saved.")

if __name__ == "__main__":
    main()

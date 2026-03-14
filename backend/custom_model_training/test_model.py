import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import DistilBertModel, DistilBertTokenizer
from PIL import Image
import torch.nn.functional as F
import os

# ====================
# ====================
# CONFIGURATION
# ====================
# UPDATE THESE PATHS AFTER UPLOADING YOUR DATASET!
# They will look something like: /kaggle/input/your-dataset-name/neuroclip_v1.pth
MODEL_PATH = "/kaggle/input/my-neuroclip-model/neuroclip_v1.pth"
CSV_PATH = "/kaggle/input/my-neuroclip-model/train_data_educational.csv"

# Force CPU to avoid CUDA errors hanging the script
DEVICE = "cpu"
print("DEBUG: Script started...")
print(f"DEBUG: Using device: {DEVICE}")

# ====================
# NETWORK DEFINITION (Must Match Training!)
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

def load_model(path):
    print(f"Loading model from {path}...")
    model = CustomNeuroClip(embed_dim=256)
    
    # Load state dict
    try:
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None
        
    model.to(DEVICE)
    model.eval()
    return model

def predict(model, image_path, text_queries):
    # 1. Prepare Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        # Try loading as image
        image = Image.open(image_path).convert("RGB")
    except:
        # Try loading as video frame (for MSR-VTT)
        import cv2
        cap = cv2.VideoCapture(image_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        cap.release()
        
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # 2. Prepare Text
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    encoded_text = tokenizer(
        text_queries, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors="pt"
    ).to(DEVICE)
    
    # 3. Inference
    with torch.no_grad():
        # Get embeddings
        img_emb, txt_emb = model(img_tensor, encoded_text['input_ids'], encoded_text['attention_mask'])
        
        # Normalize
        img_emb = F.normalize(img_emb, p=2, dim=1)
        txt_emb = F.normalize(txt_emb, p=2, dim=1)
        
        # Dot Product (Cosine Similarity)
        # Shape: (1, dim) x (num_queries, dim)^T -> (1, num_queries)
        similarity = torch.matmul(img_emb, txt_emb.T)
        
    scores = similarity.cpu().numpy().flatten()
    return scores

def main():
    model = load_model(MODEL_PATH)
    if not model: return

    import pandas as pd
    import random
    
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Loaded dataset from {CSV_PATH} with {len(df)} samples.")
    except:
        print(f"Could not load CSV from {CSV_PATH}. Please check the path.")
        return

    # Run 3 Random Test Cases
    print("\nXXX RUNNING 3 RANDOM INTELLIGENCE TESTS XXX")
    
    for i in range(3):
        print(f"\n--- Test Case #{i+1} ---")
        
        # 1. Pick a random video and its ground truth caption
        idx = random.randint(0, len(df)-1)
        row = df.iloc[idx]
        image_path = row['image_path']
        truth_caption = str(row['caption'])
        
        print(f"Video File: {os.path.basename(image_path)}")
        print(f"Ground Truth: \"{truth_caption}\"")
        
        # 2. Pick 3 Random Distractors (captions from other videos)
        distractors = []
        while len(distractors) < 3:
            d_idx = random.randint(0, len(df)-1)
            if d_idx != idx:
                distractors.append(str(df.iloc[d_idx]['caption']))
        
        # 3. Combine and Shuffle
        queries = [truth_caption] + distractors
        # We won't shuffle the list order in print to keep track, or we can just print results sorted.
        
        # 4. Predict
        scores = predict(model, image_path, queries)
        
        # 5. Print Results (Sorted by score)
        results = list(zip(queries, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Model Predictions (Higher is better):")
        for q, s in results:
            marker = " [CORRECT]" if q == truth_caption else ""
            print(f"  Score {s:.4f}: \"{q}\"{marker}")
            
    print("\n-------------------------------------------")
    print("INTERPRETATION:")
    print("If the [CORRECT] caption is consistently at the top (highest score),")
    print("then the model successfully learned semantic intelligence!")

# CALL MAIN FUNCTION TO RUN SCRIPT
if __name__ == "__main__":
    main()
else:
    main()

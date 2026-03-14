# Building a Custom Multimodal (Vision-Language) Model for NeuroClip

This guide outlines how to implement, train, and integrate your own custom Computer Vision model for semantic search, replacing the need for external APIs like OpenAI's CLIP.

We will build a **Dual-Encoder (Two-Tower) Architecture** that aligns image frames and text captions in a shared embedding space using **Contrastive Learning**.

---

## 1. Architecture Design

To match text queries with video frames, we need two neural networks that "speak the same language" (mathematically).

### The "Two-Tower" Approach
1.  **Image Encoder:** Takes a video frame (pixel data) and outputs a vector (e.g., 512 numbers).
    *   *Recommendation:* Use a standard CNN (ResNet-50) or Vision Transformer (ViT) as the backbone.
2.  **Text Encoder:** Takes a search query or transcript segment and outputs a vector of the same size.
    *   *Recommendation:* Use a DistilBERT or BERT model as the backbone.
3.  **Projection Heads:** Simple linear layers that map the output of the backbones to a shared dimension (e.g., 256 or 512).

```mermaid
graph LR
    img["Image Frame"] --> VE["Image Encoder\n(e.g., ResNet50)"]
    VE --> P1["Projection Head"]
    P1 --> V_Emb["Image Embedding\n(Vector)"]

    txt["Text Query"] --> TE["Text Encoder\n(e.g., DistilBERT)"]
    TE --> P2["Projection Head"]
    P2 --> T_Emb["Text Embedding\n(Vector)"]

    V_Emb <-->|Contrastive Loss\n(Maximize Similarity)| T_Emb
```

---

## 2. Implementation Strategy (PyTorch)

You will need to create a Python file (e.g., `model_train.py`) to define the model and the training loop.

### Step A: Define the Model Class
We will use `torch`, `torchvision` (for the image backbone), and `transformers` (for the text backbone) just for the architecture blocks.

```python
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel, DistilBertConfig

class CustomNeuroClip(nn.Module):
    def __init__(self, embed_dim=256, frozen_backbones=False):
        super(CustomNeuroClip, self).__init__()
        
        # 1. Image Encoder (ResNet-50)
        # We remove the final classification layer (fc)
        resnet = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_projection = nn.Linear(2048, embed_dim)
        
        # 2. Text Encoder (DistilBERT)
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.text_projection = nn.Linear(768, embed_dim)
        
        # Optional: Freeze backbones to speed up training if dataset is small
        if frozen_backbones:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, images, input_ids, attention_mask):
        # Image Branch
        # output shape: (batch, 2048, 1, 1) -> flatten -> (batch, 2048)
        img_feat = self.image_encoder(images).flatten(1)
        image_embeddings = self.image_projection(img_feat)
        
        # Text Branch
        # use the [CLS] token representation (index 0)
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.last_hidden_state[:, 0, :]
        text_embeddings = self.text_projection(text_feat)
        
        return image_embeddings, text_embeddings
```

### Step B: The Loss Function (Contrastive Loss)
The goal is to make matched (image, text) pairs close in vector space, and unmatched pairs far apart. We use **InfoNCE** or Symmetric Cross Entropy Loss.

```python
import torch.nn.functional as F

def contrastive_loss(image_embeddings, text_embeddings, temperature=0.1):
    # Normalize embeddings
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    
    # Calculate Similarity Matrix (Batch x Batch)
    logits = torch.matmul(image_embeddings, text_embeddings.T) / temperature
    
    # Labels: The diagonal elements are the correct matches (0,0), (1,1), etc.
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size).to(logits.device)
    
    # Calculate loss for both directions (Image->Text and Text->Image)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    
    return (loss_i + loss_t) / 2
```

### Step C: Dataset Preparation
To train this, you need a dataset of Image-Text pairs.
*   **Source:** COCO Dataset, Flickr30k, or **Create your own** by extracting frames from your video library and retrieving their corresponding subtitle sentences.
*   **Format:** A CSV file with columns `[image_path, caption]`.

```python
from torch.utils.data import Dataset
from PIL import Image

class NeuroClipDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform=None):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row['image_path']).convert("RGB")
        caption = row['caption']
        
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
```

---

## 3. Training Loop Workflow

1.  **Initialize** the `CustomNeuroClip` model.
2.  **Load** your dataset using `DataLoader`.
3.  **Loop** through epochs:
    *   Forward pass images and texts.
    *   Compute `contrastive_loss`.
    *   Backpropagate and update weights (`optimizer.step()`).
4.  **Save** the trained model weights (`torch.save(model.state_dict(), 'neuroclip_custom.pth')`).

---

## 4. Integration into NeuroClip (main.py)

Once trained, you integrate it into your backend by replacing the current embedding calls.

**Existing (main.py):**
```python
# Old: Using library
# model = SentenceTransformer(...)
# vecs = model.encode(sentences)
```

**New (Custom):**
```python
# New: Load your custom model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CustomNeuroClip()
model.load_state_dict(torch.load("neuroclip_custom.pth"))
model.to(device)
model.eval()

def encode_text_custom(text):
    inputs = tokenizer(text, return_tensors="pt", ...).to(device)
    with torch.no_grad():
        _, text_emb = model(None, inputs['input_ids'], inputs['attention_mask'])
    return text_emb.cpu().numpy()

def encode_image_custom(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb, _ = model(image, None, None)
    return img_emb.cpu().numpy()
```

## 5. Summary
By following this path, you own the entire stack:
1.  **Data:** You can train on perfectly aligned data (e.g., frames specifically from your domain of videos).
2.  **Model:** You control the size and speed (ResNet18 vs ResNet50).
3.  **Privacy:** No data ever leaves your server.

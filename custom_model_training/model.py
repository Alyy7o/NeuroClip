import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel

class CustomNeuroClip(nn.Module):
    def __init__(self, embed_dim=256, frozen_backbones=False):
        super(CustomNeuroClip, self).__init__()
        
        # 1. Image Encoder (ResNet-50)
        # We remove the final classification layer (fc)
        # Default weights=ResNet50_Weights.IMAGENET1K_V1 implied by pretrained=True in older pytorch, 
        # but better to interpret 'pretrained=True' as 'pass weights' if available or use newer syntax.
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        # ResNet50 output is 2048-dim
        self.image_projection = nn.Linear(2048, embed_dim)
        
        # 2. Text Encoder (DistilBERT)
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # DistilBERT output is 768-dim
        self.text_projection = nn.Linear(768, embed_dim)
        
        # Freeze backbones if requested (faster training, less memory, lower accuracy)
        if frozen_backbones:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, images, input_ids, attention_mask):
        # --- Image Branch ---
        # images shape: (B, 3, H, W) -> (B, 2048, 1, 1)
        img_feat = self.image_encoder(images)
        # Flatten: (B, 2048)
        img_feat = img_feat.flatten(1)
        # Project to shared space: (B, embed_dim)
        image_embeddings = self.image_projection(img_feat)
        
        # --- Text Branch ---
        # text_out tuple: (last_hidden_state, ...)
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # We take the [CLS] token embedding (index 0) as the sentence ref
        text_feat = text_out.last_hidden_state[:, 0, :]
        # Project to shared space: (B, embed_dim)
        text_embeddings = self.text_projection(text_feat)
        
        return image_embeddings, text_embeddings

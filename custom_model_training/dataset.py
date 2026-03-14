import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class NeuroClipDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform=None, root_dir=""):
        """
        Args:
            csv_file (str or Path): Path to the CSV file with annotations.
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
            transform (callable, optional): Optional transform to be applied on a sample.
            root_dir (str): Directory with all the images (prefix for image_path).
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.transform = transform
        self.root_dir = root_dir

        # Ensure required columns exist
        required_cols = {'image_path', 'caption'}
        if not required_cols.issubset(self.data.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]
        img_name = row['image_path']
        
        # Handle relative or absolute paths
        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_name)
        else:
            img_path = img_name

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for missing/corrupt images during dev
            print(f"Warning: Could not load image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='gray')

        caption = str(row['caption']) # Ensure string
        
        if self.transform:
            image = self.transform(image)
            
        # Tokenize text
        text_input = self.tokenizer(
            caption, 
            padding='max_length', 
            truncation=True, 
            max_length=128, # DistilBERT supports up to 512, but 128 is faster/sufficient for captions
            return_tensors="pt"
        )
        
        return {
            'image': image,
            'input_ids': text_input['input_ids'].squeeze(0),      # Remove batch dim added by tokenizer
            'attention_mask': text_input['attention_mask'].squeeze(0)
        }

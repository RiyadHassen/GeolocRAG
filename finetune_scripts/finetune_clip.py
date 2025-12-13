import os
import argparse
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel, AdamW
from PIL import Image




def load_data_pairs(image_file, text_file):
    """
    Reads two files line-by-line and creates pairs.
    """
    if not os.path.exists(image_file) or not os.path.exists(text_file):
        raise FileNotFoundError("Could not find one of the data files.")

    with open(image_file, 'r', encoding='utf-8') as f_img:
        # strip() removes the newline character \n at the end
        img_paths = [line.strip() for line in f_img.readlines() if line.strip()]

    with open(text_file, 'r', encoding='utf-8') as f_txt:
        captions = [line.strip() for line in f_txt.readlines() if line.strip()]

    # Sanity Check
    if len(img_paths) != len(captions):
        print(f"WARNING: Mismatch detected!")
        print(f"Images: {len(img_paths)} | Texts: {len(captions)}")
        print("Truncating to the shorter length to prevent errors...")
        min_len = min(len(img_paths), len(captions))
        img_paths = img_paths[:min_len]
        captions = captions[:min_len]
    return list(zip(img_paths, captions))

class ImageTextDataset(Dataset):
    def __init__(self, data_list, processor):
        """
        Docstring for __init__
        
        :param self: Description
        :param data_list: Description
        :param processor: Description
        """
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)
    

    def __getitem(self,idx):
        image_path, text = self.data_list[idx]

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text = [text],
            image = image,
            return_tensors="pt",
            padding = "max_length",
            truncation =True,
            max_length = 77
        )

        return {
            "pixel_values": inputs["pixel-values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }
    


def contrastive_loss(logits_per_image, logits_per_text):
    """
    Standard CLIP Loss:
    Matches (Image_i, Text_i) should have high similarity.
    Everything else should have low similarity.
    """
    batch_size = logits_per_image.shape[0]
    # Create labels: 0, 1, 2, ... batch_size-1
    # This means the 0th image matches the 0th text, etc.
    labels = torch.arange(batch_size).to(device)
    
    loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
    
    return (loss_img + loss_txt) / 2



def main():

    parseargs = argparse.ArgumentParser()
    parseargs.add_argument("--image_file_path", required=False)
    parseargs.add_argument("--text-desc", required=False)
    #config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"
    # Hyperparameters
    BATCH_SIZE = 32     # Increase to 64 or 128 if your GPU allows (Critical for Contrastive Loss)
    EPOCHS = 3          # Keep low (1-5) to prevent overfitting
    LEARNING_RATE = 5e-6 
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # Initialize Model & Processor
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)

    train_data = load_data_pairs()
    # Create Loader
    dataset = ImageTextDataset(train_data, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable_params}")
    sys.exit(0)
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    print("Starting training...")

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Move batch to GPU
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward Pass
            outputs = model(
                input_ids=input_ids, 
                pixel_values=pixel_values, 
                attention_mask=attention_mask,
                return_loss=True 
            )
            
            # Calculate Loss
            # CLIPModel calculates loss internally if return_loss=True, 
            # but manual calculation gives you more control if needed.
            loss = outputs.loss 
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")

    # --- 5. SAVE THE FINE-TUNED MODEL ---
    model.save_pretrained("./fine-tuned-clip-rag")
    processor.save_pretrained("./fine-tuned-clip-rag")
    print("Model saved!")


if __name__ == "__name__":
    main()


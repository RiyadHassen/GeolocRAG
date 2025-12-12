import argparse
import json
import os
import torch
from typing import List, Dict, Any
from dataclasses import dataclass
from PIL import Image

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info


@dataclass
class GeoReasoningDataCollator:
    """Custom data collator for Qwen2-VL that handles images and text."""
    processor: Any
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract conversations and images
        texts = [f["messages"] for f in features]
        images = [f["images"] for f in features]
        
        # Process with Qwen2-VL processor
        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )
        
        # Create labels (copy input_ids and mask padding)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        # Mask the input portion (only train on assistant responses)
        for i, feature in enumerate(features):
            # Find where assistant response starts
            assistant_start_idx = feature.get("assistant_start_idx", 0)
            if assistant_start_idx > 0:
                labels[i, :assistant_start_idx] = -100
        
        batch["labels"] = labels
        return batch


class GeoReasoningDataset(torch.utils.data.Dataset):
    """
    Dataset for geo-reasoning fine-tuning with Qwen2-VL.
    Expects JSONL format with: image_path, latitude, longitude, country, contient, reason
    """
    def __init__(self, data_path: str, processor: Any):
        self.processor = processor
        self.data = []
        
        print(f"Loading dataset from: {data_path}")
        
        # Load JSONL data
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    
                    # Validate required fields
                    if not all(k in item for k in ["image_path", "latitude", "longitude", "country", "reason"]):
                        print(f"Line {line_num}: Missing required fields, skipping")
                        continue
                    
                    # Validate image exists
                    if not os.path.exists(item["image_path"]):
                        print(f"Line {line_num}: Image not found at {item['image_path']}, skipping")
                        continue
                    
                    self.data.append(item)
                    
                except json.JSONDecodeError as e:
                    print(f"Line {line_num}: JSON decode error - {e}")
                except Exception as e:
                    print(f"Line {line_num}: Error - {e}")
        
        print(f"Successfully loaded {len(self.data)} valid samples")
        
        if len(self.data) == 0:
            raise ValueError("No valid samples found in dataset!")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception as e:
            print(f"Error loading image {item['image_path']}: {e}")
            # Return a blank image as fallback
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        
        # Format instruction
        instruction = (
            "Analyze this image and provide geographical reasoning. "
            "Identify visual clues like natural features (climate, vegetation, terrain), "
            "man-made structures (roads, buildings, signage), and landmarks. "
            "Use these observations to deduce the location. "
            "Conclude with: Country, Continent, Latitude, Longitude."
        )
        
        # Format target response
        target = (
            f"{item['reason'].strip()} "
            f"Country: {item['country']}, "
            f"Continent: {item.get('contient', 'Unknown')}, "
            f"Coordinates: {item['latitude']:.6f}, {item['longitude']:.6f}"
        )
        
        # Create Qwen2-VL conversation format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target}
                ]
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        return {
            "messages": text,
            "images": [image],
            "assistant_start_idx": len(self.processor.tokenizer.encode(
                self.processor.apply_chat_template(
                    messages[:1], 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            ))
        }


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"Trainable params: {trainable_params:,} || "
          f"All params: {all_params:,} || "
          f"Trainable %: {100 * trainable_params / all_params:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-VL for geo-reasoning")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to Qwen2-VL model (e.g., Qwen/Qwen2-VL-7B-Instruct)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to JSONL training data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save fine-tuned model")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per GPU (use 1 for large models)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Accumulate gradients over N steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Model optimization
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_qlora", action="store_true",default=True,
                        help="Use QLoRA (4-bit quantization + LoRA)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--freeze_vision_tower", action="store_true", default=True,
                        help="Freeze the vision encoder")
    
    # Precision
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true")
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Qwen2-VL Geo-Reasoning Fine-tuning")
    print("=" * 80)
    print(f"Model: {args.model_name_or_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA: {args.use_lora}, QLoRA: {args.use_qlora}")
    print("=" * 80)
    
    # Load processor
    print("\n[1/5] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    
    # Set padding token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load model with optional quantization
    print("\n[2/5] Loading model...")
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16 if args.bf16 else torch.float16,
    }
    args.use_qlora = True # Always use QLoRA for this example
    if args.use_qlora:
        print("Using 4-bit quantization (QLoRA)...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        **model_kwargs
    )
    
    # Freeze vision tower if requested
    if args.freeze_vision_tower:
        print("Freezing vision tower...")
        if hasattr(model, "visual"):
            for param in model.visual.parameters():
                param.requires_grad = False
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Apply LoRA
    if args.use_lora:
        print("\n[3/5] Applying LoRA...")
        
        if args.use_qlora:
            model = prepare_model_for_kbit_training(
                model, 
                use_gradient_checkpointing=args.gradient_checkpointing
            )
        
        # Qwen2-VL LoRA target modules
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)
    
    # Load dataset
    print("\n[4/5] Loading dataset...")
    train_dataset = GeoReasoningDataset(args.data_path, processor)
    
    # Setup training arguments
    print("\n[5/5] Setting up training...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="tensorboard",
        ddp_find_unused_parameters=False if args.use_lora else None,
    )
    
    # Initialize trainer
    data_collator = GeoReasoningDataCollator(processor=processor)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    print("\n" + "=" * 80)
    print(f"Training complete! Model saved to {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
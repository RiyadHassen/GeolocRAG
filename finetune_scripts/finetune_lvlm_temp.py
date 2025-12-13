import argparse
import json
import os
import torch
from typing import List, Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration, 
    BitsAndBytesConfig,
    AutoProcessor
)
from trl import SFTTrainer, SFTConfig

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training # if using kbit training like qlora
)
from PIL import Image # For basic image validation if needed, though Qwen handles paths

# Helper function to format data into Qwen-VL conversational style
def format_qwen_vl_conversation(image_path: str, instruction: str, cot_response: str, tokenizer: AutoTokenizer) -> List[Dict[str, str]]:
    """
    Formats a single data point into the Qwen-VL expected conversational format.
    Example User turn: "Picture 1: <img>/path/to/image.jpg</img>\nWhat is this?"
    Example Assistant turn: "This is a cat."
    """
    # Check if image_path is absolute or make it relative to a known base if necessary
    # For simplicity, assuming image_path is usable as is by the model.
    if not os.path.exists(image_path):
        print(f"Warning: Image path {image_path} does not exist. Tokenizer might handle this error or it could fail.")

    user_content = f"Picture 1: <img>{image_path}</img>\n{instruction}"
    
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": cot_response}
    ]

class LVMFineTuningDataset(torch.utils.data.Dataset):
    """
    Dataset class for fine-tuning Qwen-VL.
    It processes the GeoLingo JSON, formats it into Qwen-VL's conversational style,
    and then tokenizes it.
    """
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, model_max_length: int):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        
        print(f"Loading dataset from: {data_path}")
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.raw_dataset_items = json.load(f)
            print(f"Successfully loaded {len(self.raw_dataset_items)} items.")
        except Exception as e:
            print(f"Error loading dataset from {data_path}: {e}")
            self.raw_dataset_items = []

        self.processed_data = self._prepare_data()

    def _prepare_data(self) -> List[Dict[str, Any]]:
        processed_samples = []
        if not self.raw_dataset_items:
            return []

        for item in self.raw_dataset_items:
            image_path = item.get("image_path")
            cot_description = item.get("reason") # This is the target for the assistant
            latitude = item.get("latitude") # Could be used in instruction if needed
            longitude = item.get("longitude") # Could be used in instruction if needed
            # address = item.get("address") # Could be used in instruction if needed

            if not image_path or not cot_description:
                print(f"Skipping item due to missing image_path or cot_description: {item.get('address', 'N/A')}")
                continue

            instruction = "Given an image, craft a brief and cohesive reasoning path that deduces this location based on the visual clues present in the image. Using a tone of exploration and inference. Carefully analyze and link observations of natural features (climate, vegetation, terrain), man-made structures (roads, buildings, signage), and distinct landmarks. Allow these observations to naturally lead you to the correct country, enhancing the accuracy of your deductions. Start the reasoning without any intro, and make sure to make it brief. Finally, conclude with the most probable country name contient and latitude and longitude coordinates."
            
            user_turn = f"Picture 1: <img>{image_path}</img>\n{instruction}"
            prompt_str = f"{self.tokenizer.bos_token if self.tokenizer.bos_token else ''}<|im_start|>user\n{user_turn}<|im_end|>\n<|im_start|>assistant\n"
            # The target completion
            completion_str = f"{cot_description}{self.tokenizer.eos_token if self.tokenizer.eos_token else ''}"
            full_text = prompt_str + completion_str

            tokenized_input = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.model_max_length,
                padding=False, # Do not pad here, DataCollator will handle it
                return_tensors=None, # Return lists for collator
                add_special_tokens=False # Already added bos/eos and role tokens manually for clarity
            )
            tokenized_prompt_only = self.tokenizer(
                prompt_str,
                truncation=True,
                max_length=self.model_max_length,
                padding=False,
                return_tensors=None,
                add_special_tokens=False
            )

            input_ids = tokenized_input["input_ids"]
            labels = list(input_ids) # Make a mutable copy
            prompt_len = len(tokenized_prompt_only["input_ids"])
            
            # Mask prompt tokens in labels
            for i in range(prompt_len):
                labels[i] = -100
            
            # Sanity check: ensure EOS token is not masked if it's part of completion
            if labels[-1] != -100 and input_ids[-1] == self.tokenizer.eos_token_id:
                pass # Correct, EOS is part of target
            elif labels[-1] == -100 and completion_str.endswith(self.tokenizer.eos_token if self.tokenizer.eos_token else '####'):
                 print(f"Warning: EOS token might have been masked for item with image {image_path}. Prompt length: {prompt_len}, Total length: {len(labels)}")

            if not any(label != -100 for label in labels):
                print(f"Warning: All labels are masked for item with image {image_path}. Prompt might be too long or completion empty. Skipping.")
                continue

            processed_samples.append({
                "input_ids": input_ids,
                "attention_mask": tokenized_input["attention_mask"],
                "labels": labels,
                # "image_path": image_path # Not strictly needed if image path is in input_ids via <img> tag
            })
        
        print(f"Finished processing {len(processed_samples)} samples.")
        return processed_samples

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen-VL Model.")
    # Arguments from the shell script
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen-VL-Chat", help="Path to pretrained model or model identifier.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per GPU for training.") # Adjusted from 10 for common setups
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay.")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="AdamW beta2.")
    parser.add_argument("--warmup_ratio", type=float, default=0.01, help="Warmup ratio.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type.")
    parser.add_argument("--logging_steps", type=int, default=1, help="Log every X steps.")
    parser.add_argument("--model_max_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Save strategy.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps (increased from 30).")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Limit total number of checkpoints.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Report to (e.g., wandb, tensorboard, none).")
    
    parser.add_argument("--bf16", action='store_true', default=True, help="Use bfloat16 precision.")
    parser.add_argument("--fp16", action='store_true', default=False, help="Use float16 precision (override bf16 if both true).")
    parser.add_argument("--fix_vit", action='store_true', default=True, help="Fix Vision Transformer weights.")
    parser.add_argument("--gradient_checkpointing", action='store_true', default=True, help="Enable gradient checkpointing.")
    parser.add_argument("--use_lora", action='store_true', default=True, help="Use LoRA.")

    parser.add_argument("--lora_r", type=int, default=64, help="LoRA r.") # Common Qwen LoRA r
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.") # Common Qwen LoRA alpha
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    # Qwen-VL target modules can vary, these are common. Check model config or print model to confirm.
    parser.add_argument("--lora_target_modules", nargs='+', default=["c_attn", "attn.c_proj", "mlp.w1", "mlp.w2"], help="Modules to apply LoRA to.")

    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--evaluation_strategy", type=str, default="no")
    parser.add_argument("--deepspeed", type=str, default=None)
    args = parser.parse_args()

    if args.fp16 and args.bf16:
        print("Both fp16 and bf16 are set. Defaulting to bf16 if available, else fp16.")
        if not torch.cuda.is_bf16_supported():
            args.bf16 = False # Fallback to fp16 if bf16 not supported
        else:
            args.fp16 = False 
    elif args.fp16:
        args.bf16 = False
    elif args.bf16 and not torch.cuda.is_bf16_supported():
        print("bf16 is set but not supported. Training will use fp32 or fp16 if specified.")
        args.bf16 = False # Disable if not supported, trainer will use fp32 unless fp16 is set

    print(f"--- Qwen-VL Fine-tuning Script ---")
    print(f"Arguments: {args}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Tokenizer


    # print(f"Loading Tokenizer: {args.model_name_or_path}")
    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.model_name_or_path,
    #     model_max_length=args.model_max_length,
    #     padding_side="right", # Important for generation
    #     use_fast=False,
    #     trust_remote_code=True
    # )
    

    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.eos_token_id 
    #     print(f"Set pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
    # # Qwen-VL specific: check for special tokens for chat format like <|im_start|>, <|im_end|> if not using chat template directly in SFT string construction
    # # For Qwen, they are usually part of the vocab, not special added tokens unless you add them.

    # 2. Load and Prepare Dataset
  

    # 3. Load Model
    print(f"Loading Model: {args.model_name_or_path}")
    model_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        device_map="auto", # or "cuda:0" if you want to specify a specific GPU
        quantization_config=bnb_config,
        torch_dtype=model_dtype,
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    print("Loading and Preparing Dataset...")
    train_dataset = LVMFineTuningDataset(
        data_path=args.data_path,
        tokenizer=processor.tokenizer,
        model_max_length=args.model_max_length
    )
    if not train_dataset:
        print("Dataset is empty or failed to load. Exiting.")
        return
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
        ]
    )

    training_args = SFTConfig(
    output_dir="qwen2-7b-instruct-trl-sft-ChartQA",  # Directory to save the model
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=4,  # Batch size for training
    per_device_eval_batch_size=4,  # Batch size for evaluation
    gradient_accumulation_steps=8,  # Steps to accumulate gradients
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    max_length=None,
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=2e-4,  # Learning rate for training
    # Logging and evaluation
    logging_steps=10,  # Steps interval for logging
    eval_steps=10,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    save_strategy="steps",  # Strategy for saving the model
    save_steps=20,  # Steps interval for saving
    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    # Hub and reporting
    push_to_hub=True,  # Whether to push model to Hugging Face Hub
    report_to="trackio",  # Reporting tool for tracking metrics
    )  


    trainer = SFTTrainer(
        model = model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config
    )
    print(f"Model loaded with dtype: {model_dtype}")
    trainer.train()
    print("Training completed.")
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")
    # if args.gradient_checkpointing:
    #     model.gradient_checkpointing_enable()
    #     print("Gradient checkpointing enabled.")

    # if args.fix_vit:
    #     if hasattr(model, 'visual') and model.visual is not None:
    #         model.visual.requires_grad_(False)
    #         # also check if specific layers within model.visual need to be set, e.g. model.visual.ln_post.requires_grad_(True) for some QWEN VIT
    #         # For simplicity, freezing all of model.visual
    #         print("Vision Transformer (model.visual) weights fixed.")
    #         trainable_params_visual = sum(p.numel() for p in model.visual.parameters() if p.requires_grad)
    #         print(f"Trainable parameters in model.visual: {trainable_params_visual}")
    #     else:
    #         print("Warning: --fix_vit is set, but model does not have a 'visual' attribute or it is None. ViT weights not fixed.")
    
    # if args.use_lora:
    #     print("Applying LoRA...")
    #     # For QLoRA or k-bit training, model needs to be prepared first
    #     # if args.bits == 4 or args.bits == 8: # Example if you add kbit training args
    #     #     model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        
    #     lora_config = LoraConfig(
    #         r=args.lora_r,
    #         lora_alpha=args.lora_alpha,
    #         lora_dropout=args.lora_dropout,
    #         bias="none",
    #         task_type=TaskType.CAUSAL_LM,
    #         target_modules=[
    #             "q_proj", "k_proj", "v_proj", "o_proj",
    #         ]
    #     )
    #     model = get_peft_model(model, lora_config)
    #     model.print_trainable_parameters()
    #     print("LoRA applied.")

    # 4. Set up Training Arguments
    # print("Setting up Training Arguments...")
    # training_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     num_train_epochs=args.num_train_epochs,
    #     per_device_train_batch_size=args.per_device_train_batch_size,
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     learning_rate=args.learning_rate,
    #     weight_decay=args.weight_decay,
    #     adam_beta2=args.adam_beta2,
    #     warmup_ratio=args.warmup_ratio,
    #     lr_scheduler_type=args.lr_scheduler_type,
    #     logging_steps=args.logging_steps,
    #     report_to=args.report_to,
    #     save_strategy=args.save_strategy,
    #     save_steps=args.save_steps,
    #     save_total_limit=args.save_total_limit,
    #     bf16=args.bf16,
    #     fp16=args.fp16,
    #     remove_unused_columns=False, # Important for custom datasets
    #     gradient_checkpointing=args.gradient_checkpointing, # Already enabled on model, but good to have in args too
    #     dataloader_num_workers=2, # Example, adjust based on your system
    #     dataloader_pin_memory=True,
    # )
    
    tokenizer = processor.tokenizer
    # 5. Initialize Trainer
    print("Initializing Trainer...")
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model, # Pass the actual model
        label_pad_token_id=-100, # Standard for ignoring padding in loss
        pad_to_multiple_of=8 if (args.fp16 or args.bf16) else None
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 6. Start Fine-tuning
    print("Starting Fine-tuning...")
    try:
        train_result = trainer.train()
        print(f"Fine-tuning completed. Train result: {train_result}")

        # Save the final model and tokenizer
        trainer.save_model(args.output_dir) # Saves adapter if LoRA, or full model if not
        tokenizer.save_pretrained(args.output_dir)
        print(f"Final model and tokenizer saved to {args.output_dir}")
        
        # You might want to save full model if using LoRA and need to merge later
        # if args.use_lora:
        #    merged_model = model.merge_and_unload() # If you want to save a merged model
        #    merged_model.save_pretrained(os.path.join(args.output_dir, "merged_model"))

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

    print("--- Fine-tuning Script Finished ---")

if __name__ == "__main__":
    # Check for CUDA
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Training will run on CPU, which will be very slow.")
    main() 
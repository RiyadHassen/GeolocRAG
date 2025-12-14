from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from PIL import Image
import torch
from peft import PeftModel
import argparse
import json

class InferencePipeline:
    def __init__(self, 
                base_model_path ="./Qwen-VL/Qwen-VL-Models/Qwen2-VL-Chat-Finetuned", 
                adapter_path = "./Qwen-VL/Qwen-VL-Adapters/qwen2-vl-chat-finetuned-geolocrag-adapter"):
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_path,
            dtype=torch.bfloat16,
            quantization_config=self.bnb_config,
            device_map="auto"
        )
        
        self.processor = Qwen2VLProcessor.from_pretrained(base_model_path)

        self.model  = PeftModel.from_pretrained(self.model, adapter_path)
        self.model = self.model.merge_and_unload()

        self.model.eval()

    def predict(self,imagepath, custom_instruction = None):
        # Format instruction
        if not custom_instruction:

            instruction = (
                "You are expert in analyzing image and predicting the location from the scence identify common clues to identify the locaition and "
                "return the contient, country, city , latitude longtiude of the given image output a json format as follow don't output extra result." \
                "{result:{\"country\":\"\", \"city\":\"\", \"latitude\":, \"longitude\":}}"
            )
        else:
            instruction = custom_instruction
        image = Image.open(imagepath).convert("RGB")
        # Create Qwen2-VL conversation format
        
        template = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
            
        # Apply chat template
        text = self.processor.apply_chat_template(
            template, 
            tokenizer =False,
            add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        # 4. Generate
        with torch.no_grad():
            print("Generating location deduction...")
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            
        # 5. Decode Output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text

    def predict_batch(self, jsonl_path, output_path=None):
        """
        Predict on multiple images from JSONL file.
        
        Args:
            jsonl_path: Path to JSONL file with image_path field
            output_path: Path to save results (optional)
        
        Returns:
            list of prediction results
        """
        results = []
        
        print(f"Processing batch from: {jsonl_path}\n")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    image_path = item.get("image_path")
                    
                    if not image_path:
                        print(f" Line {idx}: No image_path, skipping")
                        continue
                    
                    print(f"[{idx}] Processing: {image_path}")
                    
                    # Predict
                    result = self.predict(image_path)
                    
                    
                    results.append(result)
                    
                    # Print prediction
                    if "error" not in result:
                        print(f"Prediction: {result['prediction'][:100]}...")
                    else:
                        print(f"Error: {result['error']}")
                    print()
                    
                except json.JSONDecodeError as e:
                    print(f"Line {idx}: JSON error - {e}\n")
                except Exception as e:
                    print(f"Line {idx}: Error - {e}\n")
        
        # Save results
        if output_path and results:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_path}")
        
        print(f"\nBatch complete: {len(results)} predictions")
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fintune argument parser")
    parser.add_argument("--base_model_path", type =str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True, help = "Path to image to test")
    
    args = parser.parse_args()
    base_model = args.base_model_path
    img_path = args.image_path
    adapter_path = args.adapter_path

    infernce_pipe = InferencePipeline(base_model_path = base_model, adapter_path =adapter_path)
    print(infernce_pipe.predict(img_path))
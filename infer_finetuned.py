
from transformers import GenerationConfig
from PIL import Image
import torch


def infer_finetuned(model_path ="./Qwen-VL/Qwen-VL-Models/Qwen2-VL-Chat-Finetuned", imagepath= "./test-coconuts.png"):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    processor = Qwen2VLProcessor.frompretrained(model_path)

    text=  processor.apply_chat_template()

     # Format instruction
    instruction = (
        "You are expert in analyzing image and predicting the location from the scence identify common clues to identify the locaition and "
        "return the contient, country, city , latitude longtiude of the given image output a json format as follow don't output extra result." \
        "{result:{\"country\":\"\", \"city\":\"\", \"latitude\":, \"longitude\":}}"
    )
    image = Image.open(imagepath).convert("RGB")
    # Create Qwen2-VL conversation format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction}
            ]
        }
    ]
        
    # Apply chat template
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
        
    message = {
        "messages": text,
        "images": [image],
        "assistant_start_idx": len(processor.tokenizer.encode(
            processor.apply_chat_template(
                messages[:1], 
                tokenize=False, 
                add_generation_prompt=True
            )
        ))
    }

    generated_ids = model.generate(**messages, max_new_tokens=1024, generation_config=GenerationConfig(temperature=0.1, top_p=0.7, repetition_penalty=1.1))

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]


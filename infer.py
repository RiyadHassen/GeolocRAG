
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM, PeftModel
import torch
import os
import json
from tqdm import tqdm
import time

def infer_model(img_path,model_path="./Qwen-VL/Qwen-VL-Models/Qwen-VL-Chat"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:1", trust_remote_code=True, local_files_only=True).eval()

    lora_weights = "./Qwen-VL/LoRA/train_reason"
    model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.bfloat16)
    lora_weights = "./Qwen-VL/LoRA/train_loc"
    model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.bfloat16)

    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

    #img_path = "./test-coconuts.png"

    # query = tokenizer.from_list_format([
    #             {'image': img_path}, 
    #             {'text': "According to the content of the image, please think step by step and deduce in which country and city the image is most likely located and offer possible explanations. Output in JSON format, e.g., {'country': '', 'city': '' ,'reasons':''}."}
    #             ])
    
    # query = tokenizer.from_list_format([
    #             {'image': img_path},
    #             {'text': (
    #                 "You are an expert in image-based geo-localization.\n"
    #                 "According to the content of the image, please think step by step and deduce "
    #                 "in which country and city the image is most likely located, and offer possible explanations.\n"
    #                 "In addition, judge whether this scene is geographically synthetic in the real world.\n"
    #                 "If you find conflicting visual clues (e.g., climate, vegetation, road markings, language, architecture) "
    #                 "that make the scene unrealistic or impossible, mark it as synthetic and add reasons.\n\n"
    #                 "Output ONLY valid JSON in the following format:\n"
    #                 "e.g., {'country': '', 'city': '', 'synthetic': '', 'reasons':''}."
    #                 )}
    #             ])

    
    
    query = tokenizer.from_list_format([
                {'image': img_path},
                {'text': (
                    "You are an expert in image-based geo-localization similar to professional GeoGuessr players.\n"
                    "Analyze the provided image and provide step-by-step reasoning akin to Chain-of-Thought to determine the most likely location. Your guess should be a country and city pair.\n"
                    "Then, use your expertise in geo-localization to scrutinize the image to think of whether or not the image was doctored or modified. Provide step-by-step deductions to identify any visual cues or observations that conflict with your estimated location.\n"
                    "Once done, go back and reformat your response to match the following JSON format:\n"
                    "e.g., { 'prediction': {'country': '', 'city': '', 'reasoning': ''}, 'realness': {'is_fake': '', 'confidence': '', 'reasoning': ''} }"
                    )}
                ])

    # query = tokenizer.from_list_format([
    #     {'image': img_path},
    #     {'text': (
    #         "You are an expert in image-based geo-localization.\n"
    #         "According to the content of the image, please think step by step and deduce "
    #         "in which country and city the image is most likely located, and offer possible explanations.\n\n"

    #         "CRITICAL RULE (Conflict-First Policy):\n"
    #         "If ANY conflicting visual clues are detected, you MUST:\n"
    #         "- set 'synthesis' to 'true'\n"
    #         "- still output 'country' and 'city' based on the dominant clues if possible\n"
    #         "- explicitly list ALL conflicting clues inside 'reasons'.\n"
    #         "If you detect conflicts but set 'synthesis' to 'false' or omit 'synthesis', "
    #         "your answer is INVALID.\n\n"

    #         "Examples of conflicting clues include, but are not limited to:\n"
    #         "- tropical vegetation (e.g., palm trees) together with snow on the ground\n"
    #         "- Middle Eastern architecture together with Japanese road markings\n"
    #         "- left-hand traffic clues together with right-hand traffic clues\n"
    #         "- climate cues that are incompatible with vegetation or landscape\n"
    #         "- any combination of cues that cannot co-occur in the real world\n\n"

    #         "STEP 1 — VISUAL CUE EXTRACTION (MANDATORY):\n"
    #         "Inside 'reasons', you MUST include a section starting with 'VISUAL CUES:'.\n"
    #         "In this section, list ALL key visual clues in the image, including but not limited to:\n"
    #         "- vegetation types\n"
    #         "- climate cues (snow, humidity, dryness, sunlight)\n"
    #         "- architecture style\n"
    #         "- language or script on signs\n"
    #         "- road markings (yellow/white, dashed/solid)\n"
    #         "- traffic side (left/right)\n"
    #         "- soil color, terrain, mountains, coastline\n"
    #         "- car style, licence plates, Google car meta if visible\n"
    #         "If a cue is unknown, explicitly write 'unknown'.\n\n"

    #         "STEP 2 — GEO-CONSISTENCY REASONING (MANDATORY):\n"
    #         "Inside 'reasons', you MUST include a section starting with 'CONFLICTS:'.\n"
    #         "In this section, you MUST:\n"
    #         "- state whether the cues are mutually consistent or not;\n"
    #         "- if there are conflicts, list them explicitly, e.g., "
    #         "'palm trees' vs 'snow on the road' (tropical vs cold climate).\" \n"
    #         "If there are NO conflicts, you MUST write 'CONFLICTS: none'.\n"
    #         "If ANY conflicting clues are present, set 'synthesis' to 'true'.\n"
    #         "If there are NO conflicts, set 'synthesis' to 'false'.\n\n"

    #         "STEP 3 — GEO-REASONING (MANDATORY):\n"
    #         "Inside 'reasons', you MUST also include a section starting with 'GEO-REASONING:'.\n"
    #         "In this section, explain how the non-conflicting cues support your final choice of country and city.\n"
    #         "If cues are contradictory, still infer a country and city from the DOMINANT cues, "
    #         "and explain how you chose the dominant cues.\n\n"

    #         "FINAL OUTPUT FORMAT (STRICT):\n"
    #         "You MUST output ONLY valid JSON in the following format, with EXACTLY these keys:\n"
    #         "e.g., {'country': '', 'city': '', 'synthesis': 'true/false', 'reasons': ''}.\n"
    #         "'reasons' MUST be a single string containing all three sections in order:\n"
    #         "  'VISUAL CUES: ...; CONFLICTS: ...; GEO-REASONING: ...'\n"
    #     )}
    # ])

    
    response, history = model.chat(tokenizer, query=query, history=None)
    print(img_path, response)
    return response

if __name__ == "__main__":
    infer_model()
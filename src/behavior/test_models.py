"""
src/behavior/test_models.py
Comprehensive Stress Test / Pilot Study.
Updated with OOM & Bfloat16 protections.
"""

import torch
import pandas as pd
import json
import sys
import gc
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from jinja2.exceptions import TemplateError

sys.path.append(str(Path.cwd()))

from src.behavior.wa_generate import (
    MODELS_CONFIG, 
    SYSTEM_PROMPT, 
    USER_TEMPLATE, 
    parse_output, 
    load_swow_cues,
    get_model_dtype_and_device # Reuse the logic
)

OUTPUT_FILE = Path("outputs/raw_behavior/model_pilot_report.csv")
SWOW_PATH = Path("data/SWOW/Human_SWOW-EN.R100.20180827.csv")

N_CUES_TO_TEST = 50       
TARGET_SAMPLES = 100      
BATCH_SIZE = 10           

def setup_environment(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def build_prompt_safe(tokenizer, cue):
    content = USER_TEMPLATE.format(cue)
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except (TemplateError, ValueError):
        merged_content = f"{SYSTEM_PROMPT}\n\n{content}"
        messages = [{"role": "user", "content": merged_content}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def run_stress_test():
    print(f"Starting Comprehensive Pilot Test (BF16 + Auto Map)")
    print(f"Output: {OUTPUT_FILE}")
    
    setup_environment()
    all_cues = load_swow_cues(SWOW_PATH)
    random.shuffle(all_cues)
    test_cues = all_cues[:N_CUES_TO_TEST]

    report_rows = []

    for model_key, model_name in MODELS_CONFIG.items():
        if model_key == "debug-model": continue 
        
        print(f"\n[PILOT] Evaluating Model: {model_key} ({model_name})")
        
        # --- Load Model Safe ---
        device_name, use_dtype = get_model_dtype_and_device()
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            
            # device_map="auto" is key for L40S splitting 
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=use_dtype,
                device_map="auto" if device_name == "cuda" else None
            )
            if device_name == "mps": model.to("mps")
            model.eval()
        except Exception as e:
            print(f"[ERROR] Load failed for {model_key}: {e}")
            report_rows.append({"model": model_key, "status": "LOAD_FAIL", "yield_rate": 0.0})
            continue

        use_chat = hasattr(tokenizer, "apply_chat_template")

        total_generations = 0
        valid_triplets = 0
        
        pbar = tqdm(test_cues, desc=f"  {model_key}")
        
        for cue in pbar:
            if use_chat:
                txt = build_prompt_safe(tokenizer, cue)
            else:
                txt = f"{SYSTEM_PROMPT}\n\n{USER_TEMPLATE.format(cue)}\nAnswer:"
            
            # Important: Send inputs to model.device (it handles the splits)
            inputs = tokenizer([txt], return_tensors="pt").to(model.device)
            input_len = inputs.input_ids.shape[1]
            
            passes = TARGET_SAMPLES // BATCH_SIZE 
            
            for _ in range(passes):
                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=40,
                            do_sample=True,
                            temperature=1.0,
                            top_p=0.9,
                            num_return_sequences=BATCH_SIZE,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    
                    gen_tokens = outputs[:, input_len:]
                    decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
                    
                    for text in decoded:
                        total_generations += 1
                        parsed = parse_output(text)
                        if len(parsed) == 3:
                            valid_triplets += 1
                            
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("    [FATAL] OOM during generation. Skipping model.")
                        break
                    else:
                        # print(f"    [WARN] {e}")
                        pass

            current_yield = (valid_triplets / total_generations) if total_generations > 0 else 0
            pbar.set_postfix({"Yield": f"{current_yield:.1%}"})

        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        final_yield = (valid_triplets / total_generations) if total_generations > 0 else 0
        print(f"  [RESULT] {model_key}: Yield Rate = {final_yield:.1%} ({valid_triplets}/{total_generations})")
        
        report_rows.append({
            "model": model_key,
            "status": "SUCCESS",
            "yield_rate": final_yield
        })

        pd.DataFrame(report_rows).to_csv(OUTPUT_FILE, index=False)

    print(f"\n[DONE] Pilot Report saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_stress_test()
"""
src/behavior/test_models.py

Comprehensive Stress Test / Pilot Study.
Runs a statistically significant sample (50 cues x 100 samples) across ALL models.

Goal: Calculate the "Yield Rate" (percentage of valid, parsable triplets)
to ensure models are behaving correctly before the full run.

Usage: python src/behavior/test_models.py
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

# Add project root to path
sys.path.append(str(Path.cwd()))

# Import configs and logic from the main pipeline to ensure parity
from src.behavior.wa_generate import (
    MODELS_CONFIG, 
    SYSTEM_PROMPT, 
    USER_TEMPLATE, 
    parse_output, 
    load_swow_cues
)

# --- CONFIGURATION ---
OUTPUT_FILE = Path("outputs/raw_behavior/model_pilot_report.csv")
SWOW_PATH = Path("data/SWOW/Human_SWOW-EN.R100.20180827.csv")

# Stress Test Scale
N_CUES_TO_TEST = 50       # 50 Cues
TARGET_SAMPLES = 100      # 100 Samples per cue (Full Depth)
BATCH_SIZE = 10           # Generate 10 samples per forward pass

def setup_environment(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def run_stress_test():
    print(f"Starting Comprehensive Pilot Test")
    print(f"Target: {len(MODELS_CONFIG)} models x {N_CUES_TO_TEST} cues x {TARGET_SAMPLES} samples")
    print(f"Output: {OUTPUT_FILE}")
    
    setup_environment()
    
    # 1. Load Data
    # We take a fixed random sample of 50 cues to ensure every model sees the same words
    all_cues = load_swow_cues(SWOW_PATH)
    random.shuffle(all_cues)
    test_cues = all_cues[:N_CUES_TO_TEST]
    print(f"Selected 50 Test Cues: {test_cues[:5]}...")

    report_rows = []

    # 2. Iterate Models
    for model_key, model_name in MODELS_CONFIG.items():
        if model_key == "debug-model": continue # Skip debug model
        
        print(f"\n[PILOT] Evaluating Model: {model_key} ({model_name})")
        
        # --- Load Model (Same logic as production) ---
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        use_dtype = torch.float32 if device == "mps" else torch.float16 # Fix for Mac/MPS

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=use_dtype,
                device_map="auto" if device == "cuda" else None
            )
            if device == "mps": model.to(device)
            model.eval()
        except Exception as e:
            print(f"[ERROR] Load failed for {model_key}: {e}")
            report_rows.append({"model": model_key, "status": "LOAD_FAIL", "yield_rate": 0.0})
            continue

        # Check Chat Template
        use_chat = True
        try:
            tokenizer.apply_chat_template([{"role":"user", "content":"test"}], tokenize=False)
        except:
            use_chat = False

        # --- Generation Loop ---
        total_generations = 0
        valid_triplets = 0
        
        # Progress bar for cues
        pbar = tqdm(test_cues, desc=f"  {model_key}")
        
        for cue in pbar:
            # Prepare Prompt
            content = USER_TEMPLATE.format(cue)
            if use_chat:
                txt = tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": content}],
                    tokenize=False, add_generation_prompt=True
                )
            else:
                txt = f"{SYSTEM_PROMPT}\n\n{content}\nAnswer:"
            
            # Prepare Batch
            # We want 100 samples. We run batches of 10.
            # So we repeat the prompt 10 times in the input tensor? 
            # OR we simply run model.generate(num_return_sequences=10) 10 times.
            # Running generate 10 times is safer for memory.
            
            inputs = tokenizer([txt], return_tensors="pt").to(device)
            input_len = inputs.input_ids.shape[1]
            
            cue_valid_count = 0
            
            # 10 passes of 10 samples = 100 samples
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
                            num_return_sequences=BATCH_SIZE, # Generate 10 variants
                            pad_token_id=tokenizer.pad_token_id
                        )
                    
                    gen_tokens = outputs[:, input_len:]
                    decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
                    
                    for text in decoded:
                        total_generations += 1
                        parsed = parse_output(text)
                        if len(parsed) == 3:
                            valid_triplets += 1
                            cue_valid_count += 1
                            
                except RuntimeError as e:
                    if "probability tensor" in str(e):
                        print("    [WARN] NaN detected, skipping batch.")
                    else:
                        print(f"    [ERROR] {e}")

            # Update pbar with current stats
            current_yield = (valid_triplets / total_generations) if total_generations > 0 else 0
            pbar.set_postfix({"Yield": f"{current_yield:.1%}"})

        # --- Cleanup ---
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        # --- Log Stats ---
        final_yield = (valid_triplets / total_generations) if total_generations > 0 else 0
        print(f"  [RESULT] {model_key}: Yield Rate = {final_yield:.1%} ({valid_triplets}/{total_generations})")
        
        report_rows.append({
            "model": model_key,
            "status": "SUCCESS",
            "cues_tested": N_CUES_TO_TEST,
            "samples_per_cue": TARGET_SAMPLES,
            "total_generations": total_generations,
            "valid_triplets": valid_triplets,
            "yield_rate": final_yield
        })

        # Intermediate Save
        pd.DataFrame(report_rows).to_csv(OUTPUT_FILE, index=False)

    print(f"\n[DONE] Pilot Report saved to {OUTPUT_FILE}")
    print(pd.DataFrame(report_rows)[['model', 'yield_rate']])

if __name__ == "__main__":
    run_stress_test()
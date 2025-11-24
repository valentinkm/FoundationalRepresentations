"""
src/behavior/test_models.py

Diagnostic Suite for Active Behavior Pipeline.
Runs a small sample of cues across ALL models defined in generate.py.
Captures RAW output to debug formatting/parsing issues.

Usage: python src/behavior/test_models.py
"""

import torch
import pandas as pd
import json
import sys
import gc
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import configuration and parsing logic from the main script
# (Assumes running from project root)
sys.path.append(str(Path.cwd()))
from src.behavior.generate import MODELS_CONFIG, SYSTEM_PROMPT, USER_TEMPLATE, parse_output

# --- CONFIG ---
OUTPUT_FILE = Path("outputs/raw_behavior/model_diagnostics.csv")
TEST_CUES = ["dog", "justice", "run", "beautiful", "telephone"] # Diverse sample
N_SAMPLES = 1 # Just one generation per cue to check formatting

def run_diagnostics():
    print(f"Starting Model Diagnostics on {len(MODELS_CONFIG)} models.")
    print(f"Test Cues: {TEST_CUES}")
    
    results = []
    
    # Iterate over every model in the config
    for model_key, model_name in MODELS_CONFIG.items():
        if model_key == "debug-model": continue # Skip the tiny debug model
        
        print(f"\n[DIAGNOSTIC] Testing Model: {model_key} ({model_name})")
        
        # 1. Load Model (Standard Logic)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            model.eval()
        except Exception as e:
            print(f"[ERROR] Could not load {model_key}: {e}")
            results.append({
                "model": model_key,
                "status": "LOAD_FAILURE",
                "error": str(e)
            })
            continue

        # 2. Check Chat Template
        use_chat = True
        try:
            tokenizer.apply_chat_template([{"role":"user", "content":"test"}], tokenize=False)
        except:
            use_chat = False
            print("[INFO] No chat template found, using raw prompting.")

        # 3. Generate for Test Cues
        for cue in TEST_CUES:
            # Build Prompt
            content = USER_TEMPLATE.format(cue)
            if use_chat:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": content}],
                    tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = f"{SYSTEM_PROMPT}\n\n{content}\nAnswer:"

            inputs = tokenizer([prompt], return_tensors="pt").to(device)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=40,
                        do_sample=True,
                        temperature=0.7, # Slightly lower temp for stability in tests
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                # Decode
                input_len = inputs.input_ids.shape[1]
                gen_tokens = outputs[:, input_len:]
                raw_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                
                # Test Parsing
                parsed = parse_output(raw_text)
                
                status = "SUCCESS" if len(parsed) == 3 else "PARTIAL_FAIL"
                if len(parsed) == 0: status = "PARSE_FAIL"
                
                results.append({
                    "model": model_key,
                    "cue": cue,
                    "status": status,
                    "parsed_count": len(parsed),
                    "parsed_output": str(parsed),
                    "raw_output": raw_text.replace('\n', '\\n') # Escape newlines for CSV
                })
                print(f"  Cue: {cue:<10} | {status:<10} | Raw: {raw_text[:50]}...")

            except Exception as e:
                print(f"  [ERROR] Generation failed for {cue}: {e}")
                results.append({
                    "model": model_key,
                    "cue": cue,
                    "status": "GEN_FAILURE",
                    "error": str(e)
                })

        # 4. Cleanup VRAM (Crucial for iterating multiple models)
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print("[INFO] Unloaded model and cleared VRAM.")

    # Save Report
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[DONE] Diagnostic report saved to {OUTPUT_FILE}")
    print("\n--- Failure Summary ---")
    print(df[df['status'] != 'SUCCESS'][['model', 'cue', 'status', 'raw_output']])

if __name__ == "__main__":
    run_diagnostics()
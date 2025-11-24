"""
src/behavior/wa_generate.py

Active Behavior Pipeline:
Generates free associations from LLMs using local HuggingFace Transformers.
Designed for Scientific Reproducibility with hardcoded configs.

Goal: Generate 100 triplet responses per cue to mirror Human SWOW data structure.
"""

import argparse
import json
import torch
import pandas as pd
import numpy as np
import random
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# =============================================================================
# 1. SCIENTIFIC CONFIGURATION (HARDCODED)
# =============================================================================

MODELS_CONFIG = {
    # --- PRODUCTION MODELS ---
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct", 
    "mistral-nemo-12b":      "mistralai/Mistral-Nemo-Instruct-v1",
    "qwen2.5-32b-instruct":  "Qwen/Qwen2.5-32B-Instruct",
    "gemma-2-27b-it":        "google/gemma-2-27b-it",
    "olmo-2-7b-instruct":    "allenai/OLMo-2-1124-7B-Instruct",
    "falcon-3-10b-instruct": "tiiuae/Falcon-3-10B-Instruct",
    
    # --- TEST/DEBUG MODEL ---
    "debug-model":           "meta-llama/Llama-3.1-8B-Instruct" 
}

# Generation Parameters
PARAMS = {
    "temperature": 1.0,       
    "top_p": 0.9,             
    "max_new_tokens": 40,     
    "repetition_penalty": 1.1 
}

# Data Scale
SAMPLES_PER_CUE = 100         
BATCH_SIZE_PER_PASS = 10      

# Prompt
SYSTEM_PROMPT = "You are a participant in a word association study."
USER_TEMPLATE = "List exactly 3 words associated with: '{}'. Return only the words separated by commas. Do not write full sentences."

# =============================================================================
# 2. UTILITIES
# =============================================================================

def setup_environment(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def load_swow_cues(csv_path: Path, limit: int = None) -> list[str]:
    print(f"Loading cues from {csv_path}...")
    df = pd.read_csv(csv_path)
    cues = sorted(df['cue'].astype(str).unique().tolist())
    if limit:
        print(f"[TEST] Limiting to first {limit} cues.")
        cues = cues[:limit]
    print(f"Loaded {len(cues)} unique cues.")
    return cues

def parse_output(text: str) -> list[str]:
    """Clean model output into a list of 3 words."""
    text = text.lower().strip()
    
    # Filter refusals
    if any(x in text for x in ["i cannot", "as an ai", "sorry", "model", "language model"]):
        return []

    # Filter thinking tokens (common in Qwen/DeepSeek)
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # Normalize delimiters: replace newlines and bullets with commas
    text = re.sub(r'[\n\r]', ',', text)
    text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE) # Remove "1. "
    text = re.sub(r'^-\s*', '', text, flags=re.MULTILINE)     # Remove "- "
    
    parts = [p.strip() for p in text.split(',')]
    
    # Clean non-alpha characters from individual words
    clean = []
    for p in parts:
        # Keep only letters, numbers, hyphens, spaces
        w = "".join([c for c in p if c.isalnum() or c in "- "]).strip()
        # Filter out meta-text
        if w and w not in ["sure", "here", "association"]:
            clean.append(w)
        
    # Deduplicate preserving order
    seen = set()
    final = []
    for w in clean:
        if w not in seen:
            final.append(w)
            seen.add(w)
        if len(final) == 3: break
            
    return final

# =============================================================================
# 3. CORE GENERATION LOOP
# =============================================================================

def run_generation(model_key: str, swow_path: Path, output_dir: Path, is_test: bool):
    setup_environment()
    
    # 1. Resolve Configuration
    if is_test:
        model_name = MODELS_CONFIG["debug-model"]
        target_samples = 2
        samples_per_pass = 2
        cue_limit = 5
        print(f"[TEST] RUNNING IN TEST MODE (Model: {model_name})")
    else:
        if model_key not in MODELS_CONFIG:
            raise ValueError(f"Model '{model_key}' not found in hardcoded config.")
        model_name = MODELS_CONFIG[model_key]
        target_samples = SAMPLES_PER_CUE
        samples_per_pass = BATCH_SIZE_PER_PASS
        cue_limit = None
        print(f"[PROD] RUNNING IN PRODUCTION MODE (Model: {model_name})")

    # 2. Load Model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # DETERMINE DTYPE: MPS requires float32 for stability in sampling
    if device == "mps":
        use_dtype = torch.float32
        print("[INFO] MPS detected: Forcing float32 for generation stability.")
    elif device == "cuda":
        use_dtype = torch.float16
    else:
        use_dtype = torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None: 
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=use_dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "mps": model.to(device)
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # Check for Chat Template support
    use_chat = True
    try:
        tokenizer.apply_chat_template([{"role":"user", "content":"test"}], tokenize=False)
    except:
        use_chat = False
        print("[WARNING] Model does not support chat templates. Falling back to raw prompting.")

    # 3. Prepare Data
    cues = load_swow_cues(swow_path, limit=cue_limit)
    
    output_file = output_dir / f"{model_key}.jsonl"
    print(f"Output target: {output_file}")
    
    processed_cues = set()
    if output_file.exists():
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_cues.add(data['cue'])
                except: pass
        print(f"Resuming: Found {len(processed_cues)} cues already processed.")
    
    cues_to_process = [c for c in cues if c not in processed_cues]

    # 4. Processing Loop
    CUE_BATCH_SIZE = 8 if not is_test else 2
    passes_needed = (target_samples + samples_per_pass - 1) // samples_per_pass

    with open(output_file, 'a') as f_out:
        for i in tqdm(range(0, len(cues_to_process), CUE_BATCH_SIZE), desc="Processing Cues"):
            batch_cues = cues_to_process[i : i + CUE_BATCH_SIZE]
            
            # Prepare Prompts
            prompts = []
            for cue in batch_cues:
                content = USER_TEMPLATE.format(cue)
                if use_chat:
                    txt = tokenizer.apply_chat_template(
                        [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": content}],
                        tokenize=False, add_generation_prompt=True
                    )
                else:
                    txt = f"{SYSTEM_PROMPT}\n\n{content}\nAnswer:"
                prompts.append(txt)

            # Tokenize
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            input_len = inputs.input_ids.shape[1]

            # REPEAT GENERATION
            for p_idx in range(passes_needed):
                current_samples = min(samples_per_pass, target_samples - (p_idx * samples_per_pass))
                if current_samples <= 0: break

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=PARAMS["max_new_tokens"],
                        do_sample=True,
                        temperature=PARAMS["temperature"],
                        top_p=PARAMS["top_p"],
                        repetition_penalty=PARAMS["repetition_penalty"],
                        num_return_sequences=current_samples,
                        pad_token_id=tokenizer.pad_token_id
                    )

                # Decode
                gen_tokens = outputs[:, input_len:]
                decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

                for idx, text in enumerate(decoded):
                    cue_idx = idx // current_samples
                    original_cue = batch_cues[cue_idx]
                    
                    responses = parse_output(text)
                    
                    if len(responses) > 0:
                        record = {
                            "cue": original_cue,
                            "responses": responses,
                            "model": model_key,
                            "config": PARAMS
                        }
                        f_out.write(json.dumps(record) + "\n")
                
                f_out.flush()

    print("[DONE] Generation Complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, required=False, help="Key from MODELS_CONFIG")
    parser.add_argument("--test", action="store_true", help="Run with debug model and small limits")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent
    swow_path = project_root / 'data' / 'SWOW' / 'Human_SWOW-EN.R100.20180827.csv'
    output_dir = project_root / 'outputs' / 'raw_behavior' / 'model_swow'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.test:
        run_generation("debug-model", swow_path, output_dir, is_test=True)
    elif args.model_key:
        run_generation(args.model_key, swow_path, output_dir, is_test=False)
    else:
        print("Please provide --model_key OR --test")
        print("Available keys:", list(MODELS_CONFIG.keys()))

if __name__ == "__main__":
    main()
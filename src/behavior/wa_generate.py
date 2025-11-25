"""
src/behavior/wa_generate.py

Active Behavior Pipeline:
Generates free associations from LLMs using local HuggingFace Transformers.
Designed for Scientific Reproducibility with hardcoded configs.
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
    # --- LLAMA FAMILY ---
    "llama-3.1-8b-instruct":      "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.3-70b-instruct":     "meta-llama/Llama-3.3-70B-Instruct",
    
    # --- MISTRAL FAMILY ---
    # Mistral Small 24B (v3) - Requires "Agree" on HF Model Card
    "mistral-small-24b-instruct": "mistralai/Mistral-Small-24B-Instruct-2501",
    
    # --- QWEN FAMILY ---
    # Qwen 2.5 is the current SOTA. Qwen 3 is not released yet.
    "qwen-3-32b-instruct":        "Qwen/Qwen2.5-32B-Instruct",
    
    # --- FALCON FAMILY ---
    "falcon-3-10b-instruct":      "tiiuae/Falcon3-10B-Instruct",
    
    # --- GEMMA FAMILY ---
    # Gemma 2 is the current SOTA.
    "gemma-3-27b-instruct":       "google/gemma-2-27b-it",
    
    # --- GPT-OSS FAMILY ---
    "gpt-oss-120b-instruct":      "openai/gpt-oss-120b", 
    "gpt-oss-20b-instruct":       "openai/gpt-oss-20b",

    # --- TEST/DEBUG MODEL ---
    "debug-model":                "meta-llama/Llama-3.2-1B-Instruct" 
}

# Generation Parameters
PARAMS = {
    "temperature": 1.0,       
    "top_p": 0.9,             
    "max_new_tokens": 40,     
    "repetition_penalty": 1.1 
}

SAMPLES_PER_CUE = 100         
BATCH_SIZE_PER_PASS = 10      

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
    
    if any(x in text for x in ["i cannot", "as an ai", "sorry", "model", "language model"]):
        return []

    if "</think>" in text:
        text = text.split("</think>")[-1]

    text = re.sub(r'[\n\r]', ',', text)
    text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^-\s*', '', text, flags=re.MULTILINE)
    
    parts = [p.strip() for p in text.split(',')]
    
    clean = []
    for p in parts:
        w = "".join([c for c in p if c.isalnum() or c in "- "]).strip()
        if w and w not in ["sure", "here", "association"]:
            clean.append(w)
        
    seen = set()
    final = []
    for w in clean:
        if w not in seen:
            final.append(w)
            seen.add(w)
        if len(final) == 3: break
            
    return final

def get_model_dtype_and_device():
    """Determine best precision/device to avoid OOM and NaNs."""
    if torch.backends.mps.is_available():
        return "mps", torch.float32 # MPS needs float32
    elif torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16 # Best for L40S/A100
        return "cuda", torch.float16 # Fallback
    return "cpu", torch.float32

# =============================================================================
# 3. CORE GENERATION LOOP
# =============================================================================

def run_generation(model_key: str, swow_path: Path, output_dir: Path, is_test: bool):
    setup_environment()
    
    if is_test:
        model_name = MODELS_CONFIG["debug-model"]
        target_samples = 2
        samples_per_pass = 2
        cue_limit = 5
    else:
        if model_key not in MODELS_CONFIG:
            raise ValueError(f"Model '{model_key}' not found config.")
        model_name = MODELS_CONFIG[model_key]
        target_samples = SAMPLES_PER_CUE
        samples_per_pass = BATCH_SIZE_PER_PASS
        cue_limit = None

    device_name, use_dtype = get_model_dtype_and_device()
    print(f"Loading {model_key} ({model_name}) on {device_name} with {use_dtype}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        # device_map="auto" automatically spreads layers across GPUs (fixes OOM)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=use_dtype,
            device_map="auto" if device_name == "cuda" else None,
        )
        if device_name == "mps": model.to("mps")
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # Chat Template Detection
    use_chat = True
    try:
        tokenizer.apply_chat_template([{"role":"user", "content":"test"}], tokenize=False)
    except:
        use_chat = False

    cues = load_swow_cues(swow_path, limit=cue_limit)
    output_file = output_dir / f"{model_key}.jsonl"
    
    processed_cues = set()
    if output_file.exists():
        with open(output_file, 'r') as f:
            for line in f:
                try: processed_cues.add(json.loads(line)['cue'])
                except: pass
    
    cues_to_process = [c for c in cues if c not in processed_cues]
    CUE_BATCH_SIZE = 8 if not is_test else 2
    passes_needed = (target_samples + samples_per_pass - 1) // samples_per_pass

    with open(output_file, 'a') as f_out:
        for i in tqdm(range(0, len(cues_to_process), CUE_BATCH_SIZE), desc="Processing Cues"):
            batch_cues = cues_to_process[i : i + CUE_BATCH_SIZE]
            
            prompts = []
            for cue in batch_cues:
                content = USER_TEMPLATE.format(cue)
                if use_chat:
                    # Robust system prompt handling
                    try:
                        prompts.append(tokenizer.apply_chat_template(
                            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
                            tokenize=False, add_generation_prompt=True
                        ))
                    except:
                        # Fallback for models like Gemma that hate system roles
                        merged = f"{SYSTEM_PROMPT}\n\n{content}"
                        prompts.append(tokenizer.apply_chat_template(
                            [{"role": "user", "content": merged}],
                            tokenize=False, add_generation_prompt=True
                        ))
                else:
                    prompts.append(f"{SYSTEM_PROMPT}\n\n{content}\nAnswer:")

            # Ensure inputs are on the same device as the model (model.device handles map='auto')
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            input_len = inputs.input_ids.shape[1]

            for p_idx in range(passes_needed):
                current_samples = min(samples_per_pass, target_samples - (p_idx * samples_per_pass))
                if current_samples <= 0: break

                try:
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
                except RuntimeError as e:
                    print(f"[WARN] Generation Error (skipping batch): {e}")
                
                f_out.flush()

    print("[DONE] Generation Complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, required=False, help="Key from MODELS_CONFIG")
    parser.add_argument("--test", action="store_true", help="Run with debug model")
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
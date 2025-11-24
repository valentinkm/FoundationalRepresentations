#!/usr/bin/env python3
# score.py
"""
- Batched scoring script for word association.
- Resume-from-checkpoint (skips swow_id already saved in CSV).
- Select model via --model [model_key].
- Verbose per-item smoketest with boundary-aligned masking.
- ALWAYS excludes commas and whitespace-only tokens from scoring
  (we still pass the full assistant response string with separators;
   we only mask those separator tokens from loss).
- Optional: override per-model batch size via --batch-size N.
- Live progress logging: rows/sec and rough ETA.
- Optional deranged dataset integration via --use-deranged (auto regenerate + cache).
- Sequential multi-model execution via --run-all with automatic cache clearing.
"""
import argparse
import gc
import importlib.util
import logging
import os
import time
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Sequence

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Logging ---

def configure_logging(log_file: Optional[str] = None,
                      console_level: int = logging.INFO,
                      file_level: int = logging.INFO) -> None:
    """
    Configure root logger to write to both console and (optional) file.
    This replaces any existing handlers to avoid duplicate logs.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # let handlers decide
    # Remove existing handlers
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Optional file handler
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setLevel(file_level)
        fh.setFormatter(fmt)
        root.addHandler(fh)

# --- Model Configuration Dictionary ---
# - model_key: used in output file naming
# - model_name: HF repo id
# - is_test: when True, process only TEST_MAX_ROWS
# - batch_size: override global BATCH_SIZE if present
MODEL_CONFIGS = {
    "gemma27b": {
        "model_key": "gemma-3-27b-it",
        "model_name": "google/gemma-3-27b-it",
        "is_test": False,
        "batch_size": 16,
        "needs_all_gpus": False,
    },
    "mistral24b": {
        "model_key": "Mistral-Small-24B-Instruct-2501",
        "model_name": "mistralai/Mistral-Small-24B-Instruct-2501",
        "is_test": False,
        "batch_size": 16,
        "needs_all_gpus": False,
    },
    "qwen32b": {
        "model_key": "Qwen3-32B",
        "model_name": "Qwen/Qwen3-32B",
        "is_test": False,
        "batch_size": 16,
        "needs_all_gpus": False,
    },
    # ---- Llama ----
    "llama31_8b_base": {
        "model_key": "Llama-3.1-8B",
        "model_name": "meta-llama/Llama-3.1-8B",
        "is_test": False,
        "batch_size": 64,
        "needs_all_gpus": False,
    },
    "llama31_8b_it": {
        "model_key": "Llama-3.1-8B-Instruct",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "is_test": False,
        "batch_size": 64,
        "needs_all_gpus": False,
    },
    "llama31_70b_base": {
        "model_key": "Llama-3.1-70B",
        "model_name": "meta-llama/Llama-3.1-70B",
        "is_test": False,
        "batch_size": 2,
        "needs_all_gpus": True,
    },
    "llama33_70b_it": {
        "model_key": "Llama-3.3-70B-Instruct",
        "model_name": "meta-llama/Llama-3.3-70B-Instruct",
        "is_test": False,
        "batch_size": 2,
        "needs_all_gpus": True,
    },        
    "olmo2_7b_base": {
        "model_key": "OLMo-2-7B",
        "model_name": "allenai/OLMo-2-1124-7B",
        "is_test": False,
        "batch_size": 64,
        "needs_all_gpus": False,
    },
    "olmo2_7b_it": {
        "model_key": "OLMo-2-7B-Instruct",
        "model_name": "allenai/OLMo-2-1124-7B-Instruct",
        "is_test": False,
        "batch_size": 64,
        "needs_all_gpus": False,
    },
    # ---- gpt ----
    "gptoss_20b": {
        "model_key": "gpt-oss-20b",
        "model_name": "openai/gpt-oss-20b",
        "is_test": False,
        "batch_size": 32,
        "needs_all_gpus": True,
    },
    "gptoss_120b": {
        "model_key": "gpt-oss-120b",
        "model_name": "openai/gpt-oss-120b",
        "is_test": False,
        "batch_size": 2,
        "needs_all_gpus": True,
    },
    # --- falcon ---
    "falcon3_10b_base": {
        "model_key": "Falcon-3-10B",
        "model_name": "tiiuae/Falcon3-10B-Base",
        "is_test": False,
        "batch_size": 64,
        "needs_all_gpus": False,
    },
    "falcon3_10b_it": {
        "model_key": "Falcon-3-10B-Instruct",
        "model_name": "tiiuae/Falcon3-10B-Instruct",
        "is_test": False,
        "batch_size": 64,
        "needs_all_gpus": False,
    },
}
# --- Orchestrator Helpers for Parallel Model Runs ---
def _make_gpu_groups(total_gpus: int, group_size: int) -> List[List[int]]:
    """Return non-overlapping groups of GPU indices of size `group_size`."""
    if total_gpus <= 0 or group_size <= 0:
        return []
    groups: List[List[int]] = []
    ids = list(range(total_gpus))
    for i in range(0, len(ids), group_size):
        grp = ids[i:i + group_size]
        if len(grp) == group_size:
            groups.append(grp)
    return groups

def _spawn_model_subprocess(
    model_key: str,
    gpu_group: List[int],
    args: argparse.Namespace,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    log_label: Optional[str] = None,
) -> subprocess.Popen:
    """
    Launch this script as a child process for a single model, bound to the specified GPU group
    via CUDA_VISIBLE_DEVICES. Optionally run as a shard (index/num_shards). Each child gets its
    own log file to avoid collisions.
    """
    script_path = str(Path(__file__).resolve())
    cmd = [sys.executable, script_path, "--model", model_key]

    # propagate common flags
    if args.verbose:
        cmd.append("--verbose")
    if args.smoketest:
        cmd.append("--smoketest")
    if args.smoketest_only:
        cmd.append("--smoketest-only")
    if args.smoketest_rows is not None:
        cmd.extend(["--smoketest-rows", str(args.smoketest_rows)])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.use_deranged:
        cmd.append("--use-deranged")
    if args.deranged_path:
        cmd.extend(["--deranged-path", str(args.deranged_path)])
    if args.derange_seed is not None:
        cmd.extend(["--derange-seed", str(args.derange_seed)])
    if args.force_derange:
        cmd.append("--force-derange")

    # shard flags
    if num_shards > 1:
        cmd.extend(["--num-shards", str(num_shards), "--shard-index", str(shard_index)])

    # per-child log file to avoid interference
    ts = time.strftime("%Y%m%d_%H%M%S")
    label = log_label or model_key
    log_name = f"{label}_shard{shard_index}of{num_shards}_{ts}.log" if num_shards > 1 else f"{label}_{ts}.log"
    log_path = os.path.join(LOGS_DIR, log_name)
    os.makedirs(LOGS_DIR, exist_ok=True)
    cmd.extend(["--log-file", log_path])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_group)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    logging.info(f"[orchestrator] Launching model '{model_key}' on GPUs {env['CUDA_VISIBLE_DEVICES']}: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env)


# --- Shard Output Helpers ---
def _shard_output_paths(model_key: str, num_shards: int, args: argparse.Namespace) -> (List[str], str):
    dataset_tag = "deranged" if getattr(args, "use_deranged", False) else "swow"
    base_name = f"{MODEL_CONFIGS[model_key]['model_key']}_{dataset_tag}_results.csv"
    shard_names = [
        f"{MODEL_CONFIGS[model_key]['model_key']}_{dataset_tag}_results_shard{i}of{num_shards}.csv"
        for i in range(num_shards)
    ]
    shard_paths = [os.path.join(OUTPUTS_DIR, n) for n in shard_names]
    merged_path = os.path.join(OUTPUTS_DIR, base_name)
    return shard_paths, merged_path


def _merge_shard_outputs(model_key: str, num_shards: int, args: argparse.Namespace) -> None:
    """Merge per-shard CSVs into the canonical results file for the model."""
    try:
        shard_paths, merged_path = _shard_output_paths(model_key, num_shards, args)
        existing = [p for p in shard_paths if os.path.exists(p) and os.path.getsize(p) > 0]
        if not existing:
            logging.warning("[orchestrator] No shard outputs found for %s — nothing to merge.", model_key)
            return
        frames = []
        for p in existing:
            try:
                frames.append(pd.read_csv(p))
            except Exception as e:
                logging.error("[orchestrator] Failed to read shard file %s: %s", p, e)
        if not frames:
            logging.warning("[orchestrator] All shard files for %s failed to read — skipping merge.", model_key)
            return
        merged = pd.concat(frames, ignore_index=True)
        if "swow_id" in merged.columns:
            merged = merged.drop_duplicates(subset=["swow_id"], keep="first")
            merged = merged.sort_values("swow_id")
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        merged.to_csv(merged_path, index=False)
        logging.info("[orchestrator] Merged %d shard file(s) into %s", len(existing), merged_path)
    except Exception as e:
        logging.warning("[orchestrator] Shard merge encountered an issue for %s: %s", model_key, e)

def orchestrate_run_all(args: argparse.Namespace, model_sequence: Sequence[str]) -> None:
    """
    Run small models in parallel. If --parallelize-same-model is set, we launch up to
    2 shards of the SAME next small model (each bound to its own GPU group). Each shard
    writes to its own CSV; after all shards finish, we merge to the canonical file.

    Big models (needs_all_gpus=True) still run exclusively across --big-gpus GPUs.
    """
    # Optionally skip big models entirely when asked
    if getattr(args, "skip_big_models", False):
        model_sequence = [
            k for k in model_sequence
            if not MODEL_CONFIGS.get(k, {}).get("needs_all_gpus", False)
        ]
        logging.info(
            "[orchestrator] Skipping big models per --skip-big-models; remaining: %s",
            ", ".join(model_sequence)
        )

    if not torch.cuda.is_available():
        logging.warning("[orchestrator] CUDA not available; falling back to sequential run.")
        df_all = load_dataset(
            use_deranged=args.use_deranged,
            deranged_path=args.deranged_path,
            derange_seed=args.derange_seed,
            force_derange=args.force_derange,
        )
        for idx, model_key in enumerate(model_sequence, start=1):
            logging.info("=== Running model %s (%d/%d) ===", model_key, idx, len(model_sequence))
            run_for_model(model_key, df_all, args)
        return

    total = torch.cuda.device_count()
    small_group = min(2, max(1, int(args.small_gpus)))  # cap to 2 GPUs per small-model job
    big_group = max(1, int(args.big_gpus))

    retried_on_allgpus = set()

    # Prepare per-model flags
    big_models = []
    small_models = []
    for key in model_sequence:
        meta = MODEL_CONFIGS.get(key, {})
        if meta.get("needs_all_gpus", False):
            big_models.append(key)
        else:
            small_models.append(key)

    small_groups = _make_gpu_groups(total, small_group)
    if not small_groups:
        logging.warning(f"[orchestrator] Could not form any small groups from {total} GPUs with group size {small_group}.")
        small_groups = [[i] for i in range(total)]

    queue = list(model_sequence)
    running: List[subprocess.Popen] = []
    running_labels: List[str] = []

    def wait_all():
        nonlocal running, running_labels, retried_on_allgpus
        for proc, label in zip(running, running_labels):
            rc = proc.wait()
            base_label = label.split("[", 1)[0]
            if rc == 0:
                continue
            logging.error(f"[orchestrator] Model '{label}' exited with code {rc}.")

            # If this was a small-model *non-shard* job, attempt a one-time fallback on all GPUs.
            is_shard = "[shard" in label
            meta = MODEL_CONFIGS.get(base_label, {})
            if (not is_shard) and (not meta.get("needs_all_gpus", False)) and (base_label not in retried_on_allgpus):
                use = list(range(min(big_group, total)))
                logging.info(f"[orchestrator] Retrying '{base_label}' on all {len(use)} GPU(s): {use}")
                proc2 = _spawn_model_subprocess(base_label, use, args)
                rc2 = proc2.wait()
                if rc2 != 0:
                    logging.error(f"[orchestrator] Fallback (all GPUs) for '{base_label}' failed with code {rc2}. Continuing.")
                else:
                    logging.info(f"[orchestrator] Fallback (all GPUs) for '{base_label}' succeeded.")
                retried_on_allgpus.add(base_label)
        running, running_labels = [], []

    while queue:
        next_key = queue[0]
        meta = MODEL_CONFIGS.get(next_key, {})

        # Big models run exclusively
        if meta.get("needs_all_gpus", False):
            wait_all()
            use = list(range(min(big_group, total)))
            proc = _spawn_model_subprocess(next_key, use, args)
            running = [proc]
            running_labels = [next_key]
            rc = proc.wait()
            if rc != 0:
                logging.error(f"[orchestrator] BIG model '{next_key}' failed with code {rc}.")
            queue.pop(0)
            continue

        # Small models: either shard the SAME model, or (legacy) run different models concurrently
        if getattr(args, "parallelize_same_model", False):
            # Pop the single model we'll shard
            queue.pop(0)
            # Determine shard count (max 2 and limited by available groups)
            shard_count = min(max(1, int(getattr(args, "max_parallel_small", 2))), 2, len(small_groups))
            logging.info("[orchestrator] Launching %d shard(s) for model '%s' across %d-GPU groups.", shard_count, next_key, small_group)
            running = []
            running_labels = []
            for s in range(shard_count):
                grp = small_groups[s % len(small_groups)]
                label = f"{next_key}[shard{s+1}/{shard_count}]"
                proc = _spawn_model_subprocess(
                    next_key, grp, args, shard_index=s, num_shards=shard_count, log_label=next_key
                )
                running.append(proc)
                running_labels.append(label)
            # Wait for all shards to finish, then merge outputs
            wait_all()
            _merge_shard_outputs(next_key, shard_count, args)
            continue
        else:
            # legacy behaviour: run different small models concurrently (capped at 2)
            max_parallel = min(len(small_groups), max(1, int(getattr(args, "max_parallel_small", 2))), 2)
            launched = 0
            running = []
            running_labels = []
            while queue and launched < max_parallel:
                key = queue[0]
                meta = MODEL_CONFIGS.get(key, {})
                if meta.get("needs_all_gpus", False):
                    break
                grp = small_groups[launched % len(small_groups)]
                proc = _spawn_model_subprocess(key, grp, args)
                running.append(proc)
                running_labels.append(key)
                queue.pop(0)
                launched += 1
            wait_all()

    logging.info("[orchestrator] All models completed.")

# --- File Paths & Settings ---
SWOW_FILE_PATH = "data/SWOW/SWOW-EN.R100.20180827.csv"
DERANGED_DEFAULT_PATH = "data/SWOW-EN.R100.20180827.deranged.csv"
OUTPUTS_DIR = "outputs"
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
BATCH_SIZE = 128
TEST_MAX_ROWS = 20

# --- Globals ---
MODEL = None
TOKENIZER = None
COLS_TO_LOAD = [
    'id', 'cue', 'R1', 'R2', 'R3',
    'age', 'gender', 'nativeLanguage', 'country', 'education'
]

# --- Helpers ---
def clear_model_cache() -> None:
    """Release model/tokenizer references and clear GPU memory."""
    global MODEL, TOKENIZER
    if MODEL is not None:
        try:
            del MODEL
        except Exception:
            pass
    if TOKENIZER is not None:
        try:
            del TOKENIZER
        except Exception:
            pass
    MODEL = None
    TOKENIZER = None
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            # Collect IPC memory and synchronize to ensure allocator releases pages
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        except Exception:
            logging.warning("Failed to empty CUDA cache.")


def ensure_deranged_dataset(
    output_path: str,
    seed: Optional[int] = None,          # unused (kept for CLI compatibility)
    force_regenerate: bool = False,      # unused (kept for CLI compatibility)
) -> str:
    """
    Do NOT try to generate anything.
    Try a few sensible locations for an existing deranged CSV, in this order:
      1) The provided --deranged-path (as-is)
      2) That path resolved relative to the current working directory
      3) That path resolved relative to this script's directory
      4) A file with the same basename in the CWD
      5) The basename in this script's directory
      6) The basename inside a local 'data/' directory next to this script
      7) The DERANGED_DEFAULT_PATH (as-is, then CWD, then next to this script)
    """
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()

    candidates = []

    # 1) Provided path as-is
    p = Path(output_path)
    candidates.append(p)

    # 2) Provided path relative to CWD (if not absolute)
    if not p.is_absolute():
        candidates.append(cwd / p)

    # 3) Provided path relative to script directory (if not absolute)
    if not p.is_absolute():
        candidates.append(script_dir / p)

    # 4) Basename in CWD
    candidates.append(cwd / p.name)

    # 5) Basename in script directory
    candidates.append(script_dir / p.name)

    # 6) Basename inside local 'data/' next to the script
    candidates.append(script_dir / "data" / p.name)

    # 7) Try DERANGED_DEFAULT_PATH and variants
    d = Path(DERANGED_DEFAULT_PATH)
    candidates.append(d)
    if not d.is_absolute():
        candidates.append(cwd / d)
        candidates.append(script_dir / d)
        candidates.append(script_dir / "data" / d.name)

    for cand in candidates:
        try:
            if cand.exists():
                logging.info("Using deranged dataset at %s", cand)
                return str(cand)
        except Exception:
            # Ignore any odd path errors and keep checking
            pass

    # If nothing was found, raise with a helpful message
    searched = "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate a deranged dataset CSV. Checked the following paths:\n"
        f"  - {searched}\n"
        "Supply an existing file via --deranged-path or place "
        "SWOW-EN.R100.20180827.deranged.csv next to wa_scoring.py or in ./data/."
    )


def load_dataset(
    use_deranged: bool,
    deranged_path: str,
    derange_seed: Optional[int],
    force_derange: bool,
) -> pd.DataFrame:
    dataset_path = SWOW_FILE_PATH
    if use_deranged:
        dataset_path = ensure_deranged_dataset(
            deranged_path,
            seed=derange_seed,
            force_regenerate=force_derange,
        )

    logging.info("Loading dataset from %s", dataset_path)
    df = pd.read_csv(dataset_path, usecols=COLS_TO_LOAD).dropna(subset=['cue', 'R1'])
    return df


def _has_chat_template(tok: AutoTokenizer) -> bool:
    try:
        tmpl = getattr(tok, "chat_template", None)
        return bool(tmpl)
    except Exception:
        return False

def _get_eot_id(tok: AutoTokenizer) -> Optional[int]:
    """Prefer model's eot; fallback to eos if present."""
    try:
        eid = tok.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eid, int) and eid >= 0:
            return eid
    except Exception:
        pass
    return tok.eos_token_id

def _first_index(seq: List[int], token_id: Optional[int], start: int) -> Optional[int]:
    if token_id is None:
        return None
    for i in range(max(0, start), len(seq)):
        if seq[i] == token_id:
            return i
    return None

def initialize_model_and_tokenizer(model_name: str) -> bool:
    """Load model+tokenizer."""
    global MODEL, TOKENIZER
    try:
        logging.info(f"Loading model '{model_name}'...")

        # Use a special loading path only for OpenAI GPT-OSS models so we keep MXFP4
        # quantization (requires the 'kernels' package and Triton >= 3.4). For all
        # other models, preserve the previous bfloat16 load behavior.
        is_gptoss = isinstance(model_name, str) and model_name.startswith("openai/gpt-oss-")

        load_kwargs = dict(
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        if is_gptoss:
            # Let Transformers choose dtype to avoid forcing dequantization to bf16
            load_kwargs["torch_dtype"] = "auto"

            # Cap per-GPU VRAM to ~95% of total so we don't spike and OOM during load
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                max_memory = {}
                for i in range(torch.cuda.device_count()):
                    total = torch.cuda.get_device_properties(i).total_memory
                    cap = int(total * 0.95)
                    max_memory[i] = cap
                load_kwargs["max_memory"] = max_memory
                pretty_caps = {k: f"{v/2**30:.1f}GiB" for k, v in max_memory.items()}
                logging.info("Using per-GPU memory caps: %s", pretty_caps)

            # Warn if the MXFP4 kernels are not available — otherwise it will fall back to bf16
            try:
                if importlib.util.find_spec("kernels") is None:
                    logging.warning(
                        "OpenAI MXFP4 'kernels' package not found. Model may dequantize to bf16 and OOM. "
                        "Install with: pip install -U triton==3.4 kernels"
                    )
            except Exception:
                pass
        else:
            # Keep previous behavior for all non-GPT-OSS models
            load_kwargs["torch_dtype"] = torch.bfloat16

        # Prefer FlashAttention-2 when available to reduce memory and improve speed;
        # fall back to default attention if it isn't installed/supported.
        try:
            MODEL = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                **load_kwargs,
            )
        except Exception as e:
            logging.warning(
                "flash_attention_2 unavailable or failed for '%s' (%s). Falling back to default attention.",
                model_name, e,
            )
            MODEL = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs,
            )
        MODEL.eval()

        TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        TOKENIZER.padding_side = 'right'
        if TOKENIZER.pad_token is None and TOKENIZER.eos_token is not None:
            TOKENIZER.pad_token = TOKENIZER.eos_token
        elif TOKENIZER.pad_token is None and TOKENIZER.eos_token is None:
            TOKENIZER.add_special_tokens({"pad_token": "<|pad|>"})
            MODEL.resize_token_embeddings(len(TOKENIZER))

        # Bypass chat template for OpenAI GPT-OSS models so scoring uses the base (no-chat) path.
        # This avoids injecting channels and "Reasoning: ..." headers into the tokenized input.
        if isinstance(model_name, str) and model_name.startswith("openai/gpt-oss-"):
            try:
                # Make chat template falsy so _has_chat_template() returns False
                TOKENIZER.chat_template = ""
                logging.info("Bypassing chat template for '%s' (OpenAI GPT-OSS). Using base input/mask path.", model_name)
            except Exception as e:
                logging.warning("Could not bypass chat template for '%s': %s", model_name, e)

        # Qwen3-32B: disable "thinking" (<think>...</think>) injection by bypassing the chat template.
        # This forces the base (no-chat) path so we only score the three response words (commas/whitespace masked).
        if isinstance(model_name, str) and model_name.startswith("Qwen/Qwen3-32B"):
            try:
                TOKENIZER.chat_template = ""
                logging.info(
                    "Bypassing chat template for '%s' (Qwen3-32B). Using base input/mask path (no <think> tags).",
                    model_name,
                )
            except Exception as e:
                logging.warning("Could not bypass chat template for '%s': %s", model_name, e)

        logging.info("Model and tokenizer loaded successfully. Chat template detected: %s",
                     _has_chat_template(TOKENIZER))
        return True
    except Exception as e:
        logging.error(f"Failed to load model '{model_name}'. Error: {e}")
        return False

def _build_chat_inputs_and_masks(
    cues: List[str],
    responses_list: List[List[str]],
) -> Dict[str, Any]:
    """
    Chat path:
      - prefix = messages up to assistant header (add_generation_prompt=True)
      - full   = prefix + assistant content (add_generation_prompt=False)
      - score indices [len(prefix): first EOT/EOS after that)
    """
    device = MODEL.device
    pad_id = TOKENIZER.pad_token_id if TOKENIZER.pad_token_id is not None else TOKENIZER.eos_token_id
    eot_id = _get_eot_id(TOKENIZER)

    full_ids_list: List[List[int]] = []
    ranges: List[tuple] = []

    for cue, responses in zip(cues, responses_list):
        prompt = (
            f"You are participating in a word association task. "
            f"Think of the first three words that come to mind when you see the word \"{cue}\". "
            f"Use mostly single words."
        )
        assistant_response = ", ".join(responses)

        msgs_prefix = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]
        prefix_ids = TOKENIZER.apply_chat_template(
            msgs_prefix, add_generation_prompt=True, return_tensors="pt"
        ).squeeze(0).tolist()

        msgs_full = msgs_prefix + [{"role": "assistant", "content": assistant_response}]
        full_ids = TOKENIZER.apply_chat_template(
            msgs_full, add_generation_prompt=False, return_tensors="pt"
        ).squeeze(0).tolist()

        start = len(prefix_ids)

        # Prefer EOT; fallback to EOS; else end of sequence
        end = _first_index(full_ids, eot_id, start)
        if end is None and TOKENIZER.eos_token_id is not None:
            end = _first_index(full_ids, TOKENIZER.eos_token_id, start)
        if end is None:
            end = len(full_ids)

        full_ids_list.append(full_ids)
        ranges.append((start, end))

    max_len = max(len(seq) for seq in full_ids_list)
    input_ids = torch.full((len(full_ids_list), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(full_ids_list):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    input_ids = input_ids.to(device)

    labels = torch.full_like(input_ids, -100)
    for i, (start, end) in enumerate(ranges):
        if start < end:
            labels[i, start:end] = input_ids[i, start:end]
    return {"input_ids": input_ids, "labels": labels}

def _build_base_inputs_and_masks(
    cues: List[str],
    responses_list: List[List[str]],
) -> Dict[str, Any]:
    """
    Base path (no chat template):
      - prefix: "User: ...\nAssistant:"
      - full: prefix + assistant content
      - score [len(prefix): end)
    """
    device = MODEL.device
    pad_id = TOKENIZER.pad_token_id if TOKENIZER.pad_token_id is not None else TOKENIZER.eos_token_id

    full_ids_list: List[List[int]] = []
    ranges: List[tuple] = []

    for cue, responses in zip(cues, responses_list):
        prompt = (
            f"You are participating in a word association task. "
            f"Think of the first three words that come to mind when you see the word \"{cue}\". "
            f"Use mostly single words."
        )
        assistant_response = ", ".join(responses)

        prefix_text = f"User: {prompt}\nAssistant:\n"
        full_text = f"{prefix_text}{assistant_response}"

        prefix_ids = TOKENIZER(prefix_text, add_special_tokens=False).input_ids
        full_ids = TOKENIZER(full_text, add_special_tokens=False).input_ids

        start = len(prefix_ids)
        end = len(full_ids)

        full_ids_list.append(full_ids)
        ranges.append((start, end))

    max_len = max(len(seq) for seq in full_ids_list)
    input_ids = torch.full((len(full_ids_list), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(full_ids_list):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    input_ids = input_ids.to(MODEL.device)

    labels = torch.full_like(input_ids, -100)
    for i, (start, end) in enumerate(ranges):
        if start < end:
            labels[i, start:end] = input_ids[i, start:end]
    return {"input_ids": input_ids, "labels": labels}

# --- Normalization Helper for WA Text Equivalence ---
def _normalize_wa_text(s: str) -> str:
    """Remove commas and all whitespace for equivalence checks."""
    return "".join(ch for ch in s if ch not in {",", " ", "\t", "\n", "\r"})

def _mask_spaces_and_commas(input_ids: torch.Tensor, labels: torch.Tensor, tok: AutoTokenizer) -> None:
    """
    ALWAYS-ON filter: set labels to -100 for tokens that decode to whitespace-only or ','.
    We still pass the full assistant response verbatim; we only mask these separator tokens
    from loss/score aggregation.
    """
    bsz, _ = labels.shape
    for i in range(bsz):
        idxs = torch.where(labels[i] != -100)[0]
        for j in idxs.tolist():
            tid = int(input_ids[i, j].item())
            piece = tok.decode([tid], skip_special_tokens=True)
            if piece.strip() == "" or piece.strip() == ",":
                labels[i, j] = -100

#
# ---- Scoring ----

# --- Diagnostic Helper for Scored Slice ---
from typing import Dict
def _diagnose_scored_slice(cue: str, responses: List[str]) -> Dict[str, str]:
    """
    Build inputs exactly like get_logprob_batched and return decoded strings for:
      - full_input_decoded
      - ignored_tokens_decoded  (labels == -100)
      - scored_tokens_decoded   (labels != -100)
      - assistant_response_text (", ".join(responses))
    """
    has_chat = _has_chat_template(TOKENIZER)
    pack = _build_chat_inputs_and_masks([cue], [responses]) if has_chat \
           else _build_base_inputs_and_masks([cue], [responses])

    inputs = pack["input_ids"]
    labels = pack["labels"]
    _mask_spaces_and_commas(inputs, labels, TOKENIZER)

    ignored_tokens = inputs[0][labels[0] == -100]
    scored_tokens = inputs[0][labels[0] != -100]

    full_decoded_text = TOKENIZER.decode(inputs[0], skip_special_tokens=False)
    ignored_decoded = TOKENIZER.decode(ignored_tokens, skip_special_tokens=False)
    scored_decoded = TOKENIZER.decode(scored_tokens, skip_special_tokens=False)

    return {
        "full_input_decoded": full_decoded_text,
        "ignored_tokens_decoded": ignored_decoded,
        "scored_tokens_decoded": scored_decoded,
        "assistant_response_text": ", ".join(responses),
    }
def get_logprob_batched(cues: List[str], responses_list: List[List[str]], verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Calculate log-probabilities for a batch, scoring ONLY the assistant response slice,
    with commas/whitespace-only tokens masked out.
    """
    has_chat = _has_chat_template(TOKENIZER)
    pack = _build_chat_inputs_and_masks(cues, responses_list) if has_chat \
           else _build_base_inputs_and_masks(cues, responses_list)

    inputs = pack["input_ids"]
    labels = pack["labels"]

    # Always exclude commas & whitespace-only tokens
    _mask_spaces_and_commas(inputs, labels, TOKENIZER)

    # Forward
    with torch.inference_mode():
        outputs = MODEL(input_ids=inputs, labels=labels)
        logits = outputs.logits

    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()

    log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)

    scored_labels = shifted_labels.clone()
    scored_labels[scored_labels == -100] = 0
    gathered_log_probs = torch.gather(log_probs, 2, scored_labels.unsqueeze(-1)).squeeze(-1)

    score_mask = (shifted_labels != -100)
    final_log_probs = gathered_log_probs * score_mask

    total_log_probs_per_item = final_log_probs.sum(dim=1)
    num_tokens_per_item = score_mask.sum(dim=1)

    if verbose and len(cues) > 0:
        logging.info("%s", "="*20 + " VERBOSE DEBUG " + "="*20)
        i = 0
        logging.info("Cue: %r", cues[i])
        logging.info("Response: %s", responses_list[i])

        full_decoded_text = TOKENIZER.decode(inputs[i], skip_special_tokens=False)
        logging.info("Full Input to Model (decoded):\n---\n%s\n---", full_decoded_text)

        ignored_tokens = inputs[i][labels[i] == -100]
        scored_tokens = inputs[i][labels[i] != -100]

        logging.info("Ignored Prompt Tokens (decoded): %r", TOKENIZER.decode(ignored_tokens))
        logging.info("Scored Response Tokens (decoded): %r", TOKENIZER.decode(scored_tokens))

        num_tok = int(num_tokens_per_item[i].item())
        tot_lp = float(total_log_probs_per_item[i].item())
        logging.info("Number of Scored Tokens: %d", num_tok)
        logging.info("Total Log Probability: %.4f", tot_lp)
        if num_tok > 0:
            logging.info("Normalized Log Probability (per scored token): %.4f", tot_lp/num_tok)
        logging.info("%s", "="*55)

    results: List[Dict[str, Any]] = []
    for i in range(len(cues)):
        total_prob = float(total_log_probs_per_item[i].item())
        num_tokens = int(num_tokens_per_item[i].item())
        normalized_prob = (total_prob / num_tokens) if num_tokens > 0 else 0.0
        results.append({
            "total_log_prob": total_prob,
            "normalized_log_prob": normalized_prob,
            "num_tokens": num_tokens
        })
    return results

# ---- Smoketest ----
def run_smoketest(df_for_smoke: pd.DataFrame, rows: int = 3) -> None:
    """
    Backwards-compatible shim: delegate to run_smoketest_with_checks().
    """
    run_smoketest_with_checks(df_for_smoke, rows)

def run_smoketest_with_checks(df_for_smoke: pd.DataFrame, rows: int = 3) -> None:
    """
    Run a strict smoketest on the first `rows` items:
      - Prints full decoded input, ignored/scored token strings (like verbose mode)
      - Verifies that, after removing commas and whitespace, the scored token string
        equals the intended assistant response string (", ".join(R1..R3)) with the same normalization.
      - Raises a RuntimeError if any row fails the check.
    """
    n = min(rows, len(df_for_smoke))
    if n == 0:
        logging.info("Smoke test skipped: no rows available.")
        return

    logging.info("===== SMOKE TEST (with token checks) BEGIN =====")
    failures = 0
    for k in range(n):
        row = df_for_smoke.iloc[k]
        cue = row['cue']
        responses = [str(r).lower().strip() for r in [row['R1'], row.get('R2', None), row.get('R3', None)]
                     if pd.notna(r) and str(r).strip()]

        diag = _diagnose_scored_slice(cue, responses)

        logging.info("--- ITEM %d/%d ---", k+1, n)
        logging.info("Cue: %r", cue)
        logging.info("Responses: %s", responses)
        logging.info("Full Input (decoded):\n---\n%s\n---", diag["full_input_decoded"])
        logging.info("Ignored Tokens (decoded): %r", diag['ignored_tokens_decoded'])
        logging.info("Scored Tokens  (decoded): %r", diag['scored_tokens_decoded'])

        expected_norm = _normalize_wa_text(diag["assistant_response_text"])
        scored_norm = _normalize_wa_text(diag["scored_tokens_decoded"])

        if expected_norm != scored_norm:
            failures += 1
            logging.error(
                "Smoketest token check FAILED for cue '%s'. "
                "Expected (no commas/space): '%s'  vs  Scored: '%s'",
                cue, expected_norm, scored_norm
            )
        else:
            logging.info("Smoketest token check PASSED for cue '%s'.", cue)

    logging.info("===== SMOKE TEST END =====")
    if failures > 0:
        raise RuntimeError(f"Smoketest failed on {failures}/{n} item(s). Aborting.")

def _is_cuda_oom(err: BaseException) -> bool:
    s = str(err)
    return ("CUDA out of memory" in s) or ("CUDA error: out of memory" in s)

def run_for_model(model_key: str, df_all: pd.DataFrame, args: argparse.Namespace) -> None:
    selected_config = MODEL_CONFIGS[model_key]
    model_name = selected_config["model_name"]
    model_display_key = selected_config["model_key"]
    is_test = selected_config["is_test"]

    effective_batch_size = selected_config.get("batch_size", BATCH_SIZE)
    if args.batch_size is not None:
        effective_batch_size = args.batch_size
    logging.info("Effective batch size for %s: %d", model_display_key, effective_batch_size)
    save_interval_rows = effective_batch_size * 50

    if is_test:
        logging.info(
            "--- RUNNING IN TEST MODE FOR MODEL: %s (%s) ---",
            model_display_key,
            model_key,
        )

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    dataset_tag = "deranged" if args.use_deranged else "swow"

    # Shard-aware output path
    num_shards = max(1, int(getattr(args, "num_shards", 1)))
    shard_index = int(getattr(args, "shard_index", 0))
    if num_shards > 1:
        output_filename = f"{model_display_key}_{dataset_tag}_results_shard{shard_index}of{num_shards}.csv"
    else:
        output_filename = f"{model_display_key}_{dataset_tag}_results.csv"
    output_path = os.path.join(OUTPUTS_DIR, output_filename)
    logging.info(f"Saving results for model '{model_display_key}' (dataset={dataset_tag}) to {output_path}")

    try:
        clear_model_cache()
        if not initialize_model_and_tokenizer(model_name):
            return

        # Always run a smoketest with strict token checks before full processing.
        run_smoketest_with_checks(df_all, rows=args.smoketest_rows)
        if args.smoketest_only:
            logging.info("Smoke test completed. Exiting as requested (--smoketest-only).")
            return

        processed_ids = set()
        if os.path.exists(output_path):
            try:
                logging.info("Output file found. Loading previously processed IDs to resume...")
                processed_df = pd.read_csv(output_path)
                if 'swow_id' in processed_df.columns:
                    processed_ids = set(processed_df['swow_id'])
                logging.info(
                    "Found %d previously processed IDs. They will be skipped.",
                    len(processed_ids),
                )
            except Exception as e:
                logging.warning(
                    "Could not read existing output file, starting from scratch. Error: %s",
                    e,
                )

        df = df_all
        # If running as a shard, filter the data deterministically by swow_id modulo num_shards
        if num_shards > 1:
            try:
                df = df.copy()
                df["id"] = df["id"].astype(int)
                df = df[df["id"] % num_shards == shard_index]
                logging.info("Shard filter active: shard_index=%d num_shards=%d -> %d rows to process",
                             shard_index, num_shards, len(df))
            except Exception as e:
                logging.error("Failed to apply shard filter (id %% %d == %d): %s", num_shards, shard_index, e)
        if processed_ids:
            df = df[~df['id'].isin(processed_ids)]

        if is_test:
            df = df.head(TEST_MAX_ROWS)
            logging.info("Test mode is on. Processing only the first %d rows.", TEST_MAX_ROWS)

        if df.empty:
            logging.info("All rows have already been processed or data is empty. Nothing to do.")
            return

        total_remaining = len(df)
        start_time = time.time()
        rows_done = 0

        results_to_save: List[Dict[str, Any]] = []
        total_rows = len(df)

        i = 0
        current_bs = effective_batch_size
        oom_retry_budget = 3  # if we hit OOM at bs=1, allow a few retries before skipping
        last_progress_log_ts = 0.0
        progress_interval = max(0.5, float(getattr(args, "progress_interval", 10.0)))

        while i < total_rows:
            bs = min(current_bs, total_rows - i)
            batch_df = df.iloc[i:i + bs]
            if batch_df.empty:
                break

            cues = batch_df['cue'].tolist()
            responses_list: List[List[str]] = []
            for _, row in batch_df.iterrows():
                responses = [
                    str(r).lower().strip()
                    for r in [row['R1'], row['R2'], row['R3']]
                    if pd.notna(r) and str(r).strip()
                ]
                responses_list.append(responses)

            now_ts = time.time()
            if (now_ts - last_progress_log_ts) >= progress_interval:
                logging.info(
                    "--- Processing rows %d to %d / %d (batch=%d) ---",
                    i,
                    i + len(batch_df) - 1,
                    total_rows,
                    bs,
                )
                last_progress_log_ts = now_ts

            try:
                batch_scores = get_logprob_batched(cues, responses_list, verbose=args.verbose)
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    logging.warning(
                        "CUDA OOM while scoring rows %d..%d with batch=%d. Emptying cache and reducing batch size...",
                        i, i + bs - 1, bs,
                    )
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                    if current_bs > 1:
                        current_bs = max(1, current_bs // 2)
                        continue  # retry same starting index with a smaller batch
                    elif oom_retry_budget > 0:
                        oom_retry_budget -= 1
                        time.sleep(1.0)
                        continue
                    else:
                        logging.error(
                            "Persistent OOM at batch size 1 for rows %d..%d. Skipping these rows.",
                            i, i + bs - 1,
                        )
                        # record skipped rows so we can resume later without reprocessing
                        for _, row in batch_df.iterrows():
                            results_to_save.append({
                                "model": model_display_key,
                                "swow_id": row['id'],
                                "cue": row['cue'],
                                "response_set": ", ".join([
                                    str(r).lower().strip()
                                    for r in [row['R1'], row['R2'], row['R3']]
                                    if pd.notna(r) and str(r).strip()
                                ]),
                                "age": row['age'],
                                "gender": row['gender'],
                                "nativeLanguage": row['nativeLanguage'],
                                "country": row['country'],
                                "education": row['education'],
                                "total_log_prob": None,
                                "normalized_log_prob": None,
                                "num_tokens": 0,
                                "error": "oom",
                            })
                        # advance past this tiny batch
                        i += bs
                        current_bs = effective_batch_size
                        oom_retry_budget = 3

                        # periodic save after skip
                        if len(results_to_save) >= save_interval_rows:
                            logging.info("--- Saving batch of %d results ---", len(results_to_save))
                            header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
                            pd.DataFrame(results_to_save).to_csv(
                                output_path,
                                mode='a',
                                header=header,
                                index=False,
                            )
                            results_to_save = []
                        continue
                else:
                    raise

            # success path
            for j, (_, row) in enumerate(batch_df.iterrows()):
                result_entry = {
                    "model": model_display_key,
                    "swow_id": row['id'],
                    "cue": row['cue'],
                    "response_set": ", ".join(responses_list[j]),
                    "age": row['age'],
                    "gender": row['gender'],
                    "nativeLanguage": row['nativeLanguage'],
                    "country": row['country'],
                    "education": row['education'],
                    **batch_scores[j],
                }
                results_to_save.append(result_entry)

            rows_done += len(batch_df)
            elapsed = time.time() - start_time
            throughput = rows_done / elapsed if elapsed > 0 else 0.0
            remain = max(0, total_remaining - rows_done)
            eta_sec = int(remain / throughput) if throughput > 0 else 0
            now_ts = time.time()
            if (now_ts - last_progress_log_ts) >= progress_interval:
                logging.info(
                    "[progress] %d/%d rows (%.2f%%), %.1f rows/s",
                    rows_done,
                    total_remaining,
                    rows_done / total_remaining * 100,
                    throughput,
                )
                logging.info("[eta] ~%s remaining", str(timedelta(seconds=eta_sec)))
                last_progress_log_ts = now_ts

            if len(results_to_save) >= save_interval_rows:
                logging.info("--- Saving batch of %d results ---", len(results_to_save))
                header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
                pd.DataFrame(results_to_save).to_csv(
                    output_path,
                    mode='a',
                    header=header,
                    index=False,
                )
                results_to_save = []

            # advance index and reset adaptive knobs
            i += len(batch_df)
            current_bs = effective_batch_size
            oom_retry_budget = 3

        if results_to_save:
            logging.info("--- Saving final batch of %d results ---", len(results_to_save))
            header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
            pd.DataFrame(results_to_save).to_csv(
                output_path,
                mode='a',
                header=header,
                index=False,
            )

        logging.info("--- PROCESSING COMPLETE ---")
        logging.info("Final results saved to %s", output_path)
    finally:
        clear_model_cache()


def _is_instruction_model(meta: Dict[str, Any]) -> bool:
    key = str(meta.get("model_key", "")).lower()
    name = str(meta.get("model_name", "")).lower()
    return ("instruct" in name) or key.endswith("-it") or ("-it" in key)

def main():
    parser = argparse.ArgumentParser(
        description="Run word association scoring with one or more HuggingFace models.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=MODEL_CONFIGS.keys(),
        help="The shorthand key for a single model run.",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run every model defined in the configuration sequentially.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable extra logging (not the per-item smoke test printouts).",
    )
    parser.add_argument(
        "--smoketest",
        action="store_true",
        help="Run a short, verbose smoke test before the full batch run.",
    )
    parser.add_argument(
        "--smoketest-only",
        action="store_true",
        help="Run only the smoke test and exit (no CSV write).",
    )
    parser.add_argument(
        "--smoketest-rows",
        type=int,
        default=3,
        help="Number of rows to inspect in the smoke test (default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the per-model batch size for this run.",
    )
    parser.add_argument(
        "--use-deranged",
        action="store_true",
        help="Use a pre-deranged SWOW CSV (no auto-generation).",
    )
    parser.add_argument(
        "--deranged-path",
        type=str,
        default=DERANGED_DEFAULT_PATH,
        help=f"Destination path for the deranged dataset (default: {DERANGED_DEFAULT_PATH}).",
    )
    parser.add_argument(
        "--derange-seed",
        type=int,
        default=None,
        help="Optional random seed when generating the deranged dataset.",
    )
    parser.add_argument(
        "--force-derange",
        action="store_true",
        help="Force regeneration of the deranged dataset even if a cached copy exists.",
    )

    parser.add_argument(
        "--orchestrate-parallel",
        action="store_true",
        help="Run small models in parallel across GPU groups; big models use all GPUs exclusively.",
    )
    parser.add_argument(
        "--small-gpus",
        type=int,
        default=2,
        help="GPUs per small-model job when orchestrating in parallel (default: 2).",
    )
    parser.add_argument(
        "--big-gpus",
        type=int,
        default=4,
        help="GPUs for big models that require all cards (default: 4).",
    )
    parser.add_argument(
        "--max-parallel-small",
        type=int,
        default=2,
        help="Maximum number of small-model jobs to run at once when orchestrating (default: 2).",
    )
    parser.add_argument(
        "--parallelize-same-model",
        action="store_true",
        help=(
            "When orchestrating, instead of running different small models concurrently, "
            "launch up to 2 shards of the SAME next model (each on its own GPU group)."
        ),
    )
    parser.add_argument(
        "--skip-big-models",
        action="store_true",
        help="When orchestrating, skip models that require all GPUs (needs_all_gpus=True).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to a log file. If omitted, a timestamped file is created in outputs/logs/.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="Seconds between progress log lines to avoid flooding the terminal (default: 10s).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Optional: run this process as a shard; total number of shards (default: 1).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Optional: shard index for this process in [0..num_shards-1] (default: 0).",
    )

    args = parser.parse_args()

    # Quiet external progress bars
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("ACCELERATE_DISABLE_RICH", "1")

    # Set up logging to file + console
    if args.log_file:
        log_file = args.log_file
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOGS_DIR, f"wa_scoring_{ts}.log")
    configure_logging(log_file=log_file)
    logging.info("Logging to file: %s", log_file)

    # --- GPU Status Logger ---
    try:
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            lines = [f"Detected {num_devices} CUDA device{'s' if num_devices != 1 else ''}:"]
            for i in range(num_devices):
                prop = torch.cuda.get_device_properties(i)
                name = prop.name
                cc = f"{prop.major}.{prop.minor}"
                total_mem_gb = prop.total_memory / (1024 ** 3)
                allocated_gb = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved_gb = torch.cuda.memory_reserved(i) / (1024 ** 3)
                lines.append(
                    f"  [{i}] {name} | compute ({cc}) | total {total_mem_gb:.1f} GiB | allocated {allocated_gb:.1f} GiB | reserved {reserved_gb:.1f} GiB"
                )
            logging.info("\n" + "\n".join(lines))
        else:
            logging.info("CUDA not available")
    except Exception as e:
        logging.warning(f"Could not log GPU status: {e}")

    if not args.run_all and args.model is None:
        parser.error("You must supply --model or --run-all.")

    model_sequence: Sequence[str]
    if args.run_all:
        keys = list(MODEL_CONFIGS.keys())
        model_sequence = sorted(
            keys,
            key=lambda k: (not _is_instruction_model(MODEL_CONFIGS[k]), k)
        )
        logging.info("Model run order (instruction-first): %s", ", ".join(model_sequence))
    else:
        model_sequence = [args.model]

    # Orchestrated parallel mode: spawn child processes per model across GPU groups
    if args.run_all and args.orchestrate_parallel:
        orchestrate_run_all(args, model_sequence)
        return

    # Default sequential mode (in-process)
    df_all = load_dataset(
        use_deranged=args.use_deranged,
        deranged_path=args.deranged_path,
        derange_seed=args.derange_seed,
        force_derange=args.force_derange,
    )

    for idx, model_key in enumerate(model_sequence, start=1):
        logging.info("=== Running model %s (%d/%d) ===", model_key, idx, len(model_sequence))
        run_for_model(model_key, df_all, args)
        # Post-run cleanup to minimize fragmentation across sequential models
        try:
            clear_model_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
            gc.collect()
            time.sleep(1.0)
        except Exception as e:
            logging.warning("Post-run cleanup encountered an issue: %s", e)

if __name__ == "__main__":
    main()

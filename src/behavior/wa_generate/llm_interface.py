"""
src/llm_interface.py
Parallel chat‑request helper for the vLLM cluster.

Key points
──────────
• Gemma fix: wraps system+user instructions into a single `user` message.
• Qwen “thinking” disabled.
• Improved tolerant parsing + inline debug printouts.
"""

from __future__ import annotations

import os
import re
import time
import random
import json
import sys
from functools import partial
from typing import List, Dict, Tuple

import pandas as pd
import requests
import concurrent.futures

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import SERVER_CONFIGS, DEMOGRAPHIC_COLS, SWOW_DATA_PATH, WORKERS

# ──────────────────────────────────────────────────────────────
# Constants & helper text
# ──────────────────────────────────────────────────────────────
NON_STEERED_SYSTEM_PROMPT = "You are a participant in a word association study."
STEERED_SYSTEM_PROMPT_BASE = (
    "You are a participant in a word association study.\n"
    "You are answering as a typical participant with the following characteristics:"
)

_REFUSAL_PHRASES = [
    "i cannot",
    "i'm sorry",
    "as an ai",
]

# ──────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────
def build_messages(model_name: str, sys_prompt: str, user_prompt: str):
    """
    Return a list of messages in the correct order for the given model.

    * Gemma: **single** user turn containing both prompts.
    * Others: system → user (OpenAI‑style).
    """
    if "gemma" in model_name.lower():
        combined = sys_prompt + "\n\n" + user_prompt
        return [{"role": "user", "content": combined}]
    else:
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]


def _strip_boilerplate(text: str) -> str:
    text = re.sub(r"<think>.*?(?:\n|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return text.lstrip(" \n\r\t")


def _parse_response(text: str) -> List[str]:
    if not text or any(p in text.lower() for p in _REFUSAL_PHRASES):
        return ["parsing_error_refusal"] * 3

    text = _strip_boilerplate(text)
    parts = re.split(r"[,\n]", text)
    clean = [re.sub(r"[^\w\s-]", "", p).strip().lower() for p in parts if p.strip()]

    if len(clean) < 3:
        return ["parsing_error_count"] * 3
    return clean[:3]


# ──────────────────────────────────────────────────────────────
# Core request function
# ──────────────────────────────────────────────────────────────
def run_parallel_request(
    request_data: Tuple[int, Dict],
    model_api_name: str,
    temperature: float,
    endpoints: List[str],
    api_key: str,
    verbose: bool = False,
) -> Tuple[int, Dict]:
    original_idx, prompt_item = request_data

    messages = build_messages(
        model_api_name, prompt_item["system_prompt"], prompt_item["user_prompt"]
    )

    payload = {
        "model": model_api_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 50,
    }

    # Disable chain‑of‑thought dumps for Qwen
    if "qwen" in model_api_name.lower():
        payload["enable_thinking"] = False
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    url = f"{endpoints[original_idx % len(endpoints)]}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    if verbose:
        print("\n──── REQUEST ──────────────────────────────────")
        print(json.dumps(payload, indent=2))
        print(f"POST → {url}\n")

    max_retries = 5
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]

            if verbose:
                print("──── RESPONSE ─────────────────────────────────")
                print(raw)
                print("──────────────────────────────────────────────\n")

            parsed = _parse_response(raw)

            if parsed[0].startswith("parsing_error") and not verbose:
                print(f"[PARSE‑ERR] {parsed[0]} → {raw[:80]!r}")

            return original_idx, {"set": parsed}

        except requests.exceptions.RequestException as e:
            wait = 5 * (attempt + 1)
            print(
                f"Network error ({attempt+1}/{max_retries}): {e}. Retrying in {wait}s…"
            )
            time.sleep(wait)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Bad JSON from server: {e}")
            return original_idx, {"set": ["json_error"] * 3}

    return original_idx, {"set": ["api_error_retries_failed"] * 3}


# ──────────────────────────────────────────────────────────────
# Public trampoline
# ──────────────────────────────────────────────────────────────
def process_prompts_parallel(
    model_key: str,
    model_api_name: str,
    prompt_items: List[Dict],
    temperature: float,
    verbose: bool = False,
) -> List[Dict]:
    cfg = SERVER_CONFIGS[model_key]
    endpoints = cfg["load_balancing_endpoints"]
    api_key = os.getenv("VLLM_API_KEY", cfg.get("api_key", ""))

    fn = partial(
        run_parallel_request,
        model_api_name=model_api_name,
        temperature=temperature,
        endpoints=endpoints,
        api_key=api_key,
        verbose=verbose,
    )

    results: Dict[int, Dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        from tqdm import tqdm

        futures = pool.map(fn, enumerate(prompt_items), chunksize=max(1, len(prompt_items) // WORKERS))
        for idx, res in tqdm(
            futures,
            total=len(prompt_items),
            desc="Parallel Requests",
            leave=False,
            disable=verbose,
        ):
            results[idx] = res

    return [results[i] for i in sorted(results.keys())]


def load_swow_data(path: str, n_cues: int | None):
    df = pd.read_csv(
        path,
        usecols=["cue", "participantID"] + DEMOGRAPHIC_COLS,
        low_memory=False,
    )
    all_cues = df["cue"].astype(str).unique().tolist()
    chosen = (
        random.sample(all_cues, n_cues) if n_cues and n_cues < len(all_cues) else all_cues
    )
    demo = (
        df[df["cue"].isin(chosen)]
        .drop_duplicates(subset=["cue", "participantID"])
        .set_index("cue")
    )
    return chosen, demo


def get_demographic_profile(cue: str, demo_df: pd.DataFrame):
    try:
        subset = demo_df.loc[[cue]]
        if subset.empty:
            return None, None
        s = subset.sample(1).iloc[0]
        d = s.to_dict()

        lines = []
        if pd.notna(a := d.get("age")):
            lines.append(f"Age: {int(a)}")
        if pd.notna(g := d.get("gender")):
            lines.append(f"Gender: {g}")
        if pd.notna(l := d.get("nativeLanguage")):
            lines.append(f"Native language: {l}")
        if pd.notna(c := d.get("country")):
            lines.append(f"Country: {c}")
        if pd.notna(e := d.get("education")):
            lines.append(f"Education: {_get_education_description(e)}")

        sys_prompt = STEERED_SYSTEM_PROMPT_BASE + "\n" + "\n".join(lines)
        return d, sys_prompt
    except (KeyError, IndexError):
        return None, None


def create_profile_id(profile: Dict) -> str:
    return "profile_" + "_".join(
        f"{k}_{str(profile.get(k, 'NA')).replace(' ', '_')}" for k in DEMOGRAPHIC_COLS
    )

def _get_education_description(code: float) -> str:
    """
    Convert SWOW education numeric code to a readable label.

        1 = Some high school
        2 = High school graduate
        3 = Some college/university
        4 = College/university graduate
        5 = Graduate degree
    """
    if pd.isna(code):
        return "Not specified"
    return {
        1: "Some high school",
        2: "High school graduate",
        3: "Some college/university",
        4: "College/university graduate",
        5: "Graduate degree",
    }.get(int(code), "Not specified")
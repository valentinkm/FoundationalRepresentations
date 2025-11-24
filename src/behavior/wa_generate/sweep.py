"""
src/generate_wa_sweep.py
Entry‑point script for full prompt‑temperature‑steering sweeps.
"""

import argparse
import json
import os
import sys
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import (
    MODELS,
    PROMPTS_TO_RUN,
    TEMPERATURES_TO_RUN,
    STEERING_TO_RUN,
    SWOW_DATA_PATH,
    OUTPUT_LEXICON_FILE,
)
from src.llm_interface import (
    load_swow_data,
    get_demographic_profile,
    create_profile_id,
    process_prompts_parallel,
    NON_STEERED_SYSTEM_PROMPT,
)


# ──────────────────────────────────────────────────────────────────────────────
def load_prompt_template(name: str) -> str:
    path = os.path.join("prompts", f"{name}.txt")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        print(f"ERROR: Prompt file not found → {path}")
        sys.exit(1)


def parse_arguments():
    p = argparse.ArgumentParser("Generate LLM word associations for a prompt sweep")
    p.add_argument("--ncues", type=int, default=100, help="Number of random cues.")
    p.add_argument("--nsets", type=int, default=100, help="Response sets per cue.")
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()), help="Models.")
    p.add_argument(
        "--output_file", default=OUTPUT_LEXICON_FILE, help="Where to append JSONL."
    )
    p.add_argument(
        "--cue_batch_size",
        type=int,
        default=2,
        help="How many cues to bundle per checkpoint batch.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print every payload/response (use with ncues==1, nsets==1).",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_arguments()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    cues, demo_df = load_swow_data(SWOW_DATA_PATH, args.ncues)
    print(f"Loaded {len(cues)} cues. Each will be run with {args.nsets} sets.")

    for model_key in args.models:
        if model_key not in MODELS:
            print(f"⚠️  Model '{model_key}' missing in config – skipping.")
            continue

        print(f"\n─── Sweep for model: {model_key} ───")

        configs = [
            (prompt, temp, steer)
            for prompt in PROMPTS_TO_RUN
            for temp in TEMPERATURES_TO_RUN
            for steer in STEERING_TO_RUN
        ]

        for prompt_name, temp, is_steered in configs:
            print(f"\nConfig → prompt:{prompt_name}  temp:{temp}  steered:{is_steered}")
            template = load_prompt_template(prompt_name)

            with tqdm(
                total=len(cues),
                desc="Cue‑batches",
                disable=args.verbose,
            ) as pbar:
                for i in range(0, len(cues), args.cue_batch_size):
                    cue_chunk = cues[i : i + args.cue_batch_size]

                    # Build prompt list for this chunk
                    prompts, meta = [], []
                    for cue in cue_chunk:
                        for _ in range(args.nsets):
                            user_prompt = template.replace("{{cue}}", cue)
                            item = {"user_prompt": user_prompt}
                            m = {"cue": cue}

                            if is_steered:
                                profile, sys_prompt = get_demographic_profile(cue, demo_df)
                                if not profile:
                                    continue
                                item["system_prompt"] = sys_prompt
                                m["profile_id"] = create_profile_id(profile)
                            else:
                                item["system_prompt"] = NON_STEERED_SYSTEM_PROMPT

                            prompts.append(item)
                            meta.append(m)

                    if not prompts:
                        pbar.update(len(cue_chunk))
                        continue

                    results = process_prompts_parallel(
                        model_key,
                        MODELS[model_key],
                        prompts,
                        temp,
                        verbose=args.verbose,
                    )

                    # Bucket results by cue
                    by_cue: dict[str, list] = {}
                    for j, res in enumerate(results):
                        cue = meta[j]["cue"]
                        by_cue.setdefault(cue, [])
                        item = {"set": res["set"]}
                        if "profile_id" in meta[j]:
                            item["profile_id"] = meta[j]["profile_id"]
                        by_cue[cue].append(item)

                    with open(args.output_file, "a") as fh:
                        for cue, resp in by_cue.items():
                            fh.write(
                                json.dumps(
                                    {
                                        "cue": cue,
                                        "cfg": {
                                            "model": model_key,
                                            "prompt": prompt_name,
                                            "temp": temp,
                                            "nsets": len(resp),
                                            "steered": is_steered,
                                        },
                                        "responses": resp,
                                    }
                                )
                                + "\n"
                            )

                    pbar.update(len(cue_chunk))

    print(f"\n✅ Sweep finished – results appended to {args.output_file}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

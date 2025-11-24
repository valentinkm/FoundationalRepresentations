import argparse
import json
import os
import sys
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import MODELS, SWOW_DATA_PATH, WORKERS
from src.llm_interface import (
    load_swow_data,
    get_demographic_profile,
    create_profile_id,
    process_prompts_parallel,
    NON_STEERED_SYSTEM_PROMPT,
)


def load_prompt_template(name: str):
    path = os.path.join("prompts", f"{name}.txt")
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        sys.exit(f"❌  Prompt file not found → {path}")


def parse_args():
    p = argparse.ArgumentParser("Generate final lexicon with optimal settings")
    p.add_argument("--models", required=True, nargs="+", choices=list(MODELS.keys()))
    p.add_argument("--prompt", required=True, help="Optimal prompt name")
    p.add_argument("--temp", required=True, type=float)
    p.add_argument("--nsets", type=int, default=100)
    p.add_argument("--output_file", required=True)
    p.add_argument("--cue_batch_size", type=int, default=2)
    p.add_argument("--ncues", type=int, default=None)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    prompt_template = load_prompt_template(args.prompt)
    cues, demo_df = load_swow_data(SWOW_DATA_PATH, args.ncues)

    for model_key in args.models:
        base, ext = os.path.splitext(args.output_file)
        outfile = f"{base}_{model_key}{ext}"

        outdir = os.path.dirname(outfile)
        if outdir:  # '' when path is purely a filename
            os.makedirs(outdir, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"Model: {model_key}")
        print(f"Cues: {len(cues)}   nsets: {args.nsets}   temp: {args.temp}")
        print(f"Writing → {outfile}")
        print("=" * 80)

        steered = any(k in args.prompt for k in ["experiential", "intuition", "typical"])

        with tqdm(total=len(cues), desc=f"{model_key}") as pbar:
            for i in range(0, len(cues), args.cue_batch_size):
                chunk = cues[i : i + args.cue_batch_size]

                prompts, meta = [], []
                for cue in chunk:
                    for _ in range(args.nsets):
                        item = {"user_prompt": prompt_template.replace("{{cue}}", cue)}
                        m = {"cue": cue}

                        if steered:
                            profile, sys_p = get_demographic_profile(cue, demo_df)
                            if not profile:
                                continue
                            item["system_prompt"] = sys_p
                            m["profile_id"] = create_profile_id(profile)
                        else:
                            item["system_prompt"] = NON_STEERED_SYSTEM_PROMPT

                        prompts.append(item)
                        meta.append(m)

                if not prompts:
                    pbar.update(len(chunk))
                    continue

                results = process_prompts_parallel(
                    model_key, MODELS[model_key], prompts, args.temp
                )

                with open(outfile, "a") as fh:
                    for j, res in enumerate(results):
                        record = {
                            "cue": meta[j]["cue"],
                            "model": model_key,
                            "prompt": args.prompt,
                            "temp": args.temp,
                            "set": res["set"],
                        }
                        if "profile_id" in meta[j]:
                            record["profile_id"] = meta[j]["profile_id"]
                        fh.write(json.dumps(record) + "\n")

                pbar.update(len(chunk))

    print("\n✅  Final data generation complete.")


if __name__ == "__main__":
    main()

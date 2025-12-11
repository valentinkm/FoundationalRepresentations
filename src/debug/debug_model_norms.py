"""
src/debug_model_norms.py

Audits the quality of model-generated norms.
Checks for:
- Variance (Lazy Models)
- Mode Skew (Refusals/Defaults)
- Value Ranges
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

NORMS_DIR = Path("outputs/raw_behavior/model_norms")
OUTPUT_DIR = Path("outputs/plots/audit")

def audit_norms():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = list(NORMS_DIR.glob("*.csv"))
    
    if not files:
        print("No norm files found.")
        return

    summary = []

    for fp in files:
        model_name = fp.stem
        print(f"\nChecking {model_name}...")
        
        try:
            df = pd.read_csv(fp)
            if 'cleaned_rating' not in df.columns:
                print("  FAIL: Missing 'cleaned_rating'")
                continue
                
            norms = df['norm'].unique()
            print(f"  Found {len(norms)} norms.")
            
            for norm in norms:
                sub = df[df['norm'] == norm]
                ratings = pd.to_numeric(sub['cleaned_rating'], errors='coerce').dropna()
                
                if len(ratings) < 10:
                    continue
                    
                std = ratings.std()
                mean = ratings.mean()
                min_val = ratings.min()
                max_val = ratings.max()
                
                # Mode Skew
                mode_counts = ratings.value_counts()
                mode_val = mode_counts.index[0]
                mode_pct = mode_counts.iloc[0] / len(ratings)
                
                status = "PASS"
                issue = ""
                
                if std < 0.1:
                    status = "FAIL"
                    issue = "Low Variance"
                elif mode_pct > 0.90:
                    status = "WARNING"
                    issue = f"Skewed ({mode_pct:.1%} is {mode_val})"
                    
                summary.append({
                    'model': model_name,
                    'norm': norm,
                    'n_samples': len(ratings),
                    'mean': mean,
                    'std': std,
                    'min': min_val,
                    'max': max_val,
                    'mode_pct': mode_pct,
                    'status': status,
                    'issue': issue
                })
                
        except Exception as e:
            print(f"  Error: {e}")

    # Save Summary
    summ_df = pd.DataFrame(summary)
    summ_df.to_csv(OUTPUT_DIR / "norms_audit_summary.csv", index=False)
    print(f"\nSaved audit summary to {OUTPUT_DIR / 'norms_audit_summary.csv'}")
    
    # Print Failures
    failures = summ_df[summ_df['status'] != 'PASS']
    if not failures.empty:
        print("\n=== Issues Found ===")
        print(failures[['model', 'norm', 'status', 'issue', 'std', 'mode_pct']].to_string())
    else:
        print("\n=== All Norms Passed Checks ===")

if __name__ == "__main__":
    audit_norms()

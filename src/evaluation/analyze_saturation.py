"""
src/evaluation/analyze_saturation.py

Analyzes the output of the dimensionality sweep.
1. Aggregates R2 scores across all norms.
2. Identifies the 'saturation point' (where gains diminish).
3. Generates a plot: R2 vs Dimensions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIG ---
INPUT_CSV = Path("outputs/results/saturation_sweep.csv")
PLOT_OUTPUT = Path("outputs/plots/saturation_curve.png")
SUMMARY_OUTPUT = Path("outputs/results/saturation_summary.txt")

def main():
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found. Run the sweep first.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # 1. Aggregate: Mean R2 per Dimension per Modality
    # We group by dim/modality and average across all norms
    summary = df.groupby(['modality', 'dim'])['r2'].agg(['mean', 'sem', 'count']).reset_index()
    
    # 2. Calculate "Gain" (Delta from previous step)
    summary.sort_values(['modality', 'dim'], inplace=True)
    summary['prev_mean'] = summary.groupby('modality')['mean'].shift(1)
    summary['gain'] = summary['mean'] - summary['prev_mean']
    
    # 3. Print Text Report
    print(f"\n{'='*60}")
    print(f"{'DIMENSION SATURATION REPORT':^60}")
    print(f"{'='*60}")
    
    pivoted = summary.pivot(index='dim', columns='modality', values='mean')
    pivoted['Delta (Act - Beh)'] = pivoted['activation'] - pivoted['behavior']
    
    print("\n--- Mean R2 Scores by Dimension ---")
    print(pivoted.round(4).to_string())
    
    print(f"\n{'='*60}")
    print("ANALYSIS OF GAINS (Diminishing Returns)")
    print(f"{'='*60}")
    
    for mod in ['behavior', 'activation']:
        print(f"\n[{mod.upper()}]")
        subset = summary[summary['modality'] == mod]
        for _, row in subset.iterrows():
            d = int(row['dim'])
            r2 = row['mean']
            gain = row['gain']
            
            gain_str = f"+{gain:.4f}" if pd.notna(gain) else "N/A"
            if pd.notna(gain) and gain < 0.005: 
                status = "SATURATED (Gain < 0.5%)"
            elif pd.notna(gain) and gain < 0.01:
                status = "Plateauing..."
            else:
                status = "Rising"
                
            print(f"  Dim {d:<4} | R2: {r2:.4f} | Gain: {gain_str:<8} | {status}")

    # 4. Save Summary to File
    with open(SUMMARY_OUTPUT, 'w') as f:
        f.write(pivoted.to_string())

    # 5. Plotting
    PLOT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Line plot with confidence intervals (error bands)
    sns.lineplot(
        data=df, 
        x='dim', 
        y='r2', 
        hue='modality', 
        marker='o',
        errorbar=('ci', 95), # 95% confidence interval across norms
        palette={'behavior': '#1f77b4', 'activation': '#ff7f0e'}
    )
    
    plt.title('Performance Saturation: Behavior vs. Activation', fontsize=14)
    plt.xlabel('Dimension ($d$)', fontsize=12)
    plt.ylabel('Predictive Power ($R^2$)', fontsize=12)
    plt.xscale('log')
    plt.xticks(sorted(df['dim'].unique()), labels=sorted(df['dim'].unique()))
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=300)
    print(f"\n[Visual] Plot saved to {PLOT_OUTPUT}")
    print(f"[Data] Summary saved to {SUMMARY_OUTPUT}")

if __name__ == "__main__":
    main()
"""
src/visualize/plot_results.py

Visualizes the "Brain vs. Behavior" Comparison.
Input: outputs/results/norm_prediction_scores.csv (Long Format)
Output: 3 Plots in outputs/figures/

1. performance_summary.png: Bar chart of Mean R2 (Human vs. Models vs. Methods).
2. brain_vs_behavior_scatter.png: Direct head-to-head scatter (Activation vs. Contrastive).
3. delta_distribution.png: Violin plot showing the "Hidden Information" gap.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import re

# --- CONFIG ---
# Find project root by walking up until we see the outputs folder
try:
    script_path = Path(__file__).resolve()
except NameError:
    script_path = Path.cwd()

BASE_DIR = None
for parent in [script_path] + list(script_path.parents):
    if (parent / "outputs").exists():
        BASE_DIR = parent
        break

if BASE_DIR is None:
    print(f"❌ Critical Error: Could not locate project root from {script_path}")
    print("   Please run this script from the project root or adjust path logic.")
    sys.exit(1)

INPUT_CSV = BASE_DIR / "outputs" / "results" / "norm_prediction_scores.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Visual Style
sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
plt.rcParams['figure.dpi'] = 300
PALETTE = {
    "Baseline": "#7f7f7f",           # Grey (Human)
    "Activation": "#d62728",         # Red (Brain)
    "Behavior_Contrastive": "#1f77b4", # Blue (Clean Behavior)
    "Behavior_Standard": "#aec7e8",    # Light Blue (Noisy Behavior)
    "Behavior_Generated": "#2ca02c"    # Green (Active)
}

def load_data():
    if not INPUT_CSV.exists():
        print(f"❌ File not found: {INPUT_CSV}")
        sys.exit(1)
    df = pd.read_csv(INPUT_CSV)
    # Normalize model names so Activation and Behavior align (e.g., "gemma-3-27-b" -> "gemma-3-27b")
    df['Model'] = df['Model'].astype(str).apply(lambda name: re.sub(r'-b$', 'b', name))
    print(f"✅ Loaded {len(df)} rows.")
    return df

def plot_performance_summary(df):
    """
    Main Bar Chart: Mean R2 across all norms for each Model/Type.
    Shows the hierarchy of representation quality.
    """
    print("Generating Summary Bar Chart...")
    plt.figure(figsize=(12, 6))
    
    # Filter out Shuffled if present (it's noise)
    df_plot = df[df['Embedding_Type'] != 'Behavior_Shuffled']
    
    # Order models alphabetically but keep Human first if possible
    models = sorted(df_plot['Model'].unique())
    if 'Human' in models:
        models.remove('Human')
        models = ['Human'] + models

    ax = sns.barplot(
        data=df_plot,
        x="Model",
        y="R2_Mean",
        hue="Embedding_Type",
        order=models,
        palette=PALETTE,
        errorbar=('ci', 95),
        capsize=0.1
    )
    
    plt.title("Semantic Encoding Power: Brain vs. Behavior", fontsize=16, fontweight='bold')
    plt.ylabel("Predictive Accuracy ($R^2$)", fontsize=14)
    plt.xlabel("")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Representation")
    plt.ylim(0, 0.6) # Adjust based on data range
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "1_performance_summary.png")
    plt.close()

def plot_head_to_head(df):
    """
    Scatter Plot: Activation (Y) vs. Behavior Contrastive (X).
    Points above diagonal = "Hidden Info" (Brain knows more than it says).
    """
    print("Generating Head-to-Head Scatter...")
    
    # 1. Pivot Data to Wide Format (Norm x Type)
    # We need columns: R2_Activation, R2_Contrastive
    df_piv = df.pivot_table(
        index=['Model', 'Norm'], 
        columns='Embedding_Type', 
        values='R2_Mean'
    ).reset_index()
    
    # Filter for valid pairs (Must have both)
    if 'Activation' not in df_piv.columns or 'Behavior_Contrastive' not in df_piv.columns:
        print("⚠️  Skipping Scatter: Missing columns for comparison.")
        return

    df_piv = df_piv.dropna(subset=['Activation', 'Behavior_Contrastive'])
    
    # Exclude Human (No activation)
    df_piv = df_piv[df_piv['Model'] != 'Human']
    
    if df_piv.empty:
        print("⚠️  Skipping Scatter: No overlapping Activation and Behavior_Contrastive rows after normalization.")
        return

    # Plot
    g = sns.lmplot(
        data=df_piv,
        x="Behavior_Contrastive",
        y="Activation",
        col="Model",
        hue="Model",
        col_wrap=2,
        height=5,
        scatter_kws={'alpha': 0.5, 's': 20},
        line_kws={'color': 'black', 'linestyle': '--'}
    )
    
    # Add diagonal line x=y
    for ax in g.axes.flat:
        lims = [0, 0.8]
        ax.plot(lims, lims, color='gray', linestyle=':', zorder=0)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.8)
    
    g.set_axis_labels("Behavior ($R^2$)", "Activation ($R^2$)")
    g.fig.suptitle("Does the Brain know more than it says?", y=1.03, fontsize=16, fontweight='bold')
    
    plt.savefig(OUTPUT_DIR / "2_brain_vs_behavior_scatter.png")
    plt.close()

def plot_delta_dist(df):
    """
    Distribution of (Activation - Behavior) Gap.
    Positive = Brain is Richer.
    Negative = Behavior is clearer.
    """
    print("Generating Delta Distribution...")
    
    # Pivot again
    df_piv = df.pivot_table(
        index=['Model', 'Norm'], 
        columns='Embedding_Type', 
        values='R2_Mean'
    ).reset_index()
    
    if 'Activation' not in df_piv.columns or 'Behavior_Contrastive' not in df_piv.columns:
        return

    df_piv['Delta'] = df_piv['Activation'] - df_piv['Behavior_Contrastive']
    df_piv = df_piv[df_piv['Model'] != 'Human'].dropna(subset=['Delta'])
    
    if df_piv.empty:
        print("⚠️  Skipping Delta Distribution: No overlapping Activation and Behavior_Contrastive rows after normalization.")
        return

    plt.figure(figsize=(10, 6))
    
    sns.violinplot(
        data=df_piv,
        x="Model",
        y="Delta",
        palette="muted",
        inner="quartile"
    )
    
    plt.axhline(0, color='black', linestyle='-', linewidth=1.5)
    plt.text(0.5, 0.25, "Brain Superior", transform=plt.gca().transAxes, ha='center', color='red', alpha=0.5)
    plt.text(0.5, 0.15, "Behavior Superior", transform=plt.gca().transAxes, ha='center', color='blue', alpha=0.5)
    
    plt.title("The 'Ineffability' Gap: Where Brain Outperforms Output", fontsize=14)
    plt.ylabel(r"$\Delta R^2$ (Activation - Behavior)")
    plt.xlabel("")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "3_delta_distribution.png")
    plt.close()

def print_top_divergent_norms(df):
    """Which concepts are 'stuck' in the brain?"""
    df_piv = df.pivot_table(
        index=['Model', 'Norm'], 
        columns='Embedding_Type', 
        values='R2_Mean'
    ).reset_index()
    
    if 'Activation' not in df_piv.columns: return

    df_piv['Delta'] = df_piv['Activation'] - df_piv['Behavior_Contrastive']
    
    print("\n=== TOP 5 'INEFFABLE' CONCEPTS (Brain >>> Behavior) ===")
    top = df_piv.sort_values('Delta', ascending=False).head(5)
    print(top[['Model', 'Norm', 'Activation', 'Behavior_Contrastive', 'Delta']])
    
    print("\n=== TOP 5 'EXPLICIT' CONCEPTS (Behavior >>> Brain) ===")
    bot = df_piv.sort_values('Delta', ascending=True).head(5)
    print(bot[['Model', 'Norm', 'Activation', 'Behavior_Contrastive', 'Delta']])

if __name__ == "__main__":
    df = load_data()
    plot_performance_summary(df)
    plot_head_to_head(df)
    plot_delta_dist(df)
    print_top_divergent_norms(df)
    print(f"\n✨ Plots saved to: {OUTPUT_DIR}")

"""
visualize/compare.py

Visualizes the output of the "Head-to-Head" comparison.
Generates plots in outputs/figures/
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# --- CONFIG ---
# Robustly find the project root (looks for 'outputs' folder)
current_path = Path(__file__).resolve()
base_dir = None

# Try specific parent levels to find the project root
for parent in [current_path.parents[1], current_path.parents[2]]:
    if (parent / "outputs").exists():
        base_dir = parent
        break

if base_dir is None:
    print(f"❌ Critical Error: Could not locate project root from {current_path}")
    print("   Please run this script from the project root or adjust path logic.")
    sys.exit(1)

input_path = base_dir / "outputs" / "results" / "comparison_results.csv"
output_dir = base_dir / "outputs" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

def load_data():
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        sys.exit(1)
    
    df = pd.read_csv(input_path)
    
    # Ensure columns exist (Integrity check)
    required = ['model', 'r2_behavior', 'r2_activation', 'delta']
    if not all(col in df.columns for col in required):
        print(f"❌ CSV is missing columns. Found: {df.columns}")
        sys.exit(1)
            
    print(f"✅ Loaded {len(df)} rows from {input_path.name}")
    return df

def plot_summary_bars(df):
    """Bar chart of Mean R2 for Behavior vs Activation."""
    print("Generating Summary Bar Chart...")
    
    # Melt for Seaborn (Long format)
    df_long = df.melt(
        id_vars=['model'], 
        value_vars=['r2_behavior', 'r2_activation'],
        var_name='Modality', 
        value_name='R2 Score'
    )
    
    # Rename for legend
    df_long['Modality'] = df_long['Modality'].replace({
        'r2_behavior': 'Passive Behavior',
        'r2_activation': 'Internal Activation'
    })

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_long, 
        x='model', 
        y='R2 Score', 
        hue='Modality',
        palette="viridis",
        errorbar=('ci', 95)
    )
    
    plt.title("Mean Prediction Accuracy: Internal vs External", fontsize=14)
    plt.ylim(0, 0.8) 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / "summary_performance.png")
    plt.close()

def plot_head_to_head_scatter(df):
    """Scatter plots per model: Activation vs Behavior."""
    print("Generating Head-to-Head Scatters...")
    
    # Create a FacetGrid
    g = sns.FacetGrid(df, col="model", col_wrap=2, height=4, aspect=1)
    g.map(sns.scatterplot, "r2_behavior", "r2_activation", alpha=0.6, s=30)
    
    # Add diagonal lines (y=x)
    def plot_diag(**kwargs):
        ax = plt.gca()
        lims = [0, 1] # Fixed limits for cleaner comparison
        ax.plot(lims, lims, '--', color="red", alpha=0.5, label="Equal Performance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    g.map(plot_diag)
    
    g.set_axis_labels("Behavior R2", "Activation R2")
    g.fig.suptitle("Norm-by-Norm Comparison (Points above line = Activation Wins)", y=1.02)
    
    plt.savefig(output_dir / "head_to_head_scatter.png")
    plt.close()

def plot_delta_distribution(df):
    """Violin plot of the Delta (Advantage) to see distribution shape."""
    print("Generating Delta Distributions...")
    
    plt.figure(figsize=(10, 6))
    
    # Order by median delta
    order = df.groupby('model')['delta'].median().sort_values(ascending=False).index
    
    sns.violinplot(
        data=df, 
        x='model', 
        y='delta', 
        order=order, 
        palette="mako", 
        inner="quartile"
    )
    
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.title("Distribution of 'Internal Advantage' (Activation R2 - Behavior R2)", fontsize=14)
    plt.ylabel("Delta (R2 Improvement)")
    plt.tight_layout()
    plt.savefig(output_dir / "delta_distribution.png")
    plt.close()

def print_stats(df):
    print("\n" + "="*40)
    print("STATISTICAL SUMMARY")
    print("="*40)
    
    summary = df.groupby("model")[["r2_behavior", "r2_activation", "delta"]].mean()
    print(summary)
    
    print("\n--- Largest 'Hidden' Insights (Top Delta) ---")
    # Show top 5 norms where Activation beats Behavior the most
    top_delta = df.sort_values("delta", ascending=False).head(5)
    print(top_delta[['model', 'norm', 'r2_behavior', 'r2_activation', 'delta']].to_string(index=False))

if __name__ == "__main__":
    df = load_data()
    
    plot_summary_bars(df)
    plot_head_to_head_scatter(df)
    plot_delta_distribution(df)
    print_stats(df)
    
    print(f"\n✨ Done! Plots saved to {output_dir}")
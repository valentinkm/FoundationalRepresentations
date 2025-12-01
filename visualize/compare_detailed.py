"""
src/visualize/plot_detailed_comparison.py

Deep Dive Visualization:
Visualizes the individual norm-level differences between Activation and Behavior.

Outputs:
1. scatter_comparison.png (The "Diagonal" plot)
2. delta_distribution.png (How consistent is the advantage?)
3. norm_divergence.png (Which specific concepts are handled better by the Brain?)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def setup_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['figure.dpi'] = 300

def load_results():
    # Robust Path Detection
    # Determine if we are in 'src/visualize' (parents[2]) or just 'visualize' (parents[1])
    script_path = Path(__file__).resolve()
    
    if script_path.parent.name == 'visualize':
        if script_path.parents[1].name == 'src':
            base_dir = script_path.parents[2] # src/visualize -> src -> root
        else:
            base_dir = script_path.parents[1] # visualize -> root
    else:
        base_dir = script_path.parent # Running from root?
        
    csv_path = base_dir / "outputs" / "results" / "comparison_results.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Results not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    # Ensure Delta exists
    if "delta" not in df.columns:
        df["delta"] = df["r2_activation"] - df["r2_behavior"]
    return df, base_dir

def plot_scatter_grid(df, output_path):
    print("Plotting Scatter Grid...")
    
    # FacetGrid allows us to show one plot per model
    g = sns.relplot(
        data=df,
        x="r2_behavior", 
        y="r2_activation",
        col="model", 
        col_wrap=2,
        hue="delta", 
        palette="vlag", 
        size="n_samples",
        sizes=(10, 100),
        alpha=0.8,
        height=5, 
        aspect=1
    )
    
    # Add Diagonal Line (x=y) to every subplot
    for ax in g.axes.flat:
        ax.plot([0, 1], [0, 1], ls="--", c="black", alpha=0.5, label="Equal Performance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"Behavioral $R^2$ (Logprobs)")
        ax.set_ylabel(r"Activation $R^2$ (Hidden States)")

    g.fig.suptitle("Brain vs. Behavior: Performance by Norm", fontsize=16, fontweight='bold', y=1.02)
    
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")

def plot_delta_distribution(df, output_path):
    print("Plotting Distribution...")
    
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(
        data=df, 
        x="delta", 
        hue="model", 
        fill=True, 
        alpha=0.3, 
        palette="viridis",
        linewidth=2
    )
    
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel(r"$\Delta R^2$ (Activation - Behavior)")
    plt.title("Distribution of Activation Advantage", fontsize=14, fontweight='bold')
    plt.text(0.02, 0.5, "Brain Wins →", transform=plt.gca().get_xaxis_transform(), color='green')
    plt.text(-0.02, 0.5, "← Behavior Wins", transform=plt.gca().get_xaxis_transform(), ha='right', color='orange')
    
    plt.savefig(output_path)
    print(f"Saved: {output_path}")

def plot_top_divergent_norms(df, output_path):
    """Shows which specific norms have the biggest gap."""
    print("Plotting Divergent Norms...")
    
    # Calculate average delta per norm across all models
    norm_diffs = df.groupby("norm_name")["delta"].mean().sort_values()
    
    # Take Top 10 (Brain wins) and Bottom 10 (Behavior wins)
    top_10 = norm_diffs.tail(10)
    bot_10 = norm_diffs.head(10)
    subset = pd.concat([top_10, bot_10]).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#56B4E9' if x > 0 else '#E69F00' for x in subset.values]
    
    sns.barplot(x=subset.values, y=subset.index, palette=colors)
    
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel(r"Mean $\Delta R^2$ (Activation - Behavior)")
    plt.title("Which Concepts are 'Hidden' in the Brain?", fontsize=14, fontweight='bold')
    
    # Annotate regions
    plt.text(0.02, len(subset)-1, "Brain Represents Better", color='#56B4E9', fontweight='bold', va='bottom')
    plt.text(-0.02, len(subset)-1, "Behavior Represents Better", color='#E69F00', fontweight='bold', ha='right', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")

def main():
    setup_style()
    try:
        df, base_dir = load_results()
        viz_dir = base_dir / "visualize"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        plot_scatter_grid(df, viz_dir / "scatter_comparison.png")
        plot_delta_distribution(df, viz_dir / "delta_distribution.png")
        plot_top_divergent_norms(df, viz_dir / "norm_divergence.png")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
"""
src/visualize/plot_results.py

Visualization suite for the Foundational Representations project.
Generates publication-quality plots from the evaluation results.

Outputs:
1. leaderboard_human_prediction.png (Overall R2)
2. heatmap_selected_norms.png (Fine-grained comparison)
3. leaderboard_self_consistency.png (Internal alignment)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# --- CONFIG ---
SELECTED_NORMS = [
    "concreteness_brysbaert",
    "valence_mohammad",
    "arousal_warriner",
    "dominance_warriner",
    "age_of_acquisition_kuperman",
    "sensorimotor_strength_lynott",
    "body_object_interaction_pexman",
    "socialness_diveica",
    "humor_engelthaler",
    "gender_association_glasgow"
]

def setup_style():
    """Set scientific plotting style."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

def clean_model_names(series):
    """Make model names readable for plots."""
    # Order of replacement matters
    s = series.str.replace("passive_", "").str.replace("active_", "")
    s = s.str.replace("_instruct", "").str.replace("_base", "")
    s = s.str.replace("human_matrix", "Human Data")
    s = s.str.replace("-", " ").str.replace("_", " ")
    # Capitalize
    return s.str.title().str.replace("Llama", "Llama").str.replace("Gpt", "GPT")

def plot_leaderboard(csv_path, output_path, title="Predicting Human Psycholinguistic Norms"):
    print(f"Plotting Leaderboard from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Aggregate across all norms
    leaderboard = df.groupby("embedding_source")["r2_mean"].mean().reset_index()
    leaderboard.rename(columns={"r2_mean": "mean"}, inplace=True)
    
    # Clean names
    leaderboard["clean_name"] = clean_model_names(leaderboard["embedding_source"])
    
    # Sort Descending (Best at Top)
    leaderboard = leaderboard.sort_values("mean", ascending=False).reset_index(drop=True)
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=leaderboard,
        x="mean",
        y="clean_name",
        palette="viridis",
        hue="clean_name",
        legend=False,
        order=leaderboard["clean_name"]  # Explicit ordering
    )
    
    plt.xlabel("Mean $R^2$ (Ridge Regression)")
    plt.ylabel("")
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlim(0, max(leaderboard["mean"]) * 1.1) # Add headroom for labels
    
    # Add value labels
    for i, v in enumerate(leaderboard["mean"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va='center', fontsize=10, color='black')
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()

def plot_heatmap(csv_path, output_path):
    print(f"Plotting Heatmap...")
    df = pd.read_csv(csv_path)
    
    # Filter to selected norms (using partial matching)
    mask = df["norm_name"].apply(lambda x: any(s in x for s in SELECTED_NORMS))
    subset = df[mask].copy()
    
    if subset.empty:
        print("Warning: No selected norms found for heatmap. Skipping.")
        return

    # Pivot: Rows=Models, Cols=Norms
    pivot = subset.pivot(index="embedding_source", columns="norm_name", values="r2_mean")
    
    # Clean Index
    pivot.index = clean_model_names(pd.Series(pivot.index))
    
    # Sort by average performance
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=False).drop(columns="mean")
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="magma",
        cbar_kws={'label': '$R^2$ Score'},
        linewidths=0.5
    )
    plt.title("Performance on Specific Psycholinguistic Dimensions", fontsize=14, fontweight='bold')
    plt.ylabel("")
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()

def plot_self_consistency(csv_path, output_path):
    if not Path(csv_path).exists():
        print("Self-consistency results not found. Skipping.")
        return

    print(f"Plotting Self-Consistency...")
    df = pd.read_csv(csv_path)
    
    # Aggregate
    leaderboard = df.groupby("embedding_source")["self_r2_mean"].mean().reset_index()
    leaderboard.rename(columns={"self_r2_mean": "mean"}, inplace=True)
    
    leaderboard["clean_name"] = clean_model_names(leaderboard["embedding_source"])
    leaderboard = leaderboard.sort_values("mean", ascending=False).reset_index(drop=True)
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=leaderboard,
        x="mean",
        y="clean_name",
        palette="mako",
        hue="clean_name",
        legend=False,
        order=leaderboard["clean_name"]
    )
    
    plt.xlabel("Mean $R^2$")
    plt.ylabel("")
    plt.title("Self-Consistency", fontsize=14, fontweight='bold')
    plt.xlim(min(0, min(leaderboard["mean"])*1.1), max(leaderboard["mean"])*1.1)

    # Add value labels
    for i, v in enumerate(leaderboard["mean"]):
        # Handle negative bars correctly for label placement
        offset = 0.01 if v >= 0 else -0.05
        ax.text(v + offset, i, f"{v:.3f}", va='center', fontsize=10, color='black')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()

def main():
    setup_style()
    
    # Auto-detect paths: assume script is in <repo>/src/visualize/plot_results.py
    # So root is parents[1]
    try:
        script_path = Path(__file__).resolve()
        # Fallback logic if running from root directly
        if script_path.parent.name != 'visualize':
            base_dir = script_path.parent
        else:
            base_dir = script_path.parents[1]
            
        results_dir = base_dir / "outputs" / "results"
        viz_dir = base_dir / "visualize"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        eval_csv = results_dir / "evaluation_results.csv"
        self_csv = results_dir / "self_consistency_results.csv"
        
        if eval_csv.exists():
            plot_leaderboard(eval_csv, viz_dir / "leaderboard_human_prediction.png")
            plot_heatmap(eval_csv, viz_dir / "heatmap_selected_norms.png")
        else:
            print(f"Error: Could not find {eval_csv}. Check your path.")
            
        if self_csv.exists():
            plot_self_consistency(self_csv, viz_dir / "leaderboard_self_consistency.png")
            
    except Exception as e:
        print(f"Critical Error: {e}")

if __name__ == "__main__":
    main()
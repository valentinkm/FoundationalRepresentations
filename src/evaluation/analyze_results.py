"""
src/analysis/visualize_results.py

Analyzes the outputs of compare_representations.py.
Generates high-resolution plots to compare Activation vs. Behavioral embeddings.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "outputs" / "results"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

HUMAN_PRED_CSV = RESULTS_DIR / "norm_prediction_scores.csv"
SELF_PRED_CSV = RESULTS_DIR / "self_prediction_scores.csv"

# Set visual style
sns.set_theme(style="whitegrid", context="talk")
PALETTE = "viridis"

def load_data():
    """Load and preprocess the CSVs."""
    dfs = {}
    
    # 1. Load Human Prediction (Task A)
    if HUMAN_PRED_CSV.exists():
        df = pd.read_csv(HUMAN_PRED_CSV)
        # Filter out Baseline if needed, or keep for reference
        dfs['human'] = df
        print(f"✅ Loaded Human Predictions: {len(df)} rows")
    else:
        print(f"⚠️ Missing {HUMAN_PRED_CSV}")

    # 2. Load Self Prediction (Task B)
    if SELF_PRED_CSV.exists():
        df = pd.read_csv(SELF_PRED_CSV)
        dfs['self'] = df
        print(f"✅ Loaded Self Predictions: {len(df)} rows")
    else:
        print(f"⚠️ Missing {SELF_PRED_CSV} (Self-prediction might not have finished yet)")
        
    return dfs

def plot_leaderboard(df):
    """Global performance overview by Model and Embedding Type."""
    plt.figure(figsize=(12, 6))
    
    # Sort by performance
    order = df.groupby('Model')['R2_Mean'].mean().sort_values(ascending=False).index
    
    ax = sns.barplot(
        data=df, 
        x='Model', 
        y='R2_Mean', 
        hue='Embedding_Type',
        order=order,
        palette="magma",
        errorbar=('ci', 95),
        capsize=0.1
    )
    
    plt.title("Global Representation Performance (Predicting Human Norms)")
    plt.ylabel(r"Mean $R^2$ Score")
    plt.xlabel("Model")
    plt.xticks(rotation=15, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Representation")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_global_leaderboard.png", dpi=300)
    print("  -> Saved 01_global_leaderboard.png")
    plt.close()

def plot_activation_vs_behavior(df):
    """
    Scatter plot comparing Activation performance vs Behavioral performance
    for the SAME model and SAME norm.
    """
    # Pivot to get columns for Activation and Behavior
    # We assume 'Behavior_Standard' is the main comparison, but valid for others too
    
    # 1. Filter for relevant types
    target_behavior = "Behavior_Standard"
    if "Behavior_Standard_300d" in df['Embedding_Type'].unique():
        target_behavior = "Behavior_Standard_300d"

    subset = df[df['Embedding_Type'].isin(['Activation', target_behavior])].copy()
    
    if subset.empty:
        print("⚠️ Skipping Activation vs Behavior plot (missing data types).")
        return

    # Pivot: Index=[Model, Norm], Columns=[Embedding_Type], Values=R2_Mean
    pivoted = subset.pivot_table(index=['Model', 'Norm'], columns='Embedding_Type', values='R2_Mean').reset_index()
    pivoted = pivoted.dropna() # Only keep rows where both exist

    plt.figure(figsize=(10, 10))
    
    # Draw diagonal reference line
    plt.plot([-0.2, 1.0], [-0.2, 1.0], ls="--", c="gray", alpha=0.5)
    
    sns.scatterplot(
        data=pivoted,
        x='Activation',
        y=target_behavior,
        hue='Model',
        style='Model',
        s=100,
        alpha=0.8
    )
    
    plt.title(f"Internal State (Activation) vs Output (Behavior)\nEach point is one Norm (e.g., 'Valence')")
    plt.xlabel(r"Activation $R^2$")
    plt.ylabel(f"{target_behavior} " + r"$R^2$")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_activation_vs_behavior.png", dpi=300)
    print("  -> Saved 02_activation_vs_behavior.png")
    plt.close()

def plot_norm_heatmap(df):
    """Heatmap showing which specific norms are easy/hard for each model."""
    # Pivot: Index=[Model + Type], Columns=[Norm], Values=R2
    df['Config'] = df['Model'] + " (" + df['Embedding_Type'] + ")"
    
    # Select top 20 norms by variance or just top 20 alphabetically to keep plot readable
    top_norms = df['Norm'].unique()[:30] 
    subset = df[df['Norm'].isin(top_norms)]

    matrix = subset.pivot_table(index='Config', columns='Norm', values='R2_Mean')
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        matrix, 
        cmap="inferno", 
        annot=True, 
        fmt=".2f", 
        linewidths=.5,
        cbar_kws={'label': r'$R^2$ Score'}
    )
    plt.title("Performance Heatmap by Specific Norm")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_norm_heatmap.png", dpi=300)
    print("  -> Saved 03_norm_heatmap.png")
    plt.close()

def plot_alignment_gap(df_human, df_self):
    """
    Does predicting your own norms (Task B) imply predicting human norms (Task A)?
    Merges Human and Self dataframes.
    """
    if df_self is None: return

    # Merge on Model, Embedding_Type, Norm
    merged = pd.merge(
        df_human, 
        df_self, 
        on=['Model', 'Embedding_Type', 'Norm'], 
        suffixes=('_Human', '_Self')
    )
    
    if merged.empty:
        print("⚠️ No overlap found between Human and Self prediction files yet.")
        return

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=merged,
        x='R2_Mean_Self',
        y='R2_Mean_Human',
        hue='Model',
        style='Embedding_Type',
        s=100
    )
    
    # Correlation
    corr = merged['R2_Mean_Self'].corr(merged['R2_Mean_Human'])
    
    plt.title(f"Alignment: Self-Knowledge vs. Human-Knowledge\nPearson r = {corr:.3f}")
    plt.xlabel(r"Predicting Self (Task B) $R^2$")
    plt.ylabel(r"Predicting Humans (Task A) $R^2$")
    plt.plot([0, 1], [0, 1], ls="--", c="red", alpha=0.3) # Identity line
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_alignment_scatter.png", dpi=300)
    print("  -> Saved 04_alignment_scatter.png")
    plt.close()

def print_summary_stats(df):
    print("\n" + "="*40)
    print("📊 STATISTICAL SUMMARY")
    print("="*40)
    
    # 1. Best Model/Layer Combo
    best = df.loc[df['R2_Mean'].idxmax()]
    print(f"🏆 Best Single Result:")
    print(f"   Model: {best['Model']}")
    print(f"   Type:  {best['Embedding_Type']}")
    print(f"   Norm:  {best['Norm']}")
    print(f"   R2:    {best['R2_Mean']:.4f}")
    
    # 2. Average by Type
    print("\n📈 Average R2 by Embedding Type:")
    print(df.groupby('Embedding_Type')['R2_Mean'].mean().sort_values(ascending=False))

    # 3. Average by Model
    print("\n🤖 Average R2 by Model (All Types):")
    print(df.groupby('Model')['R2_Mean'].mean().sort_values(ascending=False))

def main():
    print("🎨 Starting Analysis...")
    data = load_data()
    
    if 'human' not in data or data['human'].empty:
        print("❌ No human prediction data found. Aborting.")
        sys.exit(0)

    df_human = data['human']
    df_self = data.get('self', None)

    # Clean up names for plotting if necessary
    df_human['Model'] = df_human['Model'].str.replace("-instruct", "").str.replace("-base", "")

    # Run Plots
    print("\n🖼️  Generating Plots...")
    plot_leaderboard(df_human)
    plot_activation_vs_behavior(df_human)
    plot_norm_heatmap(df_human)
    
    if df_self is not None:
        plot_alignment_gap(df_human, df_self)
        
    # Text Stats
    print_summary_stats(df_human)
    print("\n✅ Analysis Complete. Check 'outputs/plots' folder.")

if __name__ == "__main__":
    main()
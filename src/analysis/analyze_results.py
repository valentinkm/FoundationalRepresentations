"""
src/analysis/analyze_results.py

Analyzes self-consistency prediction results.
- Categorizes norms.
- Computes Mean/Median R^2.
- Visualizes distributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import argparse

OUTPUT_DIR = Path("outputs/plots/analysis")

def categorize_norm(name):
    name = name.lower()
    if any(x in name for x in ['visual', 'haptic', 'auditory', 'gustatory', 'olfactory', 'interoceptive', 'sensorimotor', 'perceptual', 'action', 'motor', 'body', 'mouth', 'hand', 'foot', 'head']):
        return 'Sensorimotor'
    if any(x in name for x in ['valence', 'arousal', 'dominance', 'happiness', 'fear', 'sadness', 'anger', 'disgust', 'emotion']):
        return 'Emotion'
    if any(x in name for x in ['concreteness', 'imageability', 'familiarity', 'semantic', 'abstractness']):
        return 'Semantic'
    if any(x in name for x in ['aoa', 'age', 'acquisition']):
        return 'Age of Acquisition'
    if any(x in name for x in ['frequency', 'length', 'letters', 'phonemes', 'syllables']):
        return 'Lexical'
    if 'social' in name:
        return 'Social'
    return 'Other'

def main():
    parser = argparse.ArgumentParser(description="Analyze self-consistency results.")
    parser.add_argument("input_file", type=Path, help="Path to results CSV")
    args = parser.parse_args()
    
    RESULTS_PATH = args.input_file
    
    if not RESULTS_PATH.exists():
        print(f"File not found: {RESULTS_PATH}")
        return

    print(f"Loading {RESULTS_PATH}...")
    df = pd.read_csv(RESULTS_PATH)
    
    # Add Category
    df['category'] = df['norm_name'].apply(categorize_norm)
    
    # Filter out extremely poor results (optional, but good for visualization scale)
    # df = df[df['r2_mean'] > -1.0] 
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Overall Statistics
    print("\n--- Overall Statistics (R^2) ---")
    # Show all columns
    pd.set_option('display.max_columns', None)
    overall = df.groupby('embedding_source')['r2_mean'].agg(['mean', 'median', 'std', 'count']).sort_values('median', ascending=False)
    print(overall)
    overall.to_csv(OUTPUT_DIR / "stats_overall.csv")
    
    # 2. Category Statistics
    print("\n--- Category Statistics (Mean R^2) ---")
    cat_stats = df.pivot_table(index='embedding_source', columns='category', values='r2_mean', aggfunc='mean')
    print(cat_stats)
    cat_stats.to_csv(OUTPUT_DIR / "stats_by_category_mean.csv")
    
    print("\n--- Category Statistics (Median R^2) ---")
    cat_stats_med = df.pivot_table(index='embedding_source', columns='category', values='r2_mean', aggfunc='median')
    print(cat_stats_med)
    cat_stats_med.to_csv(OUTPUT_DIR / "stats_by_category_median.csv")
    # 3. Individual Norm Statistics
    print("\n--- Individual Norm Statistics ---")
    norm_stats = df.pivot_table(index=['category', 'norm_name'], columns='embedding_source', values='r2_mean')
    print(norm_stats)
    norm_stats.to_csv(OUTPUT_DIR / "stats_by_individual_norm.csv")

    # 4. Visualization
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Existing Category Boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='r2_mean', y='embedding_source', hue='category', orient='h')
    plt.title("R^2 Distribution by Model and Category")
    plt.xlabel("R^2 Score")
    plt.ylabel("Embedding Source")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "boxplot_r2_by_category.png")
    print(f"\nSaved plot to {OUTPUT_DIR / 'boxplot_r2_by_category.png'}")
    
    # Existing Overall Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='r2_mean', y='embedding_source', orient='h')
    plt.title("Overall R^2 Distribution by Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "boxplot_r2_overall.png")
    print(f"Saved plot to {OUTPUT_DIR / 'boxplot_r2_overall.png'}")

    # NEW: Individual Norms Bar Plot
    # Sort for better visualization (group by category, then by name)
    df_sorted = df.sort_values(by=['category', 'norm_name'])
    
    height = max(10, len(df['norm_name'].unique()) * 0.5)
    plt.figure(figsize=(12, height))
    
    sns.barplot(data=df_sorted, x='r2_mean', y='norm_name', hue='embedding_source')
    
    plt.title("Prediction Performance (R^2) by Individual Norm")
    plt.xlabel("R^2 Score")
    plt.ylabel("Norm")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Model')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "barplot_individual_norms.png")
    print(f"Saved plot to {OUTPUT_DIR / 'barplot_individual_norms.png'}")

    # NEW: Median Summary Plots (Robust to Outliers)
    # 1. Median by Category
    # Reset index to make it plot-friendly
    med_cat_df = cat_stats_med.reset_index().melt(id_vars='embedding_source', var_name='category', value_name='r2_median')
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=med_cat_df, x='r2_median', y='embedding_source', hue='category')
    plt.title("Median R^2 Score by Category (Robust to Outliers)")
    plt.xlabel("Median R^2")
    plt.ylabel("Model")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "barplot_median_by_category.png")
    print(f"Saved plot to {OUTPUT_DIR / 'barplot_median_by_category.png'}")

    # 2. Median Overall
    med_overall = df.groupby('embedding_source')['r2_mean'].median().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=med_overall, x='r2_mean', y='embedding_source')
    plt.title("Overall Median R^2 Score (Robust to Outliers)")
    plt.xlabel("Median R^2")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "barplot_median_overall.png")
    print(f"Saved plot to {OUTPUT_DIR / 'barplot_median_overall.png'}")

    # 3. Median per Norm (Task Difficulty)
    norm_diff = df.groupby('norm_name')['r2_mean'].median().reset_index().sort_values('r2_mean', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=norm_diff, x='r2_mean', y='norm_name')
    plt.title("Task Difficulty: Median R^2 per Norm (Across All Models)")
    plt.xlabel("Median R^2")
    plt.ylabel("Norm")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "barplot_median_per_norm_difficulty.png")
    print(f"Saved plot to {OUTPUT_DIR / 'barplot_median_per_norm_difficulty.png'}")
    
    print("\n--- Median R^2 per Norm (Task Difficulty) ---")
    print(norm_diff)
    norm_diff.to_csv(OUTPUT_DIR / "stats_median_per_norm.csv")


if __name__ == "__main__":
    main()

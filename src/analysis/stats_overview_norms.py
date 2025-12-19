
import pandas as pd
import glob
import os

def analyze_norms_csv(filepath):
    """
    Analyzes a single model norms CSV file.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return f"Error reading {os.path.basename(filepath)}: {e}", None

    model_name = os.path.basename(filepath)
    
    # 1. Unique norms
    unique_norms = df['norm'].unique()
    num_unique_norms = len(unique_norms)
    
    # 2. Unique cues (words)
    unique_cues = df['word'].unique()
    num_unique_cues = len(unique_cues)
    
    # 3. Answers per cue
    # We expect 1 answer per (norm, word) pair ideally, or maybe we just want to know avg answers per cue regardless of norm? 
    # Usually "answers per cue" in this context implies if there are multiple samples or if it's just one.
    # Let's count rows per (norm, word) group.
    
    # Check for duplicates based on primary key columns
    duplicate_rows = df[df.duplicated(subset=['model_key', 'norm', 'word'], keep=False)]
    num_duplicates = len(duplicate_rows)
    
    # Filter out NaNs and ensure strings for sorting
    valid_norms = [str(x) for x in unique_norms if pd.notna(x)]
    stats = {
        "model_file": model_name,
        "total_rows": len(df),
        "num_unique_norms": num_unique_norms,
        "num_unique_cues": num_unique_cues,
        "num_duplicates": num_duplicates,
        "unique_norms_list": sorted(valid_norms) if num_unique_norms < 50 else f"{num_unique_norms} norms",
        # "unique_cues_sample": list(unique_cues)[:5]
    }
    
    return stats, df

def main():
    base_dir = "outputs/raw_behavior/model_norms"
    csv_files = glob.glob(os.path.join(base_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {base_dir}")
        return

    all_models_stats = []
    
    print(f"{'Model File':<30} | {'Rows':<10} | {'Norms':<10} | {'Cues':<10} | {'Dups':<10}")
    print("-" * 85)

    global_norms = set()
    global_cues = set()

    for filepath in sorted(csv_files):
        stats, df = analyze_norms_csv(filepath)
        if df is None:
            print(stats)
            continue
            
        print(f"{stats['model_file']:<30} | {stats['total_rows']:<10} | {stats['num_unique_norms']:<10} | {stats['num_unique_cues']:<10} | {stats['num_duplicates']:<10}")
        
        global_norms.update(df['norm'].unique())
        global_cues.update(df['word'].unique())
        
        # Detailed check on counts if needed
        # counts = df.groupby(['norm', 'word']).size().reset_index(name='counts')
        # if counts['counts'].max() > 1:
        #     print(f"  WARNING: Max answers per (norm, word) is {counts['counts'].max()}")

    print("-" * 85)
    print(f"Total Unique Norms across all models: {len(global_norms)}")
    print(f"Total Unique Cues across all models: {len(global_cues)}")
    
    # Sanity check: Are all models roughly the same?
    # This is implicitly shown by the table.

if __name__ == "__main__":
    main()

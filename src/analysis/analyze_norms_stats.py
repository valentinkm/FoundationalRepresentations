
import pandas as pd
import glob
import os

def analyze_file(filepath):
    print(f"Analyzing {os.path.basename(filepath)}...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Unique norms
    unique_norms = df['norm'].nunique()
    print(f"  Unique Norms: {unique_norms}")

    # Unique cues
    unique_cues = df['word'].nunique()
    print(f"  Unique Cues: {unique_cues}")

    # Answers per cue (assuming grouping by norm and cue)
    # We want to know how many responses we have for each (norm, cue) pair
    responses_per_cue = df.groupby(['norm', 'word']).size()
    
    print(f"  Answers per cue (stats on count per norm-cue pair):")
    print(f"    Min: {responses_per_cue.min()}")
    print(f"    Max: {responses_per_cue.max()}")
    print(f"    Mean: {responses_per_cue.mean():.2f}")
    print(f"    Median: {responses_per_cue.median():.2f}")
    
    # Also check if it varies by norm
    # print("  Breakdown by norm (mean responses per cue):")
    # print(df.groupby('norm').apply(lambda x: x.groupby('word').size().mean()))

    # Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"  Duplicate Rows: {duplicates}")
    
    # Check for duplicate (norm, word) pairs if expected only 1
    # If this is a generative task defined to produce N responses, duplicates might be valid if they generated the same text?
    # But usually 'rows' implies data entries.
    
    print("-" * 30)

def main():
    files = [
        "/Users/kriegmair/Desktop/FoundationalRepresentations/outputs/raw_behavior/model_norms/gemma-3-27b-instruct.csv",
        "/Users/kriegmair/Desktop/FoundationalRepresentations/outputs/raw_behavior/model_norms/gpt-oss-20b-instruct.csv",
        "/Users/kriegmair/Desktop/FoundationalRepresentations/outputs/raw_behavior/model_norms/mistral-small-24b-instruct.csv",
        "/Users/kriegmair/Desktop/FoundationalRepresentations/outputs/raw_behavior/model_norms/qwen-3-32b-instruct.csv"
    ]

    for f in files:
        if os.path.exists(f):
            analyze_file(f)
        else:
            print(f"File not found: {f}")

if __name__ == "__main__":
    main()

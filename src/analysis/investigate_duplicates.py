
import pandas as pd
import numpy as np

file_path = 'outputs/raw_behavior/model_norms/mistral-small-24b-instruct.csv'
print(f"Analyzing {file_path}...")

try:
    df = pd.read_csv(file_path, low_memory=False)
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# Normalize string cols for robust matching
df['norm'] = df['norm'].astype(str).str.strip()
df['word'] = df['word'].astype(str).str.strip().str.lower()

# Force numeric ratings
df['cleaned_rating'] = pd.to_numeric(df['cleaned_rating'], errors='coerce')

total_rows = len(df)
key_cols = ['norm', 'word']

# Identify duplicates on keys (norm + word)
# keep='first' marks the occurrences after the first one as True (the "redundant" ones)
extra_rows_mask = df.duplicated(subset=key_cols, keep='first')
num_extra = extra_rows_mask.sum()

print(f"Total Rows: {total_rows}")
print(f"Redundant Rows (same norm/word): {num_extra}")

if num_extra == 0:
    print("No duplicates found.")
    exit()

# Get all rows that are part of a duplicate group
all_dupes_mask = df.duplicated(subset=key_cols, keep=False)
dupe_df = df[all_dupes_mask].copy()

n_groups = len(dupe_df.drop_duplicates(subset=key_cols))
print(f"Number of unique norm/word pairs with duplication: {n_groups}")

# Consistency Check
# Group by key_cols, count unique ratings
rating_counts = dupe_df.groupby(key_cols)['cleaned_rating'].nunique()

consistent_groups = (rating_counts == 1).sum()
inconsistent_groups = (rating_counts > 1).sum()

print(f"\nConsistency Breakdown:")
print(f"Groups with IDENTICAL ratings (Safe redundancy): {consistent_groups}")
print(f"Groups with CONFLICTING ratings (Instability): {inconsistent_groups}")

# Also check exact row duplication (all columns)
extra_exact_rows = df.duplicated(keep='first').sum()
print(f"\nExact Full-Row Duplicates (all metadata identical): {extra_exact_rows}")

if inconsistent_groups > 0:
    print("\n[!] Found inconsistent ratings. Examples:")
    inconsistent_indices = rating_counts[rating_counts > 1].index[:5]
    for idx_key in inconsistent_indices:
        print(f"\nKey: {idx_key}")
        group = dupe_df[(dupe_df['norm'] == idx_key[0]) & (dupe_df['word'] == idx_key[1])]
        print(group[['cleaned_rating', 'model_key']].to_string(index=False))

if inconsistent_groups == 0:
    print("\nCONCLUSION: All duplicates are consistent (exact rating matches). Safe to drop.")
else:
    print("\nCONCLUSION: Some duplicates have conflicting ratings.")

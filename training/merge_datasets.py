# ── PHASE 1: Dataset Merging ──────────────────────────────────
import pandas as pd
import kagglehub
import os

print("=" * 50)
print("SpamBuster v2 — Dataset Merger")
print("=" * 50)

# ── DATASET 1: SMS Spam Collection ───────────────────────────
print("\n[1/3] Loading SMS Spam Collection...")
df1 = pd.read_csv('training/spam_v1.csv', encoding='latin-1')
df1 = df1[['v1', 'v2']]
df1.columns = ['label', 'message']
df1['label'] = df1['label'].str.lower().str.strip()
print(f"      Loaded: {len(df1)} | Spam: {len(df1[df1.label=='spam'])} | Ham: {len(df1[df1.label=='ham'])}")

# ── DATASET 2: Spam or Not Spam (Emails) ─────────────────────
print("\n[2/3] Loading Spam or Not Spam Dataset...")
path2 = kagglehub.dataset_download(
    "ozlerhakan/spam-or-not-spam-dataset"
)
for f in os.listdir(path2):
    if f.endswith('.csv'):
        df2_raw = pd.read_csv(os.path.join(path2, f), encoding='latin-1')
        break

df2 = pd.DataFrame()
df2['message'] = df2_raw['email'].astype(str).str[:500]
df2['label'] = df2_raw['label'].apply(
    lambda x: 'spam' if int(x) == 1 else 'ham'
)
df2 = df2[df2['label'].isin(['spam', 'ham'])].dropna()
print(f"      Loaded: {len(df2)} | Spam: {len(df2[df2.label=='spam'])} | Ham: {len(df2[df2.label=='ham'])}")

# ── DATASET 3: Enron Emails (Ham, already cached) ────────────
print("\n[3/3] Loading Enron Email Dataset from cache...")

enron_path = '/Users/sherlynn/.cache/kagglehub/datasets/wcukierski/enron-email-dataset/versions/2/emails.csv'

# Only read first 20k rows — file is huge
df3_raw = pd.read_csv(enron_path, nrows=20000)
print(f"      Raw rows loaded: {len(df3_raw)}")

# Extract just the email body (everything after the blank line
# that separates headers from body)
def extract_body(raw_email):
    try:
        parts = str(raw_email).split('\n\n', 1)
        if len(parts) > 1:
            return parts[1][:500].strip()
        return parts[0][:500].strip()
    except:
        return ''

df3 = pd.DataFrame()
df3['message'] = df3_raw['message'].apply(extract_body)
df3['label'] = 'ham'  # Enron = real workplace emails = all ham

# Remove empty or very short messages
df3 = df3[df3['message'].str.len() > 20]

# Sample 2500 so we don't overwhelm the other datasets
df3 = df3.sample(n=min(2500, len(df3)), random_state=42)
print(f"      Sampled: {len(df3)} ham emails")

# ── MERGE ALL THREE ───────────────────────────────────────────
print("\n── Merging all 3 datasets...")
try:
    merged = pd.concat([df1, df2, df3], ignore_index=True)
    print(f"  Concat done: {len(merged)} rows")

    # CLEAN
    merged = merged.dropna(subset=['label', 'message'])
    merged['message'] = merged['message'].astype(str).str.strip()
    merged = merged[merged['message'].str.len() > 5]

    before = len(merged)
    merged = merged.drop_duplicates(subset=['message'])
    print(f"  Removed {before - len(merged)} duplicates")

    merged = merged[merged['label'].isin(['spam', 'ham'])]
    merged = merged.reset_index(drop=True)

    print(f"\nFinal Dataset:")
    print(f"  Total: {len(merged)}")
    print(f"  Spam:  {len(merged[merged.label=='spam'])}")
    print(f"  Ham:   {len(merged[merged.label=='ham'])}")

    # SAVE
    merged.to_csv('training/merged_dataset.csv', index=False)
    print(f"\nSaved to training/merged_dataset.csv")
    print("=" * 50)

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

from sklearn.model_selection import train_test_split

print("\n── Creating Train/Val/Test splits...")

# Step 1: Split off 20% for test
train_val, test = train_test_split(
    merged,
    test_size=0.10,
    random_state=42,
    stratify=merged['label']  # keeps spam/ham ratio balanced in each split
)

# Step 2: Split remaining into train and val
train, val = train_test_split(
    train_val,
    test_size=0.111,  # 0.111 of 90% ≈ 10% of total
    random_state=42,
    stratify=train_val['label']
)

# Save all three
train.to_csv('training/train.csv', index=False)
val.to_csv('training/val.csv', index=False)
test.to_csv('training/test.csv', index=False)

print(f"  Train: {len(train)} messages")
print(f"  Val:   {len(val)} messages")
print(f"  Test:  {len(test)} messages")
print(f"\n Splits saved to training/")
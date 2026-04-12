# ── Balanced Resplit ──────────────────────────────────────────
# Fixes class imbalance before splitting
# Undersamples ham to match spam count

import pandas as pd
from sklearn.model_selection import train_test_split

# Load full merged dataset
df = pd.read_csv('training/merged_dataset.csv')

print(f"Before balancing:")
print(f"  Total: {len(df)}")
print(f"  Spam:  {len(df[df.label=='spam'])}")
print(f"  Ham:   {len(df[df.label=='ham'])}")

# Separate spam and ham
spam_df = df[df.label == 'spam']
ham_df  = df[df.label == 'ham']

# Undersample ham to match spam count
# This means equal representation during training
ham_sampled = ham_df.sample(
    n=len(spam_df),
    random_state=42
)

# Combine balanced dataset
balanced = pd.concat([spam_df, ham_sampled], ignore_index=True)
balanced = balanced.sample(frac=1, random_state=42)  # shuffle
balanced = balanced.reset_index(drop=True)

print(f"\nAfter balancing:")
print(f"  Total: {len(balanced)}")
print(f"  Spam:  {len(balanced[balanced.label=='spam'])}")
print(f"  Ham:   {len(balanced[balanced.label=='ham'])}")

# Split 80/10/10
train_val, test = train_test_split(
    balanced,
    test_size=0.10,
    random_state=42,
    stratify=balanced['label']
)
train, val = train_test_split(
    train_val,
    test_size=0.111,
    random_state=42,
    stratify=train_val['label']
)

# Save splits
train.to_csv('training/train.csv', index=False)
val.to_csv('training/val.csv',     index=False)
test.to_csv('training/test.csv',   index=False)

print(f"\nSplits:")
print(f"  Train: {len(train)} | Spam: {len(train[train.label=='spam'])} | Ham: {len(train[train.label=='ham'])}")
print(f"  Val:   {len(val)}   | Spam: {len(val[val.label=='spam'])}   | Ham: {len(val[val.label=='ham'])}")
print(f"  Test:  {len(test)}  | Spam: {len(test[test.label=='spam'])}  | Ham: {len(test[test.label=='ham'])}")

print(f"\n✅ Balanced splits saved!")
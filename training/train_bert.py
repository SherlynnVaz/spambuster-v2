# ── PHASE 2: BERT Fine-Tuning ─────────────────────────────────
# This script:
# 1. Loads your train/val/test splits
# 2. Tokenizes text using BertTokenizer
# 3. Fine-tunes bert-base-uncased
# 4. Evaluates and saves the model

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score
import os
import time

print("=" * 55)
print("SpamBuster v2 — BERT Fine-Tuning")
print("=" * 55)

# ── DEVICE SETUP ─────────────────────────────────────────────
# Use MPS (Mac GPU) if available, otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n✅ Using device: {device}")

# ── CONFIG ───────────────────────────────────────────────────
# These are the settings that control how training works
MAX_LENGTH = 128      # max number of tokens per message
BATCH_SIZE = 16       # how many messages to process at once
EPOCHS = 3            # how many times to go through training data
LEARNING_RATE = 2e-5  # how fast the model updates its weights
MODEL_NAME = 'bert-base-uncased'  # pre-trained model to start from
SAVE_PATH = 'backend/model'       # where to save the final model
WEIGHT_FOR_SPAM = 9.0  

print(f"\n── Config:")
print(f"   Max token length: {MAX_LENGTH}")
print(f"   Batch size:       {BATCH_SIZE}")
print(f"   Epochs:           {EPOCHS}")
print(f"   Learning rate:    {LEARNING_RATE}")
print(f"   Model:            {MODEL_NAME}")

# ── LOAD DATA ────────────────────────────────────────────────
print("\n── Loading datasets...")
train_df = pd.read_csv('training/train.csv')
val_df   = pd.read_csv('training/val.csv')
test_df  = pd.read_csv('training/test.csv')

print(f"   Train: {len(train_df)} | Spam: {len(train_df[train_df.label=='spam'])} | Ham: {len(train_df[train_df.label=='ham'])}")
print(f"   Val:   {len(val_df)} | Spam: {len(val_df[val_df.label=='spam'])} | Ham: {len(val_df[val_df.label=='ham'])}")
print(f"   Test:  {len(test_df)} | Spam: {len(test_df[test_df.label=='spam'])} | Ham: {len(test_df[test_df.label=='ham'])}")

# ── LABEL ENCODING ───────────────────────────────────────────
# Convert spam/ham text labels to numbers
# ham = 0, spam = 1
label2id = {'ham': 0, 'spam': 1}

train_df['label_id'] = train_df['label'].map(label2id)
val_df['label_id']   = val_df['label'].map(label2id)
test_df['label_id']  = test_df['label'].map(label2id)

# ── TOKENIZER ────────────────────────────────────────────────
# BertTokenizer converts raw text into numbers BERT understands
# It splits words into subword pieces and adds special tokens:
# [CLS] at the start — used for classification
# [SEP] at the end — marks end of sequence
print(f"\n── Loading tokenizer: {MODEL_NAME}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
print("   Tokenizer loaded!")

# Show what tokenization looks like on a real example
sample = "WINNER!! Claim your FREE prize now!!!"
tokens = tokenizer.tokenize(sample)
print(f"\n   Example tokenization:")
print(f"   Input:  '{sample}'")
print(f"   Tokens: {tokens}")

# ── DATASET CLASS ────────────────────────────────────────────
# PyTorch needs data in a specific format
# This class handles converting our CSV rows into BERT inputs
class SpamDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # Returns total number of messages
        return len(self.data)

    def __getitem__(self, idx):
        # Returns one message tokenized and ready for BERT
        message = str(self.data.loc[idx, 'message'])
        label   = int(self.data.loc[idx, 'label_id'])

        # Tokenize — converts text to input_ids, attention_mask
        encoding = self.tokenizer(
            message,
            max_length=self.max_length,
            padding='max_length',   # pad short messages with zeros
            truncation=True,        # cut long messages at max_length
            return_tensors='pt'     # return PyTorch tensors
        )

        return {
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label':          torch.tensor(label, dtype=torch.long)
        }

# ── CREATE DATASETS & DATALOADERS ────────────────────────────
print("\n── Creating datasets and dataloaders...")
train_dataset = SpamDataset(train_df, tokenizer, MAX_LENGTH)
val_dataset   = SpamDataset(val_df,   tokenizer, MAX_LENGTH)
test_dataset  = SpamDataset(test_df,  tokenizer, MAX_LENGTH)

# DataLoader feeds data to the model in batches
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches:   {len(val_loader)}")
print(f"   Test batches:  {len(test_loader)}")

# ── LOAD BERT MODEL ──────────────────────────────────────────
# BertForSequenceClassification = BERT + a classification layer on top
# num_labels=2 means binary classification (spam or ham)
print(f"\n── Loading BERT model...")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)
model = model.to(device)  # move model to GPU
print("   BERT model loaded and moved to GPU!")

# ── OPTIMIZER & SCHEDULER ────────────────────────────────────
# AdamW is the best optimizer for transformer models
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Scheduler gradually reduces learning rate during training
# This helps the model converge more smoothly
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

# ── TRAINING LOOP ────────────────────────────────────────────
print(f"\n── Starting training for {EPOCHS} epochs...")
print("   (This will take 20-40 minutes on MPS)")
print("-" * 55)

best_val_f1 = 0

for epoch in range(EPOCHS):
    start_time = time.time()
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # ── TRAIN ──────────────────────────────────────────────
    model.train()
    total_train_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        # Move batch to GPU
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        # Zero gradients from previous step
        optimizer.zero_grad()

        # Forward pass — model makes predictions
        # Weighted loss — penalises missing spam more heavily
        weight = torch.tensor([1.0, 9.0]).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        loss = loss_fn(outputs.logits, labels)      
        total_train_loss += loss.item()

        # Backward pass — calculate gradients
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update weights
        optimizer.step()
        scheduler.step()

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"   Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)

    # ── VALIDATE ───────────────────────────────────────────
    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():  # no gradient calculation during validation
        for batch in val_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Get predicted class (0 or 1)
            preds = torch.argmax(outputs.logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    elapsed = time.time() - start_time

    print(f"   Train Loss: {avg_train_loss:.4f}")
    print(f"   Val F1:     {val_f1:.4f}")
    print(f"   Time:       {elapsed/60:.1f} mins")

    # Save best model based on validation F1
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        os.makedirs(SAVE_PATH, exist_ok=True)
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        print(f"   ✅ New best model saved! (F1: {val_f1:.4f})")

# ── FINAL EVALUATION ON TEST SET ─────────────────────────────
print("\n" + "=" * 55)
print("Final Evaluation on Test Set")
print("=" * 55)

# Load the best saved model
best_model = BertForSequenceClassification.from_pretrained(SAVE_PATH)
best_model = best_model.to(device)
best_model.eval()

test_preds = []
test_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        outputs = best_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds = torch.argmax(outputs.logits, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

print(classification_report(
    test_labels,
    test_preds,
    target_names=['Ham', 'Spam']
))

print(f"\n✅ Model saved to {SAVE_PATH}/")
print("=" * 55)
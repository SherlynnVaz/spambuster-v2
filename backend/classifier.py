# ── classifier.py ─────────────────────────────────────────────
# Loads the saved BERT model and handles predictions
# This is the brain of the backend

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ── CONFIG ────────────────────────────────────────────────────
MODEL_PATH = "backend/model"
MAX_LENGTH = 128

# Use MPS (Mac GPU) if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ── LOAD MODEL ────────────────────────────────────────────────
# These load once when the server starts
# Not on every request — that would be very slow
print(f"Loading BERT model from {MODEL_PATH}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model = model.to(device)
model.eval()  # puts model in evaluation mode (not training mode)
print(f"✅ Model loaded on {device}")

# ── PREDICT FUNCTION ──────────────────────────────────────────
def predict(text: str) -> dict:
    """
    Takes a raw text message and returns:
    - label: 'spam' or 'ham'
    - confidence: float between 0 and 1
    - spam_probability: float between 0 and 1
    - ham_probability: float between 0 and 1
    """
    # Tokenize the input text
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move to GPU
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Run through BERT — no gradient needed for inference
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    # Convert raw scores to probabilities using softmax
    # outputs.logits shape: [1, 2] — one row, two columns (ham, spam)
    probabilities = torch.softmax(outputs.logits, dim=1)
    ham_prob  = probabilities[0][0].item()
    spam_prob = probabilities[0][1].item()

    # Pick the higher probability as the label
    label = "spam" if spam_prob > ham_prob else "ham"
    confidence = spam_prob if label == "spam" else ham_prob

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "spam_probability": round(spam_prob, 4),
        "ham_probability": round(ham_prob, 4)
    }
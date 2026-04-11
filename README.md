# 🛡️ SpamBuster v2

SMS & Email Spam Classifier — Version 2.0

## Stack
- BERT (HuggingFace Transformers) — context-aware spam detection
- FastAPI — REST API backend
- React + Vite — modern web frontend
- SMS API (Twilio/Vonage) — real SMS integration

## Dataset
10,177 messages merged from 3 sources:
- [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) — 5,572 short SMS messages
- [Spam or Not Spam](https://www.kaggle.com/datasets/ozlerhakan/spam-or-not-spam-dataset) — 2,999 formal emails
- [Enron Emails](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) — 2,500 workplace emails

## Model Performance (BERT vs v1 Naive Bayes)
| Metric | v1 Naive Bayes | v2 BERT |
|--------|---------------|---------|
| Ham F1 | 99% | 100% |
| Spam F1 | 92% | 95% |
| Overall Accuracy | 98% | 99% |

## Phases
- [x] Phase 1 — Dataset merging & splitting (10,177 messages)
- [x] Phase 2 — BERT fine-tuning (99% accuracy)
- [ ] Phase 3 — FastAPI backend
- [ ] Phase 4 — SMS API integration
- [ ] Phase 5 — React + Vite frontend

## Model Files
The trained BERT model (428MB) is not stored in Git.
To regenerate:
```bash
python training/merge_datasets.py
python training/train_bert.py
```
Model saves automatically to `backend/model/`

## v1 Prototype
See [spambuster](https://github.com/SherlynnVaz/spambuster) for the v1 Naive Bayes + Streamlit prototype.

# ── main.py ───────────────────────────────────────────────────
# FastAPI application — all API endpoints live here

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime
from backend.database import get_db, save_classification, ClassificationRecord
from backend.classifier import predict

# ── APP SETUP ─────────────────────────────────────────────────
app = FastAPI(
    title="SpamBuster v2 API",
    description="BERT-powered spam classification API",
    version="2.0.0"
)

# ── CORS ──────────────────────────────────────────────────────
# CORS allows your React frontend (on port 5173)
# to talk to this API (on port 8000)
# Without this the browser blocks cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REQUEST/RESPONSE MODELS ───────────────────────────────────
# Pydantic models define the shape of data coming in and going out
class ClassifyRequest(BaseModel):
    message: str

class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    spam_probability: float
    ham_probability: float
    message: str

class SMSWebhookRequest(BaseModel):
    From: str   # sender phone number
    Body: str   # message text

# ── ENDPOINTS ─────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Simple health check — confirms API is running.
    Called by monitoring tools and the frontend on startup.
    """
    return {
        "status": "healthy",
        "model": "bert-base-uncased",
        "version": "2.0.0"
    }

@app.post("/classify", response_model=ClassifyResponse)
def classify_message(
    request: ClassifyRequest,
    db: Session = Depends(get_db)
):
    """
    Main classification endpoint.
    Accepts a message, returns spam/ham prediction.
    Called by the React frontend when user clicks Check Message.
    """
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    # Run BERT prediction
    result = predict(request.message)

    # Save to database
    save_classification(
        db=db,
        message=request.message,
        label=result["label"],
        confidence=result["confidence"],
        source="web"
    )

    return ClassifyResponse(
        label=result["label"],
        confidence=result["confidence"],
        spam_probability=result["spam_probability"],
        ham_probability=result["ham_probability"],
        message=request.message
    )

@app.post("/webhook/sms")
def sms_webhook(
    request: SMSWebhookRequest,
    db: Session = Depends(get_db)
):
    """
    SMS webhook endpoint.
    Called automatically by Twilio/Vonage when an SMS arrives.
    Classifies the message and saves it with the sender's number.
    """
    result = predict(request.Body)

    # Save to database with phone number
    save_classification(
        db=db,
        message=request.Body,
        label=result["label"],
        confidence=result["confidence"],
        source="sms",
        phone=request.From
    )

    label_text = "SPAM" if result["label"] == "spam" else "HAM"
    confidence_pct = round(result["confidence"] * 100, 1)

    return {
        "status": "classified",
        "from": request.From,
        "label": result["label"],
        "confidence": result["confidence"],
        "reply": f"SpamBuster: This message is {label_text} ({confidence_pct}% confidence)"
    }

@app.get("/history")
def get_history(
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    Returns the most recent classification history.
    Called by React dashboard to show recent messages.
    """
    records = db.query(ClassificationRecord)\
                .order_by(ClassificationRecord.created_at.desc())\
                .limit(limit)\
                .all()

    return [
        {
            "id": r.id,
            "message": r.message[:100],  # truncate long messages
            "label": r.label,
            "confidence": r.confidence,
            "source": r.source,
            "phone": r.phone,
            "created_at": r.created_at.isoformat()
        }
        for r in records
    ]

@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """
    Returns overall classification statistics.
    Called by React stats page.
    """
    total = db.query(ClassificationRecord).count()
    spam  = db.query(ClassificationRecord)\
            .filter(ClassificationRecord.label == "spam").count()
    ham   = db.query(ClassificationRecord)\
            .filter(ClassificationRecord.label == "ham").count()
    sms   = db.query(ClassificationRecord)\
            .filter(ClassificationRecord.source == "sms").count()
    web   = db.query(ClassificationRecord)\
            .filter(ClassificationRecord.source == "web").count()

    return {
        "total_classified": total,
        "spam_count": spam,
        "ham_count": ham,
        "sms_count": sms,
        "web_count": web,
        "spam_percentage": round(spam / total * 100, 1) if total > 0 else 0
    }
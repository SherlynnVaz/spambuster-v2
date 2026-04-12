# ── database.py ───────────────────────────────────────────────
# SQLite database for storing classification history
# Every message classified gets saved here

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# SQLite database file — created automatically on first run
DATABASE_URL = "sqlite:///./spambuster.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
    # check_same_thread=False needed for SQLite with FastAPI
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ── DATABASE TABLE ────────────────────────────────────────────
class ClassificationRecord(Base):
    """
    One row per message classified.
    Stores the message, result, confidence, source and timestamp.
    """
    __tablename__ = "classifications"

    id         = Column(Integer, primary_key=True, index=True)
    message    = Column(String)
    label      = Column(String)       # 'spam' or 'ham'
    confidence = Column(Float)        # 0.0 to 1.0
    source     = Column(String)       # 'web' or 'sms'
    phone      = Column(String, nullable=True)  # sender phone if SMS
    created_at = Column(DateTime, default=datetime.utcnow)

# Create table if it doesn't exist
Base.metadata.create_all(bind=engine)

# ── HELPER FUNCTIONS ──────────────────────────────────────────
def get_db():
    """Provides a database session — used by FastAPI endpoints"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_classification(db, message, label, confidence,
                        source="web", phone=None):
    """Saves one classification result to the database"""
    record = ClassificationRecord(
        message=message,
        label=label,
        confidence=confidence,
        source=source,
        phone=phone
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
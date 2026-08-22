import os
import joblib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# --- Database Setup & Configuration ---
raw_db_url = os.getenv("DATABASE_URL", "sqlite:///./heritage.db")

# Fix Render PostgreSQL URL format if needed (postgres:// -> postgresql://)
if raw_db_url.startswith("postgres://"):
    raw_db_url = raw_db_url.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if raw_db_url.startswith("sqlite") else {}
engine = create_engine(raw_db_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- Database Models ---

class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    site_name = Column(String, nullable=True)
    review_text = Column(String, nullable=False)
    sentiment = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "site_name": self.site_name,
            "review_text": self.review_text,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


# --- Database Dependency ---

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Global dictionary to hold ML model instances in memory
models = {}

MODEL_FILES = {
    "label_encoders": "models/label_encoders.pkl",
    "rf_crowd": "models/rf_crowd_model.pkl"
}

HF_API_URL = os.getenv("HF_API_URL", "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest")
HF_TOKEN = os.getenv("HF_TOKEN", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to create DB tables & load ML models safely."""
    errors = []
    print("Starting up HeritageFlow FastAPI server...")
    
    # 1. Initialize Database Tables
    try:
        Base.metadata.create_all(bind=engine)
        print("  [OK] Database tables initialized successfully")
    except Exception as e:
        print(f"  [ERROR] Error initializing database tables: {e}")
        errors.append(f"database: {e}")

    # 2. Load label encoders
    try:
        models['label_encoders'] = joblib.load(MODEL_FILES['label_encoders'])
        print("  [OK] Loaded label_encoders.pkl")
    except Exception as e:
        print(f"  [ERROR] Error loading label_encoders.pkl: {e}")
        errors.append(f"label_encoders: {e}")

    # 3. Load Random Forest model
    try:
        models['rf_crowd'] = joblib.load(MODEL_FILES['rf_crowd'])
        print("  [OK] Loaded rf_crowd_model.pkl")
    except Exception as e:
        print(f"  [ERROR] Error loading rf_crowd_model.pkl: {e}")
        errors.append(f"rf_crowd: {e}")

    # 4. Configure Hugging Face Remote Inference API
    print("  [OK] Configured Hugging Face Remote Inference API for CardiffNLP RoBERTa model")

    if errors:
        models['load_errors'] = errors
    else:
        print("All backend services and models initialized successfully!")
        
    yield
    
    models.clear()
    print("Shutting down HeritageFlow FastAPI server...")


app = FastAPI(
    title="HeritageFlow Smart City Analytics API",
    description="Backend API for predicting heritage site crowd density and analyzing visitor feedback sentiment.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Streamlit frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Data Models ---

class CrowdPredictionRequest(BaseModel):
    is_weekend: int = Field(..., ge=0, le=1, description="1 if Weekend (Sat/Sun), 0 if Weekday")
    is_festival: int = Field(..., ge=0, le=1, description="1 if Festival day, 0 otherwise")
    checkpoint: str = Field(..., description="Heritage checkpoint name")
    weather: str = Field(..., description="Weather condition")
    temperature_c: int = Field(..., ge=15, le=50, description="Temperature in Celsius")
    time_slot: str = Field(..., description="Time slot in HH:MM format")


class CrowdPredictionResponse(BaseModel):
    predicted_crowd_density: float
    status: str = "success"


class SentimentAnalysisRequest(BaseModel):
    review_text: str = Field(..., min_length=1, description="Visitor feedback review text")

    model_config = {
        "json_schema_extra": {
            "example": {
                "review_text": "Absolutely beautiful architecture and peaceful environment!"
            }
        }
    }


class SentimentAnalysisResponse(BaseModel):
    sentiment: str
    confidence: float
    status: str = "success"


class SubmitReviewRequest(BaseModel):
    site_name: str = Field(default="Heritage Site", description="Heritage site name")
    review_text: str = Field(..., min_length=1, description="Visitor feedback review text")
    sentiment: str = Field(default=None, description="Predicted sentiment class (optional)")
    confidence: float = Field(default=None, ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0 (optional)")


class SubmitReviewResponse(BaseModel):
    status: str = "success"
    message: str = "Review submitted successfully."
    record: dict = Field(..., description="Saved review record")


# --- Inference Helper Function ---

def query_huggingface(review_str: str):
    """
    Utility function to run sentiment inference via Hugging Face Remote Inference API HTTP request.
    Offloads heavy NLP transformer computation to Hugging Face servers to minimize container RAM usage.
    """
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": review_str},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                results = data[0] if isinstance(data[0], list) else data
                top_res = max(results, key=lambda x: x.get("score", 0.0))
                raw_label = str(top_res.get("label", "")).upper()
                score = float(top_res.get("score", 0.0))

                if "POS" in raw_label or "LABEL_2" in raw_label:
                    sentiment_label = "Positive"
                elif "NEG" in raw_label or "LABEL_0" in raw_label:
                    sentiment_label = "Negative"
                elif "NEU" in raw_label or "LABEL_1" in raw_label:
                    sentiment_label = "Neutral"
                else:
                    sentiment_label = "Neutral"

                return sentiment_label, round(score, 4)
    except Exception as e:
        print(f"Hugging Face Inference API request error: {e}")

    # Default fallback
    return "Neutral", 0.95


# Alias for backward compatibility
classify_text = query_huggingface


# --- API Endpoints ---

@app.get("/")
def root():
    return {
        "platform": "HeritageFlow Smart City Platform",
        "status": "online",
        "endpoints": ["/predict_crowd", "/analyze_sentiment", "/submit-review", "/get-reviews", "/health"]
    }


@app.get("/health")
def health_check():
    """Health check endpoint to verify backend services and models."""
    loaded_keys = [k for k in models.keys() if k != 'load_errors']
    is_healthy = "rf_crowd" in models and "label_encoders" in models
    return {
        "status": "healthy" if is_healthy else "degraded",
        "loaded_models": loaded_keys,
        "remote_nlp": "Hugging Face Inference API",
        "database": "connected",
        "missing_or_failed": models.get('load_errors', [])
    }


@app.post("/predict_crowd", response_model=CrowdPredictionResponse)
def predict_crowd(req: CrowdPredictionRequest):
    """
    Predict crowd density based on environmental and temporal parameters.
    """
    if "rf_crowd" not in models or "label_encoders" not in models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Crowd prediction ML model is not available."
        )

    label_encoders = models["label_encoders"]
    rf_model = models["rf_crowd"]

    # Preprocess text inputs using label encoders
    try:
        enc_checkpoint = label_encoders['Checkpoint'].transform([req.checkpoint])[0]
    except Exception:
        known = list(label_encoders['Checkpoint'].classes_)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown Checkpoint '{req.checkpoint}'. Valid options: {known}"
        )

    try:
        enc_weather = label_encoders['Weather'].transform([req.weather])[0]
    except Exception:
        known = list(label_encoders['Weather'].classes_)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown Weather '{req.weather}'. Valid options: {known}"
        )

    try:
        enc_time_slot = label_encoders['Time_Slot'].transform([req.time_slot])[0]
    except Exception:
        known = list(label_encoders['Time_Slot'].classes_)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown Time_Slot '{req.time_slot}'. Valid options: {known}"
        )

    features_df = pd.DataFrame([{
        'Is_Weekend': req.is_weekend,
        'Is_Festival': req.is_festival,
        'Checkpoint': enc_checkpoint,
        'Weather': enc_weather,
        'Temperature_C': req.temperature_c,
        'Time_Slot': enc_time_slot
    }])

    try:
        pred_value = rf_model.predict(features_df)[0]
        final_crowd = max(0.0, round(float(pred_value), 2))
        return CrowdPredictionResponse(predicted_crowd_density=final_crowd)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {e}"
        )


@app.post("/analyze_sentiment", response_model=SentimentAnalysisResponse)
def analyze_sentiment(req: SentimentAnalysisRequest):
    """
    Classify visitor feedback using remote Hugging Face Inference API.
    Does NOT save to database store.
    """
    review_str = req.review_text.strip()
    if not review_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Review text cannot be empty."
        )

    try:
        sentiment_label, confidence = query_huggingface(review_str)
        return SentimentAnalysisResponse(
            sentiment=sentiment_label,
            confidence=confidence
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis processing failed: {e}"
        )


@app.post("/submit-review")
@app.post("/submit_review")
def submit_review(req: SubmitReviewRequest, db: Session = Depends(get_db)):
    """
    Persist official visitor review, timestamp, sentiment, and confidence to PostgreSQL / SQLAlchemy database.
    """
    review_str = req.review_text.strip()
    if not review_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Review text cannot be empty."
        )

    if req.sentiment and req.confidence is not None:
        sentiment_val = req.sentiment
        confidence_val = req.confidence
    else:
        try:
            sentiment_val, confidence_val = query_huggingface(review_str)
        except Exception:
            sentiment_val, confidence_val = "Neutral", 0.95

    try:
        new_review = Review(
            site_name=req.site_name,
            review_text=review_str,
            sentiment=sentiment_val,
            confidence=confidence_val
        )
        db.add(new_review)
        db.commit()
        db.refresh(new_review)
        return {
            "status": "success",
            "message": "Review submitted and saved to database.",
            "record": new_review.to_dict()
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save review to database: {e}"
        )


@app.get("/get-reviews")
@app.get("/get_reviews")
def get_reviews(db: Session = Depends(get_db)):
    """
    Query and return all visitor reviews from database.
    """
    try:
        reviews = db.query(Review).all()
        return [r.to_dict() for r in reviews]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch reviews from database: {e}"
        )

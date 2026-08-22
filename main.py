import os
import csv
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from transformers import pipeline

# Global dictionary to hold model instances in memory
models = {}

MODEL_FILES = {
    "label_encoders": "models/label_encoders.pkl",
    "rf_crowd": "models/rf_crowd_model.pkl"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to load ML models at startup safely."""
    errors = []
    print("Starting up HeritageFlow FastAPI server...")
    
    # 1. Load label encoders
    try:
        models['label_encoders'] = joblib.load(MODEL_FILES['label_encoders'])
        print("  [OK] Loaded label_encoders.pkl")
    except Exception as e:
        print(f"  [ERROR] Error loading label_encoders.pkl: {e}")
        errors.append(f"label_encoders: {e}")

    # 2. Load Random Forest model
    try:
        models['rf_crowd'] = joblib.load(MODEL_FILES['rf_crowd'])
        print("  [OK] Loaded rf_crowd_model.pkl")
    except Exception as e:
        print(f"  [ERROR] Error loading rf_crowd_model.pkl: {e}")
        errors.append(f"rf_crowd: {e}")

    # 3. Load Hugging Face Pre-trained Sentiment Pipeline (CardiffNLP RoBERTa)
    try:
        print("  [LOADING] Hugging Face CardiffNLP RoBERTa Sentiment Model...")
        models['sentiment_pipeline'] = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        print("  [OK] Loaded Hugging Face CardiffNLP RoBERTa sentiment pipeline")
    except Exception as e:
        print(f"  [ERROR] Error loading sentiment_pipeline: {e}")
        errors.append(f"sentiment_pipeline: {e}")

    if errors:
        models['load_errors'] = errors
    else:
        print("All machine learning models loaded successfully!")
        
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
    review_text: str = Field(..., min_length=1, description="Visitor feedback review text")
    sentiment: str = Field(default=None, description="Predicted sentiment class (optional)")
    confidence: float = Field(default=None, ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0 (optional)")


class SubmitReviewResponse(BaseModel):
    status: str = "success"
    message: str = "Review submitted successfully."


# --- Inference Helper Function ---

def classify_text(review_str: str):
    """
    Utility function to run sentiment inference through CardiffNLP RoBERTa pipeline.
    Extracts label ("Positive", "Negative", "Neutral") and confidence float score.
    """
    if "sentiment_pipeline" not in models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment analysis pipeline is not available."
        )

    nlp = models["sentiment_pipeline"]
    results = nlp(review_str)
    
    if not results:
        return "Neutral", 0.95
        
    res = results[0]
    raw_label = str(res.get("label", "")).upper()
    score = float(res.get("score", 0.0))
    
    if "POS" in raw_label or "LABEL_2" in raw_label:
        sentiment_label = "Positive"
    elif "NEG" in raw_label or "LABEL_0" in raw_label:
        sentiment_label = "Negative"
    elif "NEU" in raw_label or "LABEL_1" in raw_label:
        sentiment_label = "Neutral"
    else:
        sentiment_label = "Neutral"
        
    return sentiment_label, round(score, 4)


# --- API Endpoints ---

@app.get("/")
def root():
    return {
        "platform": "HeritageFlow Smart City Platform",
        "status": "online",
        "endpoints": ["/predict_crowd", "/analyze_sentiment", "/submit_review", "/health"]
    }


@app.get("/health")
def health_check():
    """Health check endpoint to verify loaded models."""
    loaded_keys = [k for k in models.keys() if k != 'load_errors']
    is_healthy = "sentiment_pipeline" in models and "rf_crowd" in models and "label_encoders" in models
    return {
        "status": "healthy" if is_healthy else "degraded",
        "loaded_models": loaded_keys,
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

    # Feature ordering must match model training format:
    # ['Is_Weekend', 'Is_Festival', 'Checkpoint', 'Weather', 'Temperature_C', 'Time_Slot']
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
        # Crowd density cannot be negative
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
    Classify visitor feedback using DistilBERT model.
    Does NOT save to CSV data store.
    """
    review_str = req.review_text.strip()
    if not review_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Review text cannot be empty."
        )

    try:
        sentiment_label, confidence = classify_text(review_str)
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


@app.post("/submit_review", response_model=SubmitReviewResponse)
def submit_review(req: SubmitReviewRequest):
    """
    Persist official visitor review, timestamp, sentiment, and confidence to heritage_tourist_reviews.csv.
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
            sentiment_val, confidence_val = classify_text(review_str)
        except Exception:
            sentiment_val, confidence_val = "Neutral", 0.95

    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    conf_pct_str = f"{confidence_val * 100:.1f}%"
    csv_file = "data/heritage_tourist_reviews.csv"
    file_exists = os.path.exists(csv_file)

    try:
        with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(csv_file) == 0:
                writer.writerow(["Date & Time", "Visitor Review", "Sentiment", "Confidence"])
            writer.writerow([current_time_str, review_str, sentiment_val, conf_pct_str])
        return SubmitReviewResponse(status="success", message="Review submitted and saved to database.")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit review to database: {e}"
        )

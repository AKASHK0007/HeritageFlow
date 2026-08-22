# 🏛️ Smart City HeritageFlow Dashboard

An end-to-end Machine Learning & NLP intelligence platform designed for smart city management, real-time tourist crowd density prediction, and automated visitor feedback sentiment analysis at cultural heritage checkpoints.

---

## 🌟 Key Features

- **🔮 Real-Time Crowd Density Prediction**: Multi-variate Random Forest Regressor forecasting foot traffic across key heritage checkpoints based on temperature, time slot, weather conditions, weekend flags, and festival events.
- **💬 Transformer-Powered Sentiment NLP**: High-accuracy sentiment classification powered by Hugging Face `cardiffnlp/twitter-roberta-base-sentiment-latest` RoBERTa Transformer pipeline.
- **📊 Interactive Streamlit Dashboard**: Clean, responsive web UI with real-time KPI metrics, sentiment cards, public filtering tools, and admin data export capabilities.
- **⚡ High-Performance FastAPI Backend**: RESTful API service powering modular endpoints for crowd estimation (`/predict_crowd`), sentiment analysis (`/analyze_sentiment`), and review submissions (`/submit_review`).

---

## 🏗️ Project Architecture

```mermaid
graph TD
    User([Tourist / Heritage Admin]) --> UI[Streamlit Frontend Dashboard (app.py)]
    UI -->|HTTP Requests| API[FastAPI Backend Server (main.py)]
    API -->|Inference| RF[Random Forest Crowd Model (models/rf_crowd_model.pkl)]
    API -->|Inference| NLP[RoBERTa Sentiment NLP (Hugging Face / PyTorch)]
    API -->|Read / Write| CSV[Database Storage (data/heritage_tourist_reviews.csv)]
```

---

## 🤖 Machine Learning & NLP Pipeline

### 1. Crowd Density Estimation (Random Forest Regressor)
- **Model**: `RandomForestRegressor(n_estimators=100, random_state=42)`
- **Input Features**: `Is_Weekend`, `Is_Festival`, `Checkpoint`, `Weather`, `Temperature_C`, `Time_Slot`
- **Output**: Numerical Crowd Density forecast (number of estimated visitors)
- **Artifacts**: `models/rf_crowd_model.pkl`, `models/label_encoders.pkl`

### 2. Visitor Feedback Sentiment Analysis (RoBERTa Transformer)
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest` via Hugging Face `transformers.pipeline`
- **Output Labels**: `Positive`, `Neutral`, `Negative` with confidence percentage
- **Edge Case Optimization**: Resolves complex contextual feedback (e.g., *"very bad vibes at rajarani temple"*) with 90%+ confidence.

---

## 📁 Repository Structure

```text
Crowd analysis/
├── data/
│   ├── heritage_crowd_data.csv        # Synthetic crowd telemetry dataset (5,000 samples)
│   └── heritage_tourist_reviews.csv    # Live visitor reviews database
├── models/
│   ├── rf_crowd_model.pkl             # Trained Random Forest Regressor
│   ├── label_encoders.pkl             # Scikit-learn Label Encoders
│   ├── dl_sentiment_model.keras       # Legacy Keras Neural Network model
│   ├── dl_tokenizer.pkl               # Legacy Keras Tokenizer
│   └── sentiment_label_encoder.pkl    # Legacy Sentiment Label Encoder
├── notebooks/
│   └── legacy_model_augmentation.ipynb # Technical showcase for Keras data augmentation
├── scripts/
│   └── heritage_crowd_analysis.py     # Data generation, EDA, & training script
├── .gitignore                         # Git ignore configuration
├── app.py                             # Streamlit Interactive Dashboard
├── main.py                            # FastAPI Backend API Server
├── README.md                          # Project documentation
└── requirements.txt                   # Dependency manifest
```

---

## 🚀 Setup & Execution Instructions

### Prerequisites
- Python 3.9+ installed on your system.

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/[Your-Username]/heritage-flow-dashboard.git
cd heritage-flow-dashboard
pip install -r requirements.txt
```

### 2. Launch FastAPI Backend Service
```bash
uvicorn main:app --reload --port 8000
```
- API Documentation (Swagger UI): `http://127.0.0.1:8000/docs`
- Health Check Endpoint: `http://127.0.0.1:8000/`

### 3. Launch Streamlit Web Application
In a separate terminal window:
```bash
streamlit run app.py
```
- Local Web Interface: `http://localhost:8501`

---

## 👤 Author & Acknowledgments

- **Author**: `[Your Name]`
- **Title / Institution**: `[Your Title/Institution]`
- **Repository**: `[GitHub Repository URL]`

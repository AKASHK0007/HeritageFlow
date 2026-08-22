import streamlit as st
import requests
import json
import pandas as pd
import os
import random
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="HeritageFlow | Smart City Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Backend Base URL ---
API_URL = os.getenv("API_URL", "http://backend:8000")

# --- Sidebar Inputs for Crowd Prediction ---
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("Configure real-time simulation parameters below:")

checkpoints = [
    'Mukteswar Temple',
    'Parsurameswara Temple',
    'Bindu Sagar',
    'Lingaraj Temple',
    'Rajarani Temple'
]
weather_options = ['Clear', 'Cloudy', 'Rainy', 'Humid']
time_slots = ['06:00', '09:00', '12:00', '15:00', '18:00']

selected_checkpoint = st.sidebar.selectbox("📍 Select Heritage Checkpoint", checkpoints, index=3)
selected_weather = st.sidebar.selectbox("🌤️ Weather Condition", weather_options, index=0)
selected_time_slot = st.sidebar.selectbox("🕒 Time Slot", time_slots, index=1)
selected_temp = st.sidebar.slider("🌡️ Temperature (°C)", min_value=20, max_value=45, value=32, step=1)

is_weekend_val = st.sidebar.radio("🗓️ Weekend Day?", ["No (Weekday)", "Yes (Weekend)"], index=0)
is_weekend = 1 if "Yes" in is_weekend_val else 0

is_festival_val = st.sidebar.radio("🎉 Festival Day?", ["No", "Yes"], index=0)
is_festival = 1 if "Yes" in is_festival_val else 0

st.sidebar.divider()
st.sidebar.caption("⚡ **HeritageFlow Engine v1.0** | Powered by FastAPI & Streamlit")

# --- Main Dashboard Header ---
st.title("🏛️ HeritageFlow Smart City Platform")
st.caption("Real-time Crowd Analytics & Tourist Sentiment Intelligence Engine")
st.divider()

# Check API health
api_online = False
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=2)
    if health_resp.status_code == 200 and health_resp.json().get("status") == "healthy":
        api_online = True
except Exception:
    api_online = False

if not api_online:
    st.error(f"⚠️ **FastAPI Backend Offline**: Unable to connect to `{API_URL}`. Please ensure the backend server is running via `uvicorn app:app --reload`.")

# --- Row 1: Real-Time Crowd Prediction ---
st.header("📊 Live Crowd Density Prediction")

# Prepare payload for /predict_crowd
crowd_payload = {
    "is_weekend": is_weekend,
    "is_festival": is_festival,
    "checkpoint": selected_checkpoint,
    "weather": selected_weather,
    "temperature_c": selected_temp,
    "time_slot": selected_time_slot
}

predicted_density = None
error_msg = None

if api_online:
    try:
        res = requests.post(f"{API_URL}/predict_crowd", json=crowd_payload, timeout=5)
        if res.status_code == 200:
            predicted_density = res.json().get("predicted_crowd_density")
        else:
            error_msg = res.json().get("detail", "Error predicting crowd density.")
    except Exception as e:
        error_msg = str(e)

col_metric, col_details = st.columns([1.2, 1.8])

with col_metric:
    with st.container(border=True):
        st.caption("ESTIMATED FOOT TRAFFIC")
        if predicted_density is not None:
            formatted_count = f"{int(round(predicted_density)):,}"
            st.metric(label="Predicted Visitors", value=f"{formatted_count} visitors")
            
            # Determine congestion level callout & recommendation
            if predicted_density < 100:
                st.success("🟢 Low Congestion")
                st.markdown("💡 **Recommendation:** Optimal visiting window. Low waiting times expected.")
            elif predicted_density <= 250:
                st.warning("🟡 Moderate Density")
                st.markdown("💡 **Recommendation:** Normal visitor volume. Standard queue times at ticket counters.")
            else:
                st.error("🔴 High Congestion Warning")
                st.markdown("💡 **Recommendation:** Heavy crowd alert! City authorities recommend crowd control measures.")
        elif error_msg:
            st.error(f"Prediction Error: {error_msg}")
        else:
            st.info("Connecting to AI inference model...")

with col_details:
    with st.container(border=True):
        st.caption("ACTIVE SIMULATION CONTEXT")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Location:** `{selected_checkpoint}`")
            st.markdown(f"**Time Slot:** `{selected_time_slot}`")
        with c2:
            st.markdown(f"**Weather:** `{selected_weather}`")
            st.markdown(f"**Temperature:** `{selected_temp}°C`")
        with c3:
            st.markdown(f"**Weekend:** `{'Yes' if is_weekend else 'No'}`")
            st.markdown(f"**Festival:** `{'Yes' if is_festival else 'No'}`")
        
        st.divider()
        st.caption("Model: Trained Random Forest Regressor (`models/rf_crowd_model.pkl`) evaluating multi-variate environmental correlations.")

st.divider()

# --- Row 2: Live Visitor Feedback & Sentiment Analysis ---
st.header("💬 Live Visitor Feedback Sentiment Intelligence")

# --- Callback Functions for Visitor Feedback Actions ---
def handle_submit():
    target_text = st.session_state.get("visitor_review_input", "").strip()
    if not target_text:
        st.session_state["feedback_message"] = ("warning", "Please enter a review text before submitting.")
        return
    if not api_online:
        st.session_state["feedback_message"] = ("error", "Backend API is offline. Cannot submit review.")
        return

    try:
        s_resp = requests.post(
            f"{API_URL}/analyze_sentiment",
            json={"review_text": target_text},
            timeout=5
        )
        if s_resp.status_code == 200:
            res_data = s_resp.json()
            st.session_state["last_sentiment_result"] = res_data
            st.session_state["last_review_analyzed"] = target_text
            
            cur_sentiment = res_data.get("sentiment", "Neutral")
            cur_confidence = res_data.get("confidence", 0.95)
            
            sub_resp = requests.post(
                f"{API_URL}/submit_review",
                json={
                    "review_text": target_text,
                    "sentiment": cur_sentiment,
                    "confidence": cur_confidence
                },
                timeout=5
            )
            if sub_resp.status_code == 200:
                st.session_state["feedback_message"] = ("success", "✅ **Review officially submitted & saved to database!**")
                # Safely clear input text in callback phase BEFORE widget instantiation!
                st.session_state["visitor_review_input"] = ""
            else:
                st.session_state["feedback_message"] = ("error", sub_resp.json().get("detail", "Error submitting review."))
        else:
            st.session_state["feedback_message"] = ("error", s_resp.json().get("detail", "Error analyzing sentiment."))
    except Exception as e:
        st.session_state["feedback_message"] = ("error", f"Submission connection error: {e}")


col_input, col_result = st.columns([1.5, 1])

with col_input:
    with st.container(border=True):
        st.subheader("Submit Visitor Review")
        st.caption("Analyze real-time sentiment using deep neural network NLP classification (`dl_sentiment_model.keras`).")
        
        st.caption("Quick Sample Reviews:")
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        default_text = "Absolutely beautiful architecture at Lingaraj Temple! The stone carvings are breathtaking and history is amazing."
        if "visitor_review_input" not in st.session_state:
            st.session_state["visitor_review_input"] = default_text

        if btn_col1.button("✨ Positive Sample", width="stretch"):
            st.session_state["visitor_review_input"] = f"Incredible heritage site at {selected_checkpoint}! Very peaceful morning walk, loved the architecture."
        if btn_col2.button("😐 Neutral Sample", width="stretch"):
            st.session_state["visitor_review_input"] = f"Visited {selected_checkpoint} today. It was okay, standard historical site, nothing special."
        if btn_col3.button("⚠️ Negative Sample", width="stretch"):
            st.session_state["visitor_review_input"] = f"It was way too crowded at {selected_checkpoint}. Couldn't walk properly and weather was unbearable."

        with st.form(key="review_form", clear_on_submit=False):
            review_text = st.text_area(
                "Visitor Feedback / Review:",
                height=120,
                placeholder="Type a review about your experience at the heritage site...",
                key="visitor_review_input"
            )
            
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                submit_btn = st.form_submit_button("💾 Submit Review", type="primary", width="stretch", on_click=handle_submit)
            with f_col2:
                analyze_btn = st.form_submit_button("🔍 Analyze Sentiment", type="secondary", width="stretch")

with col_result:
    with st.container(border=True):
        st.subheader("Sentiment Analysis Output")
        
        # Handle Analyze Sentiment button (main script flow)
        if analyze_btn:
            target_text = st.session_state.get("visitor_review_input", "").strip()
            if not target_text:
                st.warning("Please enter a review text to analyze.")
            elif not api_online:
                st.error("Backend API is offline. Cannot analyze sentiment.")
            else:
                with st.spinner("Analyzing text with Deep Neural Network..."):
                    try:
                        s_resp = requests.post(
                            f"{API_URL}/analyze_sentiment",
                            json={"review_text": target_text},
                            timeout=5
                        )
                        if s_resp.status_code == 200:
                            res_data = s_resp.json()
                            st.session_state["last_sentiment_result"] = res_data
                            st.session_state["last_review_analyzed"] = target_text
                            st.session_state["feedback_message"] = None
                        else:
                            st.error(s_resp.json().get("detail", "Error analyzing sentiment."))
                    except Exception as e:
                        st.error(f"API Connection error: {e}")

        # Display callback notification message if present
        if st.session_state.get("feedback_message"):
            msg_type, msg_text = st.session_state["feedback_message"]
            if msg_type == "success":
                st.success(msg_text)
            elif msg_type == "warning":
                st.warning(msg_text)
            elif msg_type == "error":
                st.error(msg_text)
        
        # Render Sentiment Output Card
        if "last_sentiment_result" in st.session_state:
            result = st.session_state["last_sentiment_result"]
            sentiment = result.get("sentiment", "Unknown")
            confidence = result.get("confidence", 0.0)
            conf_pct = f"{confidence * 100:.1f}%"
            
            if sentiment.lower() == "positive":
                st.success(f"🟢 **POSITIVE SENTIMENT**\n\nConfidence Score: **{conf_pct}**\n\nThe visitor expressed high satisfaction with site maintenance, history, or atmosphere.")
            elif sentiment.lower() == "neutral":
                st.info(f"🔵 **NEUTRAL SENTIMENT**\n\nConfidence Score: **{conf_pct}**\n\nThe review is moderate with balanced or indifferent feedback regarding the site.")
            else:  # Negative
                st.error(f"🔴 **NEGATIVE SENTIMENT**\n\nConfidence Score: **{conf_pct}**\n\nIssue flagged regarding crowds, cleanliness, heat, or facility management.")
        else:
            st.info("👈 Enter review text on the left and click **Analyze Sentiment** (or press **Submit Review** to save to database).")

# --- Helper for Sentiment Color Styling ---
def color_sentiment(val):
    v = str(val).lower()
    if "positive" in v:
        return "color: #4ade80; font-weight: bold;"
    elif "negative" in v:
        return "color: #fca5a5; font-weight: bold;"
    elif "neutral" in v:
        return "color: #93c5fd; font-weight: bold;"
    return ""

csv_file_path = "data/heritage_tourist_reviews.csv"

# --- Row 3: Recent Visitor Feedback Log & Metrics ---
st.divider()

st.subheader("Recent Visitor Feedback Log")
st.caption("Live feed showing tourist review submissions, sentiment metrics, and filtering tools.")

if os.path.exists(csv_file_path):
    try:
        full_reviews_df = pd.read_csv(csv_file_path).dropna(how="all")
        
        if not full_reviews_df.empty:
            # Explicitly parse 'Date & Time' to datetime objects and sort descending
            if "Date & Time" in full_reviews_df.columns:
                full_reviews_df["_dt_parsed"] = pd.to_datetime(full_reviews_df["Date & Time"], errors="coerce")
                sorted_df = full_reviews_df.sort_values(by="_dt_parsed", ascending=False).drop(columns=["_dt_parsed"])
            else:
                sorted_df = full_reviews_df

            # Calculate KPI Metrics
            total_count = len(sorted_df)
            sentiments_lower = sorted_df["Sentiment"].astype(str).str.lower() if "Sentiment" in sorted_df.columns else pd.Series(dtype=str)
            pos_count = int((sentiments_lower == "positive").sum())
            neu_count = int((sentiments_lower == "neutral").sum())
            neg_count = int((sentiments_lower == "negative").sum())

            # 1. Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Reviews", f"{total_count:,}")
            with m2:
                pos_pct = f" ({pos_count / total_count * 100:.0f}%)" if total_count > 0 else ""
                st.metric("Positive Reviews", f"{pos_count:,}{pos_pct}")
            with m3:
                neu_pct = f" ({neu_count / total_count * 100:.0f}%)" if total_count > 0 else ""
                st.metric("Neutral Reviews", f"{neu_count:,}{neu_pct}")
            with m4:
                neg_pct = f" ({neg_count / total_count * 100:.0f}%)" if total_count > 0 else ""
                st.metric("Negative Reviews", f"{neg_count:,}{neg_pct}")

            st.write("")

            # 2. Interactive Public Filters
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                selected_sentiment = st.selectbox(
                    "Filter by Sentiment",
                    options=["All", "Positive", "Neutral", "Negative"],
                    index=0,
                    key="public_sentiment_filter"
                )
            with f_col2:
                selected_location = st.selectbox(
                    "Filter by Location",
                    options=["All"] + checkpoints,
                    index=0,
                    key="public_location_filter"
                )

            # 3. Dynamic Dataframe Filtering
            filtered_df = sorted_df.copy()
            if selected_sentiment != "All" and "Sentiment" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["Sentiment"].astype(str).str.lower() == selected_sentiment.lower()]

            if selected_location != "All":
                if "Checkpoint" in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df["Checkpoint"].astype(str).str.lower() == selected_location.lower()]
                elif "Visitor Review" in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df["Visitor Review"].astype(str).str.contains(selected_location, case=False, na=False)]

            # 4. Display Filtered Results Table
            if not filtered_df.empty:
                styled_recent = filtered_df.style.map(
                    color_sentiment, 
                    subset=["Sentiment"] if "Sentiment" in filtered_df.columns else []
                )
                st.dataframe(styled_recent, use_container_width=True, hide_index=True)
            else:
                st.info("No reviews found matching the selected sentiment filter.")
        else:
            st.info("No tourist reviews recorded yet.")
    except Exception as ex:
        st.error(f"Error loading recent feedback log: {ex}")
else:
    st.warning(f"File `{csv_file_path}` not found.")


# --- Row 4: Admin Panel - Full Database Access ---
st.divider()

with st.expander("🛠️ Admin Panel - Full Database Access"):
    if os.path.exists(csv_file_path):
        try:
            admin_df = pd.read_csv(csv_file_path).dropna(how="all")
            
            if not admin_df.empty:
                if "Date & Time" in admin_df.columns:
                    admin_df["_dt_parsed"] = pd.to_datetime(admin_df["Date & Time"], errors="coerce")
                    admin_df = admin_df.sort_values(by="_dt_parsed", ascending=False).drop(columns=["_dt_parsed"])

                c_metric, c_download = st.columns([2, 1])
                
                with c_metric:
                    st.caption("TOTAL DATABASE RECORDS")
                    st.markdown(f"**{len(admin_df):,}** recorded review entries")

                with c_download:
                    st.caption("EXPORT DATABASE")
                    csv_bytes = admin_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_bytes,
                        file_name="heritage_tourist_reviews.csv",
                        mime="text/csv",
                        width="stretch"
                    )
                
                st.divider()
                
                styled_admin = admin_df.style.map(
                    color_sentiment, 
                    subset=["Sentiment"] if "Sentiment" in admin_df.columns else []
                )
                st.dataframe(styled_admin, use_container_width=True, hide_index=True)
            else:
                st.info("No database entries to display.")
            
        except Exception as ex:
            st.error(f"Error rendering Admin Panel: {ex}")
    else:
        st.warning(f"Database file `{csv_file_path}` not available.")

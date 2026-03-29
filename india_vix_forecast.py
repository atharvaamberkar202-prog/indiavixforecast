import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# -----------------------------
# PAGE CONFIG (Bloomberg style)
# -----------------------------
st.set_page_config(
    page_title="India VIX Dashboard",
    layout="wide"
)

st.title("📊 India VIX Kalman Dashboard")

# -----------------------------
# USER CONTROLS
# -----------------------------
col1, col2 = st.columns([1,1])

with col1:
    include_today = st.checkbox("Include Current Trading Day", value=True)

with col2:
    recalc = st.button("🔄 Recalculate")

# -----------------------------
# DATA LOADING FUNCTION
# -----------------------------
@st.cache_data
def load_data(include_today_flag):

    end_date = datetime.today()
    start_date = end_date - timedelta(days=3*365)

    data = yf.download("^INDIAVIX", start=start_date, end=end_date, progress=False)

    vix = data['Close'].squeeze().astype(float).dropna()

    if not include_today_flag:
        vix = vix.iloc[:-1]

    return vix

# -----------------------------
# MODEL FUNCTION
# -----------------------------
def run_kalman(vix):

    phi = 0.99
    mu = float(vix.mean())

    transition_cov = 0.2
    obs_cov = 1.0

    state_mean = float(vix.iloc[0])
    state_cov = 1.0

    filtered_states = [state_mean]

    for t in range(1, len(vix)):

        pred_state_mean = mu + phi * (state_mean - mu)
        pred_state_cov = phi**2 * state_cov + transition_cov

        obs = float(vix.iloc[t])
        K = pred_state_cov / (pred_state_cov + obs_cov)

        state_mean = pred_state_mean + K * (obs - pred_state_mean)
        state_cov = (1 - K) * pred_state_cov

        filtered_states.append(state_mean)

    filtered_series = pd.Series(filtered_states, index=vix.index)

    # Forecast next 3 days
    n_forecast = 3
    forecast = []

    current_state_mean = state_mean
    current_state_cov = state_cov

    for _ in range(n_forecast):
        next_state_mean = mu + phi * (current_state_mean - mu)
        next_state_cov = phi**2 * current_state_cov + transition_cov

        forecast.append(next_state_mean)

        current_state_mean = next_state_mean
        current_state_cov = next_state_cov

    forecast_dates = pd.date_range(
        start=vix.index[-1],
        periods=n_forecast + 1,
        freq='B'
    )[1:]

    forecast_series = pd.Series(forecast, index=forecast_dates)

    return filtered_series, forecast_series, mu

# -----------------------------
# RUN PIPELINE
# -----------------------------
vix = load_data(include_today)
filtered_series, forecast_series, mu = run_kalman(vix)

# -----------------------------
# PLOT (BLOOMBERG STYLE)
# -----------------------------
fig, ax = plt.subplots(figsize=(12,6))

ax.plot(vix.tail(100), label="VIX", linewidth=1.5)
ax.plot(filtered_series.tail(100), label="Kalman", linewidth=2)

ax.plot(forecast_series, linestyle="--", marker='o', label="Forecast (3D)")

ax.set_title("India VIX - Kalman Forecast", fontsize=14)
ax.legend()

st.pyplot(fig)

# -----------------------------
# STATS PANEL
# -----------------------------
st.subheader("📌 Model Stats")

col1, col2, col3 = st.columns(3)

col1.metric("Current VIX", f"{vix.iloc[-1]:.2f}")
col2.metric("Filtered VIX", f"{filtered_series.iloc[-1]:.2f}")
col3.metric("Long-term Mean (μ)", f"{mu:.2f}")

# -----------------------------
# FORECAST TABLE
# -----------------------------
st.subheader("📅 3-Day Forecast")
st.dataframe(forecast_series.to_frame("Forecast"))

# -----------------------------
# DATA PREVIEW (FIRST 3 ... LAST 3)
# -----------------------------
st.subheader("📄 Data Snapshot")

preview = pd.concat([vix.head(3), vix.tail(3)])
st.dataframe(preview.to_frame("VIX"))

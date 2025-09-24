import streamlit as st
import numpy as np
import pandas as pd
from financial_models import MODELS
from ml_models import ML_MODELS, predict_next_day_close

st.set_page_config(page_title="Financial Models", layout="wide")
st.title("Financial Modeling Dashboard")

# ---------------- Financial Models Section ----------------
# Sidebar - choose financial model
model_choice = st.sidebar.selectbox("Select a Financial Model", list(MODELS.keys()))

# Common inputs
st.sidebar.header("Parameters")
S0 = st.sidebar.number_input("Initial Stock Price (S0)", value=100.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0)
T = st.sidebar.number_input("Time to Maturity (Years, T)", value=1.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)

# Extra params depending on model
extra_params = {}
if model_choice == "Monte Carlo":
    extra_params["mu"] = st.sidebar.number_input("Expected Return (μ)", value=0.07, step=0.01)
    extra_params["n_simulations"] = st.sidebar.slider("Number of Simulations", 100, 5000, 1000)
    extra_params["n_steps"] = st.sidebar.slider("Steps per Year", 50, 500, 252)
elif model_choice == "Binomial Tree":
    extra_params["N"] = st.sidebar.slider("Steps (N)", 10, 500, 100)

# Run financial model
st.subheader(f"Results: {model_choice}")
if st.button("Run Financial Model"):
    model = MODELS[model_choice]

    if model_choice == "Black-Scholes":
        call_price = model(S0, K, T, r, sigma, option="call")
        put_price = model(S0, K, T, r, sigma, option="put")
        st.write(f"**Call Option Price:** {call_price:.2f}")
        st.write(f"**Put Option Price:** {put_price:.2f}")

    elif model_choice == "Binomial Tree":
        call_price = model(S0, K, T, r, sigma, extra_params["N"], option="call")
        put_price = model(S0, K, T, r, sigma, extra_params["N"], option="put")
        st.write(f"**Call Option Price:** {call_price:.2f}")
        st.write(f"**Put Option Price:** {put_price:.2f}")

    elif model_choice == "Monte Carlo":
        prices = model(S0, extra_params["mu"], sigma, T, 
                       extra_params["n_simulations"], extra_params["n_steps"])
        final_prices = prices[-1]
        st.line_chart(prices[:, :50])  # plot 50 paths
        st.write(f"**Mean Final Price:** {np.mean(final_prices):.2f}")
        st.write(f"**Std Dev of Final Price:** {np.std(final_prices):.2f}")

# ---------------- ML Prediction Section ----------------
st.header("ML Model: Next Day Close Prediction")

# Select ticker
selected_ticker = st.selectbox("Select Ticker", list(ML_MODELS.keys()))

# Run prediction
if st.button(f"Predict Next Day Close"):
    model = ML_MODELS[selected_ticker]
    prediction = predict_next_day_close(model, selected_ticker)
    st.write(f"Predicted Next Day Close for {selected_ticker}: **{prediction:.2f}**")

    # Optional: plot last 50 historical closes
    df = pd.read_csv(f"data/processed/{selected_ticker}.csv")
    st.line_chart(df[["Close"]].tail(50))
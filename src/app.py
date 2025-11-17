import streamlit as st
import numpy as np
import pandas as pd
from financial_models import MODELS
from ml_models import ML_MODELS, predict_next_day_close
from pathlib import Path
import plotly.graph_objects as go

parent_root = Path.cwd().parent

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
    extra_params["N"] = st.sidebar.slider("Steps (N)", 2, 50, 10)  # smaller range for visualization

# Run financial model
st.subheader(f"Results: {model_choice}")
if st.button("Run Financial Model"):
    model = MODELS[model_choice]

    # ---- Black-Scholes ----
    if model_choice == "Black-Scholes":
        call_price = model(S0, K, T, r, sigma, option="call")
        put_price = model(S0, K, T, r, sigma, option="put")
        st.write(f"**Call Option Price:** {call_price:.2f}")
        st.write(f"**Put Option Price:** {put_price:.2f}")

        # Interactive Plot: Option price vs Stock price
        S_range = np.linspace(S0 * 0.5, S0 * 1.5, 100)
        call_vals = [model(s, K, T, r, sigma, option="call") for s in S_range]
        put_vals = [model(s, K, T, r, sigma, option="put") for s in S_range]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=S_range, y=call_vals, mode="lines", name="Call Option"))
        fig.add_trace(go.Scatter(x=S_range, y=put_vals, mode="lines", name="Put Option"))
        fig.update_layout(
            title="Black-Scholes Option Price vs Stock Price",
            xaxis_title="Stock Price (S)",
            yaxis_title="Option Price",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Binomial Tree ----
    elif model_choice == "Binomial Tree":
        N = extra_params["N"]
        call_price = model(S0, K, T, r, sigma, N, option="call")
        put_price = model(S0, K, T, r, sigma, N, option="put")
        st.write(f"**Call Option Price:** {call_price:.2f}")
        st.write(f"**Put Option Price:** {put_price:.2f}")

        # Interactive lattice visualization (only for small N)
        if N <= 10:
            u = np.exp(sigma * np.sqrt(T / N))
            d = 1 / u

            x_vals, y_vals, text_labels = [], [], []
            edge_x, edge_y = [], []

            for i in range(N + 1):  # steps
                for j in range(i + 1):  # nodes
                    price = S0 * (u ** j) * (d ** (i - j))
                    x_vals.append(i)
                    y_vals.append(j)
                    text_labels.append(f"{price:.2f}")

                    # edges
                    if i < N:
                        edge_x += [i, i + 1, None]
                        edge_y += [j, j, None]       # down move
                        edge_x += [i, i + 1, None]
                        edge_y += [j, j + 1, None]   # up move

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                     mode="lines",
                                     line=dict(color="gray"),
                                     hoverinfo="none"))
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals,
                                     mode="markers+text",
                                     text=text_labels,
                                     textposition="top center",
                                     marker=dict(size=12, color="skyblue"),
                                     name="Stock Prices"))
            fig.update_layout(
                title="Binomial Tree Stock Price Lattice",
                xaxis_title="Step",
                yaxis_title="Up Moves",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---- Monte Carlo ----
    elif model_choice == "Monte Carlo":
        prices = model(S0, extra_params["mu"], sigma, T,
                       extra_params["n_simulations"], extra_params["n_steps"])
        final_prices = prices[-1]

        # Interactive plot of sample paths
        fig = go.Figure()
        for i in range(min(50, prices.shape[1])):  # plot up to 50 paths
            fig.add_trace(go.Scatter(y=prices[:, i], mode="lines", line=dict(width=1), opacity=0.5))
        fig.update_layout(
            title="Monte Carlo Simulation Paths",
            xaxis_title="Time Step",
            yaxis_title="Stock Price",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

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

    # Load historical data
    df = pd.read_csv(f"{parent_root}/data/processed/{selected_ticker}.NS_prices_processed.csv")

    # Last 50 closes
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df["Date"].tail(50), y=df["Close"].tail(50),
                              mode="lines+markers", name="Close Price"))
    fig1.update_layout(title=f"{selected_ticker} - Last 50 Closes",
                       xaxis_title="Date", yaxis_title="Close Price",
                       template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

    # Full stock price trend
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["Date"], y=df["Close"],
                              mode="lines", name="Close Price"))
    fig2.update_layout(title=f"{selected_ticker} - Full Stock Price Trend",
                       xaxis_title="Date", yaxis_title="Close Price",
                       template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

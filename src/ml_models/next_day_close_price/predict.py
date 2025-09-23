import joblib
import os
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
FEATURES = ["Close","High","Low","Open","Volume","daily_return","vol_20","sentiment"]

def load_models():
    """Load all .pkl models from the models folder"""
    models = {}
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".pkl"):
            ticker = f.replace(".pkl", "")
            models[ticker] = joblib.load(os.path.join(MODEL_DIR, f))
    return models

def predict_next_day_close(model, ticker, data_dir="data/processed/"):
    """Predict next-day close price using the last row of features"""
    df = pd.read_csv(f"{data_dir}/{ticker}.csv")
    last_row = df[FEATURES].iloc[-1:]
    return model.predict(last_row)[0]

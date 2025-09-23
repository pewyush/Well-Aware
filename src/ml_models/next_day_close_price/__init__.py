from .predict import load_models, predict_next_day_close

# Load all .pkl models in the 'models' folder
ML_MODELS = load_models()  # dict: ticker -> loaded model

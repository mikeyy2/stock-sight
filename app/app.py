from flask import Flask, request, render_template
import yfinance as yf
import numpy as np
import pickle
from tensorflow.keras.models import load_model


app = Flask(__name__)

# load model & scalers once
model = load_model("models/stock_model.keras")
feature_scaler = pickle.load(open("models/feature_scaler.pkl", "rb"))
target_scaler  = pickle.load(open("models/target_scaler.pkl", "rb"))
N_STEPS        = 50

@app.route("/", methods=["GET","POST"])
def index():
    error = None
    high_pred = low_pred = None

    if request.method == "POST":
        ticker = request.form["ticker"].upper()
        df = yf.download(ticker, start="2015-01-01", progress=False)

        # Validate data
        if df.empty or len(df) < N_STEPS:
            error = "Incorrect entry, check Yahoo Finance for correct entries"
        else:
            # Compute indicators
            df["MA10"] = df["Close"].rolling(10).mean()
            df["MA50"] = df["Close"].rolling(50).mean()
            df.dropna(inplace=True)

            # Build the last window
            window = df[["Close","MA10","MA50"]].iloc[-N_STEPS:]
            scaled_window = feature_scaler.transform(window)
            X_live = scaled_window.reshape(1, N_STEPS, scaled_window.shape[1])

            # Predict and invert scaling
            scaled_pred = model.predict(X_live)[0]
            hi, lo = target_scaler.inverse_transform([scaled_pred])[0]
            high_pred, low_pred = max(hi, lo), min(hi, lo)

    return render_template("index.html",
                           error=error,
                           high=high_pred,
                           low=low_pred)


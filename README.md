# 📈 StockSight: Predicting Stock Price Movements Using Deep Learning

StockSight is a deep learning-powered stock forecasting tool that predicts the next day’s **high** and **low** prices based on historical financial data and technical indicators. Built with LSTM neural networks and deployed via a Flask web app, this project provides a real-time, interactive interface for forecasting individual stock movements.

---

## 🚀 Features

- Predict next-day **high** and **low** prices for any valid stock ticker
- Uses technical indicators (MA10, MA50) and a 50-day lookback window
- LSTM model trained with TensorFlow/Keras
- Real-time deployment via Flask web app
- Error handling for invalid tickers
- Scaled input/output with `MinMaxScaler`

---

## 📦 Tech Stack

- **Frontend**: HTML (Jinja templates via Flask)
- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow/Keras, scikit-learn
- **Data**: Yahoo Finance (`yfinance`, `pandas_datareader`)
- **Deployment**: Localhost or web server

---

## 🧠 Model Overview

- **Input**: 50-day window of scaled Close, MA10, and MA50 values
- **Output**: Scaled prediction of next-day high and low
- **Layers**: LSTM → Dropout → Dense (2 neurons)
- **Loss**: MSE with early stopping for optimal generalization

---

## 📂 Project Structure

<pre> ```bash stocksight/ ├── app/ # Flask app package │ ├── __init__.py # Initializes Flask app │ ├── routes.py # Route logic │ ├── utils.py # Data fetching, preprocessing helpers │ ├── model.py # Model loading and prediction │ ├── templates/ # HTML templates │ │ └── index.html # Main web UI │ └── static/ # (Optional) CSS/JS files │ ├── models/ # Model + scaler storage │ ├── stock_model.keras # Trained LSTM model │ ├── feature_scaler.pkl # Input scaler │ └── target_scaler.pkl # Output scaler │ ├── notebooks/ # Notebooks for EDA/training │ └── training_notebook.ipynb │ ├── requirements.txt # Python dependencies ├── run.py # Flask app entry point ├── README.md # GitHub readme └── .gitignore # Files/folders to ignore in Git ``` </pre>

## 🧪 How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/mikeyy2/stocksight.git
   cd stocksight
   
2. Create & activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate

3. Install Dependencies
   pip install -r requirements.txt

4. Run the app
   python app.py
   
5. Open server (May have to change port)
   http://127.0.0.1:5000/

## 📝 License
This project is for educational purposes (AI 201X, Iowa State University).

## ✍️ Author
Michael Proeschel


# 🥇 AuricAI: Gold Price Forecasting System

**AuricAI** is a powerful, visually appealing, and AI-driven web application that predicts gold prices using advanced deep learning models. It provides real-time analytics, interactive charts, and beautifully styled visual components — all within an intuitive Streamlit interface.

![App Preview](https://auricai-gold-forecast-wmxwvjnkmboumwwlhjec4n.streamlit.app/)

---

## 🚀 Features

- 📊 **Historical Trend Analysis**
- 🤖 **Deep Learning Forecasting** (LSTM, GRU, and DeepAR models)
- 🎯 **Interactive Price Predictions**
- 🌐 **Real-time Gold Price Monitoring**
- 📉 **Model Performance Metrics**
- 💅 **Custom Gold-Themed UI (CSS)**

---

## 🧠 Models Used

| Model   | RMSE   | MAE   | R²     |
|---------|--------|--------|--------|
| GRU     | 967.53 | 697.53 | 0.9949 |
| LSTM    | 1312.95| 924.53 | 0.9906 |
| DeepAR  | 3418.06| 3031.35| 0.9360 |

> Metrics based on test set evaluation. All values are in INR.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit + Plotly
- **Backend**: Python + TensorFlow
- **Styling**: Custom CSS (Gold-inspired palette)
- **Forecast Models**: LSTM, GRU, DeepAR
- **Data**: Yahoo Finance + Historical CSVs

---

## 📂 Project Structure

```
├── app.py                  # Streamlit main application
├── graph_utils.py          # Utility functions for Plotly visualization
├── style.css               # Custom gold-themed CSS
├── models/
│   ├── lstm_model.h5       # Trained LSTM model
│   ├── gru_model.h5        # Trained GRU model
│   ├── deepar_model.pkl    # Trained DeepAR model
│   └── scaler_X.pkl / scaler_y.pkl
├── results_summary.json    # Performance metrics for models
├── requirements.txt        # Python dependencies
```

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/arpanksasmal/auricai-gold-forecasting.git
cd auricai-gold-forecasting
```

2. **Create a virtual environment (optional)**

```bash
python -m venv auric-env
source auric-env/bin/activate  # or auric-env\Scripts\activate on Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the app**

```bash
streamlit run app.py
```

---

## 🌟 UI Preview

![UI Theme Preview](preview\image.png)

---

## 📌 Notes

- Ensure a stable internet connection for real-time price fetching.
- DeepAR model is simplified for fast prediction.
- Best viewed in a modern browser (Chrome / Edge / Firefox).

---

## 📬 Connect with Me

- 🔗 GitHub: [@arpanksasmal](https://github.com/arpanksasmal)
- 💼 LinkedIn: [Arpan Kumar Sasmal](https://www.linkedin.com/in/arpan-kumar-sasmal-2b4421240/)

---

## 🛡️ Disclaimer

> ⚠️ This is an educational tool. Forecasted values are not financial advice. Please consult professionals before making investment decisions.

---

## ❤️ Made with passion and deep learning by Arpan 😎

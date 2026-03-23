import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Bitcoin Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# CUSTOM CSS - DARK/NEON THEME
# -----------------------------
st.markdown(
    """
    <style>
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .css-1l02zno {padding-top: 0rem;}

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }

    h1, h2, h3 { color: #00ff99; }

    p, span, div { color: #ffffff !important; }

    div.stButton > button {
        background-color: #ff0066 !important;
        color: #ffffff !important;
        border-radius: 10px;
    }

    input {background-color:#1a1a2e !important; color:#00ff99 !important;}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# TITLE
# -----------------------------
st.title("💰 Bitcoin Price Prediction")
st.markdown("Predict future Bitcoin prices using a simple Linear Regression model")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('BTC-USD.csv')
    df['Close'] = df['Close'].ffill()
    return df

data = load_data()

with st.expander("📊 Show Dataset"):
    st.dataframe(data.tail(10))

# -----------------------------
# USER INPUTS
# -----------------------------
days = st.slider("Select number of days to predict", 1, 60, 30)
investment = st.number_input("Enter investment amount (₹)", min_value=100, value=1000, step=100)

# -----------------------------
# PREPARE DATA FOR MODEL
# -----------------------------
prices = data['Close'].dropna().to_numpy()[-365:]
window = 7
X, y = [], []

for i in range(len(prices) - window):
    X.append(prices[i:i+window])
    y.append(prices[i+window])

X = np.array(X)
y = np.array(y)
X = X.reshape(len(X), window)

# -----------------------------
# TRAIN MODEL
# -----------------------------
@st.cache_data
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model(X, y)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔮 Predict Future Prices"):

    # Generate predictions
    last_window = prices[-window:].copy()
    future_predictions = []

    for _ in range(days):
        pred = model.predict(last_window.reshape(1, -1))[0]
        future_predictions.append(pred)
        last_window = np.append(last_window[1:], pred)

    # -----------------------------
    # RECOMMENDATION
    # -----------------------------
    if future_predictions[-1] > prices[-1] * 1.005:
        recommendation = "BUY 🚀"
    elif future_predictions[-1] < prices[-1] * 0.995:
        recommendation = "SELL 📉"
    else:
        recommendation = "HOLD ⚖️"

    # -----------------------------
    # RISK METER (FIXED)
    # -----------------------------
    vol = np.std(future_predictions) / np.mean(future_predictions)

    

    if vol < 0.002:
        risk = "Low Risk 🟢"
    elif vol < 0.006:
        risk = "Medium Risk 🟡"
    else:
        risk = "High Risk 🔴"

    # -----------------------------
    # PREDICTED INVESTMENT VALUE
    # -----------------------------
    future_value = investment * (future_predictions[-1] / prices[-1])

    # -----------------------------
    # PLOT HISTORICAL + PREDICTED
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#111')

    ax.plot(range(len(prices)), prices, label="Historical Prices", color='#00ff99')
    ax.plot(range(len(prices), len(prices)+days), future_predictions,
            label="Predicted Prices", color='#ff0066', linestyle='--')

    ax.set_title(f"Bitcoin Price Prediction for Next {days} Days")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price (₹)")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

    # -----------------------------
    # SHOW METRICS
    # -----------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("💡 Recommendation", recommendation)
    col2.metric("⚠️ Risk", risk)
    col3.metric("💰 Predicted Value", f"₹{future_value:.2f}")
    

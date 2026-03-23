import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1. Load dataset (CHANGE PATH if needed)
data = pd.read_csv(r'C:\Users\N Divya\Desktop\Bitcoin_Project\BTC-USD.csv')

# 2. Fix column names (important)
print(data.columns)

# If column name is different, adjust here
data = data[['Close']].dropna()

# 3. Convert to numpy
prices = data['Close'].to_numpy()

# 4. Create dataset
window = 10
X = []
y = []

for i in range(len(prices) - window):
    X.append(prices[i:i+window])
    y.append(prices[i+window])

X = np.array(X)
y = np.array(y)

# 5. Ensure correct shape
X = X.reshape(len(X), window)

print("Shape:", X.shape)

# 6. Split
split = int(0.8 * len(X))
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

# 7. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Predict
y_pred = model.predict(X_test)

# 9. Accuracy
accuracy = r2_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# 10. Future prediction
predictions = []
last_window = prices[-window:].copy()

for i in range(30):
    pred = model.predict(last_window.reshape(1, -1))[0]
    predictions.append(pred)
    last_window = np.append(last_window[1:], pred)

# 11. Plot
plt.figure(figsize=(10,5))
plt.plot(predictions, label="Future Prediction")
plt.legend()
plt.title("Bitcoin Price Prediction using Dataset")
plt.show()
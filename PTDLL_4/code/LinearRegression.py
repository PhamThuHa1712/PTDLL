import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import timedelta

# Đọc dữ liệu
df = pd.read_csv('Data\\amazon_stock_2000_2025.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# SMA (Simple Moving Average)
df['SMA_5'] = df['Close'].rolling(window=5).mean()

# RSI (Relative Strength Index 14 ngày)
delta = df['Close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI_14'] = 100 - (100 / (1 + rs))

# Lag features
df['Close_lag1'] = df['Close'].shift(1)
df['Close_lag2'] = df['Close'].shift(2)

# Loại bỏ giá trị thiếu
df = df.dropna().reset_index(drop=True)

# Chọn feature và target
features = ['Open', 'High', 'Low', 'Volume', 'SP500', 'Oil', 'DollarIndex',
            'SMA_5', 'RSI_14', 'Close_lag1', 'Close_lag2']
target = 'Close'

X = df[features]
y = df[target]
dates = df['Date']

# Tách theo thời gian: 80% train, 20% test
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
dates_train, dates_test = dates[:split_index], dates[split_index:]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_train_pred = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("=== Kết quả trên tập Test ===")
print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'R²: {r2:.4f}')

# VẼ BIỂU ĐỒ SO SÁNH GIÁ THỰC TẾ VÀ GIÁ DỰ ĐOÁN
plt.figure(figsize=(14,6))
plt.plot(dates_test, y_test, label='Giá thực tế', color='blue')
plt.plot(dates_test, y_pred_test, label='Giá dự đoán', color='orange', linestyle='--')
plt.xlabel('Thời gian')
plt.ylabel('Giá đóng cửa (USD)')
plt.title('So sánh giá thực tế vs dự đoán ')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ===========================
# Dự đoán 10 ngày tiếp theo
# ===========================

future_dates = []
future_prices = []
last_row = df.iloc[-1].copy()
for i in range(10):
    new_row = {}
    # Sử dụng giá trị giả định hoặc giữ nguyên các yếu tố đầu vào không thay đổi nhiều
    new_row['Open'] = last_row['Close']
    new_row['High'] = last_row['Close'] * 1.01
    new_row['Low'] = last_row['Close'] * 0.99
    new_row['Volume'] = last_row['Volume']
    new_row['SP500'] = last_row['SP500']
    new_row['Oil'] = last_row['Oil']
    new_row['DollarIndex'] = last_row['DollarIndex']
    # SMA tính từ các giá trị gần nhất + các giá vừa dự đoán
    last_closes = df['Close'].tolist()[-4:] + future_prices[-1:] if future_prices else df['Close'].tolist()[-5:]
    new_row['SMA_5'] = np.mean(last_closes)
    new_row['RSI_14'] = last_row['RSI_14']
    new_row['Close_lag1'] = last_row['Close']
    new_row['Close_lag2'] = last_row['Close_lag1']
    # Chuẩn hóa và dự đoán
    new_df = pd.DataFrame([new_row])
    new_scaled = scaler.transform(new_df[features])
    pred_close = model.predict(new_scaled)[0]
    # Lưu kết quả
    next_date = last_row['Date'] + timedelta(days=1)
    future_dates.append(next_date)
    future_prices.append(pred_close)
    # Cập nhật last_row cho lần tiếp theo
    last_row['Date'] = next_date
    last_row['Close'] = pred_close
    last_row['Close_lag1'] = new_row['Close_lag1']
    last_row['Close_lag2'] = new_row['Close_lag2']

# VẼ BIỂU ĐỒ DỰ ĐOÁN 10 NGÀY TỚI
plt.figure(figsize=(12,5))
plt.plot(dates[-30:], y[-30:], label='Giá thực tế gần đây', color='blue')
plt.plot(future_dates, future_prices, label='Dự đoán 10 ngày tới', color='red')
plt.xlabel('Ngày')
plt.ylabel('Giá đóng cửa (USD)')
plt.title('Dự đoán giá cổ phiếu 10 ngày tiếp theo')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# IN KẾT QUẢ
print("\n=== Dự đoán giá cổ phiếu 10 ngày tiếp theo ===")
for date, price in zip(future_dates, future_prices):
    print(f"{date.strftime('%Y-%m-%d')}: {price:.2f} USD")
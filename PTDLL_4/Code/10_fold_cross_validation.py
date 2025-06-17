import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Load data
df = pd.read_csv('Data\\amazon_stock_2000_2025.csv')
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
# Loại bỏ ngoại lai bằng IQR trên tất cả cột số
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SP500', 'Oil', 'DollarIndex']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
df = df.reset_index(drop=True)
# Tính SMA và RSI
df['SMA_5'] = df['Close'].rolling(window=5).mean()
delta = df['Close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI_14'] = 100 - (100 / (1 + rs))

# Tạo lag features
df['Close_lag1'] = df['Close'].shift(1)
df['Close_lag2'] = df['Close'].shift(2)
df = df.dropna().reset_index(drop=True)

# Chọn feature và target
features = ['Open', 'High', 'Low', 'Volume', 'SP500', 'Oil', 'DollarIndex',
 'SMA_5', 'RSI_14', 'Close_lag1', 'Close_lag2']
target = 'Close'

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
data_filtered = df[features + [target]]
data_scaled = scaler.fit_transform(data_filtered)

# Tách input (X) và output (y)
X = data_scaled[:, :-1] # Tất cả cột trừ cột cuối (Close)
y = data_scaled[:, -1] # Cột Close

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=10)
results = []

def custom_mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    # Tách dữ liệu train và validation
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Huấn luyện mô hình
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Dự đoán và chuyển về giá gốc
    y_val_pred = model.predict(X_val)
    y_val_original = scaler.inverse_transform(np.hstack((np.zeros((y_val.shape[0], len(features))), y_val.reshape(-1, 1))))[:, -1]
    y_val_pred_original = scaler.inverse_transform(np.hstack((np.zeros((y_val_pred.shape[0], len(features))), y_val_pred.reshape(-1, 1))))[:, -1]

    # Đánh giá
    mse = mean_squared_error(y_val_original, y_val_pred_original)
    mae = mean_absolute_error(y_val_original, y_val_pred_original)
    r2 = r2_score(y_val_original, y_val_pred_original)
    mape = custom_mape(y_val_original, y_val_pred_original)

    results.append([fold, mse, r2, mae, mape])

# Thêm dòng Average
avg_mse = np.mean([r[1] for r in results])
avg_r2 = np.mean([r[2] for r in results])
avg_mae = np.mean([r[3] for r in results])
avg_mape = np.mean([r[4] for r in results])
results.append(['Average', avg_mse, avg_r2, avg_mae, avg_mape])

# Tạo DataFrame
df_results = pd.DataFrame(results, columns=['Fold', 'Mean Squared Error', 'R2 Score', 'Mean Absolute Error', 'Mean Absolute Percentage Error'])
print(df_results.to_string(index=False))


# # Biểu đồ
# import matplotlib.pyplot as plt

# # Lưu trữ giá trị thực tế và dự đoán từ tất cả các fold
# y_true_all = []
# y_pred_all = []
# dates_all = []

# for fold, (train_idx, val_idx) in enumerate(KFold(n_splits=10, shuffle=False).split(X), 1):
#     X_train, X_val = X[train_idx], X[val_idx]
#     y_train, y_val = y[train_idx], y[val_idx]
#     dates_val = df['Date'].iloc[val_idx]
    
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     y_val_pred = model.predict(X_val)
    
#     y_val_original = scaler.inverse_transform(
#         np.hstack((np.zeros((y_val.shape[0], len(features))), y_val.reshape(-1, 1)))
#     )[:, -1]
#     y_val_pred_original = scaler.inverse_transform(
#         np.hstack((np.zeros((y_val_pred.shape[0], len(features))), y_val_pred.reshape(-1, 1)))
#     )[:, -1]
    
#     y_true_all.extend(y_val_original)
#     y_pred_all.extend(y_val_pred_original)
#     dates_all.extend(dates_val)

# # Vẽ biểu đồ
# plt.figure(figsize=(14, 6))
# plt.plot(dates_all, y_true_all, label='Giá thực tế', color='blue')
# plt.plot(dates_all, y_pred_all, label='Giá dự đoán', color='orange', linestyle='--')
# plt.xlabel('Thời gian')
# plt.ylabel('Giá đóng cửa (USD)')
# plt.title('So sánh giá thực tế vs dự đoán (10-fold Cross-Validation)')
# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


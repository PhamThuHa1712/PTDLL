import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

# Đọc dữ liệu
try:
    df = pd.read_csv("Data\\amazon_stock_2000_2025.csv")
    df["Date"] = pd.to_datetime(df["Date"])  # Chuyển đổi Date sang kiểu datetime
    df = df.sort_values("Date")
except FileNotFoundError:
    print("Error: Data file 'amazon_stock_2000_2025.csv' not found.")
    exit(1)
except Exception as e:
    print(f"Error loading data: {str(e)}")
    exit(1)

# Tính SMA (Simple Moving Average)
df['SMA_5'] = df['Close'].rolling(window=5).mean()

# Tính RSI (Relative Strength Index 14 ngày)
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

# Loại bỏ giá trị thiếu
df = df.dropna().reset_index(drop=True)

# Chọn feature và target
features = ['Open', 'High', 'Low', 'Volume', 'SP500', 'Oil', 'DollarIndex',
            'SMA_5', 'RSI_14', 'Close_lag1', 'Close_lag2']
target = 'Close'

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
data_filtered = df[features + [target]]
data_scaled = scaler.fit_transform(data_filtered)

# Tách input (X) và output (y)
X = data_scaled[:, :-1]  # Tất cả cột trừ cột cuối (Close)
y = data_scaled[:, -1]   # Cột Close


# Hàm thực hiện 10-fold cross-validation và tạo bảng thống kê
def cross_validate_and_report(data_X, data_y, n_splits=10):
    """
    Thực hiện cross-validation k-fold và tạo báo cáo thống kê về hiệu suất mô hình Linear Regression.

    Parameters:
        data_X (numpy.ndarray): Dữ liệu đặc trưng đã được chuẩn hóa
        data_y (numpy.ndarray): Dữ liệu mục tiêu đã được chuẩn hóa
        n_splits (int, optional): Số lượng fold trong cross-validation. Mặc định là 10

    Returns:
        pandas.DataFrame: DataFrame chứa kết quả đánh giá cho mỗi fold, bao gồm:
            - Fold: Số thứ tự của fold
            - Mean Squared Error: Sai số bình phương trung bình
            - R2 Score: Hệ số xác định R2
            - Mean Absolute Error: Sai số tuyệt đối trung bình
            Cuối bảng có thêm một dòng "Average" chứa giá trị trung bình của các metrics
    """
    # Khởi tạo KFold
    kfold = KFold(n_splits=n_splits, shuffle=False)  # Không xáo trộn để giữ thứ tự thời gian

    # Lưu trữ kết quả của các fold
    fold_results = []

    # Duyệt qua từng fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(data_X), 1):
        # Tách dữ liệu train và validation
        X_train, X_val = data_X[train_idx], data_X[val_idx]
        y_train, y_val = data_y[train_idx], data_y[val_idx]

        # Khởi tạo và huấn luyện mô hình Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Dự đoán trên tập validation
        y_val_pred = model.predict(X_val)

        # Chuyển giá trị dự đoán và thực tế về dạng gốc
        y_val_original = scaler.inverse_transform(
            np.hstack((np.zeros((y_val.shape[0], len(features))), y_val.reshape(-1, 1)))
        )[:, -1]
        y_val_pred_original = scaler.inverse_transform(
            np.hstack((np.zeros((y_val_pred.shape[0], len(features))), y_val_pred.reshape(-1, 1)))
        )[:, -1]

        # Tính các chỉ số
        mse = mean_squared_error(y_val_original, y_val_pred_original)
        mae = mean_absolute_error(y_val_original, y_val_pred_original)
        r2 = r2_score(y_val_original, y_val_pred_original)

        # Lưu kết quả của fold
        fold_results.append({'Fold': fold, 'Mean Squared Error': mse, 'R2 Score': r2, 'Mean Absolute Error': mae})

    # Tạo DataFrame từ kết quả
    results_df = pd.DataFrame(fold_results)

    # Thêm dòng trung bình
    avg_row = pd.DataFrame({
        'Fold': ['Average'],
        'Mean Squared Error': [results_df['Mean Squared Error'].mean()],
        'R2 Score': [results_df['R2 Score'].mean()],
        'Mean Absolute Error': [results_df['Mean Absolute Error'].mean()]
    })

    # Nối dòng trung bình với DataFrame kết quả
    results_df = pd.concat([results_df, avg_row], ignore_index=True)

    return results_df

# Thực hiện cross-validation và in kết quả
results_df = cross_validate_and_report(X, y, n_splits=10)
print(results_df)

# Biểu đồ
import matplotlib.pyplot as plt

# Lưu trữ giá trị thực tế và dự đoán từ tất cả các fold
y_true_all = []
y_pred_all = []
dates_all = []

for fold, (train_idx, val_idx) in enumerate(KFold(n_splits=10, shuffle=False).split(X), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    dates_val = df['Date'].iloc[val_idx]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    
    y_val_original = scaler.inverse_transform(
        np.hstack((np.zeros((y_val.shape[0], len(features))), y_val.reshape(-1, 1)))
    )[:, -1]
    y_val_pred_original = scaler.inverse_transform(
        np.hstack((np.zeros((y_val_pred.shape[0], len(features))), y_val_pred.reshape(-1, 1)))
    )[:, -1]
    
    y_true_all.extend(y_val_original)
    y_pred_all.extend(y_val_pred_original)
    dates_all.extend(dates_val)

# Vẽ biểu đồ
plt.figure(figsize=(14, 6))
plt.plot(dates_all, y_true_all, label='Giá thực tế', color='blue')
plt.plot(dates_all, y_pred_all, label='Giá dự đoán', color='orange', linestyle='--')
plt.xlabel('Thời gian')
plt.ylabel('Giá đóng cửa (USD)')
plt.title('So sánh giá thực tế vs dự đoán (10-fold Cross-Validation)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
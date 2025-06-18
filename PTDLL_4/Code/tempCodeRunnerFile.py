import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta

# Hàm huấn luyện và lưu mô hình
def train_and_save_model():
    df = pd.read_csv('Data\\amazon_stock_2000_2025.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    df['Close_lag1'] = df['Close'].shift(1)
    df['Close_lag2'] = df['Close'].shift(2)
    df = df.dropna().reset_index(drop=True)

    features = ['Open', 'High', 'Low', 'Volume', 'SP500', 'Oil', 'DollarIndex', 'SMA_5', 'RSI_14', 'Close_lag1', 'Close_lag2']
    target = 'Close'

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)

    joblib.dump(model, 'Data\\amazon_stock_model.pkl')
    joblib.dump(scaler, 'Data\\scaler.pkl')
    return model, scaler, df

# Tải mô hình hoặc huấn luyện nếu chưa có
try:
    model = joblib.load('Data\\amazon_stock_model.pkl')
    scaler = joblib.load('Data\\scaler.pkl')
    df = pd.read_csv('Data\\amazon_stock_2000_2025.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
except FileNotFoundError:
    model, scaler, df = train_and_save_model()

# Hàm tính SMA_5, RSI_14, Close_lag1, Close_lag2 tự động
def calculate_indicators(last_close, current_open):
    last_row = df.iloc[-1].copy()
    
    last_5_closes = df['Close'].tail(4).tolist() + [last_close]
    sma_5 = np.mean(last_5_closes) if len(last_5_closes) == 5 else np.nan
    
    delta = current_open - last_close
    gain = delta if delta > 0 else 0
    loss = -delta if delta < 0 else 0
    avg_gain = (df['Close'].diff().where(lambda x: x > 0).rolling(window=14).mean().iloc[-1] * 13 + gain) / 14
    avg_loss = (df['Close'].diff().where(lambda x: x < 0).abs().rolling(window=14).mean().iloc[-1] * 13 + loss) / 14
    rsi_14 = 100 - (100 / (1 + (avg_gain / avg_loss if avg_loss != 0 else 0))) if not np.isnan(avg_gain) and not np.isnan(avg_loss) else np.nan
    
    close_lag1 = last_close
    close_lag2 = last_row['Close_lag1'] if 'Close_lag1' in last_row else df['Close'].iloc[-2] if len(df) > 2 else np.nan
    
    sma_5 = 0 if np.isnan(sma_5) else sma_5
    rsi_14 = 50 if np.isnan(rsi_14) else rsi_14
    close_lag2 = last_close if np.isnan(close_lag2) else close_lag2
    
    return sma_5, rsi_14, close_lag1, close_lag2

# Hàm xử lý placeholder
def on_entry_focus_in(entry, default_text):
    if entry.get() == default_text:
        entry.delete(0, tk.END)
        entry.config(foreground='black')

def on_entry_focus_out(entry, default_text):
    if not entry.get():
        entry.insert(0, default_text)
        entry.config(foreground='grey')

# Hàm dự đoán giá cho ngày hôm sau
def predict_next_day():
    try:
        inputs = {
            "Giá mở cửa (Open)": entry_open.get().strip(),
            "Giá cao nhất (High)": entry_high.get().strip(),
            "Giá thấp nhất (Low)": entry_low.get().strip(),
            "Khối lượng (Volume)": entry_volume.get().strip(),
            "S&P 500": entry_sp500.get().strip(),
            "Giá dầu (Oil)": entry_oil.get().strip(),
            "Chỉ số Đô la": entry_dollar_index.get().strip()
        }
        
        for label, value in inputs.items():
            if not value:
                messagebox.showerror("Lỗi", f"Ô {label} không được để trống!")
                return

        values = {}
        for label, value in inputs.items():
            try:
                value = value.replace(',', '.')
                values[label] = float(value)
            except ValueError:
                messagebox.showerror("Lỗi", f"Ô {label} chứa giá trị không hợp lệ: {value}")
                return

        open_price = values["Giá mở cửa (Open)"]
        high_price = values["Giá cao nhất (High)"]
        low_price = values["Giá thấp nhất (Low)"]
        volume = values["Khối lượng (Volume)"]
        sp500 = values["S&P 500"]
        oil = values["Giá dầu (Oil)"]
        dollar_index = values["Chỉ số Đô la"]

        last_close = df['Close'].iloc[-1]
        sma_5, rsi_14, close_lag1, close_lag2 = calculate_indicators(last_close, open_price)
        
        if np.isnan(sma_5) or np.isnan(rsi_14) or np.isnan(close_lag2):
            messagebox.showerror("Lỗi", "Dữ liệu tính toán chứa giá trị không hợp lệ. Vui lòng kiểm tra dữ liệu lịch sử.")
            return

        features = ['Open', 'High', 'Low', 'Volume', 'SP500', 'Oil', 'DollarIndex', 'SMA_5', 'RSI_14', 'Close_lag1', 'Close_lag2']
        input_data = pd.DataFrame([[open_price, high_price, low_price, volume, sp500, oil, dollar_index, sma_5, rsi_14, close_lag1, close_lag2]], 
                                 columns=features)
        
        if input_data.isnull().any().any():
            messagebox.showerror("Lỗi", "Dữ liệu đầu vào chứa giá trị NaN. Vui lòng kiểm tra các ô nhập liệu.")
            return

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        current_date = datetime.now().date()
        next_date = current_date + timedelta(days=1)
        result_label.config(text=f"Dự đoán giá đóng cửa ngày {next_date.strftime('%Y-%m-%d')}: {prediction:.2f} USD")
    
    except Exception as e:
        messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {str(e)}")

# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Dự đoán giá cổ phiếu Amazon")
root.geometry("600x450")
root.configure(bg='#f0f0f0')

title_label = tk.Label(root, text="Dự đoán giá cổ phiếu Amazon", font=("Arial", 16, "bold"), bg='#f0f0f0')
title_label.pack(pady=15)

frame = ttk.Frame(root, padding="10", borderwidth=2, relief="groove", style='Custom.TFrame')
frame.pack(pady=10, padx=10, fill="both", expand=True)

style = ttk.Style()
style.configure('Custom.TFrame', background='#ffffff')

labels = [
    ("Giá mở cửa (Open)", "100.0"), ("Giá cao nhất (High)", "105.0"), 
    ("Giá thấp nhất (Low)", "98.0"), ("Khối lượng (Volume)", "5000.0"), 
    ("S&P 500", "4000.0"), ("Giá dầu (Oil)", "70.0"), 
    ("Chỉ số Đô la", "95.0")
]

entries = {}
for i, (label_text, default_value) in enumerate(labels):
    ttk.Label(frame, text=label_text + ":", width=20, anchor="e", font=("Arial", 10)).grid(row=i, column=0, padx=5, pady=5, sticky="e")
    entry = ttk.Entry(frame, width=20, font=("Arial", 10))
    entry.insert(0, default_value)
    entry.config(foreground='grey')
    entry.bind("<FocusIn>", lambda event, e=entry, txt=default_value: on_entry_focus_in(e, txt))
    entry.bind("<FocusOut>", lambda event, e=entry, txt=default_value: on_entry_focus_out(e, txt))
    entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
    entries[label_text] = entry

entry_open = entries["Giá mở cửa (Open)"]
entry_high = entries["Giá cao nhất (High)"]
entry_low = entries["Giá thấp nhất (Low)"]
entry_volume = entries["Khối lượng (Volume)"]
entry_sp500 = entries["S&P 500"]
entry_oil = entries["Giá dầu (Oil)"]
entry_dollar_index = entries["Chỉ số Đô la"]

predict_button = ttk.Button(root, text="Dự đoán ngày hôm sau", command=predict_next_day, style='Custom.TButton')
predict_button.pack(pady=20)

style.configure('Custom.TButton', font=('Arial', 12, 'bold'), padding=6)

result_label = tk.Label(root, text="", font=("Arial", 12, "bold"), bg='#f0f0f0')
result_label.pack(pady=10)

root.mainloop()

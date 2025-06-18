import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import yfinance as yf

# Hàm huấn luyện và lưu mô hình cùng với DataFrame đã xử lý
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

    features = ['Open', 'High', 'Low', 'Volume', 'SP500', 'Oil', 
                'DollarIndex', 'SMA_5', 'RSI_14', 'Close_lag1', 'Close_lag2']
    target = 'Close'

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)

    joblib.dump(model, 'Data\\amazon_stock_model.pkl')
    joblib.dump(scaler, 'Data\\scaler.pkl')
    df.to_csv('Data\\amazon_stock_processed_2000_2025.csv', index=False)
    return model, scaler, df

# Tải mô hình, scaler và DataFrame đã xử lý
try:
    model = joblib.load('Data\\amazon_stock_model.pkl')
    scaler = joblib.load('Data\\scaler.pkl')
    df = pd.read_csv('Data\\amazon_stock_processed_2000_2025.csv')
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

# Hàm xử lý placeholder với định dạng đậm khi tải dữ liệu
def on_entry_focus_in(event, entry, default_text, is_auto_filled=False):
    if entry.get() == default_text and not is_auto_filled:
        entry.delete(0, tk.END)
        entry.config(foreground='black', font=("Arial", 10))
    elif entry.get() != default_text:
        entry.config(foreground='black', font=("Arial", 10))

def on_entry_focus_out(event, entry, default_text, is_auto_filled=False):
    if not entry.get() and not is_auto_filled:
        entry.insert(0, default_text)
        entry.config(foreground='grey', font=("Arial", 10))
    elif is_auto_filled and entry.get():
        entry.config(font=("Arial", 10, "bold"))


# Hàm lấy dữ liệu thời gian thực từ Yahoo Finance với kiểm tra thời gian
def fetch_today_data():
    current_time = datetime.now()
    start_date = date.today()
    
    # Lùi ngày để tìm dữ liệu gần nhất
    for i in range(7):  # Tối đa lùi 7 ngày để tránh vòng lặp vô hạn
        target_date = start_date - timedelta(days=i)
        
        if current_time.hour < 5 and i == 0:  # Nếu trước 5:00 AM, ưu tiên ngày hôm trước
            target_date -= timedelta(days=1)
        
        try:
            amzn = yf.Ticker("AMZN").history(start=target_date.strftime('%Y-%m-%d'), 
                                          end=(target_date + timedelta(days=1)).strftime('%Y-%m-%d'), 
                                          interval="1d")
            sp500 = yf.Ticker("^GSPC").history(start=target_date.strftime('%Y-%m-%d'), 
                                             end=(target_date + timedelta(days=1)).strftime('%Y-%m-%d'), 
                                             interval="1d")
            oil = yf.Ticker("CL=F").history(start=target_date.strftime('%Y-%m-%d'), 
                                          end=(target_date + timedelta(days=1)).strftime('%Y-%m-%d'), 
                                          interval="1d")
            dollar_index = yf.Ticker("DX-Y.NYB").history(start=target_date.strftime('%Y-%m-%d'), 
                                                       end=(target_date + timedelta(days=1)).strftime('%Y-%m-%d'), 
                                                       interval="1d")

            if not (amzn.empty or sp500.empty or oil.empty or dollar_index.empty):
                data = {
                    "Open": amzn["Open"].iloc[-1],
                    "High": amzn["High"].iloc[-1],
                    "Low": amzn["Low"].iloc[-1],
                    "Volume": amzn["Volume"].iloc[-1],
                    "SP500": sp500["Close"].iloc[-1],
                    "Oil": oil["Close"].iloc[-1],
                    "DollarIndex": dollar_index["Close"].iloc[-1],
                    "Date": target_date.strftime('%Y-%m-%d')
                }
                return data, None  # Trả về dữ liệu và không có lỗi
        
        except Exception as e:
            continue  # Nếu có lỗi (ví dụ: kết nối), thử ngày tiếp theo
    
    return None, "Không thể tìm thấy dữ liệu cho 7 ngày gần nhất. Vui lòng kiểm tra kết nối hoặc nhập thủ công."


# Hàm hiển thị dữ liệu hôm nay và cập nhật ô nhập liệu
def display_today_data():
    data, error = fetch_today_data()
    if error == "Người dùng chọn nhập thủ công.":
        today_data_label.config(text="Vui lòng nhập dữ liệu thủ công.")
        return
    elif error:
        messagebox.showerror("Lỗi", error)
        today_data_label.config(text="Không có dữ liệu để hiển thị.")
        return

    text = f"Dữ liệu ngày {data['Date']}:\n"
    text += f"Open: {data['Open']:.2f} USD\n"
    text += f"High: {data['High']:.2f} USD\n"
    text += f"Low: {data['Low']:.2f} USD\n"
    text += f"Volume: {data['Volume']:.0f} shares\n"
    text += f"S&P 500: {data['SP500']:.2f}\n"
    text += f"Oil: {data['Oil']:.2f} USD/barrel\n"
    text += f"Dollar Index: {data['DollarIndex']:.2f}"
    today_data_label.config(text=text)

    # Điền dữ liệu tự động với định dạng in đậm
    entry_open.delete(0, tk.END)
    entry_open.insert(0, f"{data['Open']:.2f}")
    entry_open.config(foreground='black', font=("Arial", 10, "bold"))
    entry_high.delete(0, tk.END)
    entry_high.insert(0, f"{data['High']:.2f}")
    entry_high.config(foreground='black', font=("Arial", 10, "bold"))
    entry_low.delete(0, tk.END)
    entry_low.insert(0, f"{data['Low']:.2f}")
    entry_low.config(foreground='black', font=("Arial", 10, "bold"))
    entry_volume.delete(0, tk.END)
    entry_volume.insert(0, f"{data['Volume']:.0f}")
    entry_volume.config(foreground='black', font=("Arial", 10, "bold"))
    entry_sp500.delete(0, tk.END)
    entry_sp500.insert(0, f"{data['SP500']:.2f}")
    entry_sp500.config(foreground='black', font=("Arial", 10, "bold"))
    entry_oil.delete(0, tk.END)
    entry_oil.insert(0, f"{data['Oil']:.2f}")
    entry_oil.config(foreground='black', font=("Arial", 10, "bold"))
    entry_dollar_index.delete(0, tk.END)
    entry_dollar_index.insert(0, f"{data['DollarIndex']:.2f}")
    entry_dollar_index.config(foreground='black', font=("Arial", 10, "bold"))


# Hàm dự đoán giá cho ngày hôm sau
def predict_next_day():
    try:
        # Lấy dữ liệu từ các ô nhập liệu
        inputs = {
            "Giá mở cửa (Open, USD):": entry_open.get().strip(),
            "Giá cao nhất (High, USD):": entry_high.get().strip(),
            "Giá thấp nhất (Low, USD):": entry_low.get().strip(),
            "Khối lượng (Volume, shares):": entry_volume.get().strip(),
            "S&P 500 (index):": entry_sp500.get().strip(),
            "Giá dầu (Oil, USD/barrel):": entry_oil.get().strip(),
            "Chỉ số Đô la (DollarIndex):": entry_dollar_index.get().strip()
        }
        
        # Kiểm tra các ô có rỗng hoặc vẫn là placeholder
        for label, value in inputs.items():
            if not value or value in ["100.0", "105.0", "98.0", "5000.0", "4000.0", "70.0", "95.0"]:
                messagebox.showerror("Lỗi", f"Ô {label} không được để trống hoặc vẫn là giá trị placeholder! Vui lòng nhập dữ liệu thực tế.")
                return

        values = {}
        for label, value in inputs.items():
            try:
                value = value.replace(',', '.')
                values[label] = float(value)
            except ValueError:
                messagebox.showerror("Lỗi", f"Ô {label} chứa giá trị không hợp lệ: {value}")
                return

        open_price = values["Giá mở cửa (Open, USD):"]
        high_price = values["Giá cao nhất (High, USD):"]
        low_price = values["Giá thấp nhất (Low, USD):"]
        volume = values["Khối lượng (Volume, shares):"]
        sp500 = values["S&P 500 (index):"]
        oil = values["Giá dầu (Oil, USD/barrel):"]
        dollar_index = values["Chỉ số Đô la (DollarIndex):"]

        last_close = df['Close'].iloc[-1]
        sma_5, rsi_14, close_lag1, close_lag2 = calculate_indicators(last_close, open_price)
        
        if np.isnan(sma_5) or np.isnan(rsi_14) or np.isnan(close_lag2):
            messagebox.showerror("Lỗi", "Dữ liệu tính toán chứa giá trị không hợp lệ. Vui lòng kiểm tra dữ liệu lịch sử.")
            return

        features = ['Open', 'High', 'Low', 'Volume', 'SP500', 'Oil', 'DollarIndex', 'SMA_5', 'RSI_14', 'Close_lag1', 'Close_lag2']
        input_data = pd.DataFrame([[open_price, high_price, low_price, volume, sp500, oil, 
                                    dollar_index, sma_5, rsi_14, close_lag1, close_lag2]], columns=features)
        
        if input_data.isnull().any().any():
            messagebox.showerror("Lỗi", "Dữ liệu đầu vào chứa giá trị NaN. Vui lòng kiểm tra các ô nhập liệu.")
            return

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        X_train = df[features]
        y_train = df['Close']
        X_train_scaled = scaler.transform(X_train)
        residuals = y_train - model.predict(X_train_scaled)
        std_error = np.std(residuals)
        confidence_interval = 1.96 * std_error
        lower_bound = max(0, prediction - confidence_interval)
        upper_bound = prediction + confidence_interval

        # Sử dụng ngày từ dữ liệu tải về thay vì ngày hiện tại
        current_date = pd.to_datetime(today_data_label.cget("text").split("Dữ liệu ngày ")[1].split(":")[0], format='%Y-%m-%d')
        next_date = current_date + timedelta(days=1)
        
        result_text = f"Dự đoán giá đóng cửa ngày {next_date.strftime('%Y-%m-%d')}: {prediction:.2f} USD\n"
        result_text += f"Độ tin cậy 95%: {lower_bound:.2f} - {upper_bound:.2f} USD\n"
        result_text += "(Ghi chú: Khoảng tin cậy 95% cho biết 95% khả năng giá thực tế nằm trong phạm vi này.)"
        result_label.config(text=result_text)

        update_chart(prediction, df['Close'].iloc[-1], next_date.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'),
                      lower_bound, upper_bound)
    
    except Exception as e:
        messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {str(e)}")


# Hàm cập nhật biểu đồ với khoảng tin cậy từ lower_bound đến upper_bound
def update_chart(predicted_price, last_close, next_date, last_date, lower_bound, upper_bound):
    for widget in chart_frame.winfo_children():
        widget.destroy()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    dates = [last_date, next_date]
    prices = [last_close, predicted_price]
    ax.plot(dates, prices, marker='o', label='Giá đóng cửa', color='#36A2EB')
    
    # Tạo mảng giá trị cho khoảng tin cậy
    y_fill = [lower_bound, upper_bound]
    ax.fill_between(dates, y_fill, y2=[lower_bound, lower_bound],  # Đặt y2 là lower_bound để giới hạn dưới
                    alpha=0.2, color='#36A2EB', label='Khoảng tin cậy')
    
    # Đặt giới hạn trục y từ lower_bound - 10 đến upper_bound + 10
    ax.set_ylim(bottom=lower_bound - 10, top=upper_bound + 10)
    ax.set_title("Dự đoán giá cổ phiếu Amazon", fontsize=10)
    ax.set_ylabel("Giá (USD)", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.7)

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    guide_label = tk.Label(chart_frame, text="Ghi chú: Đường xanh là giá dự đoán, vùng màu là khoảng tin cậy 95%.", 
                           font=("Arial", 8), bg='#f0f0f0', fg='grey')
    guide_label.pack()

# Tạo giao diện Tkinter với bố cục ngang
root = tk.Tk()
root.title("Dự đoán giá cổ phiếu Amazon")
root.geometry("1000x600")
root.configure(bg='#f0f0f0')

# Phần 1: Khu vực tải dữ liệu
data_frame = ttk.Frame(root, padding="10", borderwidth=2, relief="groove", style='Custom.TFrame')
data_frame.pack(pady=5, padx=10, fill="x")
title_label = tk.Label(data_frame, text="Dự đoán giá cổ phiếu Amazon", font=("Arial", 16, "bold"), bg='#f0f0f0')
title_label.pack(pady=5)
today_data_label = tk.Label(data_frame, text="Nhấn 'Tải dữ liệu hôm nay' để xem dữ liệu.", font=("Arial", 10), bg='#ffffff', justify="left")
today_data_label.pack(pady=5)
fetch_button = ttk.Button(data_frame, text="Tải dữ liệu hôm nay", command=display_today_data, style='Custom.TButton')
fetch_button.pack(pady=5)

# Phần 2: Khu vực chính với hai cột (nhập liệu và kết quả)
main_frame = ttk.Frame(root, padding="10")
main_frame.pack(pady=10, padx=10, fill="both", expand=True)

# Cột trái: Khu vực nhập dữ liệu
input_frame = ttk.Frame(main_frame, padding="10", borderwidth=2, relief="groove", style='Custom.TFrame')
input_frame.pack(side="left", padx=5, fill="y")

style = ttk.Style()
style.configure('Custom.TFrame', background='#ffffff')
style.configure('Custom.TButton', font=('Arial', 12, 'bold'), padding=6)

labels = [
    ("Giá mở cửa (Open, USD):", "100.0"), ("Giá cao nhất (High, USD):", "105.0"), 
    ("Giá thấp nhất (Low, USD):", "98.0"), ("Khối lượng (Volume, shares):", "5000.0"), 
    ("S&P 500 (index):", "4000.0"), ("Giá dầu (Oil, USD/barrel):", "70.0"), 
    ("Chỉ số Đô la (DollarIndex):", "95.0")
]

entries = {}
for i, (label_text, default_value) in enumerate(labels):
    ttk.Label(input_frame, text=label_text, width=25, anchor="e", font=("Arial", 10)).grid(row=i, column=0, padx=5, pady=5, sticky="e")
    entry = ttk.Entry(input_frame, width=20, font=("Arial", 10))
    entry.insert(0, default_value)
    entry.config(foreground='grey')
    entry.bind("<FocusIn>", lambda event, e=entry, txt=default_value: on_entry_focus_in(event, e, txt))
    entry.bind("<FocusOut>", lambda event, e=entry, txt=default_value: on_entry_focus_out(event, e, txt))
    entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
    entries[label_text] = entry

entry_open = entries["Giá mở cửa (Open, USD):"]
entry_high = entries["Giá cao nhất (High, USD):"]
entry_low = entries["Giá thấp nhất (Low, USD):"]
entry_volume = entries["Khối lượng (Volume, shares):"]
entry_sp500 = entries["S&P 500 (index):"]
entry_oil = entries["Giá dầu (Oil, USD/barrel):"]
entry_dollar_index = entries["Chỉ số Đô la (DollarIndex):"]

predict_button = ttk.Button(input_frame, text="Dự đoán ngày hôm sau", command=predict_next_day, style='Custom.TButton')
predict_button.grid(row=len(labels), column=0, columnspan=2, pady=10)

# Cột phải: Khu vực kết quả dự đoán
result_frame = ttk.Frame(main_frame, padding="10", borderwidth=2, relief="groove", style='Custom.TFrame')
result_frame.pack(side="right", padx=5, fill="both", expand=True)

result_label = tk.Label(result_frame, text="", font=("Arial", 12, "bold"), bg='#f0f0f0')
result_label.pack(pady=5)

chart_frame = tk.Frame(result_frame, bg='#f0f0f0')
chart_frame.pack(pady=5, fill="both", expand=True)

root.mainloop()


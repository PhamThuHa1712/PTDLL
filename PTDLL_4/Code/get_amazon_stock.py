import yfinance as yf
import pandas as pd

# Tải dữ liệu từ Yahoo Finance
amzn = yf.download('AMZN', start="2000-01-01", end="2025-01-01")
sp500 = yf.download('^GSPC', start="2000-01-01", end="2025-01-01")
oil = yf.download('CL=F', start="2000-01-01", end="2025-01-01")
dollar = yf.download('DX=F', start="2000-01-01", end="2025-01-01")

# Reset index (đưa cột Date ra ngoài)
amzn = amzn.reset_index()
sp500 = sp500.reset_index()
oil = oil.reset_index()
dollar = dollar.reset_index()

# Lấy các cột cần thiết và đổi tên cột cho rõ ràng
amzn = amzn[["Date", "Open", "High", "Low", "Close", "Volume"]]
sp500 = sp500[["Date", "Close"]].rename(columns={"Close": "SP500"})
oil = oil[["Date", "Close"]].rename(columns={"Close": "Oil"})
dollar = dollar[["Date", "Close"]].rename(columns={"Close": "DollarIndex"})

# Gộp dữ liệu
df = amzn.merge(sp500, on="Date", how="inner")
df = df.merge(oil, on="Date", how="inner")
df = df.merge(dollar, on="Date", how="inner")

# Làm tròn giá trị (tuỳ chọn)
df = df.round(2)

# Ghi ra file CSV (hoặc Excel nếu muốn)
df.to_csv("Data/amazon_stock_2000_2025.csv", index=False)  # hoặc .xlsx nếu cần Excel

import os
import pandas as pd
import yfinance as yf

# ==========================================
# User Settings & Directory Creation
# ==========================================
save_folder = "data"
os.makedirs(save_folder, exist_ok=True)

print("======================================")
print("Stock Data Collector Setup")
print("======================================")
print(
    "Enter the stocks you want to collect. (Type 'done' or press Enter to"
    " finish)"
)
print("Example - Ticker: 005930 (Auto-converted to 005930.KS)")
print("-" * 50)

target_stocks = {}

while True:
  ticker = (
      input(
          "Enter stock ticker (e.g., 005380, AAPL, or type 'done'): "
      )
      .strip()
      .upper()
  )

  if ticker == "DONE" or ticker == "":
    if target_stocks:
      break
    else:
      print("Please enter at least one stock.")
      continue

  # 6자리 숫자(국내 주식)인 경우 자동으로 .KS를 붙여줍니다.
  if ticker.isdigit() and len(ticker) == 6:
    ticker = ticker + ".KS"

  name = input(f"Enter name for {ticker} (e.g., Hyundai): ").strip()
  if not name:
    name = ticker

  target_stocks[ticker] = name
  print(f"-> Added: {name} ({ticker})\n")

# Dynamic date input
print("-" * 50)
start_date = (
    input("Enter start date (YYYY-MM-DD) [default: 2026-01-01]: ").strip()
    or "2026-01-01"
)
end_date = (
    input("Enter end date (YYYY-MM-DD) [default: 2026-09-22]: ").strip()
    or "2026-09-22"
)

print("\n======================================")
print("Starting stock data collection...")
print("======================================")
print(f"Save Folder : {os.path.abspath(save_folder)}")
print(f"Period      : {start_date} ~ {end_date}")
print(f"Target List : {target_stocks}")
print()

# Loop through the dictionary to collect and save data individually
for ticker, stock_name in target_stocks.items():
  print(f"[{stock_name} ({ticker})] Downloading...")

  try:
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
      print(f"-> {stock_name}: No data collected.\n")
      continue

    # Clean up multi-index columns in recent yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.droplevel(1)

    # Reset index to make 'Date' a column
    df = df.reset_index()

    # Extract required columns (Date, Close)
    df = df[["Date", "Close"]]
    df.columns = ["Date", "Close"]

    # Insert stock name column at the beginning
    df.insert(0, "Name", stock_name)

    # Format date (YYYY-MM-DD)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    # Combine folder path and filename
    file_name = os.path.join(save_folder, f"{stock_name}_stock_data.csv")

    # Save to CSV
    df.to_csv(file_name, index=False, encoding="utf-8-sig")

    print(
        f"-> Success! Saved path: {file_name} (Data count: {len(df)} rows)\n"
    )

  except Exception as e:
    print(f"-> Error occurred while collecting {stock_name}: {e}\n")

print("======================================")
print("All stock data collection and saving completed!")
print("======================================")
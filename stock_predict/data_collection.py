import yfinance as yf

def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

def preprocess_data(data):
    data = data[['Close']]
    data = data.dropna()
    return data

if __name__ == "__main__":
    ticker = "AAPL"  # Apple Inc.
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    data = get_stock_data(ticker, start_date, end_date)
    processed_data = preprocess_data(data)
    print(processed_data.head())

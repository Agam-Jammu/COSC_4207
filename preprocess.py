import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# retrieved from https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset

# Load dataset
df = pd.read_csv("data/AAPL_stock.csv", parse_dates=["Date"])

# Sort data by date
df = df.sort_values(by="Date")

# Drop usless attribute 'Adj Close' as it's redundant
df = df.drop(columns=["Adj Close"])

# Normalize numerical columns (Open, High, Low, Close, Volume)
scaler = MinMaxScaler()
df[["Open", "High", "Low", "Close", "Volume"]] = scaler.fit_transform(df[["Open", "High", "Low", "Close", "Volume"]])

# Save the preprocessed data
df.to_csv("data/preprocessed_AAPL_stock.csv", index=False)

print("complete: Data saved")

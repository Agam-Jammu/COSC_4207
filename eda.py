import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Ensure plots folder exists
PLOT_DIR = "plots/EDA"
os.makedirs(PLOT_DIR, exist_ok=True)

# Load dataset
file_path = "data/AAPL.csv" 
df = pd.read_csv(file_path, parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Display basic information
print("Dataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Plot raw stock prices
def plot_stock_prices():
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Open'], label="Open Price", alpha=0.7)
    plt.plot(df.index, df['Close'], label="Close Price", alpha=0.7)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title("Stock Prices Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(f"{PLOT_DIR}/stock_prices.png")
    plt.close()

# Calculate daily returns
df['Daily Return'] = df['Close'].pct_change()

# Plot daily returns
def plot_daily_returns():
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Daily Return'], label="Daily Return", color='tab:blue')
    plt.axhline(y=0, linestyle='--', color='gray', alpha=0.5)
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.title("Daily Returns Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(f"{PLOT_DIR}/daily_returns.png")
    plt.close()

# Compute 30-day moving average
df['30-Day MA'] = df['Close'].rolling(window=30).mean()

# Plot moving averages
def plot_moving_average():
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Close'], label="Close Price", alpha=0.6)
    plt.plot(df.index, df['30-Day MA'], label="30-Day MA", linestyle='dashed', color='red')
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title("Stock Prices with 30-Day Moving Average")
    plt.legend()
    plt.grid()
    plt.savefig(f"{PLOT_DIR}/moving_average.png")
    plt.close()

# Apply PCA on features
features = ['Open', 'High', 'Low', 'Close', 'Volume']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

df_pca = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'], index=df.index)

# Plot PCA result
def plot_pca():
    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.5, color='tab:purple')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Projection of Stock Data")
    plt.grid()
    plt.savefig(f"{PLOT_DIR}/pca_projection.png")
    plt.close()


plot_stock_prices()
plot_daily_returns()
plot_moving_average()
plot_pca()

print("Completed: Check the 'plots' folder.")

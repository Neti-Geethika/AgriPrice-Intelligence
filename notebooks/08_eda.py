import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/processed/agriprice_cleaned.csv')
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])

sns.set_style('whitegrid')

# 1. Price trend over time for top crops
target_crops = ['Onion', 'Tomato', 'Potato', 'Rice', 'Maize']
plt.figure(figsize=(12,6))
for crop in target_crops:
    crop_data = df[df['Commodity'] == crop].sort_values('Arrival_Date')
    monthly_avg = crop_data.groupby(crop_data['Arrival_Date'].dt.to_period('M'))['Modal_Price'].mean()
    plt.plot(monthly_avg.index.astype(str), monthly_avg.values, marker='o', label=crop)
plt.xticks(rotation=45)
plt.title('Monthly Average Price Trend by Crop')
plt.xlabel('Month')
plt.ylabel('Modal Price (₹/quintal)')
plt.legend()
plt.tight_layout()
plt.savefig('data/processed/eda_price_trend.png')
plt.close()
print("Saved: eda_price_trend.png")

# 2. Seasonality - average price by month, per crop
plt.figure(figsize=(12,6))
for crop in target_crops:
    crop_data = df[df['Commodity'] == crop]
    monthly_seasonal = crop_data.groupby('Month')['Modal_Price'].mean()
    plt.plot(monthly_seasonal.index, monthly_seasonal.values, marker='o', label=crop)
plt.title('Seasonal Price Pattern (Avg by Month)')
plt.xlabel('Month')
plt.ylabel('Avg Modal Price (₹/quintal)')
plt.xticks(range(1,13))
plt.legend()
plt.tight_layout()
plt.savefig('data/processed/eda_seasonality.png')
plt.close()
print("Saved: eda_seasonality.png")

# 3. Price spread across markets for a chosen crop (e.g. Onion)
plt.figure(figsize=(10,6))
onion_data = df[df['Commodity'] == 'Onion']
market_avg = onion_data.groupby('Market')['Modal_Price'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=market_avg.values, y=market_avg.index)
plt.title('Top 10 Markets by Avg Onion Price')
plt.xlabel('Avg Modal Price (₹/quintal)')
plt.tight_layout()
plt.savefig('data/processed/eda_market_comparison.png')
plt.close()
print("Saved: eda_market_comparison.png")

# 4. Volatility check - which crops swing the most
volatility = df.groupby('Commodity')['Modal_Price'].agg(['mean','std'])
volatility['cv'] = volatility['std'] / volatility['mean']
volatility_sorted = volatility.sort_values('cv', ascending=False)
print("\nMost volatile crops (top 10):")
print(volatility_sorted.head(10))

print("\nLeast volatile crops (bottom 10):")
print(volatility_sorted.tail(10))

volatility_sorted.to_csv('data/processed/crop_volatility.csv')
print("\nSaved: crop_volatility.csv")
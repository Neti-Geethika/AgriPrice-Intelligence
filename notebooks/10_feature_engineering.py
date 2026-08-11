import pandas as pd

df = pd.read_csv('data/processed/agriprice_target_crops.csv')
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])

# Sort properly - crucial for lag/rolling features to make sense
df = df.sort_values(['Commodity', 'Market', 'Arrival_Date']).reset_index(drop=True)

# Lag features (previous price for same crop+market)
df['lag_price_1'] = df.groupby(['Commodity', 'Market'])['Modal_Price'].shift(1)
df['lag_price_7'] = df.groupby(['Commodity', 'Market'])['Modal_Price'].shift(7)

# Rolling average (needs sorted data, per crop+market)
df['rolling_avg_7'] = df.groupby(['Commodity', 'Market'])['Modal_Price'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)

# Price change %
df['price_change_pct'] = df.groupby(['Commodity', 'Market'])['Modal_Price'].pct_change()

# Season feature (basic India seasons)
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    else:
        return 'PostMonsoon'

df['Season'] = df['Month'].apply(get_season)

print("Shape after feature engineering:", df.shape)
print("\nMissing values in new features:")
print(df[['lag_price_1','lag_price_7','rolling_avg_7','price_change_pct']].isnull().sum())

# Drop rows where lag_price_1 is missing (first record per crop+market, unavoidable)
df_model_ready = df.dropna(subset=['lag_price_1']).copy()
print("\nShape after dropping first-record NaNs:", df_model_ready.shape)

df_model_ready.to_csv('data/processed/agriprice_features.csv', index=False)
print("\nSaved to data/processed/agriprice_features.csv")
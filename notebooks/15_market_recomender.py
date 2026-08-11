import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('models/price_prediction_model.pkl')
le_commodity = joblib.load('models/le_commodity.pkl')
le_market = joblib.load('models/le_market.pkl')
le_district = joblib.load('models/le_district.pkl')
le_season = joblib.load('models/le_season.pkl')

df = pd.read_csv('data/processed/agriprice_features.csv')
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])

def recommend_markets(crop, top_n=5):
    """
    For a given crop, predict expected price across all markets that sell it,
    using each market's most recent known data as input, and rank them.
    """
    crop_df = df[df['Commodity'] == crop].copy()
    
    if crop_df.empty:
        print(f"No data available for {crop}")
        return None
    
    # Get the latest record per market for this crop (most recent lag/rolling features)
    latest_per_market = crop_df.sort_values('Arrival_Date').groupby('Market').tail(1).copy()
    
    latest_per_market['Commodity_enc'] = le_commodity.transform(latest_per_market['Commodity'])
    latest_per_market['Market_enc'] = le_market.transform(latest_per_market['Market'])
    latest_per_market['District_enc'] = le_district.transform(latest_per_market['District'])
    latest_per_market['Season_enc'] = le_season.transform(latest_per_market['Season'])
    
    feature_cols = ['Commodity_enc', 'Market_enc', 'District_enc', 'Season_enc',
                     'Month', 'DayOfWeek', 'lag_price_1', 'rolling_avg_7']
    
    X = latest_per_market[feature_cols]
    latest_per_market['Predicted_Price'] = model.predict(X)
    
    result = latest_per_market[['Market', 'District', 'Modal_Price', 'Predicted_Price']].sort_values(
        'Predicted_Price', ascending=False
    ).head(top_n)
    
    result = result.rename(columns={'Modal_Price': 'Last_Known_Price'})
    return result


# Test it for a few crops
for crop in ['Onion', 'Tomato', 'Potato']:
    print(f"\n=== Best markets for {crop} ===")
    result = recommend_markets(crop, top_n=5)
    print(result)
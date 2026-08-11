import pandas as pd
import joblib

# Load all pieces
df = pd.read_csv('data/processed/agriprice_with_anomalies.csv')
volatility = pd.read_csv('data/processed/crop_volatility_final.csv')

model = joblib.load('models/price_prediction_model.pkl')
le_commodity = joblib.load('models/le_commodity.pkl')
le_market = joblib.load('models/le_market.pkl')
le_district = joblib.load('models/le_district.pkl')
le_season = joblib.load('models/le_season.pkl')

df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])

# Generate predictions for every row (predicted vs actual, for Power BI comparison charts)
df['Commodity_enc'] = le_commodity.transform(df['Commodity'])
df['Market_enc'] = le_market.transform(df['Market'])
df['District_enc'] = le_district.transform(df['District'])
df['Season_enc'] = le_season.transform(df['Season'])

feature_cols = ['Commodity_enc', 'Market_enc', 'District_enc', 'Season_enc',
                 'Month', 'DayOfWeek', 'lag_price_1', 'rolling_avg_7']

df['Predicted_Price'] = model.predict(df[feature_cols])
df['Prediction_Error'] = df['Modal_Price'] - df['Predicted_Price']

# Merge in volatility/risk info
df = df.merge(volatility[['Commodity', 'Risk_Level', 'cv']], on='Commodity', how='left')

# Clean up columns for Power BI (drop encoded helper columns, keep readable ones)
final_cols = ['Arrival_Date', 'State', 'District', 'Market', 'Commodity', 'Variety',
              'Min_Price', 'Max_Price', 'Modal_Price', 'Predicted_Price', 'Prediction_Error',
              'Anomaly', 'Risk_Level', 'cv', 'Year', 'Month', 'Season']

df_final = df[final_cols].rename(columns={'cv': 'Volatility_Score'})

df_final.to_csv('data/processed/agripriceintelligence_powerbi.csv', index=False)
print("Saved final Power BI dataset:", df_final.shape)
print("\nColumns:", df_final.columns.tolist())
print("\nSample rows:")
print(df_final.head())
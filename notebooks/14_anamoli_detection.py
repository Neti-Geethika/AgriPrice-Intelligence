import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv('data/processed/agriprice_features.csv')
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])

# Run anomaly detection PER CROP (a "normal" price for Rice is very different from Onion,
# so detecting anomalies globally would be misleading)
all_results = []

for crop in df['Commodity'].unique():
    crop_df = df[df['Commodity'] == crop].copy()
    
    if len(crop_df) < 10:  # skip crops with too little data to be meaningful
        crop_df['Anomaly'] = 0
        all_results.append(crop_df)
        continue
    
    iso = IsolationForest(contamination=0.05, random_state=42)
    crop_df['Anomaly'] = iso.fit_predict(crop_df[['Modal_Price']])
    # IsolationForest returns -1 for anomaly, 1 for normal — convert to readable flag
    crop_df['Anomaly'] = crop_df['Anomaly'].map({-1: 1, 1: 0})  # 1 = anomaly, 0 = normal
    
    all_results.append(crop_df)

df_anomalies = pd.concat(all_results, ignore_index=True)

anomaly_count = df_anomalies['Anomaly'].sum()
print(f"Total anomalies detected: {anomaly_count} out of {len(df_anomalies)} records")

print("\nAnomalies per crop:")
print(df_anomalies.groupby('Commodity')['Anomaly'].sum())

# Show a few example anomalies
print("\nSample anomaly records:")
print(df_anomalies[df_anomalies['Anomaly'] == 1][
    ['Arrival_Date', 'Commodity', 'Market', 'Modal_Price']
].sort_values('Arrival_Date').head(15))

df_anomalies.to_csv('data/processed/agriprice_with_anomalies.csv', index=False)
print("\nSaved to data/processed/agriprice_with_anomalies.csv")
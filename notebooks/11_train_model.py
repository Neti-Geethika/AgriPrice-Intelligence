import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv('data/processed/agriprice_features.csv')
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])

# Encode categorical columns
le_commodity = LabelEncoder()
le_market = LabelEncoder()
le_district = LabelEncoder()
le_season = LabelEncoder()

df['Commodity_enc'] = le_commodity.fit_transform(df['Commodity'])
df['Market_enc'] = le_market.fit_transform(df['Market'])
df['District_enc'] = le_district.fit_transform(df['District'])
df['Season_enc'] = le_season.fit_transform(df['Season'])

# Features and target
feature_cols = ['Commodity_enc', 'Market_enc', 'District_enc', 'Season_enc',
                 'Month', 'DayOfWeek', 'lag_price_1', 'rolling_avg_7']
target_col = 'Modal_Price'

X = df[feature_cols]
y = df[target_col]

# IMPORTANT: sort by date and split by time, not randomly
df_sorted = df.sort_values('Arrival_Date')
X_sorted = df_sorted[feature_cols]
y_sorted = df_sorted[target_col]

split_idx = int(len(df_sorted) * 0.8)
X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]

print("Train size:", len(X_train), "Test size:", len(X_test))

# Naive baseline: predict using lag_price_1 (yesterday's/last known price)
naive_preds = X_test['lag_price_1']
naive_mae = mean_absolute_error(y_test, naive_preds)
naive_rmse = np.sqrt(mean_squared_error(y_test, naive_preds))
print(f"\nNaive baseline (last price) -> MAE: {naive_mae:.2f}, RMSE: {naive_rmse:.2f}")

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=200, random_state=42)
}

results = {}
best_model = None
best_mae = float('inf')

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"\n{name} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")
    
    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_model_name = name

print(f"\nBest model: {best_model_name} (beats naive baseline: {best_mae < naive_mae})")

# Try simpler tree configs to reduce overfitting on this dataset size
models_tuned = {
    'Random Forest (shallow)': RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=42),
    'XGBoost (shallow)': XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
}

for name, model in models_tuned.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n{name} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")
    
    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_model_name = name

print(f"\nFinal best model after tuning: {best_model_name}")
# Save the best model and encoders for later use (Streamlit app + Power BI export)
joblib.dump(best_model, 'models/price_prediction_model.pkl')
joblib.dump(le_commodity, 'models/le_commodity.pkl')
joblib.dump(le_market, 'models/le_market.pkl')
joblib.dump(le_district, 'models/le_district.pkl')
joblib.dump(le_season, 'models/le_season.pkl')

print("\nModel and encoders saved to models/")
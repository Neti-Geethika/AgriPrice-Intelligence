import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

df = pd.read_csv('data/processed/agriprice_features.csv')

# Load saved model and encoders
model = joblib.load('models/price_prediction_model.pkl')
le_commodity = joblib.load('models/le_commodity.pkl')
le_market = joblib.load('models/le_market.pkl')
le_district = joblib.load('models/le_district.pkl')
le_season = joblib.load('models/le_season.pkl')

df['Commodity_enc'] = le_commodity.transform(df['Commodity'])
df['Market_enc'] = le_market.transform(df['Market'])
df['District_enc'] = le_district.transform(df['District'])
df['Season_enc'] = le_season.transform(df['Season'])

feature_cols = ['Commodity_enc', 'Market_enc', 'District_enc', 'Season_enc',
                 'Month', 'DayOfWeek', 'lag_price_1', 'rolling_avg_7']

X = df[feature_cols]

# SHAP for linear model
explainer = shap.LinearExplainer(model, X)
shap_values = explainer.shap_values(X)

plt.figure()
shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig('data/processed/shap_summary.png')
plt.close()
print("Saved SHAP summary plot to data/processed/shap_summary.png")

# Print average absolute importance per feature
import numpy as np
importance = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("\nFeature importance (by mean |SHAP value|):")
print(importance)
import pandas as pd

df = pd.read_csv('data/processed/agriprice_features.csv')

# Calculate volatility (coefficient of variation) per crop
volatility = df.groupby('Commodity')['Modal_Price'].agg(['mean', 'std']).reset_index()
volatility['cv'] = volatility['std'] / volatility['mean']

# Classify into risk buckets
def classify_risk(cv):
    if cv < 0.30:
        return 'Low'
    elif cv < 0.45:
        return 'Medium'
    else:
        return 'High'

volatility['Risk_Level'] = volatility['cv'].apply(classify_risk)
volatility = volatility.sort_values('cv', ascending=False)

print("Volatility & Risk Classification:")
print(volatility)

volatility.to_csv('data/processed/crop_volatility_final.csv', index=False)
print("\nSaved to data/processed/crop_volatility_final.csv")
import pandas as pd

df = pd.read_csv('data/processed/agriprice_cleaned.csv')

target_crops = ['Onion', 'Tomato', 'Potato', 'Rice', 'Maize', 
                 'Cabbage', 'Carrot', 'Brinjal', 'Banana', 'Mango']

df_target = df[df['Commodity'].isin(target_crops)].copy()

print("Target crop dataset shape:", df_target.shape)
print("\nRecords per crop:")
print(df_target['Commodity'].value_counts())

print("\nRecords per crop per district:")
print(df_target.groupby(['Commodity','District']).size())

# Volatility check on just target crops
volatility = df_target.groupby('Commodity')['Modal_Price'].agg(['mean','std'])
volatility['cv'] = volatility['std'] / volatility['mean']
print("\nTarget crop volatility:")
print(volatility.sort_values('cv', ascending=False))

df_target.to_csv('data/processed/agriprice_target_crops.csv', index=False)
print("\nSaved to data/processed/agriprice_target_crops.csv")
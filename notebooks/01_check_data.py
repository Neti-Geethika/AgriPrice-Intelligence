import pandas as pd

df = pd.read_csv('data/raw/maharashtra_nashik_all.csv')

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nUnique commodities:", df['Commodity'].nunique())
print(df['Commodity'].value_counts().head(15))
print("\nDate range:")
print("Earliest:", df['Arrival_Date'].min())
print("Latest:", df['Arrival_Date'].max())
print("\nSample rows:")
print(df.head())
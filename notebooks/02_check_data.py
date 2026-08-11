import pandas as pd

df = pd.read_csv('data/raw/multi_state_recent_2yrs.csv')

print("Shape:", df.shape)
print("\nStates:", df['State'].unique())
print("\nDistricts per state:")
print(df.groupby('State')['District'].unique())

print("\nTop 20 commodities:")
print(df['Commodity'].value_counts().head(20))

print("\nDate range:")
print("Earliest:", df['Arrival_Date'].min())
print("Latest:", df['Arrival_Date'].max())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nSample rows:")
print(df.head(10))
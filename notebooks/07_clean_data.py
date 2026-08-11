import pandas as pd

df = pd.read_csv('data/processed/agriprice_combined_recent.csv')
print("Starting shape:", df.shape)

# 1. Parse date properly (drop the string version, keep parsed)
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date_parsed'], format='mixed')
df = df.drop(columns=['Arrival_Date_parsed'])

# 2. Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# 3. Convert price columns to numeric (in case any are stored as text)
for col in ['Min_Price', 'Max_Price', 'Modal_Price']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. Remove rows with missing/invalid prices
before = len(df)
df = df.dropna(subset=['Min_Price', 'Max_Price', 'Modal_Price'])
print(f"\nDropped {before - len(df)} rows with missing prices")

# 5. Remove rows with zero or negative prices (data errors)
before = len(df)
df = df[(df['Min_Price'] > 0) & (df['Max_Price'] > 0) & (df['Modal_Price'] > 0)]
print(f"Dropped {before - len(df)} rows with zero/negative prices")

# 6. Sanity check: Max_Price should be >= Min_Price
before = len(df)
df = df[df['Max_Price'] >= df['Min_Price']]
print(f"Dropped {before - len(df)} rows where Max_Price < Min_Price")

# 7. Standardize commodity names (strip whitespace, consistent casing)
df['Commodity'] = df['Commodity'].str.strip()
df['District'] = df['District'].str.strip()
df['Market'] = df['Market'].str.strip()

# 8. Remove exact duplicate rows
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows")

# 9. Add useful date features now (saves a step later)
df['Year'] = df['Arrival_Date'].dt.year
df['Month'] = df['Arrival_Date'].dt.month
df['DayOfWeek'] = df['Arrival_Date'].dt.dayofweek

print("\nFinal cleaned shape:", df.shape)
print("\nFinal commodity coverage:")
print(df['Commodity'].value_counts().head(15))

df.to_csv('data/processed/agriprice_cleaned.csv', index=False)
print("\nSaved cleaned data to data/processed/agriprice_cleaned.csv")
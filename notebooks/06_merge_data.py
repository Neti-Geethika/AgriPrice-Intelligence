import pandas as pd

# Load both files
df_existing = pd.read_csv('data/raw/multi_state_recent_2yrs.csv')
df_extra = pd.read_csv('data/raw/extra_districts.csv')

print("Existing recent data shape:", df_existing.shape)
print("Extra pull shape (all years):", df_extra.shape)

# Parse dates in the extra pull
df_extra['Arrival_Date_parsed'] = pd.to_datetime(df_extra['Arrival_Date'], format='%d/%m/%Y', errors='coerce')

# Filter extra pull to last 2 years (based on its own max date)
cutoff = df_extra['Arrival_Date_parsed'].max() - pd.DateOffset(years=2)
df_extra_recent = df_extra[df_extra['Arrival_Date_parsed'] >= cutoff].copy()

print("Extra pull filtered to recent 2 years:", df_extra_recent.shape)

# Make sure columns match before combining
print("\nColumns match:", set(df_existing.columns) == set(df_extra_recent.columns))

# Combine
df_combined = pd.concat([df_existing, df_extra_recent], ignore_index=True)

# Drop exact duplicates (in case any districts overlapped)
df_combined = df_combined.drop_duplicates()

print("\nFinal combined shape:", df_combined.shape)
print("\nRecords per state/district:")
print(df_combined.groupby(['State','District']).size())

print("\nCommodity coverage (top 15):")
print(df_combined['Commodity'].value_counts().head(15))

# Save the final working dataset
df_combined.to_csv('data/processed/agriprice_combined_recent.csv', index=False)
print("\nSaved to data/processed/agriprice_combined_recent.csv")
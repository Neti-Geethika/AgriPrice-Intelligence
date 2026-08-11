import pandas as pd

df = pd.read_csv('data/raw/multi_state_raw_all_years.csv')

print("All records per state/district (all years):")
print(df.groupby(['State','District']).size())
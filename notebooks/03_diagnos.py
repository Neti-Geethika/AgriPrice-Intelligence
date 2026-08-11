import pandas as pd

df = pd.read_csv('data/raw/multi_state_recent_2yrs.csv')
df['Arrival_Date_parsed'] = pd.to_datetime(df['Arrival_Date_parsed'])

print("Correct date range:")
print("Earliest:", df['Arrival_Date_parsed'].min())
print("Latest:", df['Arrival_Date_parsed'].max())

print("\nRecords per district (recent 2 years):")
print(df.groupby(['State','District']).size())
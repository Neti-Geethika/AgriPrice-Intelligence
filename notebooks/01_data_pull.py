import requests
import pandas as pd
import time

API_KEY = "579b464db66ec23bdd00000104cb56f29b3f49104a53f1bcada1b9c8"
BASE_URL = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

locations = {
    'Andhra Pradesh': ['Guntur', 'Krishna', 'Kurnool', 'East Godavari', 'Chittoor'],
    'Telangana': ['Hyderabad', 'Warangal', 'Nizamabad', 'Karimnagar', 'Nalgonda'],
    'Karnataka': ['Bengaluru', 'Belagavi', 'Mysuru', 'Dharwad', 'Ballari']
}

def fetch_batch(params, max_retries=3, timeout=30):
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=timeout)
            if response.status_code != 200:
                print("  Non-200:", response.text[:200])
                time.sleep(3)
                continue
            return response.json()
        except Exception as e:
            print(f"  Attempt {attempt} failed: {e}")
            time.sleep(3)
    return None


def fetch_district(state, district, max_offset=15000, limit=500):
    all_records = []
    offset = 0
    while offset <= max_offset:
        params = {
            'api-key': API_KEY,
            'format': 'json',
            'limit': limit,
            'offset': offset,
            'filters[State]': state,
            'filters[District]': district
        }
        data = fetch_batch(params)
        if data is None:
            break
        records = data.get('records', [])
        if not records:
            break
        all_records.extend(records)
        offset += limit
        time.sleep(0.5)
    return all_records


def main():
    all_data = []
    for state, districts in locations.items():
        for district in districts:
            print(f"\nFetching {district}, {state} ...")
            records = fetch_district(state, district)
            print(f"  -> Got {len(records)} records")
            all_data.extend(records)

    df = pd.DataFrame(all_data)
    print(f"\nTotal records across all districts (before date filtering): {len(df)}")
    df.to_csv('data/raw/multi_state_raw_all_years.csv', index=False)

    # Now filter to last 2 years
    df['Arrival_Date_parsed'] = pd.to_datetime(df['Arrival_Date'], format='%d/%m/%Y', errors='coerce')
    cutoff = df['Arrival_Date_parsed'].max() - pd.DateOffset(years=2)
    df_recent = df[df['Arrival_Date_parsed'] >= cutoff].copy()

    print(f"Records in last 2 years: {len(df_recent)}")
    df_recent.to_csv('data/raw/multi_state_recent_2yrs.csv', index=False)
    print("Saved recent-2-years version to data/raw/multi_state_recent_2yrs.csv")


if __name__ == "__main__":
    main()
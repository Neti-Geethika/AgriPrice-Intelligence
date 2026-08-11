import requests
import pandas as pd
from datetime import datetime

import sys
sys.path.append('..')
from config import API_KEY
BASE_URL = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

def fetch_live_price(state, district, commodity, limit=10):
    """
    Fetches the most recent available price records for a given 
    state, district, and commodity from the live API.
    Returns the latest record found, or None if unavailable.
    """
    params = {
        'api-key': API_KEY,
        'format': 'json',
        'limit': limit,
        'filters[State]': state,
        'filters[District]': district,
        'filters[Commodity]': commodity
    }
    
    try:
        response = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=20)
        if response.status_code != 200:
            print(f"API error: status {response.status_code}")
            return None
        
        data = response.json()
        records = data.get('records', [])
        
        if not records:
            print(f"No live data found for {commodity} in {district}, {state}")
            return None
        
        # Convert to DataFrame, parse dates, sort to get the most recent
        df = pd.DataFrame(records)
        df['Arrival_Date_parsed'] = pd.to_datetime(df['Arrival_Date'], format='%d/%m/%Y', errors='coerce')
        df = df.sort_values('Arrival_Date_parsed', ascending=False)
        
        latest = df.iloc[0]
        print("DEBUG latest date:", latest['Arrival_Date'], "parsed:", latest['Arrival_Date_parsed'])
        days_old = (datetime.now() - latest['Arrival_Date_parsed']).days
        
        return {
            'Commodity': latest['Commodity'],
            'Market': latest['Market'],
            'District': latest['District'],
            'State': latest['State'],
            'Date': latest['Arrival_Date'],
            'Min_Price': latest['Min_Price'],
            'Max_Price': latest['Max_Price'],
            'Modal_Price': latest['Modal_Price'],
            'Days_Old': days_old,
            'Is_Recent': days_old <= 90
        }
    
    except Exception as e:
        print(f"Error fetching live price: {e}")
        return None


# Test it
if __name__ == "__main__":
    result = fetch_live_price('Karnataka', 'Bengaluru', 'Onion')
    print("\nLive price result:")
    print(result)
    
    result2 = fetch_live_price('Telangana', 'Warangal', 'Tomato')
    print("\nLive price result:")
    print(result2)
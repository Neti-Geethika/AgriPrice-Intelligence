import requests

API_KEY ="579b464db66ec23bdd00000104cb56f29b3f49104a53f1bcada1b9c8"
url = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"

params = {
    'api-key': API_KEY,
    'format': 'json',
    'limit': 5
}
response = requests.get(url, params=params, timeout=45)
print(response.status_code)
print(response.json())


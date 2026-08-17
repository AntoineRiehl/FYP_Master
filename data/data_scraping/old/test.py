import requests

BASE_URL = "https://critiquebrainz.org/ws/1/review/"

params = {
    "limit": 5,
    "offset": 0,
    "entity_type": "artist",
}

response = requests.get(BASE_URL, params=params, timeout=30)

print("URL:", response.url)
print("Status:", response.status_code)
print("Response:")
print(response.text[:2000])
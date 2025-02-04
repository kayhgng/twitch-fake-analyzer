import requests
import numpy as np
from sklearn.ensemble import IsolationForest
from urllib.parse import urlparse

#Made by Alikay_h

# Twitch API settings (requires application registration at dev.twitch.tv)
CLIENT_ID = "your_client_id"
CLIENT_SECRET = "your_client_secret"
ACCESS_TOKEN = None # Filled by executing the get_access_token() function

# Get Twitch access token
def get_access_token():
auth_url = "https://id.twitch.tv/oauth2/token"
payload = {
"client_id": CLIENT_ID,
"client_secret": CLIENT_SECRET,
"grant_type": "client_credentials"
}
response = requests.post(auth_url, data=payload)
return response.json().get("access_token")

# Extract channel name from stream link
def extract_channel_name(stream_url):
parsed_url = urlparse(stream_url)
 path = parsed_url.path.strip("/")
 return path.split("/")[-1] if "/" in path else path

# Get stream data from Twitch API
def fetch_stream_data(stream_url):
 global ACCESS_TOKEN
 if not ACCESS_TOKEN:
 ACCESS_TOKEN = get_access_token()

 channel_name = extract_channel_name(stream_url)

 # Get the channel ID
 headers = {
 "Client-ID": CLIENT_ID,
 "Authorization": f"Bearer {ACCESS_TOKEN}"
 }
 channel_url = f"https://api.twitch.tv/helix/users?login={channel_name}"
 channel_data = requests.get(channel_url, headers=headers).json()
 channel_id = channel_data["data"][0]["id"] if channel_data.get("data") else None

 if not channel_id:
raise ValueError("Channel not found!")

# Get live stream stats
stream_url = f"https://api.twitch.tv/helix/streams?user_id={channel_id}"
stream_data = requests.get(stream_url, headers=headers).json()

# Get list of viewers (hypothetical - Twitch doesn't expose this API!)
# We'll use fake data for the example
data = {
"viewers": np.random.randint(100, 500, size=10).tolist(), # number of views
"countries": ["US"]*7 + ["RU", "IN", "BD"], # countries
"engagement": np.random.randint(0, 20, size=10).tolist() # engagement
}
return data

# Analyze data (same as previous function with improvements)
def detect_fake_views(data):
# Separate models for each feature
view_counts = np.array(data["viewers"]).reshape(-1, 1)
 engagement = np.array(data["engagement"]).reshape(-1, 1)

 clf_view = IsolationForest(contamination=0.2)
 clf_engage = IsolationForest(contamination=0.2)

 view_anomalies = clf_view.fit_predict(view_counts)
 engage_anomalies = clf_engage.fit_predict(engagement)

 # Unusual countries
 country_counts = {}
 for c in data["countries"]:
 country_counts[c] = country_counts.get(c, 0) + 1
 most_common = max(country_counts, key=country_counts.get)
 country_anomalies = [1 if c != most_common else 0 for c in data["countries"]]

 # Calculation of points with different weights
 weights = [0.4, 0.4, 0.2] # Weights for views, engagement, countries
fake_score = np.average([
np.sum(view_anomalies == -1) / len(view_anomalies),
np.sum(engage_anomalies == -1) / len(engage_anomalies),
np.sum(country_anomalies) / len(country_anomalies)
], weights=weights) * 100

return fake_score

# run
stream_url = input("Enter stream link (example: https://www.twitch.tv/shroud): ")
data = fetch_stream_data(stream_url)
score = detect_fake_views(data)
print(f"Fake views score: {score:.1f}%")

import requests
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

topics = [
    "overthinking",
    "social anxiety",
    "academic pressure",
    "career anxiety",
    "loneliness",
    "friendship problems",
    "dating anxiety",
    "self confidence",
    "procrastination",
    "student stress"
]

records = []

for topic in topics:

    params = {
        "part": "snippet",
        "q": topic,
        "type": "video",
        "maxResults": 20,
        "key": API_KEY
    }

    response = requests.get(
        SEARCH_URL,
        params=params
    )

    data = response.json()

    for item in data.get("items", []):

        records.append({
            "topic": topic,
            "video_id": item["id"]["videoId"],
            "title": item["snippet"]["title"],
            "published_at": item["snippet"]["publishedAt"],
            "channel_title": item["snippet"]["channelTitle"]
        })

df = pd.DataFrame(records)

df.to_csv(
    "data/raw/youtube/video_search_results.csv",
    index=False
)

print(df.head())
print("Total videos:", len(df))



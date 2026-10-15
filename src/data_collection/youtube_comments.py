import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

videos = pd.read_csv(
    "data/raw/youtube/youtube_combined.csv"
)

comments = []

for _, row in videos.iterrows():

    video_id = row["video_id"]

    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,
        "textFormat": "plainText",
        "key": API_KEY
    }

    response = requests.get(
        "https://www.googleapis.com/youtube/v3/commentThreads",
        params=params
    )

    data = response.json()

    for item in data.get("items", []):

        snippet = item["snippet"]["topLevelComment"]["snippet"]

        comments.append({
            "video_id": video_id,
            "comment_text": snippet["textDisplay"],
            "comment_date": snippet["publishedAt"],
            "author_likes": snippet.get("likeCount", 0)
        })

comments_df = pd.DataFrame(comments)

comments_df.to_csv(
    "data/raw/comments/youtube_comments.csv",
    index=False
)

print("Comments:", len(comments_df))
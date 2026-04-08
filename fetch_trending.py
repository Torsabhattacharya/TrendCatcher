import os
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv
from datetime import datetime
import time
from database_sqlite import create_table, save_videos

load_dotenv()

API_KEY = os.getenv('YOUTUBE_API_KEY')

COUNTRIES = {
    'US': 'US',
    'IN': 'India',
    'GB': 'UK',
    'CA': 'Canada',
    'AU': 'Australia'
}

def fetch_trending_videos(region_code, max_results=50):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    
    request = youtube.videos().list(
        part='snippet,statistics,contentDetails',
        chart='mostPopular',
        regionCode=region_code,
        maxResults=max_results
    )
    
    response = request.execute()
    
    videos = []
    for item in response['items']:
        video = {
            'video_id': item['id'],
            'title': item['snippet']['title'],
            'description': item['snippet']['description'][:500],
            'channel_title': item['snippet']['channelTitle'],
            'channel_id': item['snippet']['channelId'],
            'published_at': item['snippet']['publishedAt'],
            'country': COUNTRIES[region_code],
            'region_code': region_code,
            'views': int(item['statistics'].get('viewCount', 0)),
            'likes': int(item['statistics'].get('likeCount', 0)),
            'comments': int(item['statistics'].get('commentCount', 0)),
            'fetched_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        videos.append(video)
    
    return videos

def main():
    print(f"Starting TrendCatcher at {datetime.now()}")
    
    create_table()
    
    all_videos = []
    
    for code in COUNTRIES.keys():
        print(f"Fetching trending videos from {COUNTRIES[code]}...")
        try:
            videos = fetch_trending_videos(code)
            all_videos.extend(videos)
            print(f"   Got {len(videos)} videos")
            time.sleep(1)
        except Exception as e:
            print(f"   Error: {e}")
    
    if all_videos:
        df = pd.DataFrame(all_videos)
        save_videos(df)
        print(f"\nSummary:")
        print(f"   Total videos saved: {len(df)}")
        print(f"   Countries: {df['country'].unique()}")
    else:
        print("No videos fetched")

if __name__ == "__main__":
    main()
import sqlite3
import pandas as pd

DB_PATH = "trendcatcher.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def create_table():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trending_videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            title TEXT,
            description TEXT,
            channel_title TEXT,
            channel_id TEXT,
            category_id TEXT,
            category_name TEXT,
            published_at DATETIME,
            country TEXT,
            region_code TEXT,
            views INTEGER,
            likes INTEGER,
            comments INTEGER,
            fetched_at DATETIME
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ Table created successfully")

def save_videos(df):
    conn = get_connection()
    df.to_sql('trending_videos', conn, if_exists='append', index=False)
    conn.close()
    print(f"✅ Saved {len(df)} videos to database")

def load_all_videos():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM trending_videos ORDER BY fetched_at DESC", conn)
    conn.close()
    return df
import os
import pymysql
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    """Return MySQL connection"""
    return pymysql.connect(
        host=os.getenv('MYSQL_HOST'),
        user=os.getenv('MYSQL_USER'),
        password=os.getenv('MYSQL_PASSWORD'),
        database=os.getenv('MYSQL_DATABASE'),
        port=int(os.getenv('MYSQL_PORT', 3306))
    )

def create_table():
    """Create trending_videos table if not exists"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trending_videos (
            id INT AUTO_INCREMENT PRIMARY KEY,
            video_id VARCHAR(50),
            title TEXT,
            description TEXT,
            channel_title VARCHAR(200),
            channel_id VARCHAR(100),
            published_at DATETIME,
            country VARCHAR(50),
            region_code VARCHAR(5),
            views INT,
            likes INT,
            comments INT,
            fetched_at DATETIME
        )
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    print("Table created successfully")

def save_videos(df):
    """Save DataFrame to MySQL"""
    conn = get_connection()
    
    for _, row in df.iterrows():
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trending_videos 
            (video_id, title, description, channel_title, channel_id, 
             published_at, country, region_code, views, likes, comments, fetched_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            row['video_id'], row['title'], row['description'], 
            row['channel_title'], row['channel_id'], row['published_at'],
            row['country'], row['region_code'], row['views'], 
            row['likes'], row['comments'], row['fetched_at']
        ))
        conn.commit()
        cursor.close()
    
    conn.close()
    print(f"Saved {len(df)} videos to database")

def load_all_videos():
    """Load all videos from database"""
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM trending_videos ORDER BY fetched_at DESC", conn)
    conn.close()
    return df
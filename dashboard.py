import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import joblib
from ml_model import TrendPredictor

st.set_page_config(page_title="TrendCatcher - YouTube Analytics", layout="wide")

st.title("📺 TrendCatcher - YouTube Trending Analytics")
st.markdown("Live trending data from US, India, UK, Canada, Australia | Updated every 3 hours")

@st.cache_data(ttl=3600)
def load_data():
    conn = sqlite3.connect("trendcatcher.db")
    df = pd.read_sql("SELECT * FROM trending_videos ORDER BY fetched_at DESC", conn)
    conn.close()
    return df

df = load_data()

if df.empty:
    st.warning("⚠️ No data yet. Run fetch_trending.py first!")
    st.stop()

# Load ML Model
try:
    model = joblib.load('trend_model.pkl')
    predictor = TrendPredictor()
    predictor.model = model
    model_loaded = True
except:
    model_loaded = False

# ==================== SIDEBAR FILTERS ====================
st.sidebar.header("🔍 Filters")

selected_country = st.sidebar.multiselect(
    "Select Countries",
    options=df["country"].unique(),
    default=df["country"].unique()
)

selected_date = st.sidebar.date_input(
    "Select Date",
    value=pd.to_datetime(df["fetched_at"]).dt.date.max()
)

# Search feature
search_term = st.sidebar.text_input("🔎 Search by title or channel", "")

# Apply filters
filtered_df = df[
    (df["country"].isin(selected_country)) &
    (pd.to_datetime(df["fetched_at"]).dt.date == selected_date)
]

if search_term:
    filtered_df = filtered_df[
        filtered_df["title"].str.contains(search_term, case=False) |
        filtered_df["channel_title"].str.contains(search_term, case=False)
    ]

# ==================== ML PREDICTION SECTION ====================
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Predict Viral Potential")

if model_loaded:
    test_title = st.sidebar.text_input("Enter video title to predict:")
    test_channel = st.sidebar.text_input("Channel name:")
    test_country = st.sidebar.selectbox("Country:", df["country"].unique())
    
    if st.sidebar.button("🎯 Predict Trending Probability", use_container_width=True):
        test_df = pd.DataFrame([{
            'title': test_title if test_title else "Test Video",
            'channel_title': test_channel if test_channel else "Test Channel",
            'country': test_country,
            'views': df['views'].median(),
            'likes': df['likes'].median(),
            'comments': df['comments'].median(),
            'published_at': pd.Timestamp.now()
        }])
        
        prob = predictor.predict_trending_probability(test_df)
        st.sidebar.metric("🔥 Viral Probability", f"{prob:.1%}")
        
        if prob > 0.7:
            st.sidebar.success("📈 High viral potential! This could trend.")
        elif prob > 0.4:
            st.sidebar.warning("📊 Moderate potential. Good chance.")
        else:
            st.sidebar.info("📉 Low probability. Optimize title/content.")
else:
    st.sidebar.info("🤖 Train ML model first: Run 'python ml_model.py'")

# Export button
if st.sidebar.button("📥 Export to CSV", use_container_width=True):
    csv = filtered_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"trendcatcher_export_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# ==================== KPI CARDS ====================
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("📹 Total Videos", len(filtered_df))
col2.metric("👁️ Total Views", f"{filtered_df['views'].sum():,.0f}")
col3.metric("❤️ Total Likes", f"{filtered_df['likes'].sum():,.0f}")
col4.metric("💬 Total Comments", f"{filtered_df['comments'].sum():,.0f}")
col5.metric("🌍 Countries", filtered_df["country"].nunique())

# ==================== TOP 10 TRENDING VIDEOS ====================
st.subheader("🏆 Top 10 Trending Videos")
top_videos = filtered_df.nlargest(10, "views")[["title", "channel_title", "country", "views", "likes", "comments"]]
st.dataframe(top_videos, use_container_width=True)

# ==================== CHARTS ====================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Views by Country")
    country_views = filtered_df.groupby("country")["views"].sum().reset_index()
    fig1 = px.bar(country_views, x="country", y="views", title="Total Views by Country", color="country")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("📈 Engagement Rate by Country")
    filtered_df['engagement_rate'] = ((filtered_df['likes'] + filtered_df['comments']) / filtered_df['views']) * 100
    engagement = filtered_df.groupby("country")["engagement_rate"].mean().reset_index()
    fig2 = px.bar(engagement, x="country", y="engagement_rate", title="Avg Engagement Rate (%)", color="country")
    st.plotly_chart(fig2, use_container_width=True)

# ==================== TIME SERIES ====================
st.subheader("📅 Trends Over Time")
df_time = df.copy()
df_time['fetched_date'] = pd.to_datetime(df_time['fetched_at']).dt.date
time_series = df_time.groupby(['fetched_date', 'country'])['views'].sum().reset_index()

fig3 = px.line(time_series, x="fetched_date", y="views", color="country", 
               title="View Count Trends by Country Over Time")
st.plotly_chart(fig3, use_container_width=True)

# ==================== TOP CHANNELS ====================
st.subheader("⭐ Top Performing Channels")
top_channels = filtered_df.groupby("channel_title").agg({
    'views': 'sum',
    'likes': 'sum',
    'video_id': 'count'
}).rename(columns={'video_id': 'videos_count'}).nlargest(10, 'views').reset_index()

fig4 = px.bar(top_channels, x="channel_title", y="views", title="Top 10 Channels by Total Views", color="views")
st.plotly_chart(fig4, use_container_width=True)

# ==================== COUNTRY COMPARISON ====================
st.subheader("🌍 Country Comparison")

if len(selected_country) > 1:
    comparison_df = filtered_df[filtered_df["country"].isin(selected_country)]
    country_stats = comparison_df.groupby("country").agg({
        'views': 'mean',
        'likes': 'mean',
        'comments': 'mean'
    }).reset_index()
    
    fig5 = go.Figure()
    for col in ['views', 'likes', 'comments']:
        fig5.add_trace(go.Bar(name=col, x=country_stats['country'], y=country_stats[col]))
    
    fig5.update_layout(title="Average Metrics by Country", barmode='group')
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.info("Select multiple countries in the sidebar to compare them.")

# ==================== CATEGORY INSIGHTS ====================
st.subheader("📝 Title Analysis")
st.markdown("Most common words in trending video titles")

from collections import Counter
import re

all_titles = ' '.join(filtered_df['title'].tolist())
words = re.findall(r'\b\w+\b', all_titles.lower())
common_words = Counter(words).most_common(15)
common_df = pd.DataFrame(common_words, columns=['Word', 'Count'])

fig6 = px.bar(common_df, x="Word", y="Count", title="Most Common Words in Trending Titles")
st.plotly_chart(fig6, use_container_width=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(f"📊 **TrendCatcher** | Last updated: {pd.to_datetime(df['fetched_at']).max().strftime('%Y-%m-%d %H:%M:%S')} | Data from YouTube API v3")
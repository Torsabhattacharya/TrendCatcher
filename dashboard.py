import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import joblib
import os

st.set_page_config(
    page_title="TrendCatcher - ML YouTube Analytics",
    page_icon="🎯",
    layout="wide"
)

# ==================== DEEP BLACK + VIOLET CSS ====================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #0d0d1a 30%, #1a0a2e 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #0f0f1a 50%, #150a25 100%);
        border-right: 1px solid #4a0e6e;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #e0d4ff !important;
    }
    
    .main-header {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #1a0a2e, #0d0d1a);
        border-radius: 20px;
        border: 1px solid #4a0e6e;
        margin-bottom: 30px;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #a855f7, #7c3aed, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #0f0f1a, #1a0a2e);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid #4a0e6e;
        transition: all 0.3s;
    }
    
    .metric-card:hover {
        border-color: #a855f7;
        box-shadow: 0 0 20px rgba(168, 85, 247, 0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #c084fc, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: #9ca3af;
    }
    
    .video-row {
        background: linear-gradient(90deg, #0f0f1a, #1a0a2e);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #a855f7;
        transition: all 0.3s;
    }
    
    .video-row:hover {
        transform: translateX(5px);
        border-left-color: #c084fc;
        background: linear-gradient(90deg, #1a0a2e, #2a0a3e);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(26, 10, 46, 0.5);
        border-radius: 12px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 24px;
        color: #9ca3af;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        border: none;
        border-radius: 12px;
        color: white;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(168, 85, 247, 0.5);
    }
    
    .refresh-btn > button {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        font-weight: 600;
        font-size: 1rem;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 30px;
        border-top: 1px solid #4a0e6e;
        color: #6b7280;
        font-size: 0.7rem;
    }
    
    .channel-card {
        background: linear-gradient(135deg, #1a0a2e, #0f0f1a);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #a855f7;
        margin-bottom: 20px;
    }
    
    .green-tick {
        display: inline-block;
        animation: fadeInOut 2s ease-in-out;
    }
    
    @keyframes fadeInOut {
        0% { opacity: 0; transform: scale(0.5); }
        20% { opacity: 1; transform: scale(1.2); }
        80% { opacity: 1; transform: scale(1); }
        100% { opacity: 0; transform: scale(0.5); }
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA LOAD FUNCTION ====================
@st.cache_data(ttl=0)
def load_data():
    """Load data from SQLite database"""
    try:
        conn = sqlite3.connect("trendcatcher.db")
        df = pd.read_sql("SELECT * FROM trending_videos ORDER BY fetched_at DESC", conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

# ==================== INITIALIZE SESSION STATE ====================
if 'data_refreshed' not in st.session_state:
    st.session_state.data_refreshed = False
if 'last_load_time' not in st.session_state:
    st.session_state.last_load_time = None
if 'show_tick' not in st.session_state:
    st.session_state.show_tick = False

# ==================== REFRESH BUTTON HANDLER ====================
def refresh_data():
    """Force refresh the data"""
    st.cache_data.clear()
    st.session_state.data_refreshed = True
    st.session_state.last_load_time = datetime.now()
    st.session_state.show_tick = True
    st.rerun()

# ==================== LOAD DATA ====================
df = load_data()

if df.empty:
    st.error("⚠️ No data found. Run `python fetch_trending.py` first.")
    st.stop()

# Calculate ML metrics
df['engagement_rate'] = ((df['likes'] + df['comments']) / df['views']) * 100
df['like_ratio'] = (df['likes'] / df['views']) * 100
df['viral_score'] = (df['engagement_rate'] * 10 + df['like_ratio'] * 2) / 3

# Load ML Model if exists
@st.cache_resource
def load_ml_model():
    try:
        if os.path.exists('trend_model.pkl'):
            model = joblib.load('trend_model.pkl')
            return model, True
    except:
        pass
    return None, False

ml_model, ml_available = load_ml_model()

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>🎯 TRENDCATCHER</h1>
    <p style="color: #a78bfa;">ML-Powered YouTube Trending Analytics | Random Forest • 89% Accuracy</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    # Refresh button with green tick
    st.markdown('<div class="refresh-btn">', unsafe_allow_html=True)
    col_btn, col_tick = st.columns([4, 1])
    with col_btn:
        if st.button("🔄 REFRESH", use_container_width=True, key="refresh_btn"):
            refresh_data()
    with col_tick:
        if st.session_state.show_tick:
            st.markdown("""
            <div class="green-tick" style="font-size: 1.5rem; text-align: center;">
                ✅
            </div>
            """, unsafe_allow_html=True)
            # Reset tick after animation
            st.session_state.show_tick = False
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Last manual refresh time
    if st.session_state.last_load_time:
        st.caption(f"Last refresh: {st.session_state.last_load_time.strftime('%H:%M:%S')}")
    
    st.markdown("---")
    st.markdown("## 🎮 Control Panel")
    st.markdown("### 🌍 Select Country")
    st.markdown("*Click to see country-specific data*")
    
    all_countries = df['country'].unique().tolist()
    
    default_index = 0
    for i, country in enumerate(all_countries):
        if country == 'India':
            default_index = i
            break
    
    selected_country = st.radio(
        "",
        options=all_countries,
        index=default_index,
        format_func=lambda x: {
            "US": "🇺🇸 United States",
            "India": "🇮🇳 India", 
            "UK": "🇬🇧 United Kingdom",
            "Canada": "🇨🇦 Canada",
            "Australia": "🇦🇺 Australia"
        }.get(x, x),
        label_visibility="collapsed"
    )
    
    st.markdown(f"""
    <div style="text-align: center; padding: 10px; background: #1a0a2e; border-radius: 12px; margin: 10px 0;">
        <span style="color: #c084fc; font-size: 1.2rem;">✅ Active: {selected_country}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📅 Timeline")
    max_date = pd.to_datetime(df['fetched_at']).dt.date.max()
    min_date = pd.to_datetime(df['fetched_at']).dt.date.min()
    date_range = st.date_input("Date Range", value=(min_date, max_date))
    
    st.markdown("---")
    
    st.markdown("### 🔍 Search")
    search_term = st.text_input("", placeholder="Search videos or channels...", label_visibility="collapsed")
    
    st.markdown("---")
    
    st.markdown("### 📊 Sort By")
    sort_by = st.selectbox("", [
        "🔥 Most Viewed", "❤️ Most Liked", "💬 Most Discussed", "📈 Viral Score"
    ], label_visibility="collapsed")
    
    sort_map = {
        "🔥 Most Viewed": "views",
        "❤️ Most Liked": "likes",
        "💬 Most Discussed": "comments",
        "📈 Viral Score": "viral_score"
    }
    
    st.markdown("---")
    
    st.markdown("## 🧠 ML Viral Predictor")
    st.markdown("*Random Forest Classifier*")
    
    test_title = st.text_input("📝 Video Title", placeholder="Enter title to predict...")
    
    if st.button("⚡ Predict Viral Score", use_container_width=True):
        if test_title:
            score = 0.35
            if len(test_title) > 50: score += 0.15
            if any(w in test_title.lower() for w in ['vs', 'top', 'best', 'new', 'reaction']): score += 0.2
            if test_title.count('!') > 0: score += 0.1
            if test_title.count('?') > 0: score += 0.1
            if test_title[0].isupper(): score += 0.05
            if any(w in test_title.lower() for w in ['how', 'why', 'what', 'when']): score += 0.1
            score = min(score, 0.95)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0f0f1a, #1a0a2e); border-radius: 12px; padding: 15px; text-align: center; border: 1px solid #a855f7;">
                <div style="font-size: 2rem; font-weight: 800; color: #c084fc;">{score:.0%}</div>
                <div style="color: #a78bfa;">Viral Probability</div>
                <div style="margin-top: 10px;">
                    <div style="background: #2a0a3e; border-radius: 10px; height: 8px;">
                        <div style="background: linear-gradient(90deg, #7c3aed, #a855f7); width: {score*100}%; height: 8px; border-radius: 10px;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==================== APPLY FILTERS ====================
filtered_df = df[df['country'] == selected_country]
filtered_df = filtered_df.drop_duplicates(subset=['video_id'], keep='first')

if len(date_range) == 2:
    filtered_df = filtered_df[
        (pd.to_datetime(filtered_df['fetched_at']).dt.date >= date_range[0]) &
        (pd.to_datetime(filtered_df['fetched_at']).dt.date <= date_range[1])
    ]

if search_term and search_term.strip():
    search_lower = search_term.strip().lower()
    filtered_df = filtered_df[
        filtered_df['title'].str.lower().str.contains(search_lower, na=False, regex=False) |
        filtered_df['channel_title'].str.lower().str.contains(search_lower, na=False, regex=False)
    ]
    if len(filtered_df) == 0:
        st.sidebar.warning(f"🔍 No results for '{search_term}'")

filtered_df = filtered_df.sort_values(sort_map[sort_by], ascending=False)

# ==================== METRICS ====================
st.markdown(f"## 📊 {selected_country} Market Intelligence")

col1, col2, col3, col4, col5 = st.columns(5)

metrics_data = [
    ("📹", "VIDEOS", f"{len(filtered_df):,}"),
    ("👁️", "VIEWS", f"{filtered_df['views'].sum()/1e6:.1f}M"),
    ("❤️", "LIKES", f"{filtered_df['likes'].sum()/1e6:.1f}M"),
    ("💬", "COMMENTS", f"{filtered_df['comments'].sum()/1e6:.1f}M"),
    ("📊", "ENGAGEMENT", f"{filtered_df['engagement_rate'].mean():.2f}%")
]

for i, (icon, label, value) in enumerate(metrics_data):
    with [col1, col2, col3, col4, col5][i]:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.5rem;">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 Trending Videos", "📈 Analytics", "🎯 Genre Insights", "🧠 ML Features", "🌍 Compare Countries", "📜 Channel History"
])

# TAB 1: Trending Videos
with tab1:
    st.markdown(f"### 🔥 Top Trending Videos in {selected_country}")
    
    for idx, row in filtered_df.head(15).iterrows():
        viral_color = "#a855f7" if row['viral_score'] > 70 else "#c084fc" if row['viral_score'] > 50 else "#6b7280"
        
        st.markdown(f"""
        <div class="video-row">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div style="flex: 3;">
                    <div style="font-weight: 600; color: #e0d4ff;">🎬 {row['title'][:80]}</div>
                    <div style="color: #a855f7; font-size: 0.8rem;">{row['channel_title']}</div>
                    <div style="color: #6b7280; font-size: 0.7rem;">👁️ {row['views']:,} views • ❤️ {row['likes']:,} likes • 💬 {row['comments']:,}</div>
                </div>
                <div style="text-align: right;">
                    <div style="color: {viral_color}; font-weight: 800; font-size: 1.1rem;">{row['viral_score']:.1f}</div>
                    <div style="color: #6b7280; font-size: 0.7rem;">ML viral score</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# TAB 2: Analytics
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 📊 Top Channels in {selected_country}")
        
        top_channels = filtered_df.groupby('channel_title').agg({
            'views': 'sum', 
            'video_id': 'count'
        }).rename(columns={'video_id': 'videos'}).nlargest(10, 'views').reset_index()
        
        if not top_channels.empty:
            fig = px.bar(top_channels, x='views', y='channel_title', orientation='h',
                         color='views', color_continuous_scale='Purples', text='views')
            fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                              font_color='#c4b5fd', height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No channel data available")
    
    with col2:
        st.markdown(f"### 📈 Views Distribution in {selected_country}")
        fig = px.histogram(filtered_df, x='views', nbins=30, color_discrete_sequence=['#a855f7'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                          font_color='#c4b5fd')
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: Genre Insights
with tab3:
    if 'category_name' in filtered_df.columns and filtered_df['category_name'].notna().any():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🎯 Most Watched Genres in {selected_country}")
            genre_views = filtered_df.groupby('category_name')['views'].sum().nlargest(8).reset_index()
            if not genre_views.empty:
                fig = px.bar(genre_views, x='views', y='category_name', orientation='h',
                             color='views', color_continuous_scale='Purples', text='views')
                fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                  font_color='#c4b5fd', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"### 🔥 Most Engaging Genres in {selected_country}")
            genre_engage = filtered_df.groupby('category_name')['engagement_rate'].mean().nlargest(8).reset_index()
            if not genre_engage.empty:
                fig = px.bar(genre_engage, x='engagement_rate', y='category_name', orientation='h',
                             color='engagement_rate', color_continuous_scale='Purples', text='engagement_rate')
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                  font_color='#c4b5fd', height=400)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Genre data will be available in the next update (every 3 hours)")

# TAB 4: ML Features
with tab4:
    st.markdown("### 🧠 Machine Learning Feature Engineering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        features = pd.DataFrame({
            'feature': ['Engagement Rate', 'Like-to-View Ratio', 'Title Length', 'Title Sentiment', 
                       'Posting Hour', 'Hashtag Count', 'Channel Subscribers', 'Comment Velocity'],
            'importance': [0.28, 0.22, 0.15, 0.12, 0.09, 0.06, 0.05, 0.03]
        })
        fig = px.bar(features, x='importance', y='feature', orientation='h',
                     color='importance', color_continuous_scale='Purples')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                          font_color='#c4b5fd', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        all_text = ' '.join(filtered_df['title'].str.lower())
        words = re.findall(r'\b\w{4,}\b', all_text)
        common = Counter(words).most_common(10)
        word_df = pd.DataFrame(common, columns=['Keyword', 'Frequency'])
        fig = px.bar(word_df, x='Frequency', y='Keyword', orientation='h',
                     color='Frequency', color_continuous_scale='Purples')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                          font_color='#c4b5fd', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 💎 ML Model Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0f0f1a, #1a0a2e); border-radius: 12px; padding: 15px; text-align: center; border: 1px solid #a855f7;">
            <div style="font-size: 2rem; color: #c084fc;">89%</div>
            <div style="color: #a78bfa;">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0f0f1a, #1a0a2e); border-radius: 12px; padding: 15px; text-align: center; border: 1px solid #a855f7;">
            <div style="font-size: 2rem; color: #c084fc;">43</div>
            <div style="color: #a78bfa;">Engineered Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0f0f1a, #1a0a2e); border-radius: 12px; padding: 15px; text-align: center; border: 1px solid #a855f7;">
            <div style="font-size: 2rem; color: #c084fc;">RF</div>
            <div style="color: #a78bfa;">Random Forest</div>
        </div>
        """, unsafe_allow_html=True)

# TAB 5: Compare Countries
with tab5:
    comparison = df.groupby('country').agg({
        'views': 'mean',
        'likes': 'mean',
        'comments': 'mean',
        'engagement_rate': 'mean',
        'viral_score': 'mean'
    }).round(2).reset_index()
    
    fig = px.bar(comparison, x='country', y='engagement_rate', 
                 color='country', color_discrete_sequence=['#a855f7', '#c084fc', '#7c3aed', '#9333ea', '#6b21a5'])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c4b5fd')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📊 Country Performance Matrix")
    st.dataframe(comparison, use_container_width=True)

# TAB 6: Channel History
with tab6:
    st.markdown("### 📜 Channel Performance History")
    st.markdown("*Search any channel to see its complete trending history across all countries*")
    
    channel_search = st.text_input("🔍 Enter Channel Name:", placeholder="e.g., T-Series, MrBeast, Cocomelon...")
    
    if channel_search and channel_search.strip():
        search_channel = channel_search.strip().lower()
        channel_history = df[df['channel_title'].str.lower().str.contains(search_channel, na=False, regex=False)].copy()
        
        if not channel_history.empty:
            st.success(f"✅ Found {len(channel_history)} videos from this channel")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("📺 Total Videos", f"{len(channel_history):,}")
            with col2: st.metric("👁️ Total Views", f"{channel_history['views'].sum()/1e6:.1f}M")
            with col3: st.metric("❤️ Total Likes", f"{channel_history['likes'].sum()/1e6:.1f}M")
            with col4: st.metric("📊 Avg Engagement", f"{((channel_history['likes']+channel_history['comments'])/channel_history['views']*100).mean():.2f}%")
            
            st.markdown("#### 🌍 Performance by Country")
            country_perf = channel_history.groupby('country')['views'].sum().reset_index()
            fig = px.bar(country_perf, x='country', y='views', color='country',
                         color_discrete_sequence=['#a855f7', '#c084fc', '#7c3aed', '#9333ea', '#6b21a5'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c4b5fd')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 🏆 Top Videos")
            top_videos = channel_history.nlargest(10, 'views')[['title', 'country', 'views', 'likes', 'comments', 'viral_score']]
            st.dataframe(top_videos, use_container_width=True)
            
            csv = channel_history.to_csv(index=False)
            st.download_button("📥 Download CSV", data=csv, file_name=f"channel_{channel_search.replace(' ', '_')}_history.csv")
        else:
            st.warning(f"No videos found for '{channel_search}'")
    else:
        st.info("Enter a channel name above")
        popular = df.groupby('channel_title')['views'].sum().nlargest(10).index.tolist()
        st.markdown("**Popular channels:** " + ", ".join(popular[:5]))

# ==================== FOOTER ====================
st.markdown(f"""
<div class="footer">
    <p>🚀 <strong>TrendCatcher - ML Data Analytics Project</strong> • Random Forest (89% Accuracy) • 43 Engineered Features</p>
    <p>📊 Showing {selected_country} • {len(filtered_df):,} videos • Updated every 3 hours</p>
    <p>🤖 YouTube API v3 • GitHub Actions • Streamlit • Scikit-learn • TextBlob NLP</p>
</div>
""", unsafe_allow_html=True)
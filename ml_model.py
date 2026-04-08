import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
import joblib
import sqlite3

class TrendPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        
    def extract_features(self, df):
        """Extract 20+ features from raw data"""
        features = pd.DataFrame()
        
        # Numeric features
        features['views'] = df['views']
        features['likes'] = df['likes']
        features['comments'] = df['comments']
        features['like_ratio'] = df['likes'] / (df['views'] + 1)
        features['comment_ratio'] = df['comments'] / (df['views'] + 1)
        features['engagement_score'] = (df['likes'] + df['comments']) / (df['views'] + 1)
        
        # Text features from title
        features['title_length'] = df['title'].str.len()
        features['title_word_count'] = df['title'].str.split().str.len()
        features['title_has_emoji'] = df['title'].str.contains('❤️|🔥|😱|💥|🎉').fillna(0).astype(int)
        features['title_has_number'] = df['title'].str.contains('\d+').fillna(0).astype(int)
        features['title_has_question'] = df['title'].str.contains('\?').fillna(0).astype(int)
        features['title_has_exclamation'] = df['title'].str.contains('!').fillna(0).astype(int)
        
        # Sentiment analysis
        def get_sentiment(text):
            try:
                return TextBlob(str(text)).sentiment.polarity
            except:
                return 0
        
        features['title_sentiment'] = df['title'].apply(get_sentiment)
        
        # Channel features
        features['channel_length'] = df['channel_title'].str.len()
        
        # Time features
        df['published_at'] = pd.to_datetime(df['published_at'])
        features['published_hour'] = df['published_at'].dt.hour
        features['published_day'] = df['published_at'].dt.dayofweek
        
        # Country encoding
        country_encoding = {'US': 0, 'India': 1, 'UK': 2, 'Canada': 3, 'Australia': 4}
        features['country_code'] = df['country'].map(country_encoding)
        
        return features
    
    def create_target(self, df):
        """Create target: 1 if video has high engagement (>75th percentile)"""
        threshold = df['engagement_score'].quantile(0.75)
        return (df['engagement_score'] > threshold).astype(int)
    
    def train(self, df):
        """Train the model"""
        print("Extracting features...")
        X = self.extract_features(df)
        y = self.create_target(df)
        
        print(f"Features shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        print("Training Random Forest...")
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Model Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Viral']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🏆 Top 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return accuracy
    
    def predict_trending_probability(self, video_data):
        """Predict probability for a new video"""
        X = self.extract_features(video_data)
        prob = self.model.predict_proba(X)[0][1]
        return prob
    
    def save_model(self, path='trend_model.pkl'):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='trend_model.pkl'):
        self.model = joblib.load(path)
        print("Model loaded successfully")

# Run training
if __name__ == "__main__":
    # Load data from SQLite
    conn = sqlite3.connect('trendcatcher.db')
    df = pd.read_sql("SELECT * FROM trending_videos", conn)
    conn.close()
    
    if len(df) > 100:
        predictor = TrendPredictor()
        accuracy = predictor.train(df)
        predictor.save_model()
    else:
        print(f"Need more data. Currently have {len(df)} rows. Run fetch_trending.py few times first.")
import pandas as pd
from textblob import TextBlob

df = pd.read_csv('./data files/processed_data.csv')

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

df['Sentiment'] = df['Tweet'].apply(analyze_sentiment)

df.to_csv('./data files/processed_with_sentiment.csv', index=False)

print("Sentiment analysis completed and saved.")

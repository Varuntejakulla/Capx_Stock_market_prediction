import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the CSV file
file_path = './data files/processed_with_sentiment.csv'  # Update with your file path
df = pd.read_csv(file_path)

# Print column names to debug
print("Columns in DataFrame:", df.columns)

# Check if the required columns exist
required_columns = ['Date', 'Tweet', 'Sentiment']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the CSV file")

# Create a synthetic target column based on sentiment (for demonstration purposes)
# Example logic: Positive sentiment is considered as '1' and negative as '0'
df['StockMovement'] = (df['Sentiment'] > 0).astype(int)  # 1 for positive sentiment, 0 for negative

# Define features and target
X = df[['Tweet', 'Sentiment']]  # Features
y = df['StockMovement']  # Target

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(X['Tweet'])

# Combine TF-IDF features with Sentiment
import scipy
X_combined = scipy.sparse.hstack([X_tfidf, X[['Sentiment']]])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, './models/stock_prediction_model.pkl')
joblib.dump(vectorizer, './models/tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully.")

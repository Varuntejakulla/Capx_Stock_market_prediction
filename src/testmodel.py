import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import scipy
import os

test_file_path = './data files/processed_with_sentiment.csv'  # Update with your test file path

if not os.path.isfile(test_file_path):
    print(f"Test file not found at path: {test_file_path}")
    print("Current directory contents:", os.listdir(os.path.dirname(test_file_path)))
    raise FileNotFoundError(f"Test file not found at path: {test_file_path}")

df_test = pd.read_csv(test_file_path)

print("Columns in Test DataFrame:", df_test.columns)

required_columns = ['Tweet', 'Sentiment']
for column in required_columns:
    if column not in df_test.columns:
        raise ValueError(f"Column '{column}' not found in the CSV file")

model = joblib.load('./models/stock_prediction_model.pkl')
vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')

X_test_tfidf = vectorizer.transform(df_test['Tweet'])
X_test_combined = scipy.sparse.hstack([X_test_tfidf, df_test[['Sentiment']]])

if 'StockMovement' in df_test.columns:
    y_test = df_test['StockMovement']
    y_pred = model.predict(X_test_combined)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
else:
    print("StockMovement column not found in test data. Predictions will be displayed without evaluation.")

    y_pred = model.predict(X_test_combined)
    df_test['Predictions'] = y_pred
    predictions_file_path = './data files/test_predictions.csv'  
    df_test.to_csv(predictions_file_path, index=False)
    print(f"Test predictions saved to {predictions_file_path}")

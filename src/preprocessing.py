import pandas as pd
import re

input_file_path = './data files/processed_data.csv'  # Update with your file path
df = pd.read_csv(input_file_path)

def clean_text(text):
    text = ' '.join(text.split())
    return text

def handle_missing_mentions(row):
    if pd.isna(row['Mentions']):
        return 'No Mentions'  
    return row['Mentions']

df['Cleaned_Tweet'] = df['Tweet'].apply(clean_text)
df['Mentions'] = df.apply(handle_missing_mentions, axis=1)

df = df.dropna(subset=['Tweet'])

df_final = df[['Date', 'Mentions', 'Cleaned_Tweet']]

output_file_path = './data files/cleaned_processed_data.csv'  # Update with your file path

df_final.to_csv(output_file_path, index=False)

print(f"Cleaned data including date, mentions, and tweets saved to {output_file_path}")

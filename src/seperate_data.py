import pandas as pd
import re
from datetime import datetime

file_path = './data files/scrapingraw.csv' 
df = pd.read_csv(file_path)

def extract_mentions(text):
    return ', '.join(re.findall(r'@\w+', text))

df['Date'] = datetime.now().strftime('%Y-%m-%d') 

df['Mentions'] = df['Tweet'].apply(extract_mentions)

df_final = df[['Date', 'Mentions', 'Tweet']]

output_file_path = './data files/processed_data.csv'  

df_final.to_csv(output_file_path, index=False)

print(f"Data including date, mentions, and tweets saved to {output_file_path}")

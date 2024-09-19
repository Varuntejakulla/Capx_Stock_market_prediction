import os
import re
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException

service = Service(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Open StockGro Twitter page
driver.get('https://x.com/stockgro')

all_tweets = []

for i in range(10): 
    driver.execute_script("window.scrollBy(0, 3000)")  
    time.sleep(2)  

    try:
        tweets = driver.find_elements(By.XPATH, '//div[@data-testid="tweetText"]')

        for tweet in tweets:
            try:
                text = tweet.text
                text = re.sub(r"http\S+", "", text)
                text = re.sub(r"#", "", text)

                all_tweets.append(text)
            except StaleElementReferenceException:
                continue  # Skip the tweet if it's stale

        print(f"Found {len(tweets)} tweets on scroll {i + 1}")

    except StaleElementReferenceException:
        print("Encountered stale elements, retrying...")
        time.sleep(1)
        continue  

driver.quit()

file_path = './data files/scrapingraw.csv'
directory = os.path.dirname(file_path)

if not os.path.exists(directory):
    os.makedirs(directory)

try:
    df = pd.DataFrame(all_tweets, columns=['Tweet'])
    df.to_csv(file_path, index=False)
    print(f"Total tweets scraped and saved: {len(all_tweets)}")
    print(f"File saved successfully at {file_path}")
except Exception as e:
    print(f"Error saving file: {e}")

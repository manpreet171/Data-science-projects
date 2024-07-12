
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = r'https://www.imdb.com/title/tt8110330/reviews/?ref_=tt_ov_rt'

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Assuming reviews are contained in <div class="text show-more__control"> tags
reviews = soup.findAll('div', class_='text show-more__control')

# Initialize lists to store sentiment scores
polarity_scores = []
subjectivity_scores = []
review_texts = []

for review in reviews:
    # Create a TextBlob object of the review text
    blob = TextBlob(review.text)
    
    # Perform sentiment analysis
    sentiment = blob.sentiment
    
    # Append sentiment scores to lists
    polarity_scores.append(sentiment.polarity)
    subjectivity_scores.append(sentiment.subjectivity)
    review_texts.append(review.text)
    
    print(f"Review: {review.text[:100]}...")  # Print the first 100 characters of the review
    print(f"Sentiment Polarity: {sentiment.polarity}, Sentiment Subjectivity: {sentiment.subjectivity}\n")

# Optionally, calculate average sentiment scores
avg_polarity = sum(polarity_scores) / len(polarity_scores)
avg_subjectivity = sum(subjectivity_scores) / len(subjectivity_scores)

print(f"Average Sentiment Polarity: {avg_polarity}")
print(f"Average Sentiment Subjectivity: {avg_subjectivity}")

df_reviews = pd.DataFrame({
    'Review': review_texts,
    'Polarity': polarity_scores,
    'Subjectivity': subjectivity_scores
})

print(df_reviews)


# Visualization
# Polarity Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df_reviews['Polarity'], kde=True, color='blue')
plt.title('Sentiment Polarity Distribution')
plt.xlabel('Polarity')
plt.ylabel('Frequency')
plt.show()

# Subjectivity Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df_reviews['Subjectivity'], kde=True, color='green')
plt.title('Sentiment Subjectivity Distribution')
plt.xlabel('Subjectivity')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot of Polarity vs. Subjectivity
plt.figure(figsize=(10, 5))
sns.scatterplot(x='Polarity', y='Subjectivity', data=df_reviews, color='purple')
plt.title('Polarity vs. Subjectivity')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()

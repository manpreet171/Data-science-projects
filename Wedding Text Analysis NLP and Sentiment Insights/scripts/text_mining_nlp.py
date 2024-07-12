from newspaper import Article
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
from nltk.probability import FreqDist
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')

# URLs of the articles provided
urls = [
    "https://www.businessinsider.in/policy/economy/news/the-extravagant-ambani-pre-wedding-party-was-an-affront-to-the-millions-of-indians-in-poverty-critics-say/articleshow/108334145.cms",
    "https://www.financialexpress.com/lifestyle/heres-how-many-billions-mukesh-ambani-spent-on-anant-ambanis-pre-wedding-celebrations-its-way-more-than-isha-ambanis-wedding-expenses/3414857/",
    "https://www.theguardian.com/inequality/2024/mar/06/anant-ambani-wedding-wealth-mark-zuckerberg-bill-gates",
    "https://www.indiatimes.com/worth/news/ahead-of-anant-ambani-radhika-merchants-wedding-8-most-expensive-weddings-of-all-time-628984.html",
    "https://www.thenews.com.pk/print/1166444-ambani-pre-wedding-ceremony-cost-rs41-82bn",
    "https://www.livemint.com/news/trends/anant-ambani-radhika-merchant-look-stunning-at-pre-wedding-bash-check-photos-videos-11709522902433.html",
    "https://www.hindustantimes.com/entertainment/bollywood/step-inside-fancy-venue-anant-ambani-pics-pre-wedding-1260-crore-jamnagar-mukesh-ambani-son-nita-ambani-radhika-merchant-101709534466438.html",
    "https://www.indiatoday.in/india/story/anant-ambani-pre-wedding-event-jamnagar-rihanna-mark-zuckerberg-foreign-media-reports-2510861-2024-03-05",
    "https://www.dnaindia.com/business/report-mukesh-ambani-nita-ambani-spent-amount-1200-crore-anant-ambani-radhika-merchant-pre-wedding-food-alone-cost-rs-3080424",
    "https://www.ptcnews.tv/entertainment/anant-ambani-and-radhika-merchants-pre-wedding-festivities-know-the-total-expenditure-4212294",
    "https://www.news9live.com/knowledge/most-expensive-weddings-in-the-world-2455867",
    "https://www.moneycontrol.com/news/photos/trends/entertainment/have-a-look-at-the-most-expensive-wedding-outfits-worn-by-ambani-women-12345301.html",
    "https://www.theguardian.com/world/2024/feb/28/anant-ambani-wedding-son-indias-richest-person-mukesh-celebrity-guest-list",
    "https://www.hindustantimes.com/india-news/anant-ambani-radhika-merchant-pre-wedding-2500-dishes-menu-101709024161274.html",
    "https://www.indiatimes.com/worth/news/big-numbers-show-how-grand-anant-isha-akash-ambani-family-weddings-628976.html",
    "https://www.hindustantimes.com/entertainment/music/rihanna-is-getting-paid-52-crore-to-perform-at-anant-ambani-radhika-merchant-s-pre-wedding-party-report-101709272300684.html",
    "https://economictimes.indiatimes.com/news/india/rihanna-to-perform-at-anant-ambanis-pre-wedding-gala-heres-how-much-the-global-popstar-charges-for-an-event/articleshow/108105563.cms?from=mdr",
    "https://www.koimoi.com/bollywood-news/anant-ambani-radhika-merchant-wedding-budget-1900-higher-than-bollywoods-costliest-shaadi-heres-how-much-other-ambani-shaadis-cost/",
    "https://timesofindia.indiatimes.com/gadgets-news/when-mark-zuckerberg-wife-priscilla-chan-got-impressed-by-anant-ambanis-watch/articleshow/108200414.cms",
    "https://e.vnexpress.net/photo/celebrities/asias-richest-billionaire-ambani-heirs-fiancee-dazzles-at-152m-pre-wedding-gala-4720058.html"
]

# Initialize an empty list to store article data
articles_content = []

# Loop through the URLs and extract data
for url in urls:
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        # Store article details in a dictionary
        article_details = {
            'Title': article.title,
            'Text': article.text,
            'URL': url
        }
        
        # Append the dictionary to the list
        articles_content.append(article_details)
    except Exception as e:
        print(f"Failed to process {url}: {str(e)}")

# Convert the list of dictionaries to a DataFrame for easier handling
articles_df = pd.DataFrame(articles_content)

def preprocess_text(text):
    exclude_terms = ["ambani", "anant", "nita", "radhika", "merchant", "isha", "akash", "mukesh", "pre", "wedding", "marriage"]
    currency_indicators = ["$", "rs", "rupee", "rupees", "crore", "lakh", "million", "billion", "USD"]

    text = re.sub(r'<.*?>', '', text)
    currency_related_numbers = re.findall(r'(\b\d+\.?\d*\s*(?:' + '|'.join(currency_indicators) + ')\b)', text.lower())
    text = re.sub("[^a-zA-Z0-9\s]", " ", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens.extend(currency_related_numbers)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words and w not in exclude_terms and (bool(re.search(r'\d', w)) and w in currency_related_numbers or not bool(re.search(r'\d', w)))]
    
    return filtered_tokens

articles_df['Cleaned_Text'] = articles_df['Text'].apply(preprocess_text)

sample_bigrams = list(bigrams(articles_df['Cleaned_Text'][0]))
fdist_unigrams = FreqDist([word for token_list in articles_df['Cleaned_Text'] for word in token_list])
fdist_bigrams = FreqDist(bigrams([word for token_list in articles_df['Cleaned_Text'] for word in token_list]))

print(fdist_unigrams.most_common(10))
print(fdist_bigrams.most_common(10))

def calculate_sentiment(text_tokens):
    text = " ".join(text_tokens)
    sentiment = TextBlob(text).sentiment
    return sentiment.polarity, sentiment.subjectivity

articles_df['Sentiment'] = articles_df['Cleaned_Text'].apply(calculate_sentiment)
articles_df[['Polarity', 'Subjectivity']] = pd.DataFrame(articles_df['Sentiment'].tolist(), index=articles_df.index)

average_polarity = articles_df['Polarity'].mean()
average_subjectivity = articles_df['Subjectivity'].mean()

print(f"Average Polarity: {average_polarity}")
print(f"Average Subjectivity: {average_subjectivity}")

plt.figure(figsize=(10, 5))
sns.histplot(articles_df['Polarity'], kde=True, color='blue')
plt.title('Polarity Distribution')
plt.xlabel('Polarity')
plt.ylabel('Frequency')
plt.savefig('../results/polarity_distribution.png')
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(articles_df['Subjectivity'], kde=True, color='green')
plt.title('Subjectivity Distribution')
plt.xlabel('Subjectivity')
plt.ylabel('Frequency')
plt.savefig('../results/subjectivity_distribution.png')
plt.show()

wordcloud_unigrams = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(fdist_unigrams)
plt.figure(figsize=(10, 7), dpi=300)
plt.imshow(wordcloud_unigrams, interpolation='nearest')
plt.axis('off')
plt.title('Unigram Word Cloud')
plt.savefig('../results/unigram_word_cloud.png')
plt.show()

bigrams_dict = {' '.join(pair): freq for pair, freq in fdist_bigrams.items()}
wordcloud_bigrams = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(bigrams_dict)
plt.figure(figsize=(10, 7), dpi=300)
plt.imshow(wordcloud_bigrams, interpolation='nearest')
plt.axis('off')
plt.title('Bigram Word Cloud')
plt.savefig('../results/bigram_word_cloud.png')
plt.show()

all_cleaned_text = ' '.join([' '.join(article) for article in articles_df['Cleaned_Text']])
file_path = r'../results/all_cleaned_text.txt'

with open(file_path, 'w', encoding='utf-8') as file:
    file.write(all_cleaned_text)

print("All cleaned text has been saved to:", file_path)

all_words = [word for article in articles_df['Cleaned_Text'] for word in article]

with open("../data/positive-words.txt", "r", encoding='latin-1') as pos:
    poswords = pos.read().split("\n")

pos_tokens = " ".join([w for w in all_words if w in poswords])
wordcloud_positive = WordCloud(background_color='White', width=1800, height=1400).generate(pos_tokens)
plt.figure(figsize=(10, 8))
plt.axis("off")
plt.imshow(wordcloud_positive)
plt.title("Positive Word Cloud")
plt.savefig('../results/positive_word_cloud.png')
plt.show()

with open("../data/negative-words.txt", "r", encoding='latin-1') as neg:
    negwords = neg.read().split("\n")

neg_tokens = " ".join([w for w in all_words if w in negwords])
wordcloud_negative = WordCloud(background_color='black', width=1800, height=1400).generate(neg_tokens)
plt.figure(figsize=(10, 8))
plt.axis("off")
plt.imshow(wordcloud_negative)
plt.title("Negative Word Cloud")
plt.savefig('../results/negative_word_cloud.png')
plt.show()

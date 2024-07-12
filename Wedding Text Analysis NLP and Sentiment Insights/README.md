# Text Mining and NLP Project

This repository contains a text mining and natural language processing (NLP) project focused on analyzing news articles related to Anant Ambani's lavish pre-wedding ceremony. The project involves extracting and preprocessing text data, performing sentiment analysis, and generating visualizations.

## Project Overview

This project uses 10 news articles about Anant Ambani's pre-wedding ceremony to demonstrate text mining and NLP techniques. The main steps include:

1. **Extracting Text Data**: Using the `newspaper3k` library to scrape articles.
2. **Preprocessing Text Data**: Cleaning the text data by removing unwanted characters, stop words, and specific terms.
3. **Generating Bigrams and Frequency Distributions**: Identifying common unigrams and bigrams in the text.
4. **Sentiment Analysis**: Using `TextBlob` to analyze the sentiment of the articles.
5. **Visualization**: Creating visualizations to display the results of the analysis, including word clouds and sentiment distributions.

## Project Structure

text-mining-nlp-project/
│
├── data/
│ ├── positive-words.txt
│ ├── negative-words.txt
│
├── scripts/
│ ├── text_mining_nlp.py
│
├── results/
│ ├── all_cleaned_text.txt
│ ├── polarity_distribution.png
│ ├── subjectivity_distribution.png
│ ├── unigram_word_cloud.png
│ ├── bigram_word_cloud.png
│ ├── positive_word_cloud.png
│ ├── negative_word_cloud.png
│
├── .gitignore
├── README.md
└── requirements.txt



## Libraries Used

- **newspaper3k**: For scraping and parsing online news articles.
- **pandas**: For data manipulation and analysis.
- **nltk (Natural Language Toolkit)**: For text preprocessing, tokenization, and frequency distribution.
- **re (Regular Expressions)**: For text cleaning.
- **TextBlob**: For sentiment analysis.
- **matplotlib**: For creating static, animated, and interactive visualizations.
- **seaborn**: For making statistical graphics.
- **wordcloud**: For generating word cloud visualizations.

## How to Run the Project

1. **Clone the Repository**
    ```bash
    git clone https://github.com/manpreet171/text-mining-nlp-project.git
    ```

2. **Navigate to the Project Directory**
    ```bash
    cd text-mining-nlp-project
    ```

3. **Install the Required Packages**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Script**
    ```bash
    python scripts/text_mining_nlp.py
    ```

## Detailed Steps of the Project

### 1. Extracting Text Data

Using the `newspaper3k` library, we scrape text data from 10 online news articles about Anant Ambani's pre-wedding ceremony. The URLs of these articles are stored in a list, and for each URL, the article's title, text, and URL are extracted and stored in a pandas DataFrame.

### 2. Preprocessing Text Data

Text preprocessing involves several steps:
- **Removing HTML Tags**: Using regular expressions to clean the text.
- **Tokenization**: Splitting text into individual words (tokens).
- **Removing Stop Words**: Removing common words that do not contribute to the analysis.
- **Filtering Specific Terms**: Excluding specific terms related to names and events that are not informative.
- **Identifying Currency-Related Numbers**: Recognizing numbers that indicate currency to retain context in the analysis.

### 3. Generating Bigrams and Frequency Distributions

Using `nltk`, we generate unigrams (single words) and bigrams (pairs of consecutive words) from the cleaned text. Frequency distributions are created to identify the most common unigrams and bigrams in the text.

### 4. Sentiment Analysis

Sentiment analysis is performed using `TextBlob`. The cleaned text is analyzed to calculate sentiment polarity (how positive or negative the text is) and subjectivity (how subjective or objective the text is). These values are stored in the DataFrame, and the average polarity and subjectivity across all articles are calculated.

### 5. Visualization

Several visualizations are created to display the results of the analysis:
- **Polarity and Subjectivity Distributions**: Histograms showing the distribution of sentiment polarity and subjectivity across the articles.
- **Word Clouds**: Word clouds for unigrams, bigrams, positive words, and negative words, providing a visual representation of the most common words and phrases.

## Results

The results of the analysis are stored in the `results/` directory:
- `all_cleaned_text.txt`: Contains all the cleaned text data concatenated into a single file.
- `polarity_distribution.png`: Histogram of sentiment polarity distribution.
- `subjectivity_distribution.png`: Histogram of sentiment subjectivity distribution.
- `unigram_word_cloud.png`: Word cloud of the most common unigrams.
- `bigram_word_cloud.png`: Word cloud of the most common bigrams.
- `positive_word_cloud.png`: Word cloud of positive words.
- `negative_word_cloud.png`: Word cloud of negative words.

## Dependencies

The project requires the following Python packages:
- `newspaper3k`
- `pandas`
- `nltk`
- `textblob`
- `matplotlib`
- `seaborn`
- `wordcloud`

Install them using:
```bash
pip install -r requirements.txt

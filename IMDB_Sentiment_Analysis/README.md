# IMDB Sentiment Analysis

This project performs sentiment analysis on user reviews from IMDB for the movie "1917". The reviews are scraped from the IMDB website, analyzed for sentiment polarity and subjectivity using `TextBlob`, and visualized using `matplotlib` and `seaborn`.

## Project Structure

IMDB_Sentiment_Analysis/
│
├── scripts/
│ ├── imdb_sentiment_analysis.py
│
├── results/
│
├── .gitignore
├── README.md
└── requirements.txt



## Overview

The main steps of this project include:
1. **Scraping Reviews**: Using `requests` and `BeautifulSoup` to scrape user reviews from IMDB.
2. **Sentiment Analysis**: Analyzing the sentiment of the reviews using `TextBlob`.
3. **Visualization**: Visualizing the sentiment polarity and subjectivity distributions, and creating a scatter plot of polarity vs. subjectivity.

## Libraries Used

- **requests**: For sending HTTP requests to the IMDB website.
- **BeautifulSoup**: For parsing HTML and extracting review text.
- **pandas**: For data manipulation and analysis.
- **TextBlob**: For sentiment analysis.
- **matplotlib**: For creating visualizations.
- **seaborn**: For making statistical graphics.

## How to Run the Project

1. **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/Data-science-projects.git
    ```

2. **Navigate to the Project Directory**
    ```bash
    cd Data-science-projects/IMDB_Sentiment_Analysis
    ```

3. **Install the Required Packages**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Script**
    ```bash
    python scripts/imdb_sentiment_analysis.py
    ```

## Results

The results of the analysis, including sentiment distributions and visualizations, will be saved in the `results/` directory.

## Dependencies

The project requires the following Python packages:
- requests
- beautifulsoup4
- pandas
- textblob
- matplotlib
- seaborn

Install them using:
```bash
pip install -r requirements.txt
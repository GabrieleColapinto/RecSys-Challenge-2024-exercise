import polars as pl
from pathlib import Path

from ebrec.utils._constants import *

'''
    The articles have a sentiment label and a sentiment score.
    I can calculate the weighted average of articles read by the user for each sentiment.

    weighted_positive = sum(sentiment_score | label=positive) / total_articles

    The same goes for the other sentiments.
'''

DATA_PATH = Path(r"../data/ebnerd_demo").resolve()
ARTICLES_PATH = DATA_PATH / "articles.parquet"
HISTORY_PATH = DATA_PATH / "train" / "history.parquet"

HISTORY_SIZE = 20

'''
    To make accessing information easier I create a dictionary for every sentiment.
    
    positive_scores = {article_id: sentiment_score} for positive sentiments
    
    The same goes for the other sentiments.
'''
df_sentiments = (
    pl.read_parquet(ARTICLES_PATH)
    .select(
        pl.col(DEFAULT_ARTICLE_ID_COL),
        pl.col(DEFAULT_SENTIMENT_SCORE_COL),
        pl.col(DEFAULT_SENTIMENT_LABEL_COL)
    )
)

article_id_list = df_sentiments[DEFAULT_ARTICLE_ID_COL].to_list()
article_sentiment_score_list = df_sentiments[DEFAULT_SENTIMENT_SCORE_COL].to_list()
article_sentiment_label_list = df_sentiments[DEFAULT_SENTIMENT_LABEL_COL].to_list()

positive_articles = {}
neutral_articles = {}
negative_articles = {}

for i in range(len(article_id_list)):
    # Positive sentiments
    if article_sentiment_label_list[i] == "Positive":
        positive_articles[article_id_list[i]] = article_sentiment_score_list[i]

    if article_sentiment_label_list[i] == "Neutral":
        neutral_articles[article_id_list[i]] = article_sentiment_score_list[i]

    if article_sentiment_label_list[i] == "Negative":
        negative_articles[article_id_list[i]] = article_sentiment_score_list[i]


# I calculate the weighted average of articles read by the user for each sentiment
df_weighted_sentiments = (
    pl.read_parquet(HISTORY_PATH)
    # I create the column containing the list of positive articles read by each user
    .with_columns(
        pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.tail(HISTORY_SIZE).list.eval(
            pl.element().map
        )
    )
)






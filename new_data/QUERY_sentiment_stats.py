import polars as pl
from pathlib import Path

from ebrec.utils._constants import (
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_SENTIMENT_LABEL_COL,
    DEFAULT_SENTIMENT_SCORE_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_USER_COL
)

'''
    The articles have a sentiment label and a sentiment score.
    I can calculate the weighted sentiment average for each user and the rate of articles for each sentiment
    read by each user.

    weighted_positive_sentiment = sum(sentiment_score | label=positive) / total_articles
    positive_sentiment_articles_rate = count(label=positive) / total_articles

    The same goes for the other sentiments.
'''

# Paths
DATA_PATH = Path("../data/ebnerd_demo").resolve()
ARTICLES_PATH = DATA_PATH / "articles.parquet"
HISTORY_PATH_TRAIN = DATA_PATH / "train" / "history.parquet"
HISTORY_PATH_VALIDATION = DATA_PATH / "validation" / "history.parquet"

# Constants
HISTORY_SIZE = 20

''' Concatenated history retrieval '''
df_history = pl.concat([
    pl.read_parquet(HISTORY_PATH_TRAIN),
    pl.read_parquet(HISTORY_PATH_VALIDATION)
])

''' articles -> (article_id, sentiment_label, sentiment_score) '''
df_sent = (
    pl.scan_parquet(ARTICLES_PATH)
    .select([
        pl.col(DEFAULT_ARTICLE_ID_COL),
        pl.col(DEFAULT_SENTIMENT_LABEL_COL),
        pl.col(DEFAULT_SENTIMENT_SCORE_COL),
    ]).collect()
)

''' history -> (user_id, tail_articles) -> explode -> join -> group_by with lists '''
df_lists = (
    df_history
    .select([
        pl.col(DEFAULT_USER_COL),
        pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.tail(HISTORY_SIZE).alias(DEFAULT_ARTICLE_ID_COL),
    ])
    .explode(DEFAULT_ARTICLE_ID_COL)
    .join(df_sent, on=DEFAULT_ARTICLE_ID_COL, how="left") # This left join is to keep the users
    .group_by(DEFAULT_USER_COL)
    .agg([
        pl.col(DEFAULT_ARTICLE_ID_COL).filter(pl.col(DEFAULT_SENTIMENT_LABEL_COL) == "Positive")
        .alias("positive_article_ids"),

        pl.col(DEFAULT_ARTICLE_ID_COL).filter(pl.col(DEFAULT_SENTIMENT_LABEL_COL) == "Neutral")
        .alias("neutral_article_ids"),

        pl.col(DEFAULT_ARTICLE_ID_COL).filter(pl.col(DEFAULT_SENTIMENT_LABEL_COL) == "Negative")
        .alias("negative_article_ids"),
    ])
)

''' user -> sentiment_articles_rate, sentiment_weighted_average '''
df_user_sentiment_stats = (
    df_history
    .select([
        pl.col(DEFAULT_USER_COL),
        pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.tail(HISTORY_SIZE).alias(DEFAULT_ARTICLE_ID_COL),
    ])
    .explode(DEFAULT_ARTICLE_ID_COL)
    .join(df_sent, on=DEFAULT_ARTICLE_ID_COL, how="left") # This left join is to keep the users
    .group_by(DEFAULT_USER_COL)
    .agg([
        pl.len().alias("n_read"),

        # Counts of each article sentiment
        (pl.col("sentiment_label") == "Positive").sum().alias("n_pos"),
        (pl.col("sentiment_label") == "Neutral").sum().alias("n_neu"),
        (pl.col("sentiment_label") == "Negative").sum().alias("n_neg"),

        # Weighted sentiment average
        pl.col("sentiment_score").filter(pl.col("sentiment_label") == "Positive").mean().fill_null(0)
        .alias("mean_positive_score"),

        pl.col("sentiment_score").filter(pl.col("sentiment_label") == "Neutral").mean().fill_null(0)
        .alias("mean_neutral_score"),

        pl.col("sentiment_score").filter(pl.col("sentiment_label") == "Negative").mean().fill_null(0)
        .alias("mean_negative_score")
    ])
    .with_columns([
        # Articles rate per sentiment
        (pl.col("n_pos") / pl.col("n_read")).fill_nan(0).alias("positive_sentiment_articles_rate"),
        (pl.col("n_neu") / pl.col("n_read")).fill_nan(0).alias("neutral_sentiment_articles_rate"),
        (pl.col("n_neg") / pl.col("n_read")).fill_nan(0).alias("negative_sentiment_articles_rate")
    ])
).select(
    pl.col(DEFAULT_USER_COL),
    pl.col("mean_positive_score"),
    pl.col("mean_neutral_score"),
    pl.col("mean_negative_score"),
    pl.col("positive_sentiment_articles_rate"),
    pl.col("neutral_sentiment_articles_rate"),
    pl.col("negative_sentiment_articles_rate")
)

# I store the data permanently
df_user_sentiment_stats.write_parquet("./sentiment_stats.parquet")

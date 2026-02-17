import polars as pl
from pathlib import Path

from ebrec.utils._constants import (
    DEFAULT_BODY_COL,
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL,
    DEFAULT_USER_COL
)

DATA_PATH = Path(r"../data/ebnerd_demo").resolve()
ARTICLES_PATH = DATA_PATH / "articles.parquet"
HISTORY_PATH = DATA_PATH / "train" / "history.parquet"

# Let's keep this parameter fixed for now
HISTORY_SIZE = 20

# THRESHOLDS FOR THE ARTICLE LENGTH AND FOR THE SCROLL PERCENTAGE
ARTICLE_LENGTH_THRESHOLD = 2000
SCROLL_PERCENTAGE_THRESHOLD = 60

'''
   I classify the users according to their propensity to read long articles or short articles.
   To do so I refer to the recent history of the user and evaluate the median 
   article length and median scroll percentage.
   
   I generate the following categories of users:
        1) Short article and low scroll percentage -> Not interested user
        2) Short article and high scroll percentage -> Short articles reader
        3) Long article and low scroll percentage -> Skimmer
        4) Long article and high scroll percentage -> Long articles reader
'''

# I create the association between article ID and article length in characters
df_article_lengths = (
    pl.read_parquet(ARTICLES_PATH)
    .with_columns(
        pl.col(DEFAULT_BODY_COL).str.len_chars()
        .alias("article_length")
    )
).select(
    pl.col(DEFAULT_ARTICLE_ID_COL),
    pl.col("article_length")
)

# I create the map to associate articles to lengths for faster lookup
article_lengths_map = dict(
    zip(
        df_article_lengths[DEFAULT_ARTICLE_ID_COL].to_list(),
        df_article_lengths["article_length"].to_list()
    )
)

# I create the dataframe containing the medians of the article lengths and the scroll percentages
df_medians = (
    pl.read_parquet(HISTORY_PATH)

    # I create the column containing the list of article lengths
    .with_columns(
        pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL)
        .list.tail(HISTORY_SIZE).list.eval(
            pl.element().replace(article_lengths_map, default=0)
        )
        .alias("article_length_fixed_tail")
    )

    # I take the last HISTORY SIZE elements of the scrolls list
    .with_columns(
        pl.col(DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL).list.tail(HISTORY_SIZE).fill_null(0)
        .alias("article_scroll_percentage_fixed_tail")
    )

    # I retrieve the median scroll percentage for the user in the last HISTORY SIZE articles
    .with_columns(
        pl.col("article_scroll_percentage_fixed_tail").list.median()
        .alias("median_scroll_percentage")
    )

    # I retrieve the median article length for the user in the last HISTORY SIZE articles
    .with_columns(
        pl.col("article_length_fixed_tail").list.median()
        .alias("median_article_length")
    )
)

# Now I classify the users according to the previously calculated median values
df_article_length_propensity = df_medians.with_columns(
    (pl.when(pl.col("median_article_length") <= ARTICLE_LENGTH_THRESHOLD)
    # Short median article length
    .then(
        pl.when(pl.col("median_scroll_percentage") <= SCROLL_PERCENTAGE_THRESHOLD)
        # Not interested user
        .then(1)
        # Short articles reader
        .otherwise(2)
    )
    # Long median article length
    .otherwise(
        pl.when(pl.col("median_scroll_percentage") <= SCROLL_PERCENTAGE_THRESHOLD)
        # Skimmer
        .then(3)
        # Long articles reader
        .otherwise(4)
    )).alias("article_length_propensity")
).select(
    pl.col(DEFAULT_USER_COL),
    pl.col("article_length_propensity")
)

# Considering that this data is static I store it permanently to avoid generating it every time
df_article_length_propensity.write_parquet("article_length_propensity.parquet")

# I store the association between article ID and article length to provide this information to the artice data loader
df_article_lengths.write_parquet("article_length.parquet")

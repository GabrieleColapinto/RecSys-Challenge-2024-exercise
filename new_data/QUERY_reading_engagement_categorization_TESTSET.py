import polars as pl
from pathlib import Path

from ebrec.utils._constants import (
    DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_HISTORY_READ_TIME_COL,
    DEFAULT_USER_COL
)

DATA_PATH = Path(r"../data/ebnerd_testset").resolve()
HISTORY_PATH = DATA_PATH / "test" / "history.parquet"

'''
    I classify the users according to their reading engagement.
    
    To classify the users I use the mean number of daily impressions and the mean reading time of each user.
    
    Mean daily impressions (MDI) = Number of impressions / Date difference in days between the last impression and the first one
    Mean reading time (MRT) = Mean of the list of reading time of each user
    
    There is a total of 4 category of users:
        1) Low MRT and low MDI -> Occasional researcher
        2) Low MRT and high MDI -> Bored person (They use the website as if it was Instagram)
        3) High MRT and low MDI -> Occasional reader (They don't read frequently but they read thoroughly)
        4) High MRT and high MDI -> Usual reader
'''

# THRESHOLDS TO CLASSIFY THE USERS
MEAN_DAILY_IMPRESSIONS_THRESHOLD = 5
MEAN_READING_TIME_THRESHOLD = 45

df_reading_engagement = (
    pl.read_parquet(HISTORY_PATH)

    .with_columns(
        # I add the column containing the average number of impressions per day
        (
                pl.col(DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL).list.len() / (
                pl.col(DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL).list.last().dt.ordinal_day() -
                pl.col(DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL).list.first().dt.ordinal_day()
        )).alias("mean daily impressions")
    )

    .with_columns(
        # I add the column containing the mean reading time of the user
        (
            pl.col(DEFAULT_HISTORY_READ_TIME_COL).list.mean()
        ).alias("mean reading time")
    )

    # I add the column containing the category of the user
    .with_columns(
        pl.when(pl.col("mean reading time") <= MEAN_READING_TIME_THRESHOLD).then(
            # Skimmers
            pl.when(pl.col("mean daily impressions") <= MEAN_DAILY_IMPRESSIONS_THRESHOLD)
            .then(1) # Occasional researcher
            .otherwise(2) # Bored person (They use the website as if it was Instagram)
        ).otherwise(
            # Deep readers
            pl.when(pl.col("mean daily impressions") <= MEAN_DAILY_IMPRESSIONS_THRESHOLD)
            .then(3) # Occasional reader
            .otherwise(4) # Usual reader
        ).alias("reading_engagement")
    )
).select(
    pl.col(DEFAULT_USER_COL),
    pl.col("reading_engagement")
)

# Considering that this data is static I store it permanently to avoid generating it every time
df_reading_engagement.write_parquet("reading_engagement_TESTSET.parquet")

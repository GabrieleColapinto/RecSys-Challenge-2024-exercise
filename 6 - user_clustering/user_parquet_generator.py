from pathlib import Path
import polars as pl

from ebrec.utils._constants import DEFAULT_USER_COL

'''
    This file creates a parquet that contains all the relevant information of each user.
    This dataframe will then be used for user clustering.
'''

# Constants
DATASPLIT = "ebnerd_demo"
VARIANCE_THRESHOLD = 0.1

# Paths
PATH = Path("../data").resolve()

HISTORY_TRAIN_PATH = PATH / DATASPLIT / "train" / "history.parquet"
HISTORY_VALIDATION_PATH = PATH / DATASPLIT / "validation" / "history.parquet"

NEW_DATA_PATH = Path("../new_data").resolve()

SENTIMENT_STATS = NEW_DATA_PATH / 'sentiment_stats.parquet'
FAVOURITE_DEVICE = NEW_DATA_PATH / 'favourite_device.parquet'
TIME_PROFILE = NEW_DATA_PATH / 'time_profile.parquet'
MEDIAN_NEXT_READ_TIME = NEW_DATA_PATH / 'median_next_read_time.parquet'
PREMIUM_PROPENSITY = NEW_DATA_PATH / "premium_propensity.parquet"
READING_ENGAGEMENT = NEW_DATA_PATH / "reading_engagement.parquet"
SUBSCRIPTION_STATUS = NEW_DATA_PATH / "subscription_status.parquet"

''' I retrieve the necessary data '''
# User IDs retrieval
df_users = (pl.concat([pl.read_parquet(HISTORY_TRAIN_PATH), pl.read_parquet(HISTORY_VALIDATION_PATH)])
            .select(pl.col(DEFAULT_USER_COL)))

# I add the sentiment stats
df_sentiment_stats = pl.read_parquet(SENTIMENT_STATS)
df_users = df_users.join(
    other=df_sentiment_stats,
    on=DEFAULT_USER_COL,
    how="inner"
)

# I add the information about the favourite device
df_favourite_device = pl.read_parquet(FAVOURITE_DEVICE)
df_users = df_users.join(
    other=df_favourite_device,
    on=DEFAULT_USER_COL,
    how="inner"
)

# I add the time profile of the users
df_time_profile = pl.read_parquet(TIME_PROFILE)
df_users = df_users.join(
    other=df_time_profile,
    on=DEFAULT_USER_COL,
    how="inner"
)

# I add the median next read time
df_median_next_read_time = pl.read_parquet(MEDIAN_NEXT_READ_TIME)
df_users = df_users.join(
    other=df_median_next_read_time,
    on=DEFAULT_USER_COL,
    how="inner"
)

# I add the association between user and premium propensity
df_premium_propensity = pl.read_parquet(PREMIUM_PROPENSITY)
df_users = df_users.join(
    other=df_premium_propensity,
    on=DEFAULT_USER_COL,
    how="inner"
)

# I add the association between user and reading engagement
df_reading_engagement = pl.read_parquet(READING_ENGAGEMENT)
df_users = df_users.join(
    other=df_reading_engagement,
    on=DEFAULT_USER_COL,
    how="inner"
)

# I include the subscription status of the users
df_subscription_status = pl.read_parquet(SUBSCRIPTION_STATUS)
df_users = df_users.join(
    other=df_subscription_status,
    on=DEFAULT_USER_COL,
    how="inner"
)

# I store the data permanently
df_users.write_parquet("users_clustering_data.parquet")

'''
    The history df has the field impressions_time_fixed and many impressions are within a short period of time.
    Considering that a user can have more than one impression per session because they reload the page we can group
    consecutive impressions with a maximum gap of 1 hour and classify users according to the average time of the
    day they check the website in. The key idea is to understand when users read the news to gay key insights on the
    readers. This can be useful to send notifications at a specific time of the day like Duolingo does.

    So, the key idea is to:
        1) Group impressions into sessions by calculating the time difference between two consecutive
           impressions and checking if it is shorter than a threshold value (for instance, 1 hour).

        2) Associate to each session the time of the day and the number of impressions so that we can have a list of
           times of the day for each user with the relative number of impressions.

        3) Discard the sessions with a small number of impressions because they represent a non-meaningful usage of the
           website. If a user has a session with only one impression it means that they either opened the website
           accidentally, or they were expecting a specific new because they are interested in a particular phenomenon
           or they were not engaged in using the website and they closed it.

        4) Use the times of the day to create time windows and use those time windows to classify the users according to
           their propensity to read in that specific time window.
           (time window) 8-9 = number of sessions of the user from 8am to 9am / total number of meaningful sessions

        5) In the end we will have 24 time windows associated to a float32 value useful to the clustering algorithm.
'''

import polars as pl
from pathlib import Path

from ebrec.utils._constants import (
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL
)

# Paths
DATA_PATH = Path(r"../data/ebnerd_demo").resolve()
HISTORY_PATH_TRAIN = DATA_PATH / "train" / "history.parquet"
HISTORY_PATH_VALIDATION = DATA_PATH / "validation" / "history.parquet"

# Gap between two consecutive sessions
SESSION_GAP_THRESHOLD = pl.duration(hours=1)

# Impression threshold between a meaningful session and a meaningless session
IMPRESSIONS_THRESHOLD = 2

# Names of the additional columns
IMPRESSIONS_TIME_DIFF = "impressions_time_diff"
NEW_SESSION_MARKER = "new_session"
SESSION_ID_COLUMN = "session_id"
NUMBER_OF_IMPRESSIONS_COL = "n_impressions"
SESSION_HOUR_COL = "session_hour"
SESSIONS_COUNT_COL = "n_sessions"
TOTAL_SESSIONS_COL = "total_sessions"
PROPENSITY_COL = "propensity"

# I retrieve the necessary data
df_history = pl.concat([
    pl.scan_parquet(HISTORY_PATH_TRAIN),
    pl.scan_parquet(HISTORY_PATH_VALIDATION)
]).select([
    pl.col(DEFAULT_USER_COL),
    pl.col(DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL)
]).collect()

'''
    Calculate the gap between two consecutive impressions.
    The field impressions_time_diff is of Duration type.
    The first element of the list is null because it doesn't have a preceding value to
    calculate the difference.
'''
df_history = (df_history
    .with_columns(
        pl.col(DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL)
        .list.diff().over(DEFAULT_USER_COL)
        .alias(IMPRESSIONS_TIME_DIFF)
    )
)

# I explode the columns containing the timestamps and the time differences so that I can identify the sessions later
df_history = df_history.explode([
    pl.col(DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL),
    pl.col(IMPRESSIONS_TIME_DIFF)
])

# I insert a boolean field that acts like a marker for the beginning of the sessions
df_history = df_history.with_columns(
    (
        pl.col(IMPRESSIONS_TIME_DIFF).is_null() | (pl.col(IMPRESSIONS_TIME_DIFF) > SESSION_GAP_THRESHOLD)
    ).alias(NEW_SESSION_MARKER)
)

# I identify the sessions through the new session marker
df_history = df_history.with_columns(
    pl.col(NEW_SESSION_MARKER).cast(pl.Int32).cum_sum().over(DEFAULT_USER_COL).alias(SESSION_ID_COLUMN)
)

df_sessions = (
    df_history
    .group_by([DEFAULT_USER_COL, SESSION_ID_COLUMN])
    .agg([
        pl.len().alias(NUMBER_OF_IMPRESSIONS_COL),
        pl.col(DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL).mean().dt.hour().alias(SESSION_HOUR_COL)
    ])
    .filter(pl.col(NUMBER_OF_IMPRESSIONS_COL) >= IMPRESSIONS_THRESHOLD)
)
'''
    Filtering only the meaningful sessions reduces the number of unique users in the dataframe.
    The number of users in the original dataframe is 3152 and the number of users in the sessions
    dataframe is 1931. Considering the high amount of users removed during this operation I accept
    to reduce the number of users in the end dataframe containing all the features because using a
    left join would cause about 1/3 of the tuples to have null values. Those tuples would be meaningless
    and filling them with a default value or an average value would create noise in the end dataframe.
'''

# I count the sessions per time window
df_user_hour_counts = (
    df_sessions
    .group_by([DEFAULT_USER_COL, SESSION_HOUR_COL])
    .agg(pl.count().alias(SESSIONS_COUNT_COL))
)

# I generate a dataframe containing the total number of sessions per user
df_user_totals = (
    df_user_hour_counts
    .group_by(DEFAULT_USER_COL)
    .agg(pl.sum(SESSIONS_COUNT_COL).alias(TOTAL_SESSIONS_COL))
)

# I calculate the propensity of each user to have a session in each time window
df_user_hour_props = (
    df_user_hour_counts
    .join(df_user_totals, on=DEFAULT_USER_COL)
    .with_columns(
        (pl.col(SESSIONS_COUNT_COL) / pl.col(TOTAL_SESSIONS_COL))
        .cast(pl.Float32)
        .alias(PROPENSITY_COL)
    )
)

# I create the association between users, times and propensities to have a session in those times
df_time_profile = (
    df_user_hour_props
    .select([DEFAULT_USER_COL, SESSION_HOUR_COL, PROPENSITY_COL])
    .sort(SESSION_HOUR_COL)
    .pivot(
        values=PROPENSITY_COL,
        index=DEFAULT_USER_COL,
        columns=SESSION_HOUR_COL
    )
    .fill_null(0.0)
).with_columns(
    # The operation turns the user id into a float value. I have to reconvert it to unsigned integer.
    pl.col(DEFAULT_USER_COL).cast(pl.UInt32)
)

# I store the data permanently
df_time_profile.write_parquet("./time_profile.parquet")

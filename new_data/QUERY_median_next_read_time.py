import polars as pl
from pathlib import Path

from ebrec.utils._constants import (
    DEFAULT_USER_COL,
    DEFAULT_NEXT_READ_TIME_COL
)

'''
    The behaviors dataframe contains the field next_read_time which is the time in seconds between one reading and the
    next one. I need to retrieve the median_next_read_time for each user because it is a behavioral indicator.
'''

DATA_PATH = Path("../data/ebnerd_demo").resolve()
BEHAVIORS_TRAIN_PATH = DATA_PATH / "train" / "behaviors.parquet"
BEHAVIORS_VALIDATION_PATH = DATA_PATH / "validation" / "behaviors.parquet"

# i retrieve the necessary data
df_behaviors_train = (pl.scan_parquet(BEHAVIORS_TRAIN_PATH)
                      .select(pl.col(DEFAULT_USER_COL), pl.col(DEFAULT_NEXT_READ_TIME_COL))
                      .collect()
                      )

df_behaviors_validation = (pl.scan_parquet(BEHAVIORS_VALIDATION_PATH)
                           .select(pl.col(DEFAULT_USER_COL), pl.col(DEFAULT_NEXT_READ_TIME_COL))
                           .collect()
                           )

df_behaviors = pl.concat([df_behaviors_train, df_behaviors_validation])

# I derive the median next read time
df_behaviors = df_behaviors.group_by(pl.col(DEFAULT_USER_COL)).agg([
    pl.col(DEFAULT_NEXT_READ_TIME_COL).median().alias("median_next_read_time"),
])

# I store the data permanently
df_behaviors.write_parquet("./median_next_read_time.parquet")

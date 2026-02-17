import polars as pl
from pathlib import Path

from ebrec.utils._constants import (
    DEFAULT_USER_COL,
    DEFAULT_DEVICE_COL
)

'''
    The impressions are associated to a device type whose value can be either 1, 2 or 3.
    I can associate each user to their favourite device.
'''

DATA_PATH = Path("../data/ebnerd_demo").resolve()
BEHAVIORS_TRAIN_PATH = DATA_PATH / "train" / "behaviors.parquet"
BEHAVIORS_VALIDATION_PATH = DATA_PATH / "validation" / "behaviors.parquet"

# I collect the user_id and device_type from the behaviors dataframe
df_behaviors_train = (
    pl.scan_parquet(BEHAVIORS_TRAIN_PATH)
    .select(
        pl.col(DEFAULT_USER_COL),
        pl.col(DEFAULT_DEVICE_COL)
    ).collect()
)

df_behaviors_validation = (
    pl.scan_parquet(BEHAVIORS_VALIDATION_PATH)
    .select(
        pl.col(DEFAULT_USER_COL),
        pl.col(DEFAULT_DEVICE_COL)
    ).collect()
)

df_user_device = pl.concat([
    df_behaviors_train,
    df_behaviors_validation
])

'''
    I create an association between user_id and a list of devices of the different impressions then
    I select the favourite device as the median value of the list.
'''
df_favourite_device = (
    df_user_device
    .group_by(DEFAULT_USER_COL)
    .agg([
        pl.col(DEFAULT_DEVICE_COL).alias("device_list")
    ])
    .with_columns(
        pl.col("device_list").list.median().cast(pl.Int8).alias("favourite_device")
    )
).select(
    pl.col(DEFAULT_USER_COL),
    pl.col("favourite_device")
)

# Permanent storage
df_favourite_device.write_parquet("./favourite_device.parquet")

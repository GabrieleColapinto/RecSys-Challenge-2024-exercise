import polars as pl
from pathlib import Path

from ebrec.utils._constants import (
    DEFAULT_IS_SUBSCRIBER_COL,
    DEFAULT_USER_COL
)

PATH = Path("../data").resolve()

DATASPLIT = "ebnerd_demo"

BEHAVIORS_TRAIN_PATH = PATH / DATASPLIT / "train" / "behaviors.parquet"
BEHAVIORS_VALIDATION_PATH = PATH / DATASPLIT / "validation" / "behaviors.parquet"

df_subscribers_train = (pl.scan_parquet(BEHAVIORS_TRAIN_PATH)
                        .select(pl.col(DEFAULT_USER_COL), pl.col(DEFAULT_IS_SUBSCRIBER_COL))
                        .collect()
                        )

df_subscribers_validation = (pl.scan_parquet(BEHAVIORS_VALIDATION_PATH)
                             .select(pl.col(DEFAULT_USER_COL), pl.col(DEFAULT_IS_SUBSCRIBER_COL))
                             .collect()
                             )

df_subscribers = pl.concat([df_subscribers_train, df_subscribers_validation])

'''
The following query confirms that the values of is_subscriber for each user are unique because the number of unique
values for each user is 1. Hence, I can use a similar query to retrieve the subscription status of the users.

test_query = (df_subscribers
                .select(pl.col('user_id'), pl.col('is_subscriber'))
                .group_by('user_id')
                .agg([pl.col('is_subscriber')])
                .with_columns(
                    pl.col('is_subscriber').list.n_unique()
                )
)
'''

df_subscribers = (df_subscribers
                .select(pl.col(DEFAULT_USER_COL), pl.col(DEFAULT_IS_SUBSCRIBER_COL))
                .group_by(DEFAULT_USER_COL)
                .agg([pl.col(DEFAULT_IS_SUBSCRIBER_COL)])
                .with_columns(
                    pl.col(DEFAULT_IS_SUBSCRIBER_COL).list.all().cast(pl.Float32)
                )
)

df_subscribers.write_parquet("subscription_status.parquet")

import polars as pl
from pathlib import Path

from ebrec.utils._constants import (
    DEFAULT_PREMIUM_COL,
    DEFAULT_ARTICLE_ID_COL
)

DATA_PATH = Path(r"../data/ebnerd_testset").resolve()
ARTICLES_PATH = DATA_PATH / "articles.parquet"

# I retrieve the association between articles and premiumness
df_premium_articles = pl.read_parquet(ARTICLES_PATH).select(
    pl.col(DEFAULT_ARTICLE_ID_COL),
    pl.col(DEFAULT_PREMIUM_COL).cast(pl.Float32)
)

# I store the association permanently
df_premium_articles.write_parquet("premium_articles_TESTSET.parquet")

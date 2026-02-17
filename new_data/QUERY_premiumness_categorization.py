import polars as pl
from pathlib import Path

from ebrec.utils._constants import (
    DEFAULT_PREMIUM_COL,
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_USER_COL
)

DATA_PATH = Path(r"../data/ebnerd_demo").resolve()
BEHAVIORS_PATH = DATA_PATH / "train" / "behaviors.parquet"
ARTICLES_PATH = DATA_PATH / "articles.parquet"

# I retrieve the list of premium articles
premium_articles = (
    pl.read_parquet(ARTICLES_PATH)
    .filter(pl.col(DEFAULT_PREMIUM_COL) == True)
    .select(pl.col(DEFAULT_ARTICLE_ID_COL))
).get_column(DEFAULT_ARTICLE_ID_COL).to_list()

'''
    I classify the user according to their propensity to read premium articles.
    
    I calculate the premium frequency as the ratio between premium articles clicked
    and the total number of clicked articles.
    
    Then I classify the premium frequency as follows:
        0 <= PF < 0.2 -> Free only users -> Encoded as 1
        0.2 <= PF < 0.5 -> Mixed users -> Encoded as 2
        PF >= 0.5 -> Premium heavy users -> Encoded as 3
'''

# I generate the association between user, total clicked articles and total premium clicked articles
df_user_articles = (
    pl.read_parquet(BEHAVIORS_PATH)

    .with_columns([
        # I generate the column containing the number of premium clicked articles in the impression
        pl.col(DEFAULT_CLICKED_ARTICLES_COL)
        .list.eval(pl.element().is_in(premium_articles))
        .list.sum()
        .alias("premium_clicked_articles"),

        # I generate teh column containing the number of clicked articles in the impression
        pl.col(DEFAULT_CLICKED_ARTICLES_COL)
        .list.len()
        .alias("clicked_articles"),
    ])
# I sum the data aggregating it by user
).group_by(DEFAULT_USER_COL).agg(
    pl.col("premium_clicked_articles").sum().alias("premium_clicked_articles"),
    pl.col("clicked_articles").sum().alias("clicked_articles")
).select(
    # I generate a temporary dataframe containing only the desired association
    pl.col(DEFAULT_USER_COL),
    pl.col("clicked_articles"),
    pl.col("premium_clicked_articles")
)

'''
    Now I turn the temporary dataframe into a definitive dataframe containing the association
    between user and premium propensity.
'''
df_premium_propensity = df_user_articles.with_columns(
    # I generate the premium frequency column
    (pl.col("premium_clicked_articles") / pl.col("clicked_articles"))
    .fill_nan(0)
    .alias("premium_frequency")
).with_columns(
    # I generate the premium propensity column
    (pl.when((pl.col("premium_frequency") >= 0) & (pl.col("premium_frequency") < 0.2))
    .then(1)
    .otherwise(
        pl.when((pl.col("premium_frequency") >= 0.2) & (pl.col("premium_frequency") < 0.5))
        .then(2)
        .otherwise(3)  # In this case the premium frequency is >= 0.5
    )).alias("premium_propensity")
).select(
    pl.col(DEFAULT_USER_COL),
    pl.col("premium_propensity")
)

# Considering that this data is static I store it permanently to avoid generating it every time
df_premium_propensity.write_parquet("premium_propensity.parquet")

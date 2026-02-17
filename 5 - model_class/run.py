'''
    This file contains the main code that handles model.py and
    uses run_utils.py.
'''

from run_utils import(
    get_articles_df,
    get_item_feature_dict,
    get_train_and_evaluation_df,
    make_hparams,
    EPOCHS,
    BS_TRAIN,
    BS_TEST,
    DOCVEC_COL,
    ARTIFACTS_FOLDER,
    RANDOM_SEED,
    USER_FEAT_COLS
)

from model import Model

from new_dataloader import create_article_id_to_value_mapping
from run_utils import get_test_df

# I collect all the necessary data
hparams = make_hparams()
df_train, df_val_fit = get_train_and_evaluation_df()
df_test = get_test_df()
df_articles = get_articles_df()
item_feature_dict = get_item_feature_dict(df_articles=df_articles)

# I create the article mapping
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=DOCVEC_COL
)

# I create the model
model = Model(
    hparams=hparams,
    item_feature_dict=item_feature_dict,
    article_mapping=article_mapping,
    artifacts_folder=ARTIFACTS_FOLDER,
    seed=RANDOM_SEED,
    bs_train=BS_TRAIN,
    bs_test=BS_TEST,
    epochs=EPOCHS,
    user_feat_cols=USER_FEAT_COLS
)

# I train the model
model.fit(X=df_train, y=df_val_fit)

# I compute the predictions
y_true, y_pred = model.predict(df_test)

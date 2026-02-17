'''
    I classify the users using the PCA -> GMM -> BIC pipeline.
'''

import polars as pl
from pathlib import Path
import numpy as np
from itertools import product
from tqdm import tqdm

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ebrec.utils._constants import DEFAULT_USER_COL

# Constants
DF_PATH = Path('./users_clustering_data.parquet')
VARIANCE_TO_KEEP = 0.80
RANDOM_STATE = 42
K_MIN, K_MAX = 8, 15 # Clusters range

# I load the data
df = pl.read_parquet(DF_PATH).drop(DEFAULT_USER_COL)

# I convert the df into numpy arrays
X = df.to_numpy()

# I standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# I remove the NaN
col_means = np.nanmean(X_scaled, axis=0)
inds = np.where(np.isnan(X_scaled))
X_scaled[inds] = np.take(col_means, inds[1])

# I reduce the data dimensionality to keep the 80% of the variance of the dataset
pca = PCA(n_components=VARIANCE_TO_KEEP, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

print(f"Original features: {X.shape[1]}")
print(f"PCA components kept: {pca.n_components_}")
print(f"Explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")

# I research the best hyperparameters for the DBSCAN clustering
param_grid = {
    "n_components": [k for k in range(K_MIN, K_MAX)],
    "n_init": [5, 8, 10]
}
keys = param_grid.keys()
values = param_grid.values()

configs = [
    dict(zip(keys, combo))
    for combo in product(*values)
]

# Data derived from the grid search
bics = []
models = []

print("\nSearching the best combination of hyperparameters...")
for cfg in tqdm(configs):
    gmm = GaussianMixture(n_components=cfg["n_components"], n_init=cfg["n_init"])
    gmm.fit(X_pca)
    bics.append(gmm.bic(X_pca))
    models.append(gmm)

best_idx = int(np.argmin(bics))
best_settings = configs[best_idx]
best_gmm = models[best_idx]

print(f"\nBest number of clusters: {best_settings['n_components']}")
print(f"Best number of initializations: {best_settings['n_init']}")
print(f"BIC min: {bics[best_idx]:.2f}")

# I generate the labels for the users
labels = best_gmm.predict(X_pca)

# I add the labels to the original dataframe
new_df = pl.read_parquet(DF_PATH).with_columns(pl.Series("label", labels))

# I store the data permanently
new_df.write_parquet("./labeled_users_data.parquet")

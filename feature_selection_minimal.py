
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression, RFE, SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# -------------------------- CONFIG ---------------------------
DATA_PATH = 'data-set.csv'      # adjust path if needed
TARGET = 'SalePrice'
ID_COL = 'Id'
N_TOP_MI = 15                   # top K for Mutual Information
N_FINAL = 12                    # features to keep in wrapper
# -------------------------------------------------------------

def load_and_preprocess(path):
    """Load CSV and keep numeric features only (encoded/engineered)."""
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number])
    X = numeric.drop([TARGET], axis=1, errors='ignore')
    y = numeric[TARGET]
    if ID_COL in X.columns:
        X = X.drop(ID_COL, axis=1)
    return X, y

def variance_filter(X, thresh=0.0):
    vt = VarianceThreshold(threshold=thresh)
    X_new = vt.fit_transform(X)
    cols = X.columns[vt.get_support()]
    return pd.DataFrame(X_new, columns=cols, index=X.index)

# ------------------------ FILTER METHODS ---------------------
def pearson_filter(X, y, thr=0.3):
    corrs = X.corrwith(y).abs()
    selected = corrs[corrs > thr].index.tolist()
    return selected, corrs

def mi_filter(X, y, k=N_TOP_MI):
    skb = SelectKBest(mutual_info_regression, k=k).fit(X, y)
    feats = X.columns[skb.get_support()].tolist()
    scores = dict(zip(feats, skb.scores_[skb.get_support()]))
    return feats, scores

# ------------------------ WRAPPER METHODS --------------------
def rfe_wrapper(X, y, n=N_FINAL):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rfe = RFE(rf, n_features_to_select=n)
    rfe.fit(X, y)
    return X.columns[rfe.get_support()].tolist()

def sfs_wrapper(X, y, n=N_FINAL):
    lr = LinearRegression()
    sfs = SequentialFeatureSelector(lr, n_features_to_select=n, direction='forward', cv=3, scoring='r2')
    sfs.fit(X, y)
    return X.columns[sfs.get_support()].tolist()

# ------------------------ VISUAL -----------------------------
def plot_key_features(core_feats, extra_feats, path='important_features.png'):
    labels = core_feats + extra_feats
    y_vals = [1]*len(labels)
    plt.figure(figsize=(10,4))
    plt.bar(labels, y_vals)
    plt.xticks(rotation=45, ha='right')
    plt.yticks([])
    plt.title('Most Important Features Selected')
    for i, lbl in enumerate(core_feats):
        plt.text(i, 1.02, 'core', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

# --------------------------- MAIN ----------------------------
if __name__ == '__main__':
    print('Loading data ...')
    X_raw, y = load_and_preprocess(DATA_PATH)
    X = variance_filter(X_raw)
    print(f'Features after variance filter: {X.shape[1]}')

    pearson_feats, corrs = pearson_filter(X, y)
    mi_feats, _ = mi_filter(X[pearson_feats], y)  # faster on reduced set

    union_feats = sorted(set(pearson_feats + mi_feats))
    print(f'Pearson: {len(pearson_feats)} | MI: {len(mi_feats)} | Union: {len(union_feats)}')

    X_union = X[union_feats].fillna(X[union_feats].median())

    rfe_feats = rfe_wrapper(X_union, y)
    sfs_feats = sfs_wrapper(X_union, y)
    print(f'RFE ({len(rfe_feats)}):', rfe_feats)
    print(f'SFS ({len(sfs_feats)}):', sfs_feats)

    core = sorted(set(pearson_feats) & set(mi_feats) & set(rfe_feats) & set(sfs_feats))
    extras = [f for f in rfe_feats if f not in core]

    print('\nCore features (selected by ALL methods):', core)
    print('Additional features (from RFE):', extras)

    img_path = plot_key_features(core, extras)
    print(f'Visual saved to {img_path}')

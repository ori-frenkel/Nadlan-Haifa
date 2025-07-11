import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression, RFE, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Config
DATA_PATH = 'data/gov/transactions_with_interest_final.csv'
TARGETS = ['מחיר', 'SalePrice']  # Try both, use the one that exists
N_TOP_MI = 15
N_FINAL = 12

if __name__ == '__main__':
    print('Loading data ...')
    df = pd.read_csv(DATA_PATH)
    # Find the correct target column
    target_col = None
    for col in TARGETS:
        if col in df.columns:
            target_col = col
            break
    assert target_col is not None, 'Target column not found!'
    print(f'Using target column: {target_col}')

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    # Select target from original DataFrame
    # Clean target column: remove non-numeric characters and convert to float
    y = df[target_col].astype(str).str.replace(r'[^\d.]', '', regex=True).replace('', np.nan).astype(float)
    # Keep only numeric features (excluding target)
    numeric = df.select_dtypes(include=[np.number])
    if target_col in numeric.columns:
        X = numeric.drop([target_col], axis=1)
    else:
        X = numeric
    # Fill NaNs in features with median
    X = X.fillna(X.median())
    print(f'Shape after cleaning (numeric features): {X.shape}')
    print('All numeric features at start:', list(X.columns))

    # Variance Threshold
    vt = VarianceThreshold(threshold=0.0)
    X_vt = vt.fit_transform(X)
    vt_cols = X.columns[vt.get_support()]
    X_vt = pd.DataFrame(X_vt, columns=vt_cols, index=X.index)
    print(f'Features after variance filter: {X_vt.shape[1]}')
    print('Features after variance filter:', list(X_vt.columns))

    # Pearson Correlation
    corrs = X_vt.corrwith(y).abs()
    pearson_feats = corrs[corrs > 0.3].index.tolist()
    print(f'Pearson-selected features: {len(pearson_feats)}')
    print('Features after Pearson filter:', pearson_feats)

    # Mutual Information
    if len(pearson_feats) > 0:
        skb = SelectKBest(mutual_info_regression, k=min(N_TOP_MI, len(pearson_feats)))
        skb.fit(X_vt[pearson_feats], y)
        mi_feats = X_vt[pearson_feats].columns[skb.get_support()].tolist()
    else:
        mi_feats = []
    print(f'Mutual Information-selected features: {len(mi_feats)}')
    print('Features after MI filter:', mi_feats)

    # Wrapper Methods: RFE and SFS
    union_feats = sorted(set(pearson_feats + mi_feats))
    X_union = X_vt[union_feats] if union_feats else X_vt

    # RFE
    n_feats_rfe = min(N_FINAL, max(1, X_union.shape[1] - 1)) if X_union.shape[1] > 1 else 1
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rfe = RFE(rf, n_features_to_select=n_feats_rfe)
    rfe.fit(X_union, y)
    rfe_feats = X_union.columns[rfe.get_support()].tolist()
    print(f'RFE-selected features: {len(rfe_feats)}')
    print('Features after RFE:', rfe_feats)

    # SFS
    n_feats_sfs = min(N_FINAL, max(1, X_union.shape[1] - 1)) if X_union.shape[1] > 1 else 1
    lr = LinearRegression()
    if X_union.shape[1] > 1:
        sfs = SequentialFeatureSelector(lr, n_features_to_select=n_feats_sfs, direction='forward', cv=3, scoring='r2')
        sfs.fit(X_union, y)
        sfs_feats = X_union.columns[sfs.get_support()].tolist()
    else:
        sfs_feats = X_union.columns.tolist()
    print(f'SFS-selected features: {len(sfs_feats)}')
    print('Features after SFS:', sfs_feats)

    # Core and Additional Features
    core = sorted(set(pearson_feats) & set(mi_feats) & set(rfe_feats) & set(sfs_feats))
    extras = [f for f in rfe_feats if f not in core]
    print('\nCore features (selected by ALL methods):', core)
    print('Additional features (from RFE):', extras)

    # Visualize
    labels = core + extras
    y_vals = [1]*len(labels)
    plt.figure(figsize=(10,4))
    plt.bar(labels, y_vals)
    plt.xticks(rotation=45, ha='right')
    plt.yticks([])
    plt.title('Most Important Features Selected')
    for i, lbl in enumerate(core):
        plt.text(i, 1.02, 'core', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig('data/gov/important_features.png')
    print('Visual saved to important_features.png') 
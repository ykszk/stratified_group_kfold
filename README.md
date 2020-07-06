# Stratified Group K-fold
[![codecov](https://codecov.io/gh/yk-szk/stratified_group_kfold/branch/master/graph/badge.svg)](https://codecov.io/gh/yk-szk/stratified_group_kfold)

Split dataset into k folds with balanced label distribution (stratified) and non-overlapping groups.

StratifiedGroupKFold class is compatible with [sklearn.model_selection.KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)

Reference : [Stratified Group k-Fold Cross-Validation | Kaggle](https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation)

## Install
``` sh
pip install git+https://github.com/yk-szk/stratified_group_kfold
```

## Usage
``` python
from stratified_group_kfold import StratifiedGroupKFold


X, y, groups = load_dataset()

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True)
for train_index, test_index in sgkf.split(X, y, groups):
    do_stuff(train_index, test_index)
```

[notebook example](example.ipynb)

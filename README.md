# Stratified Group K-fold
Split dataset into k folds with balanced label distribution (stratified) and non-overlapping groups.

StratifiedGroupKFold class is compatible with [sklearn.model_selection.KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)

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
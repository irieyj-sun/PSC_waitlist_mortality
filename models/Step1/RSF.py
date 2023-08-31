import pandas as pd
import numpy as np
from statistics import mean, stdev
from itertools import combinations
from sklearn.preprocessing import StandardScaler

import sksurv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
import warnings
from sklearn.exceptions import ConvergenceWarning

import random
from scipy.linalg import svd

def load_SRTR_dynamic(df):
    """
    Loads dynamic time series data from the STATHIST_LIIN
    table.
    """
    
    px_ids = {}
    for split in ["train", "val", "test"]:
        with open(f"/Users/iriesun/MLcodes/PSC2.0/data/data_splits/{split}_split.txt") as f:
            px_ids[split] = [float(id) for id in f]

    train_df= df[df["PX_ID"].isin(px_ids["train"])]
    val_df = df[df["PX_ID"].isin(px_ids["val"])]
    test_df = df[df["PX_ID"].isin(px_ids["test"])]
    cols = list(train_df.columns)

    return train_df, val_df, test_df, cols

longi_subset= pd.read_csv('/Users/iriesun/MLcodes/PSC2.0/data/longi_subset.csv',index_col=0)
train_df, val_df, test_df, cols = load_SRTR_dynamic(longi_subset)

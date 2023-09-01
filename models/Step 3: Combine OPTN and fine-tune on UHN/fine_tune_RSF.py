import pandas as pd
import numpy as np 

from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter

from sksurv.ensemble import RandomSurvivalForest

from sksurv.metrics import brier_score
from sksurv.metrics import concordance_index_censored
import warnings
import sksurv.metrics
from sklearn.exceptions import ConvergenceWarning

from tableone import TableOne
import matplotlib.pyplot as plt

import shap 
import pickle


"""
Load data
"""

###SRTR
srtr= pd.read_csv('/PSC/data/srtr_subset.csv',index_col=0)
srtr['wl_to_event'] = srtr['wl_to_event'] - srtr['time_since_baseline']

def combine48(row):
    if row['optn_region'] == 4:
        return 13
    if row['optn_region'] == 8:
        return 13
    else:
        return row['optn_region']

srtr["optn_region"] = srtr.apply (lambda row: combine48(row), axis=1)

def load_SRTR_dynamic(df):
    """
    Loads dynamic time series data from the SRTR dataset
    table.
    """
    
    px_ids = {}
    for split in ["train", "val", "test"]:
        with open(f"/PSC/data/data_splits/{split}_split.txt") as f:
            px_ids[split] = [float(id) for id in f]

    train_df= df[df["PX_ID"].isin(px_ids["train"])]
    val_df = df[df["PX_ID"].isin(px_ids["val"])]
    test_df = df[df["PX_ID"].isin(px_ids["test"])]
    cols = list(train_df.columns)

    return train_df, val_df, test_df, cols


srtr_train_df, srtr_val_df, srtr_test_df, cols = load_SRTR_dynamic(srtr)

srtr_train_df.loc[(srtr_train_df['event'] ==2),'event'] = 0
srtr_val_df.loc[(srtr_val_df['event'] ==2),'event'] = 0
srtr_test_df.loc[(srtr_test_df['event'] ==2),'event'] = 0

###UHN
uhn= pd.read_csv('/PSC/data/uhn_subset.csv',index_col=0)
uhn['wl_to_event'] = uhn['wl_to_event'] - uhn['time_since_baseline']
psc_uhn['optn_region']=12

def load_UHN(df):

    # Load the STATHIST_LIIN table into memory.
    # df = pd.read_sas(DATA_DIR + "stathist_liin.sas7bdat")

    # Load the patient IDs associated with the train, validation, and test 
    # splits into memory.
    px_ids = {}
    for split in ["train","test"]:
        with open(f"/Users/iriesun/MLcodes/PSC2.0/data/data_splits2/{split}_split.txt") as f:
            px_ids[split] = [float(id) for id in f]

    train_df= df[df["PX_ID"].isin(px_ids["train"])]
    test_df = df[df["PX_ID"].isin(px_ids["test"])]
    cols = list(train_df.columns)

    return train_df,test_df, cols
  
uhn_train_df,uhn_test_df, cols = load_UHN(psc)
uhn_train_df.loc[(uhn_train_df['event'] ==2),'event'] = 0
uhn_test_df.loc[(uhn_test_df['event'] ==2),'event'] = 0


"""
Scale features 
"""
get_x = lambda df: (df
                    .drop(columns=['PX_ID','wl_to_event','event'])
                    .values.astype('float32'))

get_time = lambda df: (df['wl_to_event'].values)
get_event = lambda df: (df['event'].values)

srtr_train_feature = get_x(srtr_train_df)
srtr_train_T = srtr_train_df['wl_to_event']
srtr_train_E = srtr_train_df['event']
srtr_train_Y= np.array([(bool(e), t) for (e, t) in zip(srtr_train_E, srtr_train_T)], dtype=[("e", bool), ("t", float)])

srtr_test_feature = get_x(srtr_test_df)
srtr_test_T = srtr_test_df['wl_to_event']
srtr_test_E = srtr_test_df['event']
srtr_test_Y= np.array([(bool(e), t) for (e, t) in zip(srtr_test_E, srtr_test_T)], dtype=[("e", bool), ("t", float)])

uhn_train_feature = get_x(uhn_train_df)
uhn_train_T = uhn_train_df['wl_to_event']
uhn_train_E = uhn_train_df['event']
uhn_train_Y= np.array([(bool(e), t) for (e, t) in zip(uhn_train_E, uhn_train_T)], dtype=[("e", bool), ("t", float)])

uhn_test_feature = get_x(uhn_test_df)
uhn_test_T = uhn_test_df['wl_to_event']
uhn_test_E = uhn_test_df['event']
uhn_test_Y= np.array([(bool(e), t) for (e, t) in zip(uhn_test_E, uhn_test_T)], dtype=[("e", bool), ("t", float)])


scaler = StandardScaler()

scaler.fit(srtr_train_feature)
scaler.transform(srtr_test_feature)
scaler.transform(uhn_train_feature)
scaler.transform(uhn_test_feature)


"""
Training and fine-tuning
"""
print('Training')

fine_tune_model = RandomSurvivalForest(random_state=1, max_features=2, min_samples_leaf=75,warm_start=True)
fine_tune_model.fit(srtr_train_feature, srtr_train_Y)

fine_tune_model.n_estimators += 200
fine_tune_model.fit(uhn_train_feature, uhn_train_Y)

"""
Testing
"""
def dynamic_score_dynamic_predictions(col, df, t_times, delta_t_times, NUM_ITERATIONS):
    out_mat = np.zeros((len(t_times), len(delta_t_times), NUM_ITERATIONS))

    df = df[~df[col].isnull()]
    for i, t in enumerate(t_times):
        # Step 1: Look only at rows with time less than t (we can't look into the future).
        df_t = df[df["time_since_baseline"] <= t]
        # Step 1.5 - remove NaN values in the MELD score.
        df_t = df_t[~df_t[col].isna()]
        # Step 2: take the maximum time (delta_T) per PX_ID as the rows.
        idx = df_t.groupby("PX_ID")["time_since_baseline"].transform(max) == df_t["time_since_baseline"]
        df_t = df_t[idx]

        for j, delta_t in enumerate(delta_t_times):

            for k in range(NUM_ITERATIONS):
                df_resample = df_t.sample(df_t.shape[0], replace=True)
                risks = np.array(df_resample[col]).reshape(-1, 1)
                risks = np.broadcast_to(risks, (risks.shape[0], len(delta_t_times)) )
                
                out_mat[i, j, k] = c_index(risks[:, j], np.asarray(df_resample["wl_to_event"]), np.asarray(df_resample["event"]), t+delta_t)

    return out_mat

def c_index(Prediction, Time_survival, Death, Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    '''
    N = len(Prediction)
    A = np.zeros((N,N))
    Q = np.zeros((N,N))
    N_t = np.zeros((N,N))
    Num = 0
    Den = 0
    for i in range(N):
        A[i, np.where(Time_survival[i] < Time_survival)] = 1
        Q[i, np.where(Prediction[i] > Prediction)] = 1
  
        if (Time_survival[i]<=Time and Death[i]==1):
            N_t[i,:] = 1

    Num  = np.sum(((A)*N_t)*Q)
    Den  = np.sum((A)*N_t)

    if Num == 0 and Den == 0:
        result = -1 # not able to compute c-index!
    else:
        result = float(Num/Den)

    return result

print('SRTR C-index')
srtr_test_df["risk"] = fine_tune_model.predict(srtr_test_feature)

t_times = [0, 30, 60, 90, 183, 365]       # Reference times
delta_t_times = [30, 60, 90, 183, 365,float('inf')]    # Prediction horizon times

dynamic_score_dynamic_predictions("risk",srtr_test_df, t_times, delta_t_times,50).mean(axis=2)

  
print('UHN C-index')
uhn_test_df["risk"] = fine_tune_model.predict(uhn_test_feature)

dynamic_score_dynamic_predictions("risk",uhn_test_df, t_times, delta_t_times,50).mean(axis=2)

filename = "PSC_RSF.pkl"
pickle.dump(fine_tune_model, open(filename, 'wb'))

print('Complete')

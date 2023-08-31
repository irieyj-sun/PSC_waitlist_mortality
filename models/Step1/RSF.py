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

import pickle 

"""
Load data
"""

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

#create event specific df 
death_train = train_df.copy()
death_val = val_df.copy()
death_te = test_df.copy()

death_train.loc[(death_train['event'] ==2),'event'] = 0
death_val.loc[(death_val['event'] ==2),'event'] = 0
death_te.loc[(death_te['event'] ==2),'event'] = 0

#create covariate matrix and T,E variables 
scaler = StandardScaler()
#train 
death_train['wl_to_event'] = death_train['wl_to_event']-death_train['time_since_baseline']

d_train_matrix = death_train.drop(['PX_ID','event','wl_to_event','optn_region'],axis=1)
scaler.fit(d_train_matrix)
scaler.transform(d_train_matrix)

d_train_T = death_train['wl_to_event']
d_train_E = death_train['event']
d_train_Y = np.array([(bool(e), t) for (e, t) in zip(d_train_E, d_train_T)], dtype=[("e", bool), ("t", float)])

#val 
death_val['wl_to_event'] = death_val['wl_to_event']-death_val['time_since_baseline']
d_val_matrix = death_val.drop(['PX_ID','event','wl_to_event','optn_region'],axis=1)
scaler.transform(d_val_matrix)

d_val_T = death_val['wl_to_event']
d_val_E = death_val['event']
d_val_Y = np.array([(bool(e), t) for (e, t) in zip(d_val_E, d_val_T)], dtype=[("e", bool), ("t", float)])


#test
death_te['wl_to_event'] = death_te['wl_to_event']-death_te['time_since_baseline']
d_te_matrix = death_te.drop(['PX_ID','event','wl_to_event','optn_region'],axis=1)
scaler.transform(d_te_matrix)

d_te_T = death_te['wl_to_event']
d_te_E = death_te['event']
d_te_Y = np.array([(bool(e), t) for (e, t) in zip(d_te_E, d_te_T)], dtype=[("e", bool), ("t", float)])


"""
Train and Evaluate
"""
##### WEIGHTED C-INDEX & BRIER-SCORE
def CensoringProb(Y, T):

    T = T.reshape([-1]) # (N,) - np array
    Y = Y.reshape([-1]) # (N,) - np array

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=(Y==0).astype(int))  # censoring prob = survival probability of event "censoring"
    G = np.asarray(kmf.survival_function_.reset_index()).transpose()
    G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]  #fill 0 with ZoH (to prevent nan values)
    
    return G


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


def dynamic_score_dynamic_predictions(col, df, t_times, delta_t_times, NUM_ITERATIONS):
    out_mat = np.zeros((len(t_times), len(delta_t_times), NUM_ITERATIONS))

    for i, t in enumerate(t_times):
        # Step 1: Look only at rows with time less than t (we can't look into the future).
        df_t = df[df["time_since_baseline"] <= t]
    
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


out_file = "rsf_hparam.txt"

print('Training')

for max_features in (2,5):
    
    for min_samples_leaf in (50,75,100,125):
        
        with open(out_file, "a") as f:

            f.write(f"MODEL: max_features={max_features}, min_sample_leaf={min_samples_leaf}\n")

            print(f"Training RSF, max_features={max_features},min_sample_leaf={min_samples_leaf}")

            model = RandomSurvivalForest(random_state=1, max_features=max_features, min_samples_leaf=min_samples_leaf)

            model.fit(d_train_matrix, d_train_Y)

            train_risk_scores = model.predict(d_train_matrix)

            val_risk_scores = model.predict(d_val_matrix)

            cindex_train=concordance_index_censored(event_indicator=d_train_Y['e'],
                                              event_time=d_train_Y['t'],
                                              estimate=train_risk_scores)

            cindex_val=concordance_index_censored(event_indicator=d_val_Y['e'],
                                              event_time=d_val_Y['t'],
                                              estimate=val_risk_scores)

            print("Training C-Index: " + str(cindex_train[0])+ "Validation C-Index: " + str(cindex_val[0]))

            try:

                with open(out_file, "a") as f:

                    f.write("Training C-Index: " + str(cindex_train[0])+'\n')

                    f.write("Validation C-Index: " + str(cindex_val[0])+'\n')

            except ValueError:

                with open(out_file, "a") as f:

                    f.write("NaN\n")

rsf1_best=RandomSurvivalForest(random_state=1, max_features=2, min_samples_leaf=100)
rsf1_best.fit(d_train_matrix, d_train_Y)

t_times = [0, 30, 60, 90, 183, 365]                     # Reference times
delta_t_times = [30, 60, 90, 183, 365, float("inf")]    # Prediction horizon times

death_te["risk"] = rsf1_best.predict(d_te_matrix)

out_mats = {}
out_mats['test'] = dynamic_score_dynamic_predictions("risk",death_te, t_times, delta_t_times, 100)

print('Training complete')
print('SRTR C-index performance')
out_mats['test'].mean(axis=2)

filename = "PSC_SRTR_RSF.pkl"
pickle.dump(rsf1_best, open(filename, 'wb'))



"""
Test on UHN data
"""

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

psc_uhn= pd.read_csv('/Users/iriesun/MLcodes/PSC/data/uhndf_alt.csv',index_col=0)
psc_uhn['wl_to_event'] = psc_uhn['wl_to_event'] - psc_uhn['time_since_baseline']
psc_uhn.loc[(psc_uhn['event'] ==2),'event'] = 0

uhn_train_df,uhn_test_df, cols = load_UHN(psc_uhn)

get_x = lambda df: (df
                    .drop(columns=['PX_ID','wl_to_event','event'])
                    .values.astype('float32'))

uhn_test_features = get_x(uhn_test_df)

scaler.transform(uhn_test_features)

uhn_test_T = uhn_test_df['wl_to_event']
uhn_test_E = uhn_test_df['event']
uhn_test_Y = np.array([(bool(e), t) for (e, t) in zip(uhn_test_E, uhn_test_T)], dtype=[("e", bool), ("t", float)])


uhn_test_df["risk"] = rsf1_best.predict(uhn_test_features)

t_times = [0, 30, 60, 90, 183, 365]       # Reference times
delta_t_times = [30, 60, 90, 183, 365,float('inf')]    # Prediction horizon times

out_mats=dynamic_score_dynamic_predictions("risk",uhn_test_df, t_times, delta_t_times,100)

print('UHN C-index performance')
out_mats.mean(axis=2)

import os
import theano
import lasagne

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from deep_surv import DeepSurv

"""
Load data
"""

def load_SRTR_dynamic(df):
    """
    Loads dynamic time series data from the SRTR dataset 
    table.
    """
    # Load the patient IDs associated with the train, validation, and test 
    # splits into memory.
    px_ids = {}
    for split in ["train", "val", "test"]:
        with open(f"/PSC/data/data_splits/{split}_split.txt") as f:
            px_ids[split] = [float(id) for id in f]

    train_df= df[df["PX_ID"].isin(px_ids["train"])]
    val_df = df[df["PX_ID"].isin(px_ids["val"])]
    test_df = df[df["PX_ID"].isin(px_ids["test"])]
    cols = list(train_df.columns)

    return train_df, val_df, test_df, cols

longiall= pd.read_csv('/PSC/data/srtr_subset.csv',index_col=0)

train_df, val_df, test_df, cols = load_SRTR_dynamic(longiall)

train_df['wl_to_event'] = train_df['wl_to_event']-train_df['time_since_baseline']
val_df['wl_to_event'] = val_df['wl_to_event']-val_df['time_since_baseline']
test_df['wl_to_event'] = test_df['wl_to_event']-test_df['time_since_baseline']


death_train = train_df.copy()
death_val = val_df.copy()
death_te = test_df.copy()

death_train.loc[(death_train['event'] ==2),'event'] = 0
death_val.loc[(death_val['event'] ==2),'event'] = 0
death_te.loc[(death_te['event'] ==2),'event'] = 0

def f_get_Normalization(X, norm_mode):
    num_Patient, num_Feature = np.shape(X)

    if norm_mode == 'standard': #zero mean unit variance
        for j in range(num_Feature):
            if np.std(X[:,j]) != 0:
                X[:,j] = (X[:,j] - np.mean(X[:, j]))/np.std(X[:,j])
            else:
                X[:,j] = (X[:,j] - np.mean(X[:, j]))
    elif norm_mode == 'normal': #min-max normalization
        for j in range(num_Feature):
            X[:,j] = (X[:,j] - np.min(X[:,j]))/(np.max(X[:,j]) - np.min(X[:,j]))
    else:
        print("INPUT MODE ERROR!")

    return X

def format_deepsurv(current_df, current_set, train_df):
    
    df = current_df.drop("PX_ID", axis=1)
    
    T = np.array(df["wl_to_event"])
    E = np.array(df["event"])
    X = np.array(df.drop(["wl_to_event", "event"], axis=1))
    
    num_Patient, num_Feature = np.shape(X)
    
    X_train = np.array(train_df.drop(["wl_to_event", "event",'PX_ID'], axis=1))
    
    if current_set == 'train':
        
        for j in range(num_Feature):
            if np.std(X[:,j]) != 0:
                X[:,j] = (X[:,j] - np.mean(X[:, j]))/np.std(X[:,j])
            else:
                X[:,j] = (X[:,j] - np.mean(X[:, j]))
        return {
        'x' : X.astype(np.float32),
        't' : T.astype(np.float32),
        'e' : E.astype(np.int32)
    }
    
    else:
        for j in range(num_Feature):
            if np.std(X[:,j]) != 0:
                X[:,j] = (X[:,j] - np.mean(X_train[:, j]))/np.std(X_train[:,j])
            else:
                X[:,j] = (X[:,j] - np.mean(X_train[:, j]))
        return {
        'x' : X.astype(np.float32),
        't' : T.astype(np.float32),
        'e' : E.astype(np.int32)
    } 

dtrain = format_deepsurv(death_train,'train',death_train)
dval = format_deepsurv(death_val,'val',death_train)
dtest = format_deepsurv(death_te,'test',death_train)

"""
Hyperparameter search
"""


out_file = "DS_hparam.txt"

with open(out_file, "w+") as f:
    f.write("Features: {features}")

hidden_layers_sizes_space = [
    [20,20],
    [50, 50],
    [100,100],
    [20,20,20],
    [50,50,50],
]

for hidden_layers_sizes in hidden_layers_sizes_space:
    for lr in (1e-3, 1e-4):
        for l2 in (0,5):
            for epochs in (200,500):
                with open(out_file, "a") as f:
                    f.write(f"MODEL: hidden_layers={hidden_layers_sizes}, lr={lr}, l2={l2}, epochs={epochs}\n")
                model = DeepSurv(
                    hidden_layers_sizes=hidden_layers_sizes,
                    learning_rate=lr,
                    momentum=0,
                    L2_reg=l2,
                    n_in=dtrain['x'].shape[1],
                    lr_decay=0.0
                )

                model.train(dtrain, valid_data=dval, n_epochs=1000)
                try:
                    with open(out_file, "a") as f:
                        f.write("Training C-Index: " + str(model.get_concordance_index(dtrain['x'], dtrain['t'], dtrain['e']))+'\n')
                        f.write("Validation C-Index: " + str(model.get_concordance_index(dval['x'], dval['t'], dval['e']))+'\n')
                
                except ValueError:
                    with open(out_file, "a") as f:
                        f.write("NaN\n")

## print("Best hyperparameters: hidden_layers=[20, 20], lr=0.001, l2=5, epochs=200")

    
"""
## Training and evaluation  
"""
model = DeepSurv(
    hidden_layers_sizes=[20, 20],
    learning_rate=0.001,
    momentum=0,
    L2_reg=5,
    n_in=dtrain['x'].shape[1],
    lr_decay=0.0
)
print("Training")
model.train(dtrain, n_epochs=200)

death_te["risk"] = model.predict_risk(dtest['x']).squeeze()


def tv_c_statistic(Prediction, Time_survival, Death, Time):
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
        # Make sure the DataFrame contains only:
        # 1. Patients who have not died / been censored prior to the reference time.
        # df_t = df[df["T"] > t]
        # THE ABOVE IS NO LONGER NECESSARY SINCE WE ACCOUNT FOR THIS IN OUR CALCULATION OF T

        # Now replace with patients' most recent MELD/MELDNa/MELD3.0 scores
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

                out_mat[i, j, k] = tv_c_statistic(risks[:, j], np.asarray(df_resample["wl_to_event"]), np.asarray(df_resample["event"]), t+delta_t)
                #print(f"{t},{delta_t}: {out_mat[i,j,k]}")

    return out_mat

t_times = [0,30, 60, 90, 183, 365]                     # Reference times
delta_t_times = [30, 60, 90, 183, 365,float('inf')]    # Prediction horizon times

results_mat = dynamic_score_dynamic_predictions("risk", death_te, t_times, delta_t_times, 100)

print("Training complete")
print("SRTR C-index performance")
results_mat.mean(axis=2)

"""
Testing on UHN data
"""
uhndata=pd.read_csv('/PSC/data/uhn_subset.csv',index_col=0)

uhn_te_d = uhndata.copy()
uhn_te_d.loc[(uhn_te_d['event'] ==2),'event'] = 0
uhn_te= format_deepsurv(uhn_te_d)
test_result= dynamic_score_dynamic_predictions("risk", uhn_te_d, t_times, delta_t_times, 100)

print("UHN C-index performance")
test_result.mean(axis=2)
  


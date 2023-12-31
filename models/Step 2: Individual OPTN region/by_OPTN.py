import pandas as pd
import numpy as np 

from lifelines import KaplanMeierFitter

from sksurv.ensemble import RandomSurvivalForest

from sksurv.metrics import brier_score
from sksurv.metrics import concordance_index_censored
import warnings
from sklearn.exceptions import ConvergenceWarning


import matplotlib.pyplot as plt
from matplotlib import rcParams
"""
Load data
"""

srtrdf = pd.read_csv('/PSC/data/srtr_subset.csv',index_col=0)
uhndf = pd.read_csv('/PSC/data/uhn_subset.csv',index_col=0)

uhndf['wl_to_event'] = uhndf['wl_to_event'] - uhndf['time_since_baseline']
srtrdf['wl_to_event'] = srtrdf['wl_to_event'] - srtrdf['time_since_baseline']

longiall = srtrdf.copy()
uhn = uhndf.copy()

longiall.loc[(longiall['event'] ==2),'event'] = 0
uhn.loc[(uhn['event'] ==2),'event'] = 0

uhn['optn_region']=12  #treat UHN as an additional region

test_df = longiall.sample(frac=0.2)
train_df = longiall.drop(test_df.index)

test_df=test_df.append(uhn) # use UHN only for testing 

### C-index 
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

"""
Create OPTN dataframe
"""
optn_list = test_df.optn_region.unique()

get_x = lambda df: (df
                    .drop(columns=['PX_ID','wl_to_event','event'])
                    .values.astype('float32'))

get_time = lambda df: (df['wl_to_event'].values)
get_event = lambda df: (df['event'].values)

train_feature_df = {elem : pd.DataFrame() for elem in optn_list}
train_target_df = {elem : pd.DataFrame() for elem in optn_list}

test_feature_df = {elem : pd.DataFrame() for elem in optn_list}
test_target_df = {elem : pd.DataFrame() for elem in optn_list}

for key in test_target_df.keys():
    
    #full df
    #optn_list_dict[key] = full_df[:][full_df.optn_region == key]
    
    #feature df
    train_feature_df[key] = get_x(train_df[:][train_df.optn_region == key])
    test_feature_df[key] = get_x(test_df[:][test_df.optn_region == key])
    
    #target df
    train_time = get_time(train_df[:][train_df.optn_region == key])
    train_event = get_event(train_df[:][train_df.optn_region == key])
    train_target_df[key] =  pd.DataFrame({'event':train_event,'wl_to_event':train_time})
    
    test_time = get_time(test_df[:][test_df.optn_region == key])
    test_event = get_event(test_df[:][test_df.optn_region == key])
    test_target_df[key] =  pd.DataFrame({'event':test_event,'wl_to_event':test_time})

rsf_out_mat = np.zeros([11,12])
cph_out_mat = np.zeros([11,12])
"""
Training and testing
"""
print("Training and testing by OPTN")
rsf = RandomSurvivalForest(random_state=1, max_features=5, min_samples_leaf=2)
cph = CoxnetSurvivalAnalysis(alpha_min_ratio=0.001,l1_ratio=0.9)

times = np.arange(1, 12)

counter=0

t_times = [0, 30, 60, 90, 183, 365]                     # Reference times
delta_t_times = [30, 60, 90, 183, 365, float("inf")]    # Prediction horizon times

for key in train_feature_df.keys():
    
    if key!=12: 
    
        train_features = train_feature_df[key]

        train_target = train_target_df[key]

        train_T = train_target['wl_to_event']
        train_E = train_target['event']
        train_Y = np.array([(bool(e), t) for (e, t) in zip(train_E, train_T)], dtype=[("e", bool), ("t", float)])

        print('Training OPTN region: {}'.format(key))

        rsf.fit(train_features, train_Y)
        cph.fit(train_features, train_Y)

        ##### find test dfs #####
        #test_optn_list = optn_list.tolist()

        #del test_optn_list [counter]

        #counter +=1

        #test_optn_df = {elem : pd.DataFrame() for elem in test_optn_list}

        ##### test all other OPTNs #####
        for test_key in test_feature_df.keys():

            test_features = test_feature_df[test_key]

            print('Testing OPTN region: {}'.format(test_key))

            test_df_record = test_df[:][test_df.optn_region == test_key]

            #rsf
            test_df_record["risk"] = rsf.predict(test_features)

            key = int(key)-1
            test_key = int(test_key)-1
            rsf_out_mat[key][test_key] = dynamic_score_dynamic_predictions("risk",test_df_record, t_times, delta_t_times,10).mean(axis=(0,1,2)
                                                                                                                                  
            #cph
            test_df_record = test_df_record.drop('risk',axis=1)                                                                                                          
            test_df_record["risk"] = cph.predict(test_features)                                                                                                                     
            cph_out_mat[key][test_key] = dynamic_score_dynamic_predictions("risk",test_df_record, t_times, delta_t_times,10).mean(axis=(0,1,2)))

print("RSF By OPTN C-index")
rsf_mat=pd.DataFrame(rsf_out_mat)

rsf_mat.columns =['OPTN 1', 'OPTN 2', 'OPTN 3','OPTN 4','OPTN 5',
        'OPTN 6','OPTN 7','OPTN 8','OPTN 9','OPTN 10','OPTN 11','UHN']

# Change the row indexes
rsf_mat.index = ['OPTN 1', 'OPTN 2', 'OPTN 3','OPTN 4','OPTN 5',
        'OPTN 6','OPTN 7','OPTN 8','OPTN 9','OPTN 10','OPTN 11']

print("CPH By OPTN C-index")
cph_mat=pd.DataFrame(cph_out_mat)

cph_mat.columns =['OPTN 1', 'OPTN 2', 'OPTN 3','OPTN 4','OPTN 5',
        'OPTN 6','OPTN 7','OPTN 8','OPTN 9','OPTN 10','OPTN 11','UHN']

# Change the row indexes
cph_mat.index = ['OPTN 1', 'OPTN 2', 'OPTN 3','OPTN 4','OPTN 5',
        'OPTN 6','OPTN 7','OPTN 8','OPTN 9','OPTN 10','OPTN 11']

"""
Heatmap
"""
def create_cindex_heatmap(df,model_name):

    rcParams['font.family'] = 'Times New Roman'

    # Find the highest value in each column
    highest_values = df.idxmax(axis=1)

    # Create a heatmap
    fig, ax = plt.subplots(figsize=(8,6))
    heatmap = sns.heatmap(df, annot=True, fmt='.3f', cmap='PuBu')

    # Highlight the highest values
    for i, row in enumerate(df.index):
        col = highest_values[i]
        value = df.loc[row, col]
        ax.text(df.columns.get_loc(col) + 0.5, i + 0.5, f'{value:.3f}', ha='center', va='center',
                color='yellow', fontweight='bold')

    ax.set_xlabel('Testing OPTN regions C-index')
    ax.set_ylabel('Training OPTN regions')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)

    ax.set_title('Average C-index performance of ' + model_name + ' model trained on individual OPTN regions  ')

    # Adjust plot layout
    plt.tight_layout()

    # Show the heatmap
    plt.savefig("%s.png" % model_name)

create_cindex_heatmap(rsf_mat,"RSF")
create_cindex_heatmap(cph_mat,"CPH")


print('Complete')

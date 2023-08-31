import pandas as pd
import numpy as np

import datetime

DATA_DIR = ''

"""
Institution data
"""
center=pd.read_sas(DATA_DIR + 'institution.sas7bdat')

def convert_primary_province(row):
    name = str(row['PRIMARY_STATE'])
    name = name.replace('b','')
    name = name.replace("'", "")
    return name

center['OPTN_STATE'] = center.apply (lambda row: convert_primary_province(row), axis=1)

optn_state=center['OPTN_STATE'].unique()
optn_state= np.delete(optn_state, 43)
optn_region=[]

for state in optn_state:
    if state == 'VA':
        optn_region.append(11.0)
    else:
        region = (center['REGION'].loc[center['OPTN_STATE'] == state]).unique()
        region.astype(int)
        optn_region.extend(region)

optn_df= pd.DataFrame({'CAN_PERM_STATE':optn_state , 'optn_region': optn_region})

"""
Process baseline data
"""
# outout data file is the SRTR PSC cohort with inclusion excluion criteria applied 
candata = pd.read_excel(DATA_DIR + 'output.xlsx")

#check dates
candata = candata.loc[candata["CAN_ACTIVATE_DT"]<candata["CAN_ENDWLFU"]]

activedate=pd.to_datetime(datetime.date(2002, 2,27))
candata=candata.loc[candata["CAN_ACTIVATE_DT"]>activedate]

candata=candata.loc[candata['CAN_INIT_ACT_STAT_CD']!=(6010|6011)] #exclude patients listed under Status 1 and 1A

endofstudy=pd.to_datetime(datetime.date(2020, 11,30))

candata = candata.loc[candata["CAN_REM_DT"]<endofstudy]

print(f"Data shape before processing:{candata.shape}")

#covert blood type variable into A,AB,B,O
def abo(row):
    if row['CAN_ABO'] in ['A','A1','A2']:
        return 'A'
    if row['CAN_ABO'] in ['A1B','A2B','AB']:
        return 'AB'
    if row['CAN_ABO'] =='B':
        return 'B'
    else:
        return 'O'

candata["CAN_ABO"] =candata.apply (lambda row: abo(row), axis=1)

#add OPTN data
candata=candata.merge(optn_df, on='CAN_PERM_STATE',how='inner')

class FeatureExtractor():
    
    def extract_total_features(self, candata):
        """ 
        Accepts as input a dataframe of the form in candata. Include and format relevant variables 
        for models.
        """
        processor = FeatureProcessor()
       
        out = pd.DataFrame(candata["PX_ID"])
        
        out = pd.concat([out, candata["optn_region"]], axis=1)
        
        #age
        out = pd.concat([out, candata["CAN_AGE_AT_LISTING"]], axis=1)
        out = pd.concat([out, candata["CAN_AGE_IN_MONTHS_AT_LISTING"]], axis=1)
        
        #gender
        out = pd.concat([out,
            processor.gender_binary(candata["CAN_GENDER"])], axis=1)
    
        #blood type
        out = pd.concat([out, 
            pd.get_dummies(candata["CAN_ABO"], prefix="CAN_ABO")], axis=1)
        
        #race and ethnicity
        out = pd.concat([out, 
            processor.race_binary(candata["CAN_RACE_SRTR"])], axis=1)
        out = pd.concat([out, 
            processor.ethnicity_binary(candata["CAN_ETHNICITY_SRTR"])], axis=1)
        out = pd.concat([out, candata["CAN_RACE_AMERICAN_INDIAN"]], axis=1)
        out = pd.concat([out, candata["CAN_RACE_ASIAN"]], axis=1)
        out = pd.concat([out, candata["CAN_RACE_BLACK_AFRICAN_AMERICAN"]], axis=1)
        out = pd.concat([out, candata["CAN_RACE_HISPANIC_LATINO"]], axis=1)
        out = pd.concat([out, candata["CAN_RACE_NATIVE_HAWAIIAN"]], axis=1)
        out = pd.concat([out, candata["CAN_RACE_WHITE"]], axis=1)
        
        #education 
        out = pd.concat([out,
            processor.ordinal_meanpadding(candata["CAN_EDUCATION"], {996, 998})], axis=1)
        
        #demographics
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_WORK_INCOME"].fillna(b'U'))], axis=1)
        out = pd.concat([out, 
            pd.get_dummies(candata["CAN_WORK_YES_STAT"], prefix="CAN_WORK_YES_STAT")], axis=1)
        out = pd.concat([out, 
            pd.get_dummies(candata["CAN_SECONDARY_PAY"], prefix="CAN_SECONDARY_PAY")], axis=1)
        
        #medical/physical condition
        #continous, fill na with mean
        out = pd.concat([out, candata["CAN_WGT_KG"].fillna(candata["CAN_WGT_KG"].mean())], axis=1)
        out = pd.concat([out,candata["CAN_BMI"].fillna(candata["CAN_BMI"].mean())], axis=1)
        out = pd.concat([out, candata["CAN_INIT_ACT_STAT_CD"].fillna(candata["CAN_INIT_ACT_STAT_CD"].mean())], axis=1)
        out = pd.concat([out, candata["CAN_TOT_ALBUMIN"].fillna(candata["CAN_TOT_ALBUMIN"].mean())], axis=1)
        
        #categorical 
        out = pd.concat([out, 
            pd.get_dummies(candata["CAN_EMPL_STAT"], prefix="CAN_EMPL_STAT")], axis=1)
        out = pd.concat([out, 
            pd.get_dummies(candata["CAN_FUNCTN_STAT"], prefix="CAN_FUNCTN_STAT")], axis=1)
        out = pd.concat([out, 
            pd.get_dummies(candata["CAN_MED_COND"], prefix="CAN_MED_COND")], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_MALIG"])], axis=1)
        out = pd.concat([out, 
            pd.get_dummies(candata["CAN_MALIG_TY"], prefix="CAN_MALIG_TY")], axis=1)
        out = pd.concat([out, 
            processor.yesno_to_numeric(candata["CAN_LIFE_SUPPORT"])],axis=1)
        out = pd.concat([out, 
            processor.yesno_to_numeric(candata["CAN_LIFE_SUPPORT_OTHER"])],axis=1)
        out = pd.concat([out,
            pd.get_dummies(candata["CAN_INIT_STAT"], prefix="CAN_INIT_STAT")],axis=1)
        out = pd.concat([out, 
            pd.get_dummies(candata["CAN_PHYSC_CAPACITY"], prefix="CAN_PHYSC_CAPACITY")], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_PREV_TXFUS"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_PREV_ABDOM_SURG"])], axis=1)
        out = pd.concat([out, 
            processor.yesno_to_numeric(candata["CAN_VENTILATOR"])],axis=1)
        
       
        #diagnosis
        out = pd.concat([out,
            pd.get_dummies(candata["CAN_DGN"], prefix="CAN_DGN")],axis=1)
        out = pd.concat([out,
            pd.get_dummies(candata["CAN_DGN2"], prefix="CAN_DGN2")],axis=1)
        
        #comorbidities
        out = pd.concat([out,
            pd.get_dummies(candata["CAN_ANGINA"], prefix="CAN_ANGINA")],axis=1)
        out = pd.concat([out,
            pd.get_dummies(candata["CAN_ANGINA_CAD"], prefix="CAN_ANGINA_CAD")],axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_ASCITES"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_BACTERIA_PERIT"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_CEREB_VASC"])], axis=1)
        out = pd.concat([out,
            pd.get_dummies(candata["CAN_DIAB_TY"], prefix="CAN_DIAB_TY")],axis=1)
        out = pd.concat([out,
            pd.get_dummies(candata["CAN_DIAB"], prefix="CAN_DIAB")],axis=1)
        out = pd.concat([out,
            pd.get_dummies(candata["CAN_DIAL"], prefix="CAN_DIAL")],axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_DRUG_TREAT_HYPERTEN"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_DRUG_TREAT_COPD"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_ENCEPH"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_MUSCLE_WASTING"])], axis=1)
        out = pd.concat([out,
            pd.get_dummies(candata["CAN_PEPTIC_ULCER"], prefix="CAN_PEPTIC_ULCER")],axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_PERIPH_VASC"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_PORTAL_VEIN"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_PULM_EMBOL"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_TIPSS"])], axis=1)
        out = pd.concat([out, 
            processor.yesnounknown_to_numeric(candata["CAN_VARICEAL_BLEEDING"])], axis=1)
        
        return out

class DynamicFeatureExtractor():
    
    def extract_features(self, df):
        
        processor = FeatureProcessor()

        out = pd.DataFrame(df["PX_ID"])
        # Add in the reference features first
        out = pd.concat([out, df["CANHX_BEGIN_DT"]], axis=1)
        out = pd.concat([out, df["CANHX_BEGIN_DT_TM"]], axis=1)
        out = pd.concat([out, df["CANHX_END_DT"]], axis=1)
        out = pd.concat([out, df["CANHX_END_DT_TM"]], axis=1)
        out = pd.concat([out, df["CAN_LISTING_DT"]], axis=1)
        out = pd.concat([out, df["CAN_REM_DT"]], axis=1)

        # Add in the covariate features
        
        out = pd.concat([out, df["CANHX_ALBUMIN_BOUND"].fillna(df["CANHX_ALBUMIN_BOUND"].mean())], axis=1)
        out = pd.concat([out, df["CANHX_BILI_BOUND"].fillna(df["CANHX_BILI_BOUND"].mean())], axis=1)
        out = pd.concat([out, df["CANHX_CREAT_BOUND"].fillna(df["CANHX_CREAT_BOUND"].mean())], axis=1)
        out = pd.concat([out, df["CANHX_INR_BOUND"].fillna(df["CANHX_INR_BOUND"].mean())], axis=1)
        out = pd.concat([out, df['CANHX_SERUM_SODIUM'].fillna(df['CANHX_SERUM_SODIUM'].mean())], axis=1)
        #out = pd.concat([out, df["CANHX_OPTN_LAB_MELD"].fillna(df["CANHX_OPTN_LAB_MELD"].mean())], axis=1)
        #out = pd.concat([out, df["CANHX_PREV_LOWER_MELD_SCORE"].fillna(df["CANHX_PREV_LOWER_MELD_SCORE"].mean())], axis=1)
        out = pd.concat([out, df["CANHX_SRTR_LAB_MELD"].fillna(df["CANHX_SRTR_LAB_MELD"].mean())], axis=1)

        return out

  class FeatureProcessor():
    """
    Utility class to gather together feature processing functions.
    """
    def yesno_to_numeric(self, col):
        mapping = {'N': 0, 'Y': 1}
        def safe_mapping(key):
            try:
                return mapping[key]
            except KeyError:
                return -1
        return col.apply(safe_mapping)

    def yesnounknown_to_numeric(self, col):
        mapping = {'N': 0, 'Y': 1, 'U': 0.5}
        def safe_mapping(key):
            try:
                return mapping[key]
            except KeyError:
                return 0.5 # If we don't know - assume that it's average
        return col.apply(safe_mapping)
    
    def yesnounknown_to_numeric_a(self, col):
        mapping = {'N': 0, 'Y': 1, 'A': 0.5}
        def safe_mapping(key):
            try:
                return mapping[key]
            except KeyError:
                return 0.5 # If we don't know - assume that it's average
        return col.apply(safe_mapping)

    def ordinal_meanpadding(self, col, exceptions):
        def apply_ordinal(elem):
            if elem in exceptions:
                return np.nan
            return elem
        return col.apply(apply_ordinal).fillna(col.mean())
    
    def race_binary(self, col):
        return col.apply(lambda x: 1 if x == 'WHITE' else 0)

    def ethnicity_binary(self, col):
        return col.apply(lambda x : 1 if x == 'NLATIN' else 0)

    def gender_binary(self, col):
        return col.apply(lambda x : 1 if x == 'F' else 0)

extractor = FeatureExtractor()

out = extractor.extract_total_features(candata)

# Time to event variables
def find_event (row):
  if pd.isnull(row['REC_TX_DT']) == False :
    return 2
  if row['CAN_REM_CD'] in [8,9]:
    return 1
  if row['CAN_REM_CD']== 13:
     if (pd.isnull(row["PERS_SSA_DEATH_DT"])== False) | (pd.isnull(row["CAN_DEATH_DT"])== False):
        return 1
  return 0
candata["event"] =candata.apply (lambda row: find_event(row), axis=1)

def find_event_time (row):
   if pd.isnull(row['REC_TX_DT']) == False :
      return row['REC_TX_DT']
   if row['CAN_REM_CD']== 13:
       if (pd.isnull(row["PERS_SSA_DEATH_DT"])== False):
            return row["PERS_SSA_DEATH_DT"]
       if (pd.isnull(row["CAN_DEATH_DT"])== False):
            return row["CAN_DEATH_DT"]
   if pd.isnull(row["CAN_REM_DT"])== False:
      return row["CAN_REM_DT"]
   return row["CAN_ENDWLFU"]
   
candata["lastdate"] =candata.apply (lambda row: find_event_time(row), axis=1)

candata["wl_to_event"]=pd.Series((candata['lastdate']-candata["CAN_LISTING_DT"]).dt.days)

staticall= pd.concat([out, candata["wl_to_event"]], axis=1)
staticall = pd.concat([staticall, candata["event"].astype('category')], axis=1)

staticall['CAN_INIT_ACT_STAT_CD']=staticall['CAN_INIT_ACT_STAT_CD']-6200
staticall=staticall.loc[staticall['CAN_INIT_ACT_STAT_CD']>0]

"""
Longtiduinal data
"""

longidata= pd.read_sas(DATA_DIR +'cand_tx_statjust_stathist.sas7bdat')


#check dates
longidata = longidata.loc[longidata["CAN_ACTIVATE_DT"]<longidata["CAN_ENDWLFU"]]

endofstudy=pd.to_datetime(datetime.date(2020, 11,30))

longidata = longidata.loc[longidata["CAN_REM_DT"]<endofstudy]

dynamicextractor = DynamicFeatureExtractor()
longivariable=dynamicextractor.extract_features(longidata)

longicombined = pd.merge(staticforuse, longivariable,on="PX_ID", how = "left")

longicombined["time_since_baseline"] = longicombined.apply(lambda row : float((row["CANHX_BEGIN_DT_TM"] - row["CAN_LISTING_DT"]).days), axis=1)


#drop date variables
longiall = longicombined.drop(["CANHX_BEGIN_DT",
                                 "CANHX_BEGIN_DT_TM",
                                 "CANHX_END_DT",
                                 "CANHX_END_DT_TM",
                                 "CAN_LISTING_DT", 
                                 "CAN_REM_DT"],axis=1)


longiall['CANHX_SRTR_LAB_MELD']=longiall['CANHX_SRTR_LAB_MELD']-6200

longiall=longiall[longiall['wl_to_event']>=0]
longiall=longiall[longiall['time_since_baseline']<=longiall['wl_to_event']]

longiall.to_csv("longiall.csv")

# We perform a 70%-15%-15% train-val-test split.
train_splt, val_splt = 0.7, 0.85

np.random.seed(1234)

patient_identifiers = np.array(staticforuse["PX_ID"])

np.random.shuffle(patient_identifiers)

train, val, test = np.split(patient_identifiers,
                [int(train_splt*len(patient_identifiers)), 
                int(val_splt*len(patient_identifiers))]
                )

with open("/PSC/data/data_splits/train_split.txt", "w") as f:
    f.write("\n".join(train.astype('str')))

with open("/PSC/data/data_splits/val_split.txt", "w") as f:
    f.write("\n".join(val.astype('str')))

with open("/PSC/data/data_splitss/test_split.txt", "w") as f:
    f.write("\n".join(test.astype('str')))


#subset of variables shared with UHN
longi_subset=longiall[['PX_ID', 'CAN_AGE_AT_LISTING','CAN_GENDER','CAN_RACE_WHITE',
 'CAN_INIT_ACT_STAT_CD', 
 'CAN_WGT_KG', 'CAN_BMI','CAN_ASCITES',
 'CAN_ABO_A', 'CAN_ABO_AB', 'CAN_ABO_B', 'CAN_ABO_O',
 #longitudinal 
 'CANHX_ALBUMIN_BOUND','CANHX_CREAT_BOUND',
 'CANHX_INR_BOUND', 'CANHX_SRTR_LAB_MELD',
 'CANHX_BILI_BOUND', 
 'CANHX_SERUM_SODIUM',
 #event indicator, time 
'wl_to_event', 'event','time_since_baseline',
'optn_region']]

#convert lab variables to the same unit 
longi_subset['CANHX_ALBUMIN_BOUND']=longi_subset['CANHX_ALBUMIN_BOUND']*10
longi_subset['CANHX_CREAT_BOUND']=longi_subset['CANHX_CREAT_BOUND']*88.42
longi_subset['CANHX_BILI_BOUND']=longi_subset['CANHX_BILI_BOUND']*17.1
longi_subset['CANHX_SERUM_SODIUM']=longi_subset['CANHX_SERUM_SODIUM'].astype(float)


longi_subset.to_csv('longi_subset.csv')

print('Processing complete')




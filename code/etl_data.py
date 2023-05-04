import os
import pickle
import pandas as pd
import numpy as np

path = "../data/all_hourly_data.h5"
print("processing patients")
patients = pd.read_hdf(path, key="patients", columns=['gender', 'ethnicity', 'age', 'mort_icu', 'los_icu'])
patients['los_icu'] = (patients['los_icu']>7).astype(int)
patients['ethnicity'] = patients['ethnicity'].str[:5]
dummies1 = pd.get_dummies(patients['gender']).astype(int)
dummies2 = pd.get_dummies(patients['ethnicity']).astype(int)
patients = pd.concat([patients, dummies1, dummies2], axis='columns')
patients['HISPA'] = np.sum(patients[['CARIB', 'HISPA', 'SOUTH']], axis=1)
patients = patients[['age', 'F', 'ASIAN', 'BLACK', 'WHITE', 'HISPA', 'mort_icu', 'los_icu']]

print("processing interventions")
interventions = pd.read_hdf(path, key="interventions")
interventions = interventions.merge(patients[['age', 'F', 'ASIAN', 'BLACK', 'WHITE', 'HISPA']], how='left', on = ['subject_id', 'hadm_id','icustay_id'])
interventions['interventions'] = interventions.values.tolist()
data = interventions.groupby(['subject_id', 'hadm_id','icustay_id'])['interventions'].apply(list).reset_index()

print("processing vitals_labs")
vitals_labs = pd.read_hdf(path, key="vitals_labs")
vitals_labs = vitals_labs.fillna(0)
vitals_labs = vitals_labs.merge(patients[['age', 'F', 'ASIAN', 'BLACK', 'WHITE', 'HISPA']], how='left', on = ['subject_id', 'hadm_id','icustay_id'])
vitals_labs['vitals'] = vitals_labs.values.tolist()
grouped_vitals = vitals_labs.groupby(['subject_id', 'hadm_id','icustay_id'])['vitals'].apply(list).reset_index()

print("merging final dataset")
data = data.merge(grouped_vitals, how='left', on = ['subject_id', 'hadm_id','icustay_id'])
data = data.merge(patients[['mort_icu', 'los_icu']], how='left', on = ['subject_id', 'hadm_id','icustay_id'])

print("splitting datasets")
PATH_OUTPUT = "../data/processed/"
train, validate, test = np.split(data.sample(frac=1, random_state=42), [int(.7*len(data)), int(.8*len(data))])

def save_dataset(df, name = 'train'):
	print("saving "+name)
	df[['subject_id', 'hadm_id','icustay_id']].to_pickle(PATH_OUTPUT+"ids."+name)
	df['mort_icu'].to_pickle(PATH_OUTPUT+"mort_icu."+name)
	df['los_icu'].to_pickle(PATH_OUTPUT+"los_icu."+name)
	df[['interventions', 'vitals']].to_pickle(PATH_OUTPUT+"seqs."+name)

save_dataset(train, name='train')
save_dataset(validate, name='validate')
save_dataset(test, name='test')

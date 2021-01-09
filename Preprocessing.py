import sys
import pandas as pd
from pandas import DataFrame
import numpy as np

dir='/***Abdo_US_MLMD/' #Output directory
sys.stdout = open(dir+"preprocess_outputs.txt", "w") #Saves all outputs in a txt file
name="Abdo_US" #MLMD use case (abdominal ultrasounds)

#Loading Data from CSV files
df_x_triage_input = pd.read_csv('/triage.csv', encoding = "latin-1") #Triage Input Features
df_x_triage_CUI = pd.read_csv('/triage_CUI.csv') #CUI codes for triage notes free text
df_labs = pd.read_csv('/ab.csv') #Lab values for patients
idx_diagnosis = pd.read_csv('/idx.csv') #Final Diagnosis for patients
triage_notes = pd.read_csv('/notes_triage.csv', encoding = "latin-1") #Triage Input Features

#Processing Labs Data
lab_res = df.groupby('CSN').lab.apply(lambda x: pd.Series({'Abdo_US':x.str.contains('Abdomen and Pelvis').any()})).reset_index()
lab_res = lab_res.pivot('CSN','level_1','lab').reset_index()
tot = idx.merge(lab_res,how='left',on='CSN').fillna(False)

#A narrow focused differential diagnoses used to act as the label for the model. The search terms will need to change based on diagnosis labeling practices within your organization
tot['PrimaryDx_Label'] = tot['PrimaryDx'].apply(lambda x: 1 if 'ppendicitis' in x or 'varian' in x or 'ntussusception' in x or 'olvulus'in x else 0)

#Printing the value counts to be saved in the output file
print ('THESE ARE THE VALUE COUNTS')
print ("Total Differential Diagnoses: ", tot['PrimaryDx_Label'].value_counts())
tot[name]=tot[name].apply(lambda x:1 if x==True else 0)
print ("Total Tests Identified: ", tot[name].value_counts())

#Creating the combined label [1 if a patient has either a diagnosis OR a test done, 2 if a patient has both diagnosis and test done, 0 if neither test or diagnosis present)
tot['CM_Label']=tot['PrimaryDx_Label']+tot[name]
tot['CM_Label'] = tot['CM_Label'].apply(lambda x: 1 if x == 1 or x == 2 else 0)
print ("Value counts for the combined label: ", tot['CM_Label'].value_counts())

tot = tot[tot.Arrived != False] #Removes incomplete charts from our dataset (patients that did not have arrival times present)

#Orders patients based on arrival time stamps within the dataset
tot['Arrived'] = pd.to_datetime(tot['Arrived'])
tot = tot.sort_values(by='Arrived', ascending=True)
print(tot['Arrived'])

#Joining dataframes together using a  unique patient encounter number
tot = tot.join(df_x_triage_input.set_index('CSN'), on='CSN')
tot = tot.join(df_x_triage_CUI.set_index('CSN'),on='CSN')
tot.to_pickle(dir+'tot_dat.pkl')

sys.stdout.close()

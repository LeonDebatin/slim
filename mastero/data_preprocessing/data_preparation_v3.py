import pandas as pd
import arff
from utils import *
import sys
import os
sys.path.insert(0, os.path.abspath("../.."))

info_df = []

print("Processing auction dataset...")
#https://archive.ics.uci.edu/dataset/713/auction+verification
auction = pd.read_csv('../data2/data_original/auction.csv')
auction.rename(columns={'verification.result': 'target'}, inplace=True)
auction.replace({True: 1, False: 0}, inplace=True)
auction = auction[[c for c in auction.columns if c != 'target'] + ['target']]
auction.drop(columns=['verification.time'], inplace=True) # other feature

categoricals = []
binaries = []

basic_information(auction)
auction.to_csv('../data2/data_prepared/auction.csv', index=False)
info_df.append(info_dict('auction', auction, categoricals, binaries))

print("Processing autism dataset...")
#https://archive.ics.uci.edu/dataset/426/autism+screening+adult
with open("../data2/data_original/autism.arff", "r") as file:
    dataset = arff.load(file)
autism = pd.DataFrame(dataset['data'], columns=[x[0] for x in dataset['attributes']])
autism['Class/ASD'].replace({'NO': 0, 'YES': 1}, inplace=True)
autism.rename(columns={'Class/ASD': 'target'}, inplace=True)
autism.drop(columns=['ethnicity', 'relation', 'age', 'contry_of_res'], inplace=True) # NAs

cols = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
        'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'result']

autism[cols] = autism[cols].apply(pd.to_numeric, errors='coerce').astype('int32')

# These columns should be one-hot encoded later, so save as categoricals
categoricals = ['gender', 'jundice', 'austim', 'used_app_before', 'age_desc']
# Don't do one-hot encoding here - keep the original columns
numericals = ['result']
# All the A-score columns are binary
binaries = [col for col in autism.columns if col != 'target' and col not in categoricals and col != 'result']

autism = autism[[c for c in autism.columns if c != 'target'] + ['target']]
autism.drop_duplicates(inplace=True)
autism.reset_index(drop=True, inplace=True)

basic_information(autism)
autism.to_csv('../data2/data_prepared/autism.csv', index=False)
info_df.append(info_dict('autism', autism, categoricals, binaries))

print("Processing biomed dataset...")
#https://www.openml.org/search?type=data&status=active&id=481
with open("../data2/data_original/biomed.arff", "r") as file:
    dataset = arff.load(file)
biomed = pd.DataFrame(dataset['data'], columns=[x[0] for x in dataset['attributes']])
biomed.rename(columns={'class': 'target'}, inplace=True)
biomed['target'].replace({'carrier': 1, 'normal': 0}, inplace=True)
biomed.drop(columns=['Observation_number', 'Hospital_identification_number_for_blood_sample', 'Date_that_blood_sample_was_taken'], inplace=True)
biomed = biomed.fillna(biomed.mean(numeric_only=True))

numericals = list(set(biomed.columns) - {'target'})
categoricals = []
binaries = []

basic_information(biomed)
biomed.to_csv('../data2/data_prepared/biomed.csv', index=False)
info_df.append(info_dict('biomed', biomed, categoricals, binaries))

print("Processing credit dataset...")
#https://www.openml.org/search?type=data&status=active&id=46422
with open("../data2/data_original/credit.arff", "r") as file:
    dataset = arff.load(file)
credit = pd.DataFrame(dataset['data'], columns=[x[0] for x in dataset['attributes']])
credit.rename(columns={'label': 'target'}, inplace=True)
credit.drop(columns=['id', 'fea_2'], inplace=True)
credit = credit[[c for c in credit.columns if c != 'target'] + ['target']]

numericals = ['fea_4', 'fea_8', 'fea_10', 'fea_11']
categoricals = []  # No true categoricals in credit
binaries = list(set(list(credit.columns)) - set(numericals) - {'target'})

basic_information(credit)
credit.to_csv('../data2/data_prepared/credit.csv', index=False)
info_df.append(info_dict('credit', credit, categoricals, binaries))

print("Processing darwin dataset...")
#https://www.openml.org/search?type=data&status=active&id=46606
darwin = pd.read_csv('../data2/data_original/darwin.csv')
darwin['class'].replace({'P': 0, 'H': 1}, inplace=True)
darwin.rename(columns={'class': 'target'}, inplace=True)
darwin.drop(columns=['ID'], inplace=True)

numericals = list(set(darwin.columns) - {'target'})
categoricals = []
binaries = []

basic_information(darwin)
darwin.to_csv('../data2/data_prepared/darwin.csv', index=False)
info_df.append(info_dict('darwin', darwin, categoricals, binaries))

print("Processing heart dataset...")
#https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
heart = pd.read_csv('../data2/data_original/heart.csv')
heart.rename(columns={'DEATH_EVENT': 'target'}, inplace=True)

numericals = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']
categoricals = []  # No true categoricals in heart
binaries = list(set(heart.columns) - set(numericals) - {'target'})

basic_information(heart)
heart.to_csv('../data2/data_prepared/heart.csv', index=False)
info_df.append(info_dict('heart', heart, categoricals, binaries))

print("Processing pc4 dataset...")
#https://www.openml.org/search?type=data&sort=runs&id=1049&status=active
with open("../data2/data_original/pc4.arff", "r") as file:
    dataset = arff.load(file)
pc4 = pd.DataFrame(dataset['data'], columns=[x[0] for x in dataset['attributes']])
pc4.rename(columns={'c': 'target'}, inplace=True)
pc4['target'] = pc4['target'].replace({'TRUE': 1, 'FALSE': 0})
pc4.drop_duplicates(inplace=True)
pc4.reset_index(drop=True, inplace=True)

numericals = list(set(pc4.columns) - {'target'})
categoricals = []
binaries = []

basic_information(pc4)
pc4.to_csv('../data2/data_prepared/pc4.csv', index=False)
info_df.append(info_dict('pc4', pc4, categoricals, binaries))

print("Processing thyroid dataset...")
#https://openml.org/search?type=data&status=active&id=46082
with open("../data2/data_original/thyroid.arff", "r") as file:
    dataset = arff.load(file)
thyroid = pd.DataFrame(dataset['data'], columns=[x[0] for x in dataset['attributes']])

# These columns should be one-hot encoded later, so save as categoricals
# Don't do one-hot encoding here
numericals = ['Age']
categoricals = list(set(list(thyroid.columns)) - set(['Age', 'Recurred']))
binaries = []  # Recurred becomes target

thyroid['Recurred'] = thyroid['Recurred'].apply(lambda x: 1 if x == 'Yes' else 0)
thyroid.rename(columns={'Recurred': 'target'}, inplace=True)
thyroid = thyroid[[c for c in thyroid.columns if c != 'target'] + ['target']]
thyroid.drop_duplicates(inplace=True)
thyroid.reset_index(drop=True, inplace=True)

basic_information(thyroid)
thyroid.to_csv('../data2/data_prepared/thyroid.csv', index=False)
info_df.append(info_dict('thyroid', thyroid, categoricals, binaries))

print("Processing wilt dataset...")
#https://www.openml.org/search?type=data&status=active&id=40983&sort=runs
wilt = pd.read_csv('../data2/data_original/wilt.csv')
wilt.rename(columns={'class': 'target'}, inplace=True)
wilt['target'] = wilt['target'].replace({'w': 1, 'n': 0})
wilt = wilt[[c for c in wilt.columns if c != 'target'] + ['target']]
wilt.drop_duplicates(inplace=True)
wilt.reset_index(drop=True, inplace=True)

numericals = list(set(wilt.columns) - {'target'})
categoricals = []
binaries = []

basic_information(wilt)
wilt.to_csv('../data2/data_prepared/wilt.csv', index=False)
info_df.append(info_dict('wilt', wilt, categoricals, binaries))

print("Processing wisconsin dataset...")
#https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
wisconsin = pd.read_csv('../data2/data_original/wisconsin.data', header=None)
wisconsin.rename(columns={0:'ID', 10:'target'}, inplace=True)
wisconsin['target'].replace({4: 0, 2: 1}, inplace=True)
wisconsin.drop(columns=['ID'], inplace=True)
wisconsin[6] = pd.to_numeric(wisconsin[6], errors='coerce').fillna(0).astype('int32')
wisconsin.drop_duplicates(inplace=True)
wisconsin.reset_index(drop=True, inplace=True)

numericals = list(set(wisconsin.columns) - {'target'})
categoricals = []
binaries = []

basic_information(wisconsin)
wisconsin.to_csv('../data2/data_prepared/wisconsin.csv', index=False)
info_df.append(info_dict('wisconsin', wisconsin, categoricals, binaries))

print("Saving data info...")
info_df = pd.DataFrame(info_df)
info_df.to_csv('../data2/data_info.csv', index=False)
print("Data preprocessing completed successfully!")
print(f"Processed {len(info_df)} datasets")
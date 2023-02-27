import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

## Reading the dataset
df_small = pd.read_csv('dataset_full .csv')
df_full = pd.read_csv('dataset_full .csv')

## concatenating the two datasets

frames = [df_small, df_full]
result = pd.concat(frames)

## checking the null values in the dataset
result.isnull().sum()

## Doing the feature extraction
## table1
table1 = result[result.columns[:20]]
table1['phishing'] = result['phishing']
## table2
table2 = result[result.columns[20:40]]
table2['phishing'] = result['phishing']
## Table3
table3 = result[result.columns[40:60]]
table3['phishing'] = result['phishing']
### Table4
table4 = result[result.columns[60:80]]
table4['phishing'] = result['phishing']
## Table5
table5 = result[result.columns[80:100]]
table5['phishing'] = result['phishing']

## Extracting the columns in the table3
tf4 = table4.corr()['phishing'][table4.corr()['phishing'] > .4]
tf4.index

## Extracting the columns in the table3 and  table4 and appending into the list to construct the new dataset

tf3 = table3.corr()['phishing'][table3.corr()['phishing'] > .4]
tf3.index
l = []
for i in tf3.index:
    l.append(i)
for i in tf4.index:
    l.append(i)
l = l[:-1]


## Final dataset after feature extraction
df=result[l]

## Phishing column is repeated twice so we are removing the phishing from the dataset
df=df.drop(['qty_hyphen_directory'],axis=1)
df=df.drop(['phishing'],axis=1)
df=df.drop(['qty_slash_file'],axis=1)


## Adding the table1 columns into dataset
df['qty_slash_url']=result['qty_slash_url']
df['length_url']=result['length_url']
df['phishing']=result['phishing']


## splitting the dataset
X=df[df.columns[:-1]]

## Dependent column
y=df[df.columns[-1]]

## Splitting the data into 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)

rnd=RandomForestClassifier(max_depth=30, random_state=0)
rnd.fit(X_train,y_train)
pred=rnd.predict(X_test)

with open("random_forest2", 'wb') as f:
     pickle.dump(rnd, f)
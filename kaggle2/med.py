import pandas as pd 
import numpy as np 
import math 
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train.drop(['id'], axis=1, inplace=True)
print(df_train.isnull().sum())

from sklearn.ensemble import RandomForestClassifier

features = df_train.drop('Status',axis=1).columns
y = df_train['Status']

print(features)

X = pd.get_dummies(df_train[features])
X_test = pd.get_dummies(df_test[features])

print()
print(X_test.isnull().sum())

X.where(pd.notnull(X),X.median(),axis="columns",inplace=True)

print()
print(X_test.isnull().sum())

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X,y)
predictions = model.predict(X_test)

output = pd.DataFrame({'ID':df_test.id, 'Status':predictions})
output.to_csv('Submission.csv')

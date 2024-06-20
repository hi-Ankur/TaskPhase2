import pandas as pd 
import numpy as np 
import math 
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv('train.csv')
print(df_train.isnull().sum())

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

lb = LabelEncoder()

df_train["Drug_Encoded"] = lb.fit_transform(df_train["Drug"])
df_train["Sex_Encoded"] = lb.fit_transform(df_train["Sex"])
df_train["Ascites_Encoded"] = lb.fit_transform(df_train["Ascites"])
df_train["Hepatomegaly_Encoded"] = lb.fit_transform(df_train["Hepatomegaly"])
df_train["Spiders_Encoded"] = lb.fit_transform(df_train["Spiders"])
df_train["Edema_Encoded"] = lb.fit_transform(df_train["Edema"])
df_train["Status_Encoded"] = lb.fit_transform(df_train["Status"])

variables = ["Drug","Sex","Ascites","Hepatomegaly","Spiders","Edema","Status"]

print(variables)

new_df = df_train.drop(variables, axis=1)
print(new_df.isnull().sum())

arr = np.array(new_df)

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=10)
imputer.fit_transform(arr[:10])

print(arr)
df = pd.DataFrame(arr)
print(df.describe())
print(df.isnull().sum())

from sklearn.ensemble import RandomForestClassifier


















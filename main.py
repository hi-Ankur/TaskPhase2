import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns 



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


print(train.isnull().sum())
print()
print(test.isnull().sum())

X = train.drop(['yield','id'],axis=1)
print()
print(X) 

X_test = test.drop(['id'], axis=1)
print(X_test)

y = train['yield']
print(y)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)

model.fit(X,y)
predictions = model.predict(X_test)

print(predictions)

output = pd.DataFrame({'id':test.id, 'yield':predictions})
output.to_csv('submissio.csv', index=False)




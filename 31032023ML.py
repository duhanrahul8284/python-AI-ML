import pandas as pd
import numpy as np
df=pd.read_csv("insurance.csv")

X=df.iloc[:,:-1]
Y=df['charges']
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X['gender']=le.fit_transform(X['gender'])

lo=LabelEncoder()
X['smoker']=lo.fit_transform(X['smoker'])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer([("encode",OneHotEncoder(),[5])],remainder='passthrough')
X=ct.fit_transform(X)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

'''
******Regression*******
linear regression
support Vector regression
Decision tree
random forest
xgbosst

****Classifications********
logistic regression
support Vector class
noive bayes
k nearest neigbhour
decision tree
random forest
adaboost
xgboost

'''

#########training##########
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
'''
#regressor = LinearRegression()
regressor = SVR(C=5000)
regressor.fit(X_train, Y_train)
Y_pred= regressor.predict(X_test)

'''


from sklearn.tree import DecisionTreeRegressor
regreesor=DecisionTreeRegressor(max_depth=5)
regreesor.fit(X_train, Y_train)

'''
decision tree regressor is also use classifer only difference is
while chosing best column instad of antropy we will check standard davation on dependent variable
'''


'''
*************Regression************
mean square error
mean absolute error
root mean square error
r2 score


***************Classifiction***************
accuracy
recall
precision
F1 score
confusion matrix
roc curve

'''
#R2 score tell us the amount of varience of dependent variable that is explanined by dependent variable
# range -infinty to +1 (near 1 mean good accuracy and near 0 mean bad model)


Y_pred= regreesor.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(Y_test, Y_pred))

import joblib
joblib.dump(regreesor,"regressor.joblib")
joblib.dump(ct,"onehotencode.joblib")
joblib.dump(sc,"scaler.joblib")



















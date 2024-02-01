import pandas as pd
import numpy as np
import joblib


regressor=joblib.load("regressor.joblib")
ct=joblib.load("onehotencode.joblib")
sc=joblib.load("scaler.joblib")

test=pd.DataFrame({"age":[int(input("Enter your age"))],
                   "gender":[input("Enter gender")],
                   "bmi":[float(input("enter bmi"))],
                   "children":[int(input("Enter children"))],
                   "smoker":[input("do u smoke")],
                   "region":[input("which is ur region")]

                   })

if test.loc[0,"smoker"]=='yes':
    test.loc[0,"smoker"]=1
else:
    test.loc[0,"smoker"]=0

if test.loc[0,"gender"]=='male':
    test.loc[0,"gender"]=1
else:
    test.loc[0,"gender"]=0

test['gender']=test['gender'].astype("int32")

test['smoker']=test['smoker'].astype("int32")


test=ct.transform(test)
test=sc.transform(test)
print(regressor.predict(test))





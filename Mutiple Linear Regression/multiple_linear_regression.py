
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('50_Startups.csv')
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]
States=pd.get_dummies(x['State'],drop_first=True)
x=x.drop('State',axis=1)
x=pd.concat([x,States],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
Regression=LinearRegression()
Regression.fit(x_train,y_train)

y_predict=Regression.predict(x_test)
print(np.mean(np.abs((y_predict-y_test)/y_test))*100)


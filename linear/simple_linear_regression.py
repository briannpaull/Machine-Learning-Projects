import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
simplelinearRegression=LinearRegression()
simplelinearRegression.fit(x_train,y_train)

y_predict=simplelinearRegression.predict(x_test)
plt.scatter(x_train,y_train,color='red')
plt.title("SALARY VS EXPENDITURE")
plt.xlabel("YEAR OF EXPERIENCE")
plt.ylabel("SALARY")
plt.plot(x_train,simplelinearRegression.predict(x_train))
plt.show()

print(np.mean(np.abs((y_predict-y_test)/y_test))*100)


'''error=np.absolute(np.subtract(y_test,y_predict))
for i in error:
    print("|E| :",i)
    i=i+1


#mape solution 1
np.mean((error)/y_test)*100
print(np.mean)


#mape solution 2
np.array(y_test),np.array(y_predict)
np.mean(np.abs(y_test-y_predict)/y_test)*100'''


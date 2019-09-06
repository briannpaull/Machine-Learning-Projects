# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values


from sklearn.linear_model import LinearRegression
linear_reg1=LinearRegression()
linear_reg1.fit(x,y)


from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
x_poly=poly_reg.fit_transform(x)

linear_reg2=LinearRegression()
linear_reg2.fit(x_poly,y)


plt.scatter(x,y,color="red")
plt.plot(x,linear_reg1.predict(x),color="blue")
plt.show()


plt.scatter(x,y,color="red")
plt.plot(x,linear_reg2.predict(x_poly),color="green")
plt.show()
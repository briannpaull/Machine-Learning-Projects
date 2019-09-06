import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset=pd.read_csv('data.csv')
#dataset.head() to see data
dataset=dataset.dropna(axis=1)#to remove empty column
dataset=dataset.drop('id',axis=1) #removed ID because it will effect model



from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
dataset.iloc[:,0]= labelencoder_Y.fit_transform(dataset.iloc[:,0].values)
dataset.iloc[:,0] #onehot encoding method 1

'''
diagnosis=pd.get_dummies(dataset['diagnosis'],drop_first=True)
dataset=dataset.drop('diagnosis',axis=1)
dataset=pd.concat([dataset,diagnosis],axis=1)'''   #onehot encoding method2

# 1 is M..non curable cancer
#0 is B..curabke

X=dataset.iloc[:,1:32].values
Y=dataset.iloc[:,0].values  #splitting into depended and independent variable

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10, random_state = 0) #splitting into test nd train

from sklearn.preprocessing import StandardScaler   #scalling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




#using k neighbour
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)
y_pred_Kneighbour=knn.predict(X_test)
print('K Nearest Neighbor Accuracy:', knn.score(X_test, Y_test))

print("-"*20) #gapping

#using logistic regression
from sklearn.linear_model import LogisticRegression  
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)
y_pred_Logistic_regression=classifier.predict(X_test)
print('Logistic Regression Accuracy:', classifier.score(X_test, Y_test))

print("-"*20)


#using svm
from sklearn.svm import SVC
svc_lin = SVC(kernel = 'linear', random_state = 0)
svc_lin.fit(X_train, Y_train)
y_pred_SVC=svc_lin.predict(X_test)
print('Support Vector Machine Accuracy:', svc_lin.score(X_test, Y_test))

print("-"*20)


#using decission tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
tree.fit(X_train, Y_train)
y_pred_Decission_tree=tree.predict(X_test)
print('Decision Tree Classifier Accuracy:', tree.score(X_test, Y_test))

print("-"*20)


#Using RandomForestClassifier 
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(X_train, Y_train)
y_pred_random_forest=forest.predict(X_test)
print('Random Forest Classifier Accuracy:', forest.score(X_test, Y_test))

print("-"*20)





import csv
import numpy as np
import pandas as pd
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

DATA=pd.read_csv('iris_data.csv')
Y=DATA['1']
X=DATA.drop('1',axis=1)
print ('size of the input data')
print (X.shape)
print ('size of the output data')
print (Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=5)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
# scores = cross_val_score(clf,X,Y,cv=3)
score=clf.score(X_test, y_test)
print ('Test score')
print (score)
score1=clf.score(X_train, y_train)
print ('Training score')
print (score1)
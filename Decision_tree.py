import csv
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split

# cutomize the Decision tree, make the maximum number of nodes as 10
clf = DecisionTreeClassifier(random_state=0,max_leaf_nodes=10,max_depth=5)

DATA=pd.read_csv('iris_data.csv')
Y=DATA['1']
X=DATA.drop('1',axis=1)
print ('size of the input data')
print (X.shape)
print ('size of the output data')
print (Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=5)

clf.fit(X_train, y_train)
with open("iris.dot", 'w') as f:
     f = tree.export_graphviz(clf, out_file=f)

import pydotplus 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("iris.pdf") 

# scores = cross_val_score(clf, X, Y, cv=10)
# print (scores)
score=clf.score(X_test, y_test)
print ('Test score')
print (score)
score1=clf.score(X_train, y_train)
print ('Training score')
print (score1)
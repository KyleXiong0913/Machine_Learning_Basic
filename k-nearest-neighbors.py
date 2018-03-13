import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

n_neighbors = 15

# import some data to play with
DATA=pd.read_csv('iris_data.csv')
Y=DATA['1']
X=DATA.drop('1',axis=1)
print ('size of the input data')
print (X.shape)
print ('size of the output data')
print (Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors)
clf.fit(X_train, y_train)

score=clf.score(X_test, y_test)
print ('Test score')
print (score)
score1=clf.score(X_train, y_train)
print ('Training score')
print (score1)

# The let us do the cross validation
# Below is the default set for the cv
# cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
scores = cross_val_score(clf, X, Y, cv=3)
print (scores)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
    
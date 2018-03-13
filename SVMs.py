import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# import some data to play with
# import the dataset we are going to use

# DATA=np.genfromtxt('wine.csv',delimiter=',')
DATA=pd.read_csv('wine.csv')
Y=DATA['1']
X=DATA.drop('1',axis=1)
print ('size of the input data')
print (X.shape)
print ('size of the output data')
print (Y.shape)

# Below is the first customized kernel I created
def my_kernel_1(X, Y):
    """We create a custom kernel:

                 (2  0)
    k(X, Y) = X  (    ) Y.T
                 (0  1)"""
    M = np.array([[2, 0], [0, 1.0]])
    return np.dot(np.dot(X, M), Y.T)

# Below is the linear kernel function I created
def my_kernel_2(x, y):
    return np.dot(x, y.T)

# Below is the RBF kernel function I created
def my_kernel_3(x, y):
    gamma = 0.01
    return np.exp((gamma* np.power(np.linalg.norm(x-y),2)))

h = .02  # step size in the mesh

# we create an instance of SVM and fit out data.
# clf = svm.SVC(kernel=my_kernel_2)
DATA=pd.read_csv('iris_data.csv')
Y=DATA['1']
X=DATA.drop('1',axis=1)
print ('size of the input data')
print (X.shape)
print ('size of the output data')
print (Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
pred=clf.predict(X_train)
# print (pred)
print ('The accuracy of the training data')
print (clf.score(X_train, y_train))
print ('The accuracy of the test data')
print (clf.score(X_test, y_test))

# The let us do the cross validation
# Below is the default set for the cv
# cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
clf_1 = svm.SVC(kernel='linear',C=2)
clf_2 = svm.SVC(kernel='rbf')
scores = cross_val_score(clf_1, X, Y, cv=3)
print ('The scores of the cross validation result')
print (scores)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

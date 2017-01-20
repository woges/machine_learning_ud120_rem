#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

import sys
import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

from time import time

def submitAccuracy(pred, labels_test):
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    return {"acc":round(acc,3)}

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary



try:

### SVM
##from sklearn.svm import SVC
###clf = SVC(kernel="linear", gamma = 1.0)
##clf = SVC(kernel="rbf", gamma = 100.0, C = 0.1)
    
##    # kNN
    from sklearn.neighbors import KNeighborsClassifier
##
    clf = KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='auto')
##
    
##    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
##    clf = AdaBoostClassifier()#,n_estimators=100)
##    clf = RandomForestClassifier(min_samples_split=100, max_features = 0.4)


    #### now your job is to fit the classifier
    #### using the training features/labels, and to
    #### make a set of predictions on the test data
    t0 = time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"

    #### store your predictions in a list named pred

    t0 = time()
    pred = clf.predict(features_test)
    print "prediction time:", round(time()-t0, 3), "s"

    print("accuracy: "+ str(submitAccuracy(pred,labels_test)))

    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass

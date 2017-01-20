import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

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


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn import tree

clf_2 = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split = 20)
#clf_50 = tree.DecisionTreeClassifier(min_samples_split = 50)
#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
t0 = time()
clf_2.fit(features_train, labels_train)
#clf_50.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

#### store your predictions in a list named pred

t0 = time()
pred_2 = clf_2.predict(features_test)
#pred_50 = clf_50.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"


print("accuracy 2: "+ str(submitAccuracy(pred_2,labels_test)))

#print("accuracy 50: "+ str(submitAccuracy(pred_50,labels_test)))
### draw the decision boundary with the text points overlaid
prettyPicture(clf_2, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())

#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from class_vis import prettyPicture, output_image
sys.path.append("../tools/")
from email_preprocess import preprocess

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

def submitAccuracy(pred, labels_test):
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    return acc

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################
from sklearn.svm import SVC

clf = SVC(kernel="rbf", C = 10000)#, gamma = 1000.0, C = 0.1)

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

#### store your predictions in a list named pred
t0 = time()
pred = clf.predict(features_test)
print("prediction time:", round(time()-t0, 3), "s")

print("accuracy: C=10000  "+ str(submitAccuracy(pred,labels_test)))
summe = 0
for data in pred:
    summe += data

print("Summe: " +str(summe))
#print("10th element: " + str(pred[10]))
#print("26th element: " + str(pred[26]))
#print("50th element: " + str(pred[50]))

#print("Jetzt geht image los:")
### draw the decision boundary with the text points overlaid
#prettyPicture(clf, features_test, labels_test)
#output_image("test.png", "png", open("test.png", "rb").read())



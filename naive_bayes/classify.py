from __future__ import division
def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from time import time
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score

    
    ### create classifier
    clf = GaussianNB()

    ### fit the classifier on the training features and labels
    t0 = time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"

    ### use the trained classifier to predict labels for the test features
    t0 = time()
    pred = clf.predict(features_test)
    print "prediction time:", round(time()-t0, 3), "s"

    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module

# 1. Moeglichkeit accuracy
#    no_corr = 0
#    for kk in range(0, len(pred)):
#        if int(pred[kk])==int(labels_test[kk]):
#            no_corr +=1
#    accuracy = (no_corr)/(len(features_test))
#    print(accuracy)
    
# 2. Moeglichkeit accuracy
    accuracy = clf.score(features_test,labels_test)
#    print(accuracy_2)

# 3. Moeglichkeit accuracy
#    print(accuracy_score(pred, labels_test))
    
    return accuracy

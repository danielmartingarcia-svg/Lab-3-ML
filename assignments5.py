# Lab 3: Bayes Classifier and Boosting
# Check out `labfuns.py` if you are interested in the details.

import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random

def computePrior(labels, W=None):
    # in: labels - N vector of class labels
    # out: prior - C x 1 vector of class priors

    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    for jdx,clas in enumerate(classes):
        idx = labels==clas # true/false with the length of y
        classIdx = np.where(labels==clas)[0]
        NptsClass = classIdx.shape[0] # number of data points in class
        prior[jdx] = NptsClass/Npts
    return prior


def mlParams(X, labels, W):
    # in:      X - N x d matrix of N data points
    #     labels - N vector of class labels
    # out:    mu - C x d matrix of class means (mu[i] - class i mean)
    #      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)

    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)    # number of data points, number of dimensions
    classes = np.unique(labels) # unique class labels
    Nclasses = np.size(classes) # number of classes

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))

    for jdx,clas in enumerate(classes):
        idx = labels==clas # true/false with the length of y
        classIdx = np.where(labels==clas)[0]
        Xclass = X[classIdx,:] # the data points of class clas
        Wclass = W[classIdx] # the weights of the data points of class clas
        NptsClass = Xclass.shape[0] # number of data points in class
        mu[jdx,:] = np.sum(Xclass*Wclass,axis=0)/np.sum(Wclass)
        x_centered = Xclass - mu[jdx,:]
        # Simplicity: feature dimensions are independent, so we only compute the diagonal of the covariance matrix
        # Diagonal covariance matrix: 
        sigma[jdx,:,:] = np.diag(np.sum(x_centered**2*Wclass,axis=0)/np.sum(Wclass))
        # Unsimplified: 
        # sigma[jdx,:,:] = np.dot(x_centered.T, x_centered)/NptsClass

    return mu, sigma


def classifyBayes(X, prior, mu, sigma):
    # in:      X - N x d matrix of M data points
    #      prior - C x 1 matrix of class priors
    #         mu - C x d matrix of class means (mu[i] - class i mean)
    #      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
    # out:     h - N vector of class predictions for test points

    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    for jdx in range(Nclasses):
        x_centered = X - mu[jdx,:]
        # Simplicity: feature dimensions are independent, so we only compute the diagonal of the covariance matrix
        variances = np.diag(sigma[jdx,:,:]) # Ensure diagonal matrix
        logDetSigma = np.sum(np.log(variances))
        # Unsimplified:
        # logDetSigma = np.log(np.linalg.det(sigma[jdx,:,:]))
        logPrior = np.log(prior[jdx])
        logProb[jdx,:] =-0.5*logDetSigma - 0.5*np.sum((x_centered**2)/variances,axis=1) + logPrior
        # Unsimplified and with dot product and transpose:
        # logProb[jdx,:] =-0.5*logDetSigma - 0.5*np.sum(np.dot(x_centered, np.linalg.inv(sigma[jdx,:,:]))*x_centered,axis=1) + logPrior

    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb,axis=0)
    return h

class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)

# Get X and labels 
X, labels = genBlobs(centers=5)
# Initialize weights to uniform weight vector with w=1/N
W = np.ones((X.shape[0], 1)) / X.shape[0]
# Compute mu and sigma
mu, sigma = mlParams(X,labels,W)
plotGaussian(X,labels,mu,sigma)

# Create an instance of your classifier
bc = BayesClassifier()

# --- Test Accuracy ---
print("\n--- Iris Dataset Results ---")
testClassifier(bc, dataset='iris')

print("\n--- Vowel Dataset Results ---")
testClassifier(bc, dataset='vowel')

# --- Visualize Decision Boundary (Iris) ---
# This fulfills the requirement to plot the boundary for the 2D iris data
plotBoundary(bc, dataset='iris')

# Boosting functions to implement

def trainBoost(base_classifier, X, labels, T=10):
    # in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
    #                   X - N x d matrix of N data points
    #              labels - N vector of class labels
    #                   T - number of boosting iterations
    # out:    classifiers - (maximum) length T Python list of trained classifiers
    #              alphas - (maximum) length T Python list of vote weights

    # these will come in handy later on
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        # TODO: Fill in the rest, construct the alphas etc.
        # ==========================
        
        # alphas.append(alpha) # you will need to append the new alpha
        # ==========================
        
    return classifiers, alphas


def classifyBoost(X, classifiers, alphas, Nclasses):
    # in:       X - N x d matrix of N data points
    # classifiers - (maximum) length T Python list of trained classifiers as above
    #      alphas - (maximum) length T Python list of vote weights
    #    Nclasses - the number of different classes
    # out:  yPred - N vector of class predictions for test points
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))

        # TODO: implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================
        
        # ==========================

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)


# The implemented functions can now be summarized another classifer, the `BoostClassifier` class. This class enables boosting different types of classifiers by initializing it with the `base_classifier` argument. No need to add anything here.


# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# ## Run some experiments
# 
# Call the `testClassifier` and `plotBoundary` functions for this part.


#testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)



#testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)



#plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris',split=0.7)


# Now repeat the steps with a decision tree classifier.


#testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)



#testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)



#plotBoundary(DecisionTreeClassifier(), dataset='iris',split=0.7)



#plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)


# ## Bonus: Visualize faces classified using boosted decision trees
# 
# Note that this part of the assignment is completely voluntary! First, let's check how a boosted decision tree classifier performs on the olivetti data. Note that we need to reduce the dimension a bit using PCA, as the original dimension of the image vectors is `64 x 64 = 4096` elements.


#testClassifier(BayesClassifier(), dataset='olivetti',split=0.7, dim=20)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='olivetti',split=0.7, dim=20)


# You should get an accuracy around 70%. If you wish, you can compare this with using pure decision trees or a boosted bayes classifier. Not too bad, now let's try and classify a face as belonging to one of 40 persons!


#X,y,pcadim = fetchDataset('olivetti') # fetch the olivetti data
#xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,0.7) # split into training and testing
#pca = decomposition.PCA(n_components=20) # use PCA to reduce the dimension to 20
#pca.fit(xTr) # use training data to fit the transform
#xTrpca = pca.transform(xTr) # apply on training data
#xTepca = pca.transform(xTe) # apply on test data
# use our pre-defined decision tree classifier together with the implemented
# boosting to classify data points in the training data
#classifier = BoostClassifier(DecisionTreeClassifier(), T=10).trainClassifier(xTrpca, yTr)
#yPr = classifier.classify(xTepca)
# choose a test point to visualize
#testind = random.randint(0, xTe.shape[0]-1)
# visualize the test point together with the training points used to train
# the class that the test point was classified to belong to
#visualizeOlivettiVectors(xTr[yTr == yPr[testind],:], xTe[testind,:])


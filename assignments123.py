# # Lab 3: Bayes Classifier and Boosting
# ## Import the libraries
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


def mlParams(X, labels, W=None):
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
        NptsClass = Xclass.shape[0] # number of data points in class
        mu[jdx,:] = np.sum(Xclass,axis=0)/NptsClass
        x_centered = Xclass - mu[jdx,:]
        # Simplicity: feature dimensions are independent, so we only compute the diagonal of the covariance matrix
        # Diagonal covariance matrix: 
        sigma[jdx,:,:] = np.diag(np.sum(x_centered**2,axis=0)/NptsClass)
        # Unsimplified: sigma[jdx,:,:] = np.dot(x_centered.T, x_centered)/NptsClass

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
        
        variances = np.diag(sigma[jdx,:,:]) # Ensure diagonal matrix
        logDetSigma = np.sum(np.log(variances))
        logPrior = np.log(prior[jdx])
        logProb[jdx,:] =-0.5*logDetSigma - 0.5*np.sum((x_centered**2)/variances,axis=1) + logPrior
    
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

X, labels = genBlobs(centers=5)
mu, sigma = mlParams(X,labels)
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
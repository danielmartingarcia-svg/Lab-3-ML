import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random


def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts, 1)) / Npts
    else:
        assert(W.shape[0] == Npts)
    
    classes = np.unique(labels)
    Nclasses = np.size(classes)
    prior = np.zeros((Nclasses, 1))
    
    # The prior probabilities should still sum to one 
    # We count "how many times we count" each point based on W 
    total_W = np.sum(W)
    for jdx, clas in enumerate(classes):
        idx = (labels == clas)
        prior[jdx] = np.sum(W[idx]) / total_W
        
    return prior

def mlParams(X, labels, W=None):
    assert(X.shape[0] == labels.shape[0])
    Npts, Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts, 1)) / float(Npts)

    mu = np.zeros((Nclasses, Ndims))
    sigma = np.zeros((Nclasses, Ndims, Ndims))

    for jdx, clas in enumerate(classes):
        idx = (labels == clas)
        Xclass = X[idx, :]
        Wclass = W[idx] 
        
        sumWclass = np.sum(Wclass)

        # MAP parameter estimation with weighted instances (Equation 13) 
        mu[jdx, :] = np.sum(Xclass * Wclass, axis=0) / sumWclass

        # MAP parameter estimation with weighted instances (Equation 14)
        x_centered = Xclass - mu[jdx, :]
        var_diag = np.sum((x_centered**2) * Wclass, axis=0) / sumWclass
        sigma[jdx, :, :] = np.diag(var_diag)

    return mu, sigma

def classifyBayes(X, prior, mu, sigma):
    Npts = X.shape[0]
    Nclasses, Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    for jdx in range(Nclasses):
        x_centered = X - mu[jdx, :]
        variances = np.diag(sigma[jdx, :, :])
        
        # Discriminant function based on log posterior (Equation 11)
        logDetSigma = np.sum(np.log(variances))
        logPrior = np.log(prior[jdx])
        
        # Diagonal covariance allows computing dimensions separately 
        logProb[jdx, :] = -0.5 * logDetSigma - 0.5 * np.sum((x_centered**2) / variances, axis=1) + logPrior
    
    return np.argmax(logProb, axis=0)

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

def trainBoost(base_classifier, X, labels, T=10):
    Npts, Ndims = np.shape(X)
    classifiers = []
    alphas = []

    # Step 0: Initialize all weights uniformly
    wCur = np.ones((Npts, 1)) / float(Npts)

    for t in range(T):
        # Step 1: Train weak learner using distribution w 
        current_classifier = base_classifier.trainClassifier(X, labels, wCur)
        classifiers.append(current_classifier)

        # Step 2: Get weak hypothesis and compute weighted error epsilon 
        vote = current_classifier.classify(X)
        correct = (vote == labels).astype(float).reshape(-1, 1)
        # Weighted error epsilon 
        error = np.sum(wCur * (1.0 - correct))

        # Step 3: Choose alpha
        # Add a tiny constant to avoid division by zero or log of zero
        alpha = 0.5 * (np.log(1.0 - error + 1e-10) - np.log(error + 1e-10))
        alphas.append(alpha)

        # Step 4: Update weights and normalize
        # Weight increase for incorrect, decrease for correct
        wCur = wCur * np.exp(alpha * (1.0 - 2.0 * correct))
        wCur = wCur / np.sum(wCur) # Normalization factor Z

    return classifiers, alphas

def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    T = len(classifiers)
    votes = np.zeros((Npts, Nclasses))

    # Final classification H(x) aggregates votes weighted by alpha
    for t in range(T):
        preds = classifiers[t].classify(X)
        for i in range(Npts):
            votes[i, preds[i]] += alphas[t] # Weighted vote

    return np.argmax(votes, axis=1)

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


bc = BayesClassifier()
boosted_bc = BoostClassifier(bc, T=10)
    
print("\n--- Boosted Iris Results ---")
testClassifier(boosted_bc, dataset='iris')
    
print("\n--- Boosted Vowel Results ---")
testClassifier(boosted_bc, dataset='vowel')
    
plotBoundary(boosted_bc, dataset='iris')
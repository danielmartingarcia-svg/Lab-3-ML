# Lab 3: Bayes Classifier and Boosting

A machine learning lab implementing Bayesian classification and boosting algorithms, with support for multiple classification datasets.

## Overview

This lab project implements:
- **Bayes Classifier**: Maximum likelihood estimation with Gaussian class distributions
- **Boosting**: Adaptive boosting (AdaBoost) framework for ensemble methods
- **Support for multiple base classifiers**: Bayes classifier and decision tree classifiers

## Project Structure

### Main Files
- **`lab3.py`**: Main implementation file containing the core algorithms to complete
  - Bayes classifier functions: `computePrior()`, `mlParams()`, `classifyBayes()`
  - Boosting functions: `trainBoost()`, `classifyBoost()`
  - Wrapper classes: `BayesClassifier`, `BoostClassifier`

- **`labfuns.py`**: Helper functions library providing:
  - Plotting utilities (gaussian distributions, decision boundaries, covariance ellipses)
  - Data loading and preprocessing functions
  - Testing framework and evaluation metrics
  - Support for multiple datasets

### Datasets
Four datasets are included, each with feature files (X) and label files (Y):
- **`irisX.txt` / `irisY.txt`**: Classic Iris flower dataset (150 samples, 4 features, 3 classes)
- **`vowelX.txt` / `vowelY.txt`**: Vowel recognition dataset (528 samples, 10 features, 11 classes)
- **`wineX.txt` / `wineY.txt`**: Wine classification dataset (178 samples, 13 features, 3 classes)
- **`olivettifacesX.txt` / `olivettifacesY.txt`**: Olivetti faces dataset (400 samples, 4096 features, 40 classes)

## Implementation Tasks

### Part 1: Bayes Classifier
Complete these functions in `lab3.py`:

1. **`computePrior(labels, W=None)`**
   - Compute class priors from training labels
   - Returns: C×1 vector of class probabilities

2. **`mlParams(X, labels, W=None)`**
   - Compute maximum likelihood estimates for class means and covariances
   - Returns: mu (C×d matrix), sigma (C×d×d tensor)

3. **`classifyBayes(X, prior, mu, sigma)`**
   - Classify test points using log posterior probabilities
   - Returns: N vector of class predictions

### Part 2: Boosting
Complete these functions in `lab3.py`:

1. **`trainBoost(base_classifier, X, labels, T=10)`**
   - Train T boosted classifiers with adaptive sample weighting
   - Compute voting weights (alphas) based on classifier accuracy
   - Returns: list of classifiers and their weights

2. **`classifyBoost(X, classifiers, alphas, Nclasses)`**
   - Classify using weighted ensemble votes
   - Returns: N vector of class predictions

## Usage

### Testing Individual Components
```python
# Test Bayes classifier
testClassifier(BayesClassifier(), dataset='iris', split=0.7)

# Test boosted Bayes classifier
testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris', split=0.7)

# Test decision tree classifier
testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)

# Test boosted decision trees
testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel', split=0.7)
```

### Visualizing Decision Boundaries
```python
# Plot decision boundaries for different classifiers
plotBoundary(BayesClassifier(), dataset='iris', split=0.7)
plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris', split=0.7)
```

### Bonus: Face Classification
Test on the Olivetti faces dataset (requires PCA dimensionality reduction):
```python
testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), 
               dataset='olivetti', split=0.7, dim=20)
```

## Key Concepts

### Bayes Classification
- Uses multivariate Gaussian class distributions
- Computes log posterior: $\log p(y|x) = \log p(x|y) + \log p(y) - \log p(x)$
- Selects class with maximum posterior probability

### Boosting (AdaBoost)
- Iteratively trains classifiers on weighted samples
- Samples with higher misclassification errors get increased weights
- Final prediction: weighted majority vote of all classifiers
- Alpha weight depends on individual classifier accuracy

## Requirements
- NumPy
- SciPy
- Matplotlib
- scikit-learn (for decision trees and PCA)
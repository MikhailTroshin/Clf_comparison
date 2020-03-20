# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:34:32 2020

@author: TroshinMV

Module for different linear and metric classifier's performance comparison
Tested classifiers:
    - Logistic regression
    - Kernel SVM
    - KNN
    - Naive Bayes
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons, make_circles

#___________________ Open datasets___________________

#dataset = pd.read_csv('Social_Network_Ads.csv')
moons = make_moons(n_samples = 500, shuffle = True, random_state = 14)
circles = make_circles(n_samples = 500, shuffle = True, random_state = 14, factor = 0.7)
dataset_names = ['Moons', 'Circles']
datasets = (moons, circles)

#________________ Prepare classifiers________________

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

clf_names = ['LogRegression','RBF_SVM','KNN','Bayes']
clfs = (LogisticRegression(penalty='l2', C=100, random_state=14),
        SVC(C=5, kernel='rbf', random_state=14),
        KNeighborsClassifier(n_neighbors=20, weights='distance', p=2),
        GaussianNB())

#__________________ Prepare scores__________________

# Choose metric
from sklearn.metrics import roc_auc_score, classification_report

# DataFrame of scores (roc_auc scores)
all_scores = pd.DataFrame(np.zeros((2,4)), index=dataset_names, columns=clf_names)

# Make a score func
def clf_score(clf, dataset_name, clf_name):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    all_scores.at[dataset_name, clf_name] = roc_auc_score(y_test, y_pred)
    pass

    
#_________________ Plot graphics __________________

def make_meshgrids(x,y,h=0.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

score_report = open('ROC_AUC_score_report.txt', 'w+')

# dataset is not large, so we will not use KFold, simple splitter will be enough
from sklearn.model_selection import train_test_split

figure = plt.figure(figsize=(48, 18))
i = 1
for dataset, dataset_name in zip(datasets, dataset_names):
    X, y = dataset[0], dataset[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)
    xx, yy = make_meshgrids(X[:,0], X[:,1])
    score_report.write(dataset_name + '\n')
    
    for clf, clf_name in zip(clfs, clf_names):
        ax = plt.subplot(4,2,i)
        clf_score(clf, dataset_name, clf_name)
        score_report.write(clf_name + ' roc-auc score: ' + str(round(all_scores.at[dataset_name, clf_name], 3)) + '\n')
        
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
        Z_line = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        Z_line = Z_line.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.7)
        ax.contourf(xx, yy, Z_line, cmap=plt.cm.gray, alpha=0.25)
        ax.scatter(X[:,0], X[:,0], c=y, cmap=plt.cm.coolwarm, alpha=0.4, edgecolors='black')
        ax.set_title(clf_name)
        ax.set_xlim(X[:,0].min()-0.5, X[:,0].max()+0.5)
        ax.set_ylim(X[:,1].min()-0.5, X[:,1].max()+0.5)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        i += 1
    i += 1
    
score_report.close()
plt.show()


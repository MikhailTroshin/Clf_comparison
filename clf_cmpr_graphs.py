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

#___________________ Prepare data ___________________

dataset = pd.read_csv('Social_Network_Ads.csv')

# missing Data - no missing data
dataset.isnull().sum()

# Categorical variables
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
dataset.Gender = enc.fit_transform(dataset.Gender)

# features, target
X = dataset.iloc[:, 2:-1]
y = dataset.Purchased

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X.iloc[:,:] = sc.fit_transform(X.iloc[:,:])

# dataset is not large, so we will not use KFold, simple splitter will be enough
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)

#___________________ Prepare scores___________________

# Choose metric
from sklearn.metrics import accuracy_score, classification_report

# DataFrame of scores (accuracy scores)
names = ['LogRegression','RBF_SVM','KNN','Bayes']
all_scores = pd.DataFrame(np.zeros((1,4)), columns=names)


#___________________ Build classifiers ___________________

# Make a score func
def clf_score(clf, name):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    all_scores.at[0, name] = accuracy_score(y_test, y_pred)
    pass

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

clfs = (LogisticRegression(penalty='l2', C=100, random_state=14),
        SVC(C=5, kernel='rbf', random_state=14),
        KNeighborsClassifier(n_neighbors=20, weights='distance', p=2),
        GaussianNB())
    
#___________________ Plot graphics ___________________

def make_meshgrids(x,y,h=0.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

score_report = open('Score_report.txt', 'w+')

X0, X1 = X['Age'], X['EstimatedSalary']
xx, yy = make_meshgrids(X0, X1)

figure = plt.figure(figsize=(24, 18))
i=1
for name, clf in zip(names, clfs):
    ax = plt.subplot(2,2,i)
    clf_score(clf, name)
    score_report.write(name + ' score: ' + str(round(all_scores[name][0], 3)) + '\n')
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    Z_line = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    Z_line = Z_line.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.7)
    ax.contourf(xx, yy, Z_line, cmap=plt.cm.gray, alpha=0.25)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, alpha=0.4, edgecolors='black')
    ax.set_title(name)
    ax.set_xlim(X0.min()-0.5, X0.max()+0.5)
    ax.set_ylim(X1.min()-0.5, X1.max()+0.5)
    ax.set_xlabel('Age')
    ax.set_ylabel('Estimated Salary')
    i += 1
    
score_report.close()
plt.show()
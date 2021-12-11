from __future__ import division

# By pass warnings
#====================================================================
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

print("\n------ TRAINING -stage-2-Classification.py ------")


# Define Important Library
#====================================================================
import pandas as pd
import numpy as np
import os
import sys
import csv
import pickle


# Start from 1 always, no random state
#====================================================================
np.random.seed(1)

# library import
#====================================================================
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import make_pipeline

import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import auc, roc_curve,roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score

from sklearn.model_selection import KFold


import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image


# The Dataset 
#====================================================================

dirname = os.path.dirname(__file__)
np.set_printoptions(threshold=sys.maxsize)

#import stage1DT as c2

#X = c2.df_data
#y = c2.df_target
[X,y] = pickle.load(open('stage-1-train-green.pickle', 'rb'))
#train1_feature_name = list(df.columns.values[:-1])
print(y.shape)
print(X.shape)
print(np.unique(y))


# Define classifiers within a list
#====================================================================
Classifiers = [
               GradientBoostingClassifier(random_state=0),
               DecisionTreeClassifier(random_state=0),
               GaussianNB(),
               KNeighborsClassifier(n_neighbors=3),
               RandomForestClassifier(random_state=0, criterion='entropy'),
               ExtraTreesClassifier(random_state=0),
               LogisticRegression(C=1),
               AdaBoostClassifier(random_state=0),
               svm.SVC(kernel='linear', C=1, probability=True),
               ]

# Spliting with 10-Folds :
#====================================================================
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=False)


# Pick all classifier within the Classifier list and test one by one
#====================================================================

for classifier in Classifiers:
    
    precision_SE = []
    recall_SE = []
    auPR_SE = []
    
    precision_EX = []
    recall_EX = []
    auPR_EX = []
    
    precision_MA = []
    recall_MA = []
    auPR_MA= []
    
    precision_HE = []
    recall_HE = []
    auPR_HE = []
    
    y_all_proba_SE = []
    y_all_proba_EX = []
    y_all_proba_MA = []
    y_all_proba_HE = []
    y_all_test = []
    y_all_pred = []
    
    fold = 1
    print('____________________________________________')
    #print('Classifier: '+classifier.__class__.__name__)
    #model = OneVsOneClassifier(classifier)
    model = OneVsRestClassifier(classifier)
    
    for train_index, test_index in cv.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]
        
        y_train = y[train_index]
        y_test = y[test_index]
        
        # print the fold number and numbber of feature after selection
        # -----------------------------------------------------------------
        print("----- Working on Fold -----------> {0}:".format(fold))
        
        # Applying Sampling to achieve class balance
        # -----------------------------------------------------------------
        # counting unique occurences in the target column
        (unique, counts) = np.unique(y_train, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        #print(frequencies)
        
        avg = 0
        for val in frequencies:
            avg = avg + val[1]
        avg = int(avg/4)
        # define sampling strategy
        strategy = dict()
        strategy[1] = frequencies[0][1]
        strategy[2] = avg # Hard Exudates Under Sampling
        strategy[3] = frequencies[2][1]
        strategy[4] = frequencies[3][1]
        #print(strategy)
        
        # sampling on the chunk
        sampling = RandomUnderSampler(sampling_strategy=strategy)
        #sampling = SMOTE(sampling_strategy=strategy)
        XX_train, yy_train = sampling.fit_resample(X_train, y_train)
        
        # Train model
        # -----------------------------------------------------------------
        model.fit(XX_train, yy_train)
        
        # Evaluation
        # -----------------------------------------------------------------
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        y_proba = np.nan_to_num(y_proba)
        
        y_all_proba_SE.append(y_proba[:, 0])
        y_all_proba_EX.append(y_proba[:, 1])
        y_all_proba_MA.append(y_proba[:, 2])
        y_all_proba_HE.append(y_proba[:, 3])
        y_all_test.append(y_test)
        y_all_pred.append(y_pred)
                
        precision, recall, fscore, support = score(y_test, y_pred)
        
        precision_SE.append(precision[0])
        recall_SE.append(recall[0])
        
        precision_EX.append(precision[1])
        recall_EX.append(recall[1]) 
        
        precision_MA.append(precision[2])
        recall_MA.append(recall[2])
        
        precision_HE.append(precision[3])
        recall_HE.append(recall[3])
        

        fold += 1
    
    print('\n\nClassifier: '+classifier.__class__.__name__)
    
    filename = 'green/green-{0}.pickle'.format(classifier.__class__.__name__)
    pickle.dump(model, open(filename, 'wb'))
    
    y_all_test = np.concatenate(y_all_test)
    y_all_proba_SE = np.concatenate(y_all_proba_SE)
    y_all_proba_EX = np.concatenate(y_all_proba_EX)
    y_all_proba_MA = np.concatenate(y_all_proba_MA)
    y_all_proba_HE = np.concatenate(y_all_proba_HE)
    y_all_pred = np.concatenate(y_all_pred)
    
    lr1_precision, lr1_recall, _ = precision_recall_curve(y_all_test, y_all_proba_SE, pos_label=1)
    lr2_precision, lr2_recall, _ = precision_recall_curve(y_all_test, y_all_proba_EX, pos_label=2)
    lr3_precision, lr3_recall, _ = precision_recall_curve(y_all_test, y_all_proba_MA, pos_label=3)
    lr4_precision, lr4_recall, _ = precision_recall_curve(y_all_test, y_all_proba_HE, pos_label=4)
    
    print('Soft Exudates --> auPR: {0:3.4}'.format(auc(lr1_recall, lr1_precision)*100))
    print('Hard Exudates --> auPR: {0:3.4}'.format(auc(lr2_recall, lr2_precision)*100))
    print('Microaneurysms --> auPR: {0:3.4}'.format(auc(lr3_recall, lr3_precision)*100))
    print('Haemorrhages --> auPR: {0:3.4}'.format(auc(lr4_recall, lr4_precision)*100))

    #plot the precision-recall curves
    plt.figure()
    plt.plot(lr1_recall, lr1_precision, marker='.', label='Soft Exudates')
    plt.plot(lr2_recall, lr2_precision, marker='o', label='Hard Exudates')
    plt.plot(lr3_recall, lr3_precision, marker='*', label='Microaneurysm')
    plt.plot(lr4_recall, lr4_precision, marker='*', label='Haemorrhages')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Performance of "+classifier.__class__.__name__+" on Training Dataset")
    plt.legend()
    #plt.savefig('{0}_train.png'.format(classifier.__class__.__name__))
    plt.show()
    
    print('____________________________________________\n\n')
    
    
#print('Soft Exudates --> precision: {0:3.4}, recall: {1:3.4}'.format(np.mean(precision_SE), np.mean(recall_SE)))
#print('Hard Exudates --> precision: {0:3.4}, recall: {1:3.4}'.format(np.mean(precision_EX), np.mean(recall_EX)))
#print('Microaneurysms --> precision: {0:3.4}, recall: {1:3.4}'.format(np.mean(precision_MA), np.mean(recall_MA)))
#print('Haemorrhages --> precision: {0:3.4}, recall: {1:3.4}'.format(np.mean(precision_HE), np.mean(recall_HE)))
    
    

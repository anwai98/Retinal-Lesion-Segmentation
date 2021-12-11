from __future__ import division

# By pass warnings
#====================================================================
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

print("\n------ TESTING -stage-2-Classification.py ------")


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
[X_test,y_test] = pickle.load(open('stage-1-test-green.pickle', 'rb'))
#train1_feature_name = list(df.columns.values[:-1])
print(y_test.shape)
print(X_test.shape)
print(np.unique(y_test))


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


# Pick all classifier within the Classifier list and test one by one
#====================================================================

for classifier in Classifiers:
    
    print('____________________________________________')
    #print('Classifier: '+classifier.__class__.__name__)
    
    filename = 'green/green-{0}.pickle'.format(classifier.__class__.__name__)
    model = pickle.load(open(filename, 'rb'))

    # Evaluation
    # -----------------------------------------------------------------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    y_proba = np.nan_to_num(y_proba)

    print('\n\nClassifier: '+classifier.__class__.__name__)
    
    lr1_precision, lr1_recall, _ = precision_recall_curve(y_test, y_proba[:, 0], pos_label=1)
    lr2_precision, lr2_recall, _ = precision_recall_curve(y_test, y_proba[:, 1], pos_label=2)
    lr3_precision, lr3_recall, _ = precision_recall_curve(y_test, y_proba[:, 2], pos_label=3)
    lr4_precision, lr4_recall, _ = precision_recall_curve(y_test, y_proba[:, 3], pos_label=4)
    
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
    plt.title("Performance of "+classifier.__class__.__name__+" on Test Dataset")
    plt.legend()
    #plt.savefig('{0}_train.png'.format(classifier.__class__.__name__))
    plt.show()
    
    print('____________________________________________\n\n')
    


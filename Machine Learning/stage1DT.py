
from __future__ import division

# By pass warnings
#====================================================================
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

print("stage-1-DT-Train.py")

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
from sklearn.preprocessing import StandardScaler

import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score


import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image


# The Dataset 
#====================================================================
import datasetTestTrainGreen as datasets

train_data = datasets.train_data
train_target = datasets.train_target


# Normalizing and Binarizing the dataset
#====================================================================
train_target_bin = np.copy(train_target)
for i in range(0,train_target_bin.shape[0],1):
    if(train_target_bin[i] != 0):
       train_target_bin[i] = 1;
print("----- Binary transformed -----")
print(np.unique(train_target_bin))

sc = StandardScaler()
train_data = sc.fit_transform(train_data)


# Decision Tree classifier for Lesion vs Non Lesion
#====================================================================

df_data = np.array([])
df_target = np.array([])

auPR = []
model = []

print('_________________________________________________________')


# counting unique occurences in the target column
(unique, counts) = np.unique(train_target_bin, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

# define sampling strategy
strategy = dict()
strategy[0] = frequencies[1][1]*2
strategy[1] = frequencies[1][1]
print(strategy)

# sampling on the chunk
sampling = RandomUnderSampler(sampling_strategy=strategy)
X_train_data, y_train_target_bin = sampling.fit_resample(train_data, train_target_bin)

np.random.seed(40)
model = DecisionTreeClassifier(random_state=0)

model.fit(train_data, train_target_bin)

# Now make predictions with trained model
y_pred = model.predict(train_data)


#auPR = average_precision_score(train_target_bin, y_pred)
#print ("auPR for lesion: {0:3.6}".format(auPR))

recall_1 = recall_score(train_target_bin, y_pred, pos_label=1)
print ("sensitivity for lesion: {0:3.4}%".format(recall_1*100))
recall_0 = recall_score(train_target_bin, y_pred, pos_label=0)
print ("sensitivity for non lesion: {0:3.4}%".format(recall_0*100))


print("\nin prediction")
print(np.unique(y_pred))

three = 0
found_0 = 0
found_1 = 0
among_0 = 0
among_1 = 0
#for i in range(0, y_pred.shape[0], 1):


print("\ntrain-target unique")
print(np.unique(train_target))

print("\ny-pred unique")
print(np.unique(y_pred))

for i in range(0, y_pred.shape[0], 1):
    #print(y_pred[i])

    if(train_target_bin[i] == 1):
        among_1 = among_1 + 1
    if((train_target_bin[i] == y_pred[i]) and (train_target_bin[i] == 1)):
        found_1 = found_1 + 1
        
    if(train_target_bin[i] == 0):
        among_0 = among_0 + 1
    if((train_target_bin[i] == y_pred[i]) and (train_target_bin[i] == 0)):
        found_0 = found_0 + 1
    
    if(y_pred[i] != 0):
        if(three == 0):
            df_data = train_data[i]
            df_target = train_target[i]
            three = 1
        else:
            df_data = np.vstack((df_data,train_data[i]))
            df_target = np.vstack((df_target,train_target[i]))
    

print(df_data.shape)
print(df_target.shape)
print("\ntrain-target unique - after stage 1")
print(np.unique(df_target))
#print(three)
print("\n")
print("{0} among {1} of lesions match".format(found_1, among_1))
print("{0} among {1} of non-lesions match".format(found_0, among_0))

with open('stage-1-train-green.pickle', 'wb') as f:
    pickle.dump([df_data,df_target], f)

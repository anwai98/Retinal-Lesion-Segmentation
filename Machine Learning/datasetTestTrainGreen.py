
from __future__ import division

# By pass warnings
#====================================================================
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

print("getting the training and testing dataset.py")

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
dirname = os.path.dirname(__file__)
np.set_printoptions(threshold=sys.maxsize)


# TRAINING Dataset 
#====================================================================
print("\nTRAINING Dataset")

df4 = pd.read_csv('green/SE_results-green.csv', header=None)
df1 = pd.read_csv('green/EX_results-green.csv', header=None)
df2 = pd.read_csv('green/MA_results-green.csv', header=None)
df3 = pd.read_csv('green/HE_results-green.csv', header=None)

train1_data = np.asarray(df1.iloc[:, 6:-1])
train1_target = np.asarray(df1.iloc[:,-1]).reshape(-1, 1)
print("----- Hard Exudates - 2 ----- ")
print(np.unique(train1_target))
#print(train1_data[0][0])
print(train1_target.shape)

train2_data = np.asarray(df2.iloc[:, 6:-1])
train2_target = np.asarray(df2.iloc[:,-1]).reshape(-1, 1)
print("----- Microaneurysms - 3 -----")
print(np.unique(train2_target))
#print(train2_data.shape)
print(train2_target.shape)

train3_data = np.vstack((train1_data,train2_data))
train3_target = np.vstack((train1_target,train2_target))

train4_data = np.asarray(df3.iloc[:, 6:-1])
train4_target = np.asarray(df3.iloc[:,-1]).reshape(-1, 1)
print("----- Haemorrhages - 4 -----")
print(np.unique(train4_target))
#print(train4_data.shape)
print(train4_target.shape)


train5_data = np.vstack((train3_data,train4_data))
train5_target = np.vstack((train3_target,train4_target))


train6_data = np.asarray(df4.iloc[:, 6:-1])
train6_target = np.asarray(df4.iloc[:,-1]).reshape(-1, 1)
print("----- Soft Exudates - 1 -----")
print(np.unique(train6_target))
#print(train6_data.shape)
print(train6_target.shape)


train_data = np.vstack((train5_data,train6_data))
train_target = np.vstack((train5_target,train6_target))
#train_target = LabelEncoder().fit_transform(train_target)
print("----- The entire dataset -----")
print(train_data.shape)
print(train_target.shape)
print(np.unique(train_target))


# TESTING Dataset 
#====================================================================
print("\nTESTING Dataset")

df5 = pd.read_csv('green/SE_test_set_green.csv', header=None)
df6 = pd.read_csv('green/EX_test_set_green.csv', header=None)
df7 = pd.read_csv('green/MA_test_set_green.csv', header=None)
df8 = pd.read_csv('green/HE_test_set_green.csv', header=None)

test1_data = np.asarray(df5.iloc[:, 6:-1])
test1_target = np.asarray(df5.iloc[:,-1]).reshape(-1, 1)
print("----- Soft Exudates - 1 ----- ")
print(np.unique(test1_target))
#print(train1_data[0][0])
print(test1_target.shape)

test2_data = np.asarray(df6.iloc[:, 6:-1])
test2_target = np.asarray(df6.iloc[:,-1]).reshape(-1, 1)
print("----- Hard Exudates - 2 -----")
print(np.unique(test2_target))
#print(train2_data.shape)
print(test2_target.shape)

test3_data = np.vstack((test1_data,test2_data))
test3_target = np.vstack((test1_target,test2_target))

test4_data = np.asarray(df7.iloc[:, 6:-1])
test4_target = np.asarray(df7.iloc[:,-1]).reshape(-1, 1)
print("----- Microaneurysms - 3 -----")
print(np.unique(test4_target))
#print(train4_data.shape)
print(test4_target.shape)


test5_data = np.vstack((test3_data,test4_data))
test5_target = np.vstack((test3_target,test4_target))


test6_data = np.asarray(df8.iloc[:, 6:-1])
test6_target = np.asarray(df8.iloc[:,-1]).reshape(-1, 1)
print("----- Haemorrhages - 4 -----")
print(np.unique(test6_target))
#print(train6_data.shape)
print(test6_target.shape)


test_data = np.vstack((test5_data,test6_data))
test_target = np.vstack((test5_target,test6_target))
#train_target = LabelEncoder().fit_transform(train_target)
print("----- The entire dataset -----")
print(test_data.shape)
print(test_target.shape)
print(np.unique(test_target))



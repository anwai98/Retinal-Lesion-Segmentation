# This is script which iterates through all the images and calls the necessary scripts to extract features and create the csv file 

"""
SE = 1
EX = 2
MA = 3
HE = 4
"""


import numpy as np 
import pandas as pd 

import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image

import os
import sys
import csv
import string
dirname = os.path.dirname(__file__)
np.set_printoptions(threshold=sys.maxsize)

import eachRegionGreen as c1

distances = [1, 2, 3]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
properties = ['contrast', 'energy', 'homogeneity', 'correlation']
theta = []

all_types = ['SE','EX','MA','HE']


######################################################
################# READING ALL IMAGES #################
######################################################

for l_type in all_types:

    print("Extracting dataset for {0}".format(l_type))
    
    if(l_type == "SE"):
        l_par = '1'
    elif(l_type == "EX"):
        l_par = '2'
    elif(l_type == "MA"):
        l_par = '3'
    elif(l_type == "HE"):
        l_par = '4'
    
    with open('{0}_test_set_green.csv'.format(l_type), 'a+') as csvfile:
        
        print("\nCreating dataset for {0}".format(l_type))
    
        for i in range(55,82,1):
        
            if ((i>=1) and (i<=9)):
                s = '0' + str(i)
            else:
                s = str(i)
    
            filename_org = "IDRiD_" + s + ".jpg"
            filename_s = "IDRiD_" + s + "_" + l_type + "_s.png"
            filename_gt = "IDRiD_" + s + "_" + l_type + ".tif"
        
    
            print("----- Working on {0}....... --------".format(filename_org))
            #print(filename_s)
            #print(filename_gt)
    
            img = cv2.imread(os.path.join(dirname, 'images', 'test', filename_org))
            img_s = cv2.imread(os.path.join(dirname, 'segments', 'test', '{0}'.format(l_type), filename_s))
            gt = cv2.imread(os.path.join(dirname, 'groundtruths', 'test', '{0}'.format(l_type), filename_gt))
    

            
            if(type(gt) != type(None)):
                df = c1.getAllRegions(img, gt, img_s, filename_org , distances, angles, properties, l_par)
                for i in range(0,df.shape[0],1):
                    for j in range(0,df.shape[1],1):
                        if(j!=0):
                            csvfile.write(",")
                        csvfile.write("{0}".format(df[i][j]))
                    csvfile.write("\n")
        
            
        
    
                   

    
    
    



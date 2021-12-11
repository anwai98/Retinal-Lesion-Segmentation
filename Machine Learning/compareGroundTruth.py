# this file contains the function to comapre the candidate region with the ground truth


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

def compareGT(image, target):
    
    """plt.imshow(image)
    plt.axis("off")
    plt.show()"""
    
    #print(image.shape)
    #print(target.shape)
    
    ret1,image = cv2.threshold(image,1,255,cv2.THRESH_BINARY)
    ret2,target = cv2.threshold(target,1,255,cv2.THRESH_BINARY)
    
    value = 0
    
    n = image.shape[0]
    m = image.shape[1]
    #print(n)
    #print(m)
    count = 0
    
    total = n*m
    
    #print(target.shape)
    
    for i in range(0, n, 1):
        for j in range(0, m, 1):
            if(target[i][j] == image[i][j]) and (image[i][j] == 255):
                count = count + 1
            
    #print(count)
    #print(total)
    value = count/total
    
    if(value>=0.3):
        return 1
    return 0


#img = cv2.imread("IDRiD_10_EX.tif", 0)
#gt = cv2.imread("IDRiD_33_EX.tif", 0)

#i = compareGT(img, gt)
#print(i)




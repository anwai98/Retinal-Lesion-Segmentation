# -*- coding: utf-8 -*-
"""AIA-Soft Exudates.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jxjve_lNlh6c8piIqiKr_2JQpUB5YdEs
"""

# Importing the Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

# Evaluation Metrics - To Measure the Sensitivity
def evaluation(image, mask):
    
    zeros_list_img, one_list_img, zeros_list_mk, one_list_mk = [], [], [], []
    
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            val_mk = mask[i][j]
            val_img  = image[i][j]
            if val_mk == 0:
                zeros_list_mk.append((i,j))
            else:
                one_list_mk.append((i,j))
            if val_img == 0:
                zeros_list_img.append((i,j))
            else:
                one_list_img.append((i,j))

    TP = len(set(one_list_img).intersection(set(one_list_mk)))
    TN = len(set(zeros_list_img).intersection(set(zeros_list_mk)))
    FP = len(set(one_list_img).intersection(set(zeros_list_mk)))
    FN = len(set(zeros_list_img).intersection(set(one_list_mk)))
    R = TP/(TP + FN)
    return R

# Function for Visualising Images
def imshow(img):
  plt.axis('off')
  plt.imshow(img, 'gray')
  plt.show()

# Image and Label Path
image_path = "/content/drive/My Drive/AIA Team Project/data/images/test/"
images = os.listdir(image_path)
images.sort()

se_labels_path = "/content/drive/My Drive/AIA Team Project/data/groundtruths/test/soft exudates/"

for img_number in range(55,82,1):

  if ((img_number>=1) and (img_number<=9)):
    s_num = '0' + str(img_number)
  else:
    s_num = str(img_number)
  
  img_file = image_path + 'IDRiD_'+ str(s_num) +'.jpg'
  se_file = se_labels_path + 'IDRiD_'+ str(s_num) +'_SE.tif'

  img_se = cv2.imread(se_file)
  
  # Handling the Exceptions of Absence of Cotton Wool Spots in many patients
  if(type(img_se) != type(None)):

    # Mask
    img_se_gray = cv2.cvtColor(img_se, cv2.COLOR_BGR2GRAY)
    # Reading the Image
    img = cv2.imread(img_file)

    # Smoothing - Blur
    img_blur = cv2.blur(img,(30,50))

    # Contrast Stretch
    b,g,r = cv2.split(img_blur)
    a = 1.7
    img_mark = float(a) * g
    # Cut-Off Pixel > 255
    img_mark[img_mark > 255] = 255
    # Data Type 
    img_mark = np.round(img_mark)
    img_mark = img_mark.astype(np.uint8)

    # Gradient Operations (Sobel Operators)
    x = cv2.Sobel(img_mark,cv2.CV_32F,1,0)
    y = cv2.Sobel(img_mark,cv2.CV_32F,0,1)
    
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    
    img_grad_wt = cv2.addWeighted(absX,0.9,absY,0.1,0)

    # Watershed
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_open = cv2.morphologyEx(img_grad_wt, cv2.MORPH_OPEN, kernel, iterations=1)

    # Distance Transform
    dist = cv2.distanceTransform(img_open, cv2.DIST_L2, 3)
    dist_result = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
    
    ret, img_final = cv2.threshold(dist, dist.max()*0.001, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_final = cv2.morphologyEx(img_final, cv2.MORPH_OPEN, kernel, iterations=2)

    # # Matching and Evaluating the Results
    # print('Soft Exudates Detection: ')
    # imshow(img_final)

    # print('Soft Exudates Labels: ')
    # imshow(img_se_gray)

    # print('------------------------')
    # R = evaluation(img_final, img_se_gray)
    # print('Sensitivity = ', end = '')
    # print(R*100)

    # Saving the Segmented Images
    # cv2.imwrite("/content/drive/My Drive/AIA Team Project/segments/TEST DATA/SE/" + 'IDRiD_' + str(img_number) + '_SE_s.png', img_final)
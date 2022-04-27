# -*- coding: utf-8 -*-

# Importing Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

# Image and Label Path
image_path = "/content/drive/My Drive/AIA Team Project/data/images/test/"
ma_labels_path = "/content/drive/My Drive/AIA Team Project/data/groundtruths/test/microaneurysms/"
images = os.listdir(image_path)
ma_labels = os.listdir(ma_labels_path)

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

# Function for Visualising the Images
def imshow(img):
  plt.axis('off')
  plt.imshow(img, 'gray')
  plt.show()

for img_number in range(55,82,1):

  # Fetching the Image and Label Paths
  images.sort()
  img_file = image_path + "IDRiD_" + s + ".jpg"
  ma_labels.sort()
  ma_file = ma_labels_path + "IDRiD_" + s + "_MA.tif"

  # Reading the Image
  img = cv2.imread(img_file)
  b,g,r = cv2.split(img)

  # Reading the Label
  img_ma = cv2.imread(ma_file)
  # Grayscale Label
  img_ma_gray = cv2.cvtColor(img_ma, cv2.COLOR_BGR2GRAY)
  
  # Resizing to 576*720
  img_resize = cv2.resize(g, (576,720), interpolation=cv2.INTER_CUBIC)

  # Applying Adaptive Histogram Equalization
  clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(10,10))
  img_clahe = clahe.apply(img_resize)

  # Canny Edge Detection
  img_canny = cv2.Canny(img_clahe, 20, 130)

  # Gaussian Blur - Smoothing
  img_gb = cv2.GaussianBlur(img_canny,(3,3),0)

  # Disc-Shaped Structuring Element (SE) of Radius 3
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
  # Morphological Opening Operation (Erosion followed by Dilation)
  img_opening = cv2.morphologyEx(img_gb, cv2.MORPH_OPEN, kernel)

  # Subtracting from the image of microaneurysms and artifacts
  img_result = cv2.subtract(img_opening, img_clahe)

  # Resizing to Original Size
  img_final = cv2.resize(img_result, (4288,2848), interpolation=cv2.INTER_CUBIC)

  # # Matching and Evaluating the Results
  # print('Microaneurysms Detection:')
  # imshow(img_final)

  # print('Microaneurysms Labels:')
  # imshow(img_ma_gray)

  # print('------------------------')
  # R = evaluation(img_final, img_ma_gray)
  # print('Sensitivity = ', end = '')
  # print(R*100)

  # Saving the Segmented Images
  # cv2.imwrite("/content/drive/My Drive/AIA Team Project/segments/TEST DATA/MA/" + 'IDRiD_' + str(img_number) + '_MA_s.png', img_final)
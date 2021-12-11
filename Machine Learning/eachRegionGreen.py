# this script is called for each image, and it extracts all the regions, does feature extraction and assign the label of the regions as well

import cv2
import numpy as np
import pandas as pd 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage.transform import integral_image
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
from skimage import io, color, img_as_ubyte

import compareGroundTruth as c2

def getAllRegions(image, groundTruth, img_s, filename, distances, angles, properties, target_val):
    
    hh, ww = img_s.shape[:2]
    # convert to grayscale
    gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
    groundTruth_m = cv2.cvtColor(groundTruth, cv2.COLOR_BGR2GRAY)
    
    # create a binary thresholded image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    rrr,groundTruth_m = cv2.threshold(groundTruth_m, 1, 255, cv2.THRESH_BINARY)
    
    # get external contours
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contour_img = img_s.copy()
    rotrect_img = image.copy()
    rotrect_gt_img = groundTruth.copy()
    
    rows = len(contours)
    train_set = np.array([])

    cn = 1
    
    matching = 0
    for c in contours:
        
        #print("Working on contour {0}".format(cn));
        
        # draw contour on input
        cv2.drawContours(contour_img,[c],0,(0,0,255),2)
    
        imageID = filename[-12:-4]
        #print(imageID)
        
        temp = np.array([])
        temp = np.append(temp, imageID)
        temp = np.append(temp, cn)
        
        #print("line 50 {0}".format(temp.shape))
        
        c_area = cv2.contourArea(c)
        c_hull = cv2.convexHull(c)
        c_hull_area = cv2.contourArea(c_hull)
        if(c_area == 0):
            c_solidity = 0
        else:
            c_solidity = float(c_area)/c_hull_area
            
        
        #print("line 63 {0}".format(temp.shape))

        # get rotated rectangle from contour
        # get its dimensions
        # get angle relative to horizontal from rotated rectangle
        rotrect = cv2.minAreaRect(c)
        (center), (width,height), angle = rotrect
        box = cv2.boxPoints(rotrect)
        boxpts = np.int0(box)

        # draw rotated rectangle on copy of image
        cv2.drawContours(rotrect_img,[boxpts],0,(0,255,0),2)
        cv2.drawContours(rotrect_gt_img,[boxpts],0,(0,255,0),2)
        
        
        # draw mask as filled rotated rectangle on black background the size of the input
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask,[boxpts],0,255,-1)

        # apply mask to cleaned image
        blob_img = cv2.bitwise_and(thresh, mask)
        
        # threshold it again
        deskewed = cv2.threshold(blob_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        
        # get bounding box of contour of deskewed rectangle
        cntrs = cv2.findContours(deskewed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
        cntr = cntrs[0]
        x,y,w,h = cv2.boundingRect(cntr)
        #print(cntrs)
        
        # crop to white region
        crop = deskewed[y:y+h, x:x+w]
        crop_gt = groundTruth_m[y:y+h, x:x+w]
        crop_org = image[y:y+h, x:x+w]
        
        
        temp = np.append(temp, y)
        temp = np.append(temp, y+h)
        temp = np.append(temp, x)
        temp = np.append(temp, x+w)
        
        temp = np.append(temp, c_area)
        temp = np.append(temp, c_solidity)
        
    
        for i in range(crop.shape[0]):
            for j in range(crop.shape[1]):
                if crop[i][j] == 0:
                    #print("0")
                    crop_org[i][j][0] = 0
                    crop_org[i][j][1] = 0
                    crop_org[i][j][2] = 0
                    
        #print(crop.shape)
        #print(crop_org.shape)
     
        crop_org[:,:,0] = 0
        crop_org[:,:,2] = 0
        crop_org = cv2.cvtColor(crop_org, cv2.COLOR_BGR2GRAY)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(crop_org,mask = crop)
        mean_val = cv2.mean(crop_org,mask = crop)
        #print(mean_val[0])
        
        temp = np.append(temp, max_val)
        #print("line 115 {0}".format(temp.shape))
        temp = np.append(temp, min_val)
        #print("line 117 {0}".format(temp.shape))
        temp = np.append(temp, mean_val[0])
        #print("line 119 {0}".format(temp.shape))
        
        glcm = greycomatrix(crop_org, distances=distances, angles=angles,symmetric=True,normed=True)
                
        contrast = greycoprops(glcm, properties[0])
        energy = greycoprops(glcm, properties[1])
        homogeneity = greycoprops(glcm, properties[2])
        correlation = greycoprops(glcm, properties[3])
        
        
        for i in range(0, len(distances), 1):
            for j in range(0, len(angles), 1):
                #print(contrast[i][j])
                temp=np.append(temp,contrast[i][j])
                
        for i in range(0, len(distances), 1):
            for j in range(0, len(angles), 1):
                #print(contrast[i][j])
                temp=np.append(temp,energy[i][j])
                
        for i in range(0, len(distances), 1):
            for j in range(0, len(angles), 1):
                #print(contrast[i][j])
                temp=np.append(temp,homogeneity[i][j])
                
        for i in range(0, len(distances), 1):
            for j in range(0, len(angles), 1):
                #print(contrast[i][j])
                temp=np.append(temp,correlation[i][j])
        
        #print("line 146 {0}".format(temp.shape))
        
        """
        #Generate Gabor features
        
        kernels = []  #Create empty list to hold all kernels that we will generate in a loop
        phi = 0
        lamda = np.pi/2.0
        gamma = 1
        sigma = 1
        for theta in range(1,5,1):  
            theta = theta / 4. * np.pi
            resizeK1 = crop_org.copy
            resizeK1 = cv2.resize(crop_org, (25,25))
            kernel1 = cv2.getGaborKernel((25,25), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)  
            fimg_K1 = cv2.filter2D(resizeK1, cv2.CV_8UC3, kernel1)   
            filtered_img_K1 = fimg_K1.reshape(-1)
            #print(filtered_img_K1)
            
            for i in range(0, filtered_img_K1.shape[0], 1):
                    temp=np.append(temp,filtered_img_K1[i])
            #250,400
            #250,250
            #400,400
            #5,5    
            
            # Calculating GLCM energy and correlation in gabor filer
            glcmK1 = greycomatrix(fimg_K1, distances=distances, angles=angles,symmetric=True,normed=True)
            
            energyK1 = greycoprops(glcmK1, properties[1])
            for i in range(0, len(distances), 1):
                for j in range(0, len(angles), 1):
                    temp=np.append(temp,energyK1[i][j])
            corK1 = greycoprops(glcmK1, properties[3])
            for i in range(0, len(distances), 1):
                for j in range(0, len(angles), 1):
                    temp=np.append(temp,corK1[i][j])
        
        """           
        #print("bounding box = {0},{1},{2},{3}".format(y,y+h,x,x+w))
        """resize_crop = cv2.resize(crop_org, (5,5))
        feature_coord, _ = haar_like_feature_coord(crop_org.shape[0], crop_org.shape[1], 'type-3-x')
        haar_features = draw_haar_like_feature(crop_org, 0, 0, crop_org.shape[0]-1, crop_org.shape[1]-1, feature_coord, max_n_features=1)
        print("-----  {0}".format(haar_features.shape))    
        """
               
        if(c2.compareGT(crop, crop_gt) == 1):
            temp=np.append(temp,target_val)
            matching = matching + 1    
            
        else:
            temp=np.append(temp,'0')
        
        #print("line 154 {0}".format(temp.shape))
        
        if(cn == 1):
            train_set = temp
        else:
            train_set=np.vstack((train_set,temp))
        
        #print(train_set.shape)
        cn = cn + 1
    
    print("{0} MATCHES FOUND out of {1} regions\n".format(matching, cn-1))
    
    """
    plt.imshow(rotrect_img, 'gray')
    plt.axis("off")
    plt.show()
    
    plt.imshow(rotrect_gt_img, 'gray')
    plt.axis("off")
    plt.show()
    """
    
    return train_set
    


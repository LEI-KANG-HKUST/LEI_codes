# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:44:51 2020
image fusion with laplacian pyramilds
@author: KL
"""

import cv2
import numpy as np

path1 = r'C:\Users\TW\Desktop\test11.jpg'
path2 = r'C:\Users\TW\Desktop\test22.jpg'
G0 = cv2.imread(path1)
G1 = cv2.imread(path2)

GARA = G0.copy()
GARB = G1.copy()

gpA = [G0]
gpB = [G1]

# Get five gaussian dowmsamling layers
for i in range(5):   # i will be 0 1 2 3 4, run 5 times
     GARA = cv2.pyrDown(GARA)
     GARB = cv2.pyrDown(GARB)
     gpA.append(GARA)
     gpB.append(GARB)
    
# gpA has six layers in total, the first layer is the image itself, others are five downsampling layer    
    
    
lapA = [gpA[5]]
lapB = [gpB[5]]
for  i in range(5,0,-1):  # i will be 5 4 3 2 1 
    #lapA__ = cv2.pyrUp(gpA[i])            #results of cv2.substract is strange
    #lapA_ = cv2.subtract(gpA[i-1],lapA__)                            # image array sub or add operator!!!!! difference with cv2.subtract()
    lapA_ = gpA[i-1]-cv2.pyrUp(gpA[i])
    lapA.append(lapA_)
    lapB_ = gpB[i-1]-cv2.pyrUp(gpB[i])                           # image array sub or add operator!!!!! difference with cv2.add()
    lapB.append(lapB_)


# combine the corresponding six layers in the middle together.  one gaussian layer and five laplacian layers
LF = []
for l1,l2 in zip(lapA,lapB):
    nrow,cols,ND = l1.shape
    l_ = np.hstack((l1[:,0:cols//2,:],l2[:,cols//2:,:]))   # col/2 will report errors,must use floor divide operator // 
    LF.append(l_)

#recover the image, LF[0] is the ganssian layer, LF[1:5] is laplacian layer

ls_ = LF[0]
for i in range(5):  # i will be 0 1 2 3 
    ls_ = cv2.pyrUp(ls_)
    ls_ = LF[i+1]+ls_


real = np.hstack((G0[:,:cols//2],G1[:,cols//2:]))
cv2.imwrite('real.jpg',real)
cv2.imwrite('fake.jpg',ls_)

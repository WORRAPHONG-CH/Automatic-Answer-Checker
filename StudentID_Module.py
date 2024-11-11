# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 20:56:43 2023

@author: Worra
"""

import numpy as np
import cv2 as cv


def ID_student(img,img_contour):  
    #P = 0 
    #img_contour = img.copy()
    # imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    
    # thresh2 = cv.adaptiveThreshold(imgray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,41,5)
    #ret,thresh2 = cv.threshold(imgray,150,255,cv.THRESH_BINARY_INV)
    
    thresh2 = img.copy()  #img -> already threshold
    contours, hierarchy = cv.findContours(thresh2, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cente = []
    Box = []
    N = []
    area_list = []
    student_ID = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    digit1,digit2,digit3,digit4,digit5,digit6,digit7,digit8 = [],[],[],[],[],[],[],[]
    digit9,digit10,digit11,digit12,digit13,digit14,digit15,digit16,digit17 = [],[],[],[],[],[],[],[],[]
    obj_contourID = []
    
    for i in contours:
        area = cv.contourArea(i)
        if area > 1000 and area < 2000:  #1000,1600  
            #print(area)
            rect = cv.minAreaRect(i)
            box = cv.boxPoints(rect)
            box = np.array(np.intp(box))
            center,size,angle = cv.minAreaRect(i)
            #center = np.array(center)
            area_list.append(area)
            #print(center[1])
            if center[1] > 150:
                Box.append(box)
                cente.append(center)
                obj_contourID.append(i)
                #cv.drawContours(img_contour,[i],-1,(0,255,0),2)
                
    #print("lenID:",len(obj_contourID))
    #cv.circle(img_contour,(100,140), 5, (255,0,255), -1)
    # cv.imshow("Img Contour",img_contour)
    #
    #print("area:",len(area_list))
    #print("Cente:",len(cente))
    if len(cente) >= 170:
        for s,S in enumerate(cente):
            x,y = cente[s][0],cente[s][1]
            Newarrange = [x,y,Box[s]]
            #Newarrange = np.array(Newarrange)
            # cv.circle(img, (int(x),int(y)), 5, (255,0,0), -1)
            # cv.putText(img,"{}".format(s),(int(x),int(y)),cv.FONT_HERSHEY_PLAIN,1.5,(0,0,255),2)
            N.append(Newarrange)  
        #N = np.array(N)    
        N = sorted(N, key=lambda x: x[0]) #sort x      
    
    
        for k,K in enumerate(N): #create 17 digits    
            if k in np.arange(0,10):
                digit1.append(K)
    
            elif k in np.arange(10,20):
                digit2.append(K)          
    
            elif k in np.arange(20,30):
                digit3.append(K)
                
            elif k in np.arange(30,40):
                digit4.append(K)
               
            elif k in np.arange(40,50):
                digit5.append(K)
                
            elif k in np.arange(50,60):
                digit6.append(K)
                
            elif k in np.arange(60,70):
                digit7.append(K)
               
            elif k in np.arange(70,80):
                digit8.append(K)
                
            elif k in np.arange(80,90):
                digit9.append(K)
                
            elif k in np.arange(90,100):
                digit10.append(K)
               
            elif k in np.arange(100,110):
                digit11.append(K)
                
            elif k in np.arange(110,120):
                digit12.append(K)
                
            elif k in np.arange(120,130):
                digit13.append(K)
               
            elif k in np.arange(130,140):
                digit14.append(K)
                
            elif k in np.arange(140,150):
                digit15.append(K)
                
            elif k in np.arange(150,160):
                digit16.append(K)
                       
            elif k in np.arange(160,170):
                digit17.append(K)
                
        digit1 = sorted(digit1, key=lambda x: x[1]) #sort y
        digit2 = sorted(digit2, key=lambda x: x[1]) #sort y
        digit3 = sorted(digit3, key=lambda x: x[1]) #sort y
        digit4 = sorted(digit4, key=lambda x: x[1]) #sort y
        digit5 = sorted(digit5, key=lambda x: x[1]) #sort y
        digit6 = sorted(digit6, key=lambda x: x[1]) #sort y
        digit7 = sorted(digit7, key=lambda x: x[1]) #sort y
        digit8 = sorted(digit8, key=lambda x: x[1]) #sort y
        digit9 = sorted(digit9, key=lambda x: x[1]) #sort y
        digit10 = sorted(digit10, key=lambda x: x[1]) #sort y
        digit11 = sorted(digit11, key=lambda x: x[1]) #sort y
        digit12 = sorted(digit12, key=lambda x: x[1]) #sort y
        digit13 = sorted(digit13, key=lambda x: x[1]) #sort y
    
        digit_ID = [digit1,digit2,digit3,digit4,digit5,digit6,digit7,digit8,digit9,digit10,digit11,digit12,digit13]
        global non_pixelID,pixel_maxList,pixel_statList
        non_pixelID = []
        non_pixelTemp = []
        for ind,digit in enumerate(digit_ID,1):
            for p in range(len(digit)):
                #print(len(digit1))
                pts1 = np.float32(digit[p][2])
                pts2 = np.float32([[0,0],[300,0],[300,300],[0,300]])
                M = cv.getPerspectiveTransform(pts1,pts2)
                dst = cv.warpPerspective(thresh2,M,(300,300))
                non_zeropixel = cv.countNonZero(dst)
                non_pixelTemp.append(non_zeropixel)
            # Out Loop
            non_pixelTemp2 = non_pixelTemp.copy()
            non_pixelID.append(non_pixelTemp2)
            non_pixelTemp.clear()
        
        # find max non-pixel in each quest
        pixel_maxList = []
        for nq1 in non_pixelID:
            pixel_max = max(nq1)
            pixel_maxList.append(pixel_max)
            
        # Calculate threshold pass pixel by avg, +-error
        percent_PN = 0.1
        pixel_avg = int(sum(pixel_maxList)/len(pixel_maxList))
        pixel_avgP = int(pixel_avg*(1.1+percent_PN))
        pixel_avgN = int(pixel_avg*(1-percent_PN))  
        
        pixel_statList = [pixel_avg,pixel_avgP,pixel_avgN]
        cutoff_p,cutoff_n = pixel_avgP,pixel_avgN
    
        for ind2,non_zeroDigit in enumerate(non_pixelID):
            for np2 in range(len(non_zeroDigit)):
                if non_zeroDigit[np2] >= cutoff_n and non_zeroDigit[np2] <= cutoff_p:
                    cv.drawContours(img_contour,[digit_ID[ind2][np2][2]], 0, (255,0, 0), 2) 
                    student_ID[ind2] = np2
                    
                    
                elif non_zeroDigit[np2] < cutoff_n and non_zeroDigit[np2] > cutoff_p:
                    print("have something wrong with student ID")
                    student_ID[ind2] = '_'
                        #break
        
       
                
    id_num = ''.join(map(str,student_ID))
    id_num = int(id_num)   
    cv.putText(img_contour,str(id_num),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv.LINE_AA) 
    #print("ID:",id_num)
    #print(non_pixelID)
    # print("STAT:",pixel_statList)
    # print("Max:",pixel_maxList)
    
    return img_contour,student_ID,id_num

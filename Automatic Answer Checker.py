from tkinter import *
import tkinter as tk
import cv2 
from tkinter import filedialog
from PIL import Image, ImageTk
import serial
import time
import pandas as pd 
import numpy as np 
import scipy as stats
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os 


start_time = time.time()

#video = None
#arduino = serial.Serial(port = 'COM3', timeout=0)
w,h = 1920,1080
roi_x1,roi_x2 = 270,1900
roi_y1,roi_y2 = 0,1080
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#cap.set(cv2.CAP_PROP_SETTINGS,1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)#2560
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,h)#1440
cap.set(cv2.CAP_PROP_AUTOFOCUS,0)
cap.set(cv2.CAP_PROP_FOCUS, 35)
cap.set(cv2.CAP_PROP_FPS, 60.0) 
#cap.set(cv2.CAP_PROP_SHARPNESS, 130)
#cap.set(cv2.CAP_PROP_CONTRAST, 200)
cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
    #fps = int(cap.get(5))
# ================================= FUNCTION PAPER ================================ #

def contour_area(img_contour,contours,a1,a2):
    area_list = []
    obj_contour = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > a1 and area < a2:
            #print(area)
            area_list.append(area)
            obj_contour.append(cnt)
            cv2.drawContours(img_contour,cnt,-1,(0,255,0),2)
            cv2.circle(img_contour,(100,160), 5, (255,0,255), -1)
            # cv2.circle(img_contour,(270,150), 5, (255,0,255), -1)
            # cv2.circle(img_contour,(1770,140), 5, (255,0,255), -1)
            # cv2.circle(img_contour,(1000,1050), 5, (255,0,255), -1)
            
    #area_list = sorted(area_list, key = lambda a:a)
    #print("No. in list contour:",len(obj_contour))
    return area_list,obj_contour

def warp_answerBig(warp_img,corner_list,w,h,upside_down):
    #if len(corner_list) == 4:
        pts1 = np.float32([corner_list[0],corner_list[1],corner_list[2],corner_list[3]])
        if upside_down == 0 :
            #print("#### NORMAL ####")
            #pts2 = np.float32([[0,900],[0,0],[800,0],[800,900]])
            pts2 = np.float32([[0,h],[0,0],[w,h],[w,0]])
        elif upside_down == 1: 
            #print("#### UPSIDE DOWN ####")
            pts2 = np.float32([[w,0],[w,h],[0,0],[0,h]])
            
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        warp_res = cv2.warpPerspective(warp_img,matrix,(w,h))
    
        return warp_res


def sort_area(obj_contour1):
    area_objList = []
    for obj in obj_contour1:
        area = cv2.contourArea(obj)
        area_obj = [area,obj]
        area_objTemp = area_obj.copy() # no need clear new values available
        area_objList.append(area_objTemp)
    
    # Sort by area 
    areaObj_sort = sorted(area_objList,key=lambda a:a[0])
    
    area_list = []
    obj_list = []
    for area_objL in areaObj_sort:
        area_list.append(area_objL[0])
        obj_list.append(area_objL[1])
    
    return area_list,obj_list
    
def check_upsidedown(center):
    if center[0][0] > center[2][0]:
        upside_down = 0 
    elif center[0][0] < center[2][0]:
        upside_down = 1 

    return upside_down
    
def moment_center(obj_contour):
    obj_centerList = []
    obj_cutList = []
   
    for i,obj in enumerate(obj_contour):
        x,y,w,h = cv2.boundingRect(obj)
        # find center 
        M = cv2.moments(obj)
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        #if cx > x_cut1 and cx < x_cut2:
        obj_centerList.append(np.array((cx,cy),dtype = "int64"))
        obj_cutList.append(obj)
        # else:
        #     pass
    
    # print("before:",len(obj_contour))
    # print("after:",len(obj_centerList))
    return obj_centerList,obj_cutList

def edge_contour(obj_contour):
    edge_temp1 = []
    edge_list = []
    length_list = []
    
    for index in range(len(obj_contour)):
        length_con = cv2.arcLength(obj_contour[index],True)
        edge = cv2.approxPolyDP(obj_contour[index],0.02*length_con,True)
        for ind1 in range(len(edge)):  
            edge_re = np.ravel(edge[ind1])
            edge_temp1.append(edge_re)
            
        # OUT LOOP
        edge_temp2 = edge_temp1.copy()
        edge_list.append(edge_temp2)
        edge_temp1.clear()
        
        #edge_list1.append(edge)
        length_list.append(length_con)
   
    #print("Test List:",edge_list)
    return edge_list,length_list


def sort_edge(edge_list):
    edge_sortXY = []
    edge_Xtemp1 = []
    
    for ind1,edge1 in enumerate(edge_list): # depend on len contour
        #print("Y old:",ind1,":",edge1)
        edge_sortY = sorted(edge1, key=lambda y:y[1]) # only 4 in each obj
        #print("edge_sortY",ind1,":",edge_sortY)
        #print("____________________________")
        #edge_Xtemp = edge_sortY.copy()
        if len(edge_sortY) == 4: 
            if edge_sortY[0][0] > edge_sortY[1][0]: #if x 1st ele more than x 2nd ele 
                #print("X Ele 0 > Ele 1")
                edge_Xtemp1.append(edge_sortY[1])
                edge_Xtemp1.append(edge_sortY[0])
                
                if edge_sortY[2][0] > edge_sortY[3][0]: #if x 3rd ele more than x 4th ele 
                    #print("X Ele 2 > Ele 3")
                    edge_Xtemp1.append(edge_sortY[3])
                    edge_Xtemp1.append(edge_sortY[2])
                
                elif edge_sortY[2][0] < edge_sortY[3][0]: #if x 3rd ele less than x 4th ele
                    #print("X Ele 2 < Ele 3")
                    edge_Xtemp1.append(edge_sortY[2])
                    edge_Xtemp1.append(edge_sortY[3])
                    
            elif edge_sortY[0][0] < edge_sortY[1][0]: #if x 1st ele less than x 2nd ele 
                #print("X Ele 0 < Ele 1")
                edge_Xtemp1.append(edge_sortY[0])
                edge_Xtemp1.append(edge_sortY[1])
        
                if edge_sortY[2][0] > edge_sortY[3][0]: #if x 3rd ele more than x 4th ele 
                    #print("X Ele 2 > Ele 3")
                    edge_Xtemp1.append(edge_sortY[3])
                    edge_Xtemp1.append(edge_sortY[2])
                
                elif edge_sortY[2][0] < edge_sortY[3][0]: #if x 3rd ele less than x 4th ele
                    #print("X Ele 2 < Ele 3")
                    edge_Xtemp1.append(edge_sortY[2])
                    edge_Xtemp1.append(edge_sortY[3])

        # print("X",ind1,edge_Xtemp1)
        # print("----------------")
        edge_Xtemp2 = edge_Xtemp1.copy()
        edge_sortXY.append(edge_Xtemp2)
        edge_Xtemp1.clear()

    # print("****************")
    # print("XY",ind1,edge_sortXY)
    # print("****************")
    
    return edge_sortXY

def split_ch(img_cut,img_contour,objRec_contour,form_paper):
    w_list,h_list = [],[]
    x_list,y_list = [],[]
    cx_list,cy_list = [],[]
    
    for i,obj in enumerate(objRec_contour):
        x,y,w,h = cv2.boundingRect(obj)
        # find center 
        M = cv2.moments(obj)
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        
        #cv2.circle(img_contour, (cx, cy), 2, (255, 255, 255), -1)
        x_list.append(x),y_list.append(y),w_list.append(w),h_list.append(h)
        cx_list.append(cx), cy_list.append(cy)
        #cv2.rectangle(img_contour,(x,y),(x+w,y+h),(255,0,0),2)
        #cv2.putText(img_contour,"{}".format(i),(int(x),int(y)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
    
    ''' Store only mark rectangle by using avg_width to separate the other that not rectangle '''
    avg_w = int(sum(w_list) / len(w_list))
    i1 = 0
    '''Store contour which has w > avg [rectangle] *** Sorted *** '''
    TopLeft_list = []
    heightWidth_List = []
    center_sortList = []
    
    for ind1 in range(len(w_list)):
        w1,h1 = w_list[ind1],h_list[ind1]
        x1,y1 = x_list[ind1],y_list[ind1]
        cx1,cy1 = cx_list[ind1],cy_list[ind1]
        if w1> avg_w: #26 avg
            TopLeft_list.append(np.array((x1,y1),dtype = "int32"))
            heightWidth_List.append(np.array((w1,h1),dtype = "int32"))
            center_sortList.append(np.array((cx1,cy1),dtype = "int32"))
            cv2.rectangle(img_contour,(x1,y1),(x1+w1,y1+h1),(255,0,255),5)
            # cv2.putText(img_contour,"{}".format(i1),(int(cx1),int(cy1)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
            # cv2.circle(img_contour, (cx1, cy1), 2, (255, 255, 255), -1)
            i1 +=1
        else:
            pass
        
    # sort index of each center 
    # Top - Bottom
    center_sortY = sorted(center_sortList,key=lambda cy2:cy2[1]) # Top - Bottom
    center_sortAll = []
    center_tempRow1 = []
    j1=1
    for ind2 in range(len(center_sortY)):
        cx2,cy2 = center_sortY[ind2][0],center_sortY[ind2][1]
        center_tempRow1.append(np.array((cx2,cy2),dtype = "int32"))
        
        if j1 % 5 == 0 and form_paper == 1: #A4
            center_tempRow2 = center_tempRow1.copy()
            center_sortX = sorted(center_tempRow2,key=lambda cx2:cx2[0]) # Left-Right
            center_sortAll.append(center_sortX)
            center_tempRow1.clear()
            
        elif j1 % 7 == 0 and form_paper == 2 : #A5
            center_tempRow2 = center_tempRow1.copy()
            center_sortX = sorted(center_tempRow2,key=lambda cx2:cx2[0]) # Left-Right
            center_sortAll.append(center_sortX)
            center_tempRow1.clear()
        # out loop row
        j1 +=1
        
    #for cenAll in enumerate(center_sortAll):
    # ===== ROW0 COL0 ========#
    compensation = center_sortAll[1][0][1] - center_sortAll[0][0][1] # Compensation in y(row1col0)- y(row0col0)
    #print(compensation)
    for col0 in range(len(center_sortAll[0])-1):
        start_point0 = center_sortAll[0][col0][0],(center_sortAll[0][col0][1] - compensation)
        if col0 < len(center_sortAll[0]):
            end_point0 = center_sortAll[0][col0+1][0],(center_sortAll[0][col0+1][1] - compensation)
            #print(start_point0,end_point0)
        else:
            end_point0 = start_point0
            
        cv2.line(img_cut,start_point0,end_point0,(0,0,0), 2)
        
    indAll = 0
    for cenAll in center_sortAll:
        for ind3 in range(len(cenAll)):# 7
            if ind3 < len(cenAll)-1:
                start_point = cenAll[ind3][0],cenAll[ind3][1]+5
                end_point= cenAll[ind3+1][0],cenAll[ind3+1][1]+5
                #print(ind3)
                
            else:
                start_point = cenAll[ind3][0],cenAll[ind3][1] +5
                end_point= start_point
                
            cx3,cy3 = cenAll[ind3][0],cenAll[ind3][1]
            #cv2.line(img_cut,start_point,end_point,(255,255,255), 2)
            cv2.line(img_cut,start_point,end_point,(0,0,0), 2)
            #cv2.line(img_contour,start_point,end_point,(255,255,255), 2)
            cv2.putText(img_contour,"{}".format(indAll),(int(cx3),int(cy3)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
            cv2.circle(img_contour, (cx3,cy3), 2, (255, 255, 255), -1)
            indAll +=1
    
    
    return img_cut,img_contour,center_sortAll

def split_ID(thresh2,img_contour):
    '''thresh2 -> warpID[]'''
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    center_mark = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > 200 and area < 350: # 100,300   default 200,350
            M = cv2.moments(i)
            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"]) 
            #cv.circle(img, (100, 145), 2, (0, 0, 255),-1)
            if cY > 160:
                # put text and highlight the center
                cv2.circle(thresh2, (cX, cY), 2, (255, 255, 255),-1)
                #cv.drawContours(thresh2,[i],-1,(0,255,255),1)
                cv2.drawContours(img_contour,[i],-1,(0,255,255),2)
                center_mark.append((cX,cY)) 
                
        else:
            pass
        
        
    #print(len(center_mark))
    center_mark = sorted(center_mark, key=lambda x: x[0]) #sort x
    if len(center_mark) == 63:
        global collum1,collum2,collum3,collum4,collum5,collum6,collum7
        collum1,collum2,collum3,collum4,collum5,collum6,collum7 = [],[],[],[],[],[],[]
        for k,K in enumerate(center_mark): #create 5 collumn
            if k in np.arange(0,9): #0-8
                collum1.append(K)
            elif k in np.arange(9,18):#9-17
                collum2.append(K)
            elif k in np.arange(18,27):
                collum3.append(K)
            elif k in np.arange(27,36):
                collum4.append(K)
            elif k in np.arange(36,45):
                collum5.append(K)
            elif k in np.arange(45,54):
                collum6.append(K)
            elif k in np.arange(54,63):
                collum7.append(K)
                
        collum1 = sorted(collum1, key=lambda x: x[1]) #sort y
        collum2 = sorted(collum2, key=lambda x: x[1]) #sort y
        collum3 = sorted(collum3, key=lambda x: x[1]) #sort y
        collum4 = sorted(collum4, key=lambda x: x[1]) #sort y
        collum5 = sorted(collum5, key=lambda x: x[1]) #sort y
        collum6 = sorted(collum6, key=lambda x: x[1]) #sort y
        collum7 = sorted(collum7, key=lambda x: x[1]) #sort y
    
        compensation = collum1[1][1]-collum1[0][1]
        row1col1 = (collum1[0][0],collum1[0][1]-compensation)
        row1col2 = (collum2[0][0],collum2[0][1]-compensation)
        row1col3 = (collum3[0][0],collum3[0][1]-compensation)
        row1col4 = (collum4[0][0],collum4[0][1]-compensation)
        row1col5 = (collum5[0][0],collum5[0][1]-compensation)
        row1col6 = (collum6[0][0],collum6[0][1]-compensation)
        row1col7 = (collum7[0][0],collum7[0][1]-compensation)
        
        row1allcol =  [row1col1,row1col2,row1col3,row1col4,row1col5,row1col6,row1col7]
        
        cv2.circle(img_contour, row1col1, 2, (255, 255, 255),-1)
        cv2.circle(img_contour, row1col2, 2, (255, 255, 255),-1)
        cv2.circle(img_contour, row1col3, 2, (255, 255, 255),-1)
        cv2.circle(img_contour, row1col4, 2, (255, 255, 255),-1)
        cv2.circle(img_contour, row1col5, 2, (255, 255, 255),-1)
        cv2.circle(img_contour, row1col6, 2, (255, 255, 255),-1)
        cv2.circle(img_contour, row1col7, 2, (255, 255, 255),-1)
    
        for a in range(7):# draw line point to point row1 only
            cv2.line(thresh2,row1allcol[0],row1allcol[1],(0, 0, 0),thickness=10)
            cv2.line(thresh2,row1allcol[1],row1allcol[2],(0, 0, 0),thickness=10)
            cv2.line(thresh2,row1allcol[2],row1allcol[3],(0, 0, 0),thickness=10)
            cv2.line(thresh2,row1allcol[3],row1allcol[4],(0, 0, 0),thickness=10)
            cv2.line(thresh2,row1allcol[4],row1allcol[5],(0, 0, 0),thickness=10)
            cv2.line(thresh2,row1allcol[5],row1allcol[6],(0, 0, 0),thickness=10)
    
        for a in range(9):# draw line point to point
            cv2.line(thresh2,collum1[a],collum2[a],(0, 0, 0),thickness=2)
            cv2.line(thresh2,collum2[a],collum3[a],(0, 0, 0),thickness=2)
            cv2.line(thresh2,collum3[a],collum4[a],(0, 0, 0),thickness=2)
            cv2.line(thresh2,collum4[a],collum5[a],(0, 0, 0),thickness=2)
            cv2.line(thresh2,collum5[a],collum6[a],(0, 0, 0),thickness=2)
            cv2.line(thresh2,collum6[a],collum7[a],(0, 0, 0),thickness=2)
            
            # cv2.line(img_contour,collum1[a],collum2[a],(0, 0, 0),thickness=2)
            # cv2.line(img_contour,collum2[a],collum3[a],(0, 0, 0),thickness=2)
            # cv2.line(img_contour,collum3[a],collum4[a],(0, 0, 0),thickness=2)
            # cv2.line(img_contour,collum4[a],collum5[a],(0, 0, 0),thickness=2)
            # cv2.line(img_contour,collum5[a],collum6[a],(0, 0, 0),thickness=2)
            # cv2.line(img_contour,collum6[a],collum7[a],(0, 0, 0),thickness=2)
        
    return thresh2,center_mark

def process_image(frame,sort_edgeList,form_paper,upside_down):
    if form_paper == 2: #A5
        # Warp Pages For Save
        warp_page =  warp_answerBig(frame,sort_edgeList[4],1400,1000,upside_down)
        
        # Warp ID Field
        warp_IDField = warp_answerBig(frame,sort_edgeList[0],1000,800,upside_down)#800,600
        warp_IDField_contour = warp_IDField.copy()
        warp_IDGray = cv2.cvtColor(warp_IDField,cv2.COLOR_BGR2GRAY)
        warp_IDTh = cv2.adaptiveThreshold(warp_IDGray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,47,5) # 37 stick together
        
        # Warp Answer Field
        warp_ansField = warp_answerBig(frame,sort_edgeList[2],1400,1000,upside_down)#800,600
        warp_ansField_contour = warp_ansField.copy()
        warp_ansGray = cv2.cvtColor(warp_ansField,cv2.COLOR_BGR2GRAY)
        warp_ansTh = cv2.adaptiveThreshold(warp_ansGray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,41,5)
        
        
    elif form_paper == 1: #A4
        # Warp Pages For Save
        warp_page =  warp_answerBig(frame,sort_edgeList[4],1400,2000,upside_down) #1400,1800
        
        # Warp ID Field
        warp_IDField = warp_answerBig(frame,sort_edgeList[0],1000,800,upside_down)#800,600
        warp_IDField_contour = warp_IDField.copy()
        warp_IDGray = cv2.cvtColor(warp_IDField,cv2.COLOR_BGR2GRAY)
        warp_IDTh = cv2.adaptiveThreshold(warp_IDGray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,37,5)
        #ret,warp_IDTh = cv2.threshold(warp_IDGray,100,255,cv2.THRESH_BINARY_INV)
        
        # Warp Answer Field
        warp_ansField = warp_answerBig(frame,sort_edgeList[2],1400,2000,upside_down)#800,600
        warp_ansField_contour = warp_ansField.copy()
        warp_ansGray = cv2.cvtColor(warp_ansField,cv2.COLOR_BGR2GRAY)
        warp_ansTh = cv2.adaptiveThreshold(warp_ansGray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,47,5)
        #ret,warp_ansTh = cv2.threshold(warp_ansGray,100,255,cv2.THRESH_BINARY_INV)
        
    warp_ID = [warp_IDField,warp_IDGray,warp_IDTh,warp_IDField_contour]
    warp_ans = [warp_ansField,warp_ansGray,warp_ansTh,warp_ansField_contour]
    
    return warp_page,warp_ID,warp_ans

def plot_data(form_paper,scoreStd_list):
    global df,path_full,code_year,code_semester,code_subject,code_section
    # =================== CALCULATE STATISTIC ===================== #
    if len(scoreStd_list) > 0:
        score_max = max(scoreStd_list)
        score_min = min(scoreStd_list)
        score_mean = np.round(np.mean(scoreStd_list),decimals=2)
        score_median = np.median(scoreStd_list)
        score_mode = statistics.mode(scoreStd_list)
        score_sd = np.round(np.std(scoreStd_list),decimals=3)
        
    else:
        score_max,score_min,score_mean,score_median,score_mode,score_sd = 0,0,0,0,0,0
        
    # =================== SAVE STATISTIC TO DF ===================== #
    #name_col = ["No","Student_ID","Name","Surname,"Correct","False","Correct %","False %","Total","A","B","C","D","E"]
    #len_rows = len(df)
    list_series = [pd.Series(["NO. Student",len(scoreStd_list)],index=df.columns[:2]),
                    pd.Series(["MAX",score_max],index=df.columns[:2]),
                    pd.Series(["MIN",score_min],index=df.columns[:2]),
                    pd.Series(["MEAN",score_mean],index=df.columns[:2]),
                    pd.Series(["MEDIAN",score_median],index=df.columns[:2]),
                    pd.Series(["MODE",score_mode],index=df.columns[:2]),
                    pd.Series(["SD",score_sd],index=df.columns[:2])]
    
    df = df.append(list_series,ignore_index=True)
    
    
    print("\n___________________________________________________")
    print("####### SUMMARY STATISTICS #######")
    print("--------------------------------")
    print("NUMBER OF STUDENT:",len(scoreStd_list))
    print("MAX:",score_max)
    print("MIN:",score_min)
    print("MEAN:",score_mean)
    print("MEDIAN:",score_median)
    print("MODE:",score_mode)
    print("SD:",score_sd)
    print("--------------------------------")
    # =================== PLOT GRAPH ===================== #
    # --------- PLOT SCORE --------- #
    len_plot = np.arange(0, len(scoreStd_list))
    plt.figure(figsize=(12,6))
    plt.subplot(121) # row,col,sequence
    plt.scatter(len_plot,scoreStd_list)
    plt.title("SCORE PLOT"),plt.suptitle("STAT SCORE")
    plt.xticks(len_plot)
    plt.xlabel("STUDENT"),plt.ylabel("SCORE")
    plt.axhline(score_mean,color="red",linestyle="dashed",linewidth=2)
    plt.grid(True)
    plt.legend()
    #plt.show()
    
    # --------- PLOT RANGE OF SCORE--------- #
    if form_paper == 1: # A4
        
        score_150_200,score_100_150,score_50_100,score_0_50 = 0,0,0,0
        for sc in scoreStd_list:
            if sc >= 150:
                score_150_200 += 1
            elif sc >= 100 and sc < 150:
                score_100_150 += 1
            elif sc >= 50 and sc < 100:
                score_50_100 += 1
            elif sc >=0 and sc < 50:
                score_0_50 += 1
                
        score_rangeName = ["0-50","50-100","100-150","150-200"]
        score_range = [score_0_50,score_50_100,score_100_150,score_150_200]
        #plt.figure(figsize=(9,3))
        plt.subplot(122)
        plt.bar(score_rangeName,score_range)
        plt.title("PLOT RANGE SCORE")
        plt.xlabel("RANGE SCORE"),plt.ylabel("NO.STUDENT")
        plt.show()
        plt.savefig(os.path.join(path_full,"{}_{}_{}_sec{}.png".format(code_year,code_semester,code_subject,code_section)))
 
    elif form_paper == 2: # A5
        score_100_120,score_50_100,score_0_50 = 0,0,0
        for sc in scoreStd_list:
            if sc >= 100:
                score_100_120 += 1
            elif sc >= 50 and sc < 100:
                score_50_100 += 1
            elif sc >=0 and sc < 50:
                score_0_50 += 1
                
        score_rangeName = ["0-50","50-100","100-120"]
        score_range = [score_0_50,score_50_100,score_100_120]
        #plt.figure(figsize=(9,3))
        plt.subplot(122)
        plt.bar(score_rangeName,score_range)
        plt.title("PLOT RANGE SCORE")
        plt.xlabel("RANGE SCORE"),plt.ylabel("NO.STUDENT")
        plt.show()
        plt.savefig(os.path.join(path_full,"{}_{}_{}_sec{}.png".format(code_year,code_semester,code_subject,code_section)))
    
    # plt.subplot(122)
    # plt.grid()
    # plt.hist(scoreStd_list,bins=10,edgecolor='black')
    # plt.title("RANGE SCORE")
    # plt.xlabel("RANGE SCORE"),plt.ylabel("STUDENT")
    # plt.show()
    # plt.savefig(os.path.join(path_full,"{}_{}_{}_sec{}.png".format(code_year,code_semester,code_subject,code_section)))
# =============================== GUI FUNCTION ================================= #

def select_file():
    global csv_pathFile,directory_path
    # Create a dialog box
    csv_pathFile = filedialog.askopenfilename()
    directory_path = os.path.dirname(csv_pathFile)
    #directory_path = filedialog.askdirectory()
    text_select = "✓|Select File"+ "\n"
    box_display.insert(tk.END,text_select)
    box_display.see(tk.END)  # Scroll to the bottom
    #f=open(window1.csv_file,'r')

def manual():
    global mode_AM
    mode_AM = 0
    box_display.insert(tk.END,"✓|MANUAL MODE\n")
    box_display.see(tk.END)  # Scroll to the bottom

def auto():
    global mode_AM,E_COM,B_COM,L_COM
    mode_AM = 1
    box_display.insert(tk.END,"✓|AUTO MODE\n")
    box_display.see(tk.END)  # Scroll to the bottom
    L_COM = tk.Label(text="COM")
    L_COM.place(x=220, y=90) #label
    E_COM = tk.Entry(window1,width=4,font=10) #entry first box
    E_COM.place(x=260,y=90)
    B_COM = tk.Button(window1, text="OK",bg='#007fff', font=2,command = ok_com)
    B_COM.place(x=305,y=85)

def ok_com():
    global  E_COM,code_COM,arduino
    code_COM = int(E_COM.get())
    text_COM = "✓|SELECT COM"+ str(code_COM) +"\n"
    box_display.insert(tk.END,text_COM)
    box_display.see(tk.END)  # Scroll to the bottom
    arduino = serial.Serial(port = 'COM{}'.format(code_COM), timeout=0)
    
def next_paper():
    global mode_AM
    if mode_AM == 1: # AUTO MODE
        global arduino
        var = '2'
        arduino.write(str.encode(var))
    else: # MANUAL MODE
        pass

def A4():
    global form_paper
    st=bt5["bg"]
    if(st=="#007fff"):
        bt5["bg"]="blue"
        bt6["bg"]="#007fff"
        form_paper = 1
        text_form = "✓|PAPER FORM: A4"+ "\n"
        box_display.insert(tk.END,text_form)
        box_display.see(tk.END)  # Scroll to the bottom
        
    else:
        bt5["bg"]="#007fff"
            
def A5():
    global form_paper
    st=bt6["bg"]
    if(st=="#007fff"):
        bt6["bg"]="blue"
        bt5["bg"]="#007fff"
        form_paper = 2
        text_form = "✓|PAPER FORM: A5" + "\n"
        box_display.insert(tk.END,text_form)
        box_display.see(tk.END)  # Scroll to the bottom
    else:
        bt6["bg"]="#007fff"
    
    
def ABCD():
    global num_choices
    st=bt7["bg"]
    if(st=="#007fff"):
        bt7["bg"]="blue"
        bt8["bg"]="#007fff"
        num_choices = 4
        text_form = "✓|CHOICE FORM: ABCD"+ "\n"
        box_display.insert(tk.END,text_form)
        box_display.see(tk.END)  # Scroll to the bottom
    else:
        bt7["bg"]="#007fff"
    
def ABCDE():
    global num_choices
    st=bt8["bg"]
    if(st=="#007fff"):
        bt8["bg"]="blue"
        bt7["bg"]="#007fff"
        num_choices = 5
        text_form = "✓|CHOICE FORM: ABCDE"+ "\n"
        box_display.insert(tk.END,text_form)
        box_display.see(tk.END)  # Scroll to the bottom
    else:
        bt8["bg"]="#007fff"
        
def Def_checkbox_set1():
    global checkbox_state1,path_set
    if checkbox_state1.get():
        path_set = "SET1"
        print("✓|PAPER SET1 [0-200]")
        box_display.insert(tk.END,"✓|PAPER SET1 [0-200]\n")
        box_display.see(tk.END)  # Scroll to the bottom
    else:
        print("✕|PAPER SET1 [0-200]")
        
def Def_checkbox_set2():
    global checkbox_state2,path_set
    if checkbox_state2.get():
        path_set = "SET2"
        print("✓|PAPER SET2 [201-400]")
        box_display.insert(tk.END,"✓|PAPER SET2 [201-400]\n")
        box_display.see(tk.END)  # Scroll to the bottom
    else:
        print("✕|PAPER SET2 [201-400]")

def ok_semester():
    global code_semester,code_year
    code_year = int(E_year.get())
    text_year = "✓|Academic Year: "+ str(code_year) +"\n"
    code_semester = int(E_semester.get())
    text_semester = "✓|Semester: "+ str(code_semester) +"\n"
    box_display2.insert(tk.END,text_year)
    box_display2.insert(tk.END,text_semester)
    
def ok_section():
    global code_section,code_year,code_semester,code_subject,path_full,path_reject,directory_path,path_set
    code_subject = int(E_subject.get())
    text_subject = "✓|Subject Code: "+ str(code_subject) +"\n"
    code_section = int(E_section.get())
    text_section = "✓|Section: "+ str(code_section) +"\n"
    box_display2.insert(tk.END,text_subject)
    box_display2.insert(tk.END,text_section)
    
    # ============== CREATE FILE PATH FOR SAVE ============== #
    #path_current = os.path.dirname(os.path.abspath(__file__))
     
    path_semester = "{}_{}_{}_{}\sec{}".format(code_semester,code_year,code_subject,path_set,code_section)
    path_full = os.path.join(directory_path,path_semester)
    #print("Full path: ",path_full)
    path_reject = os.path.join(path_full,"REJECT")
    print("Full Path:",path_full)
    print("Reject Path:",path_reject)
    if not os.path.exists(path_full):
        print("======= FOLDER IS NOT EXISTS =======")
        os.makedirs(path_full)
        os.makedirs(path_reject)
        print("======= CREATING FOLDER =======")
    else:
        print("======= FOLDER ALREADY EXISTS =======")
    
def run():
    var = '1'
    arduino.write(str.encode(var))

def stop():
    var = 'S'
    arduino.write(str.encode(var))
    
def ok_numQuest():
    global num_questions, form_paper
    num_questions = int(E_numQuest.get())
    # #print(num_questions)
    # tk.Label(text=num_questions).place(x=350, y=240)
    if form_paper == 1:
        if num_questions <= 200: 
            text_numquest = "✓|NO. QUESTION: "+ str(num_questions)+"\n"
            box_display.insert(tk.END,text_numquest)
            box_display.see(tk.END)  # Scroll to the bottom
        else:
            text_numquest = "✕|MAXIMUM QUESTION: 200 \n"
            text_numquest2 = "!! PLEASE INSERT AGAIAN !!"+"\n"
            box_display.insert(tk.END,text_numquest)
            box_display.insert(tk.END,text_numquest2)
            E_numQuest.delete(0,"end")
            box_display.see(tk.END)  # Scroll to the bottom
            # E_numPart.delete(0,"end")
            
    elif form_paper == 2:
        if num_questions <= 120: 
            text_numquest = "✓|NO. QUESTION: "+ str(num_questions)+"\n"
            box_display.insert(tk.END,text_numquest)
            box_display.see(tk.END)  # Scroll to the bottom
        else:
            text_numquest = "✕|MAXIMUM QUESTION: 120 \n"
            text_numquest2 = "!! PLEASE INSERT AGAIAN !!"+"\n"
            box_display.insert(tk.END,text_numquest)
            box_display.insert(tk.END,text_numquest2)
            box_display.see(tk.END)  # Scroll to the bottom
            E_numQuest.delete(0,"end")
 

def ok_numPart(): # Enter num part 

    global num_part,label_list,QEachPart_list,B3,num_questions
    global entry_list,df,name_part,e1,csv_pathFile
    
    num_part = int(E_numPart.get()) # enter part
    text_numpart = "✓|NO. PART: "+ str(num_part) +"\n"
    box_display.insert(tk.END,text_numpart)
    box_display.see(tk.END)  # Scroll to the bottom
    # =================== Data CSV File  ===================== #
    df = pd.read_csv(str(csv_pathFile),index_col = False)
    #df = pd.read_csv("C:/KMUTNB/Project KMUTNB/AnswerChecker/Student/Data_Student/Excel_Std/Student_Data.csv",index_col = "No")
    #row_count = sum(1 for row in range(len(df)))
    #row_range = np.arange(0,row_count)
    
    #num_teacher = int(input("Enter No. Part: "))
    #num_part = int(input("Enter NO of Part:"))
    num_eachPart = np.arange(0,num_part)
    name_part = []
    name_col = ["Correct","False","Correct %","False %","Total","A","B","C","D","E"]
    
    len_cols = len(df.columns)-1 # insert col4 ++
    # ========== Add Column Part in DataFrame ========== # 
    for n1,p in enumerate(num_eachPart,1):
        col_part = "Part"+str(p+1)
        df.insert(len_cols+n1,col_part," ")
        name_part.append(col_part)
    len_cols2 = len(df.columns)-1        
    for n2,n_col in enumerate(name_col,1):
        df.insert(len_cols2+n2,n_col," ")
    print("--------------------------------")
    
    p1 = 30
    p2 = 30
    QEachPart_list = []
    entry_list = []
    label_list = []
    
    if num_part == 1:
        num_eachP = num_questions
        text_p1 = str(num_questions)
        label_part = tk.Label(text="Part1")
        label_part.place(x=20, y=360+p1)
        entry_part = tk.Entry(window1, font=30,width=10)
        entry_part.place(x=60, y=360+p1)
        t1 = tk.Text(window1,height=1,width=10,bg="White")
        t1.place(x=60, y=360+p1)
        t1.insert(tk.END,text_p1)
        QEachPart_list.append(num_eachP)
        # entry_list.append(entry_part)
        # label_list.append(label_part)
        e1 = str(entry_part.get()) 
        B3 = tk.Button(window1, text="OK",bg='#007fff', font=10,command = lambda :[ok_EachPart(entry_list)])
        B3.place(x=205,y=355+p1)
        
    else:
        for indp,p in enumerate(range(num_part),1):
            if indp <= 5:
                label_part = tk.Label(text="Part{}".format(indp))
                label_part.place(x=20, y=360+p1) #label
                entry_part = tk.Entry(window1, font=30,width=6)
                entry_part.place(x=60, y=360+p1) #entry first box
                entry_list.append(entry_part)
                label_list.append(label_part)
                p1=p1+30
                
            else:
                label_part = tk.Label(text="Part{}".format(indp))
                label_part.place(x=120, y=360+p2) #label
                entry_part = tk.Entry(window1, font=30,width=6)
                entry_part.place(x=160, y=360+p2) #entry first box
                entry_list.append(entry_part)
                label_list.append(label_part)
                p2= p2+30
            
            
            
        B3 = tk.Button(window1, text="OK",bg='#007fff', font=10,command = lambda :[ok_EachPart(entry_list)])
        B3.place(x=225,y=510)
            #print(p1)

def ok_EachPart(entry_list):
    global QEachPart_list,num_questions,B3,label_list,TL_x,TL_y
    #QEachPart_list = []
    #for e in range(len(entry_list)):
    for ind,e in enumerate(entry_list):
        #print(e)
        num_EachP = int(e.get()) 
        #print(num_EachP)
        QEachPart_list.append(num_EachP)
    
    
    
    if sum(QEachPart_list) == num_questions:
        text_eachPart = str(QEachPart_list) +"\n"  
        box_display.insert(tk.END,text_eachPart)
        box_display.see(tk.END)  # Scroll to the bottom
        # start_video()
        #cv2.namedWindow("Contour AnswerField",cv2.WINDOW_FREERATIO)
        #cv2.namedWindow("Contour IDField",cv2.WINDOW_FREERATIO)
        video_steam()
        
    else:
        B3.destroy()
        for c in range(len(entry_list)):
            entry_list[c].destroy()
            label_list[c].destroy()
            
        if sum(QEachPart_list) < num_questions:
            text_need = "!! NEED MORE "+str(num_questions-sum(QEachPart_list))+" !! \n"
            box_display.insert(tk.END,text_need)
            box_display.see(tk.END)  # Scroll to the bottom
            QEachPart_list.clear()
            # E_numQuest.delete(0,"end")
        
        elif sum(QEachPart_list) > num_questions:
            text_decrese = "!! DECRESE MORE "+str(sum(QEachPart_list)- num_questions)+" !!\n"
            box_display.insert(tk.END,text_decrese)
            box_display.see(tk.END)  # Scroll to the bottom
            QEachPart_list.clear()
            

       
    
def Clear_part():
    global B3 ,cap,num_part,entry_list,cap
    global mode_AM
    box_display.delete(1.0,"end")
    E_numQuest.delete(0,"end")
    
    E_numPart.delete(0,"end")
    
    # MODE 
    if mode_AM == 1: #AUTO
        global L_COM,E_COM,B_COM
        L_COM.destroy()
        E_COM.destroy()
        B_COM.destroy()
    else:
        pass
        
    if num_part >1:
        B3.destroy()
    cap.release() 
    VideoCamera_contour.place_forget()
    #score_display.place_forget()
    
    for c in range(len(entry_list)):
        entry_list[c].destroy()
        label_list[c].destroy()
    
    #tk.label_part.destroy()
    #tk.entry_part.destroy()
    cap.release() 
    cv2.destroyAllWindows()
    # VideoCamera_contour.after_cancel()
    VideoCamera_contour.place_forget()
    #score_display.place_forget()
    # video = None
 
def display_Key():
    global img_key,key_label,form_paper
    if form_paper == 1:#A4
        imgKey_resize = cv2.resize(img_key[0], (400,500))
    elif form_paper == 2:#A5
        imgKey_resize = cv2.resize(img_key[0], (400,300))
        
    imgcvt_show = cv2.cvtColor(imgKey_resize, cv2.COLOR_BGR2RGB)
    img_show = Image.fromarray(imgcvt_show)
    show_img = ImageTk.PhotoImage(image=img_show)
    
    key_label.config(image=show_img) #using config so the image doesnt overlap
    key_label.img = show_img #keeping a reference
    
def display_Score():
    global img_score,score_label,form_paper
    if form_paper == 1:#A4
        imgScore_resize = cv2.resize(img_score[2], (400,500))
    elif form_paper == 2:#A5
        imgScore_resize = cv2.resize(img_score[2], (400,300))
        
    imgcvt_show = cv2.cvtColor(imgScore_resize, cv2.COLOR_BGR2RGB)
    img_show = Image.fromarray(imgcvt_show)
    show_img = ImageTk.PhotoImage(image=img_show)
    
    score_label.config(image=show_img) #using config so the image doesnt overlap
    score_label.img = show_img #keeping a reference

def display_ID():
    global imgRes_ID,ID_label
    imgID_resize = cv2.resize(imgRes_ID, (350,220))
    imgcvt_show = cv2.cvtColor(imgID_resize, cv2.COLOR_BGR2RGB)
    img_show = Image.fromarray(imgcvt_show)
    show_img = ImageTk.PhotoImage(image=img_show)
    
    ID_label.config(image=show_img) #using config so the image doesnt overlap
    ID_label.img = show_img #keeping a reference
    
    
def video_steam():
    #while(1):
    global cap
    ret,frame = cap.read()
    #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # frame_contour = frame.copy()
    roi = frame[roi_y1:roi_y2,roi_x1:roi_x2].copy()#[y1:y2,x1,x2]
    #roi = cv2.rotate(roi, cv2.ROTATE_180)
    # roi_resize = cv2.resize(roi,(w,h))
    frame_contour = roi.copy()
    
    if ret == True:
        gray1 = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        # gray1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #blur1  = cv2.GaussianBlur(gray1,(5,5),1)
        
        # Answer Sheet 
        #ret, th1 = cv2.threshold(blur1,170,255,cv2.THRESH_BINARY)
        th1 = cv2.adaptiveThreshold(gray1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,5)
        
        # Find Contour Board Edge
        contourBE_Check ,heirarchy_Check = cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) # Answer sheet EXTERNAL TREE
        
        # ========================= GLOBAL PARAMETER ============================== #
        
        # ---------- GLOBAL BEFORE PAPER COND. --------------#
        global box_display,box_display2
        global df,num_part,num_choices,num_questions,name_part
        global std_count,repeat_ID,wrong_ID  
        global pass_count ,paper_count,key_count,capture_count # paper_count -> count normal
        global studentID_list,scoreStd_list
        global reject_cond,reject_paper,rejectID_count,rejectArea_count,rejectEdge_count
        
        # ---------- GLOBAL AFTER PAPER COND. --------------#
        global area_list_Check,obj_contour_Check,areaID_Check,obj_contourID_Check
        global area_list,areaQ,areaID,center_sortAll
        global key_data,imgRes_ID,ID_label,studentID_list,no_ansRepeatList
        global img_score,score_label,img_key,key_label,path_full,path_reject
        
        # ========================================================================= #
        
        # Get Corner point
        if form_paper == 1: #A4 Bubble 
            a1_BD ,a2_BD = 80000,1200000
            a1_rec,a2_rec = 250,450  #default 250,400
            a1_ID,a2_ID = 1000,1800 #default 1000,1800
            a1_Q,a2_Q = 600,1200  #default  600,1000
            area_list_Check,obj_contour_Check = contour_area(frame_contour,contourBE_Check,a1_BD ,a2_BD)
            
        elif form_paper == 2:#A5 Bubble 
            a1_BD ,a2_BD = 77000,1000000
            a1_rec,a2_rec = 250,400 
            a1_ID,a2_ID = 1000,1800
            a1_Q,a2_Q = 600,1000
            area_list_Check,obj_contour_Check = contour_area(frame_contour,contourBE_Check,a1_BD ,a2_BD)
        
        
        #line_check = int(w/2)
        line_check = int(h/2)-90
        point1,point2 = (0,line_check),(w,line_check)
        #point1,point2 = (line_check,0),(line_check,h)
        cv2.line(frame_contour,point1,point2,(0,255,255),7)
        cv2.putText(frame_contour,str(std_count),(line_check,200),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),4,cv2.LINE_AA) 
        #cv2.imshow("Camera",frame)
        #cv2.imshow("Camera Contour",frame_contour)
        
        # =================== LINE COUNT PROCESS ===================== #
        if len(obj_contour_Check) >= 5 and len(obj_contour_Check) <= 6:
            #print("Find Center")
            area_list_Check,obj_contour_Check = sort_area(obj_contour_Check)
            obj_centerCheck,obj_contour_Check = moment_center(obj_contour_Check)
            upside_down_check = check_upsidedown(obj_centerCheck)
            
            # Use center of BD(obj_centerCheck[4][0]) to pass the line 
            #if obj_centerCheck[0][0] >= line_check and pass_count == 0:# For paper entry below need to check x follow w
            cv2.circle(frame_contour,obj_centerCheck[4],10,(0,0,255),-1)
            if obj_centerCheck[4][1] >= line_check and pass_count == 0: #For paper entry beside need to check y follow h
                pass_count = 1
                paper_count +=1

        elif len(obj_contour_Check) == 0 and pass_count == 1:     
            pass_count = 0
            capture_count = 0
        # ============================================================ #
        #cv2.imshow("Camera Contour",frame_contour)
        frame_resize = cv2.resize(frame_contour, (350,300))
        
        #frame_resize = cv2.resize(frame, (400,300))
        frame_Show = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)
        frame_contourShow = Image.fromarray(frame_Show)
        image = ImageTk.PhotoImage(image=frame_contourShow)
        VideoCamera_contour.configure(image=image)
        VideoCamera_contour.image = image
        VideoCamera_contour.after(10, video_steam)
         
        # ============================================================ #
        #cv2.imshow("Camera Contour",frame_contour)
        
        if len(obj_contour_Check) >= 5 and len(obj_contour_Check) <= 6:
            
           # Find edge and sort edge
            edge_list_Check,length_list_Check = edge_contour(obj_contour_Check)
            sort_edgeList_Check = sort_edge(edge_list_Check)
            # ========= Check Edge ========= #
            sum_ed_check = 0
            for sort_ed_check1 in sort_edgeList_Check:
                sum_ed_check += len(sort_ed_check1)
            if len(sort_edgeList_Check) == 5 and len(sort_edgeList_Check[0]) == 4 and sum_ed_check == 20:
                '''warpID/ans -> [0] = RBG ,[1] = GrayScale, [2] = AdaptiveThreshold [3] = Contour Copy for draw'''
                #warp_page_Check,warp_ID_Check,warp_ans_Check = process_image(frame,sort_edgeList_Check,form_paper)
                warp_page_Check,warp_ID_Check,warp_ans_Check = process_image(roi,sort_edgeList_Check,form_paper,upside_down_check)
                
                # Check contour choices 
                contours_rec,heirarchy = cv2.findContours(warp_ans_Check[2],cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                areaRec_check,objRec_contour_check = contour_area(frame_contour,contours_rec,a1_rec,a2_rec) # 200,400 for detect rectangle
                
                #Split ID Check
                warp_ID_Cut,center_mark = split_ID(warp_ID_Check[2],warp_ID_Check[3])
                
                # Split Answer Check
                warp_ans_QCut = warp_ans_Check[2].copy()
                warp_ans_QCut,frame_contour,center_sortAll = split_ch(warp_ans_QCut,warp_ans_Check[3],objRec_contour_check,form_paper)
                #print(len(center_sortAll))
                
                contour_quest_Check,heirarchy = cv2.findContours(warp_ans_QCut,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                contour_ID_Check,heirarchy = cv2.findContours(warp_ID_Cut,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
               
                #cv2.imshow("FrameCut",warp_ans_QCut)
                #cv2.imshow("TH_ID",warp_ID_Check[2])
                #cv2.imshow("TH_Ans",warp_ans_Check[2])
                
                # =================== A4 PAPER BUBBLE ===================== #
                if form_paper == 1: #A4 Bubble 
                    areaQ_Check,obj_contourQuest_Check = contour_area(warp_ans_Check[3],contour_quest_Check,a1_Q,a2_Q) #200,500
                    areaID_Check,obj_contourID_Check = contour_area(warp_ID_Check[3],contour_ID_Check,a1_ID,a2_ID)
                    #cv2.imshow("Contour AnswerField",frame_contour)
                    #cv2.imshow("Contour IDField",warp_ans_QCut)
                    #cv2.imshow("Contour AnswerField",warp_ans_Check[3])
                    #cv2.imshow("TH SCORE",warp_ans_Check[2])
                    
                    #print(len(areaQ_Check),len(obj_contourQuest_Check))
                    if len(obj_contourQuest_Check) == 1000 and len(obj_contourID_Check) >= 170:
                        if capture_count == 0 and pass_count == 1:
                            screenshot =  cap.read()[1]  # Screenshot Image 
                            screenshot_roi = screenshot[roi_y1:roi_y2,roi_x1:roi_x2].copy()
                            screenshot_contour = screenshot_roi.copy()
                            #screenshot_contour = screenshot.copy()
                            #cv2.imshow("Screenshot CHECK",screenshot_roi)
                            capture_count = 1
                            
                            screenshot_gray = cv2.cvtColor(screenshot_roi,cv2.COLOR_BGR2GRAY)
                            screenshot_th = cv2.adaptiveThreshold(gray1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,5)
                            #cv2.imshow("Contour AnswerField",warp_ansField_contour)
                            contour_BE,heirarchy = cv2.findContours(screenshot_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                            #contour_BE,heirarchy = cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                            area_list,obj_contour = contour_area(screenshot_contour,contour_BE,a1_BD ,a2_BD)
                            area_list,obj_contour = sort_area(obj_contour)
                            obj_center,obj_contour = moment_center(obj_contour)
                            upside_down = check_upsidedown(obj_center)
                            edge_list1,length_list1 = edge_contour(obj_contour)
                            sort_edgeList1 = sort_edge(edge_list1)
                            # ========= Check Edge ========= #
                            sum_ed = 0
                            for sort_ed1 in sort_edgeList1:
                                sum_ed += len(sort_ed1)
                                
                            if len(sort_edgeList1) == 5 and len(sort_edgeList1[0]) == 4 and sum_ed == 20:
                                '''warpID/ans -> [0] = RBG ,[1] = GrayScale, [2] = AdaptiveThreshold [3] = Contour Copy for draw'''
                                #warp_page1,warp_ID1,warp_ans1 = process_image(screenshot,sort_edgeList1,form_paper)
                                warp_page1,warp_ID1,warp_ans1 = process_image(screenshot_roi,sort_edgeList1,form_paper,upside_down)
                                areaRec,objRec_contour = contour_area(screenshot_contour,contours_rec,a1_rec,a2_rec) # 200,400 for detect rectangle
                                
                                # Split ID 
                                warp_ID_Cut1,center_mark1 = split_ID(warp_ID1[2],warp_ID1[3])
                                
                                # Split Ans
                                areaRec,objRec_contour = contour_area(screenshot_contour,contours_rec,a1_rec,a2_rec) # 200,400 for detect rectangle
                                warp_ans_QCut1 = warp_ans1[2].copy() #[2] = AdaptiveThreshold
                                warp_ans_QCut1,screenshot_contour,center_sortAll = split_ch(warp_ans_QCut1,warp_ans1[3],objRec_contour,form_paper)
                                
                                contour_quest,heirarchy = cv2.findContours(warp_ans_QCut1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                                contour_ID,heirarchy = cv2.findContours(warp_ID_Cut1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                                
                                #-----contour_area(img_contour,contours,a1,a2)-----#
                                areaQ,obj_contourQuest = contour_area(warp_ans1[3],contour_quest,a1_Q,a2_Q) #200,500
                                areaID,obj_contourID = contour_area(warp_ID1[3],contour_ID,a1_ID,a2_ID) #
                             
                                #cv2.imshow("TH SCORE",warp_ans_QCut1)
                                print("\n___________________________________________________")
                                print("****** CHECK PROPERTY SCREENSHOT ******")
                                print("--------------------------------")
                                print("No.contour Big,Quest,ID,Rec:",len(area_list),"|",len(areaQ),"|",len(areaID),"|",len(center_sortAll))
                                print("--------------------------------")
                                if len(obj_contourQuest) == 1000 and len(obj_contourID) >= 170 : 
                                    # =================== ID PROCESSING ===================== #
                                    import StudentID_Module
                                    #global key_data,imgRes_ID,ID_label,studentID_list
                                    #img_id = cv2.imread(path_ID)
                                    img_drawID = warp_ID1[0]
                                    img_id = warp_ID_Cut1
                                    imgRes_ID,ID_Std,ID_Num = StudentID_Module.ID_student(img_id,img_drawID)
                                    #studentID_list.append(ID_Num)
                                    print("NUM:",ID_Num)
                                    if ID_Num in studentID_list:
                                        repeat_ID = 1
                                    elif ID_Num not in studentID_list:
                                        repeat_ID = 0
                                    
                                    if len(ID_Std) == 13 and ID_Std[0] == 0 and key_count == 0:
                                        print("##### KEY MODULE #####")
                                        import A4Bubble_KeyModule
                                        #path_Key = "D:/KMUTNB/Project KMUTNB/AnswerChecker/Student/A5/test/test_save/Key/std{}.jpg"
                                        #img_key = cv2.imread(path_Key)
                                        '''img_key -> [0] = RBG ,[1] = GrayScale, [2] = AdaptiveThreshold [3] = Contour Copy for draw'''
                                        img_key = [warp_ans1[0],warp_ans1[1],warp_ans_QCut1,warp_ans1[3]]
                                        key_markList,key_indexList,key_corner = A4Bubble_KeyModule.main(img_key,num_questions,num_choices)
                                        key_data = [key_markList,key_indexList,key_corner]
                                        key_count += 1   
                                        std_count += 1
                                       
                                        print("----> FINISH KEY <----")
                                        print("***** THIS ID IS KEY WAIT FOR STUDENT PAPER *****\n")
                                        print("\n___________________________________________________")
                                        # ========== Display Key on GUI ========= #
                                        #imgKey_resize = cv2.resize(img_key[3],(400,500))
                                        #cv2.imshow("STUDENT SCORE",img_key[0])
                                        key_label = tk.Label(window1,bg="white")
                                        key_label.pack()
                                        key_label.place(x=760,y=30) #(400,300)  
                                        window1.after(1,display_Key)
                                        # =================== DISPLAY TEXT IN BIG BOX  ===================#
                                        box_display2.insert(tk.END,"##### KEY MODULE #####\n")
                                        box_display2.insert(tk.END,"----> FINISH KEY <----\n")
                                        box_display2.insert(tk.END,"***** THIS ID IS KEY WAIT FOR STUDENT PAPER *****\n")
                                        box_display2.see(tk.END)  # Scroll to the bottom
                                        next_paper()
                                  
                                    elif len(ID_Std) == 13 and ID_Std[0] != 0 and key_count == 1 and repeat_ID == 0:
                                        df_mask = df["Student_ID"].values == ID_Num #find value that match with ID in df 
                                        df_pos = np.flatnonzero(df_mask) # get index(row) that equal ID 
                                        row_count = sum(1 for row in range(len(df)))
                                        row_range = np.arange(0,row_count)
                                        if df_pos in row_range:
                                            print("====== Student:{} ======".format(std_count))
                                            for std in range(len(df_pos)):
                                                std_name ,std_surname = df.iloc[df_pos[std],2],df.iloc[df_pos[std],3] # 1 is col "Name" | 2 is col "Surname"
                                                print("Name:",std_name,std_surname)
                                                print("ID:",ID_Num)
                                            print("--------------------------------")
                                            
                                            # ========== Display ID on GUI ========= #
                                            imgID_resize = cv2.resize(imgRes_ID, (350,220))
                                            #cv2.imshow("ID STUDENT",imgID_resize)
                                            ID_label = tk.Label(window1,bg="white")
                                            ID_label.pack()
                                            ID_label.place(x=420,y=300) #(400,300)
                                            #display_image()
                                            window1.after(1,display_ID)
                                            
                                            #global key_data
                                            # =================== CALCULATE SCORE ===================== #
                                            print("#### CALCULATE SCORE ####")
                                            import A4Bubble_ScoreModule
                                            #global img_score,score_label
                                            studentID_list.append(ID_Num)
                                            #img_std = cv2.imread(path_answerField)
                                            img_std = [warp_ans1[0],warp_ans1[1],warp_ans_QCut1,warp_ans1[3]]
                                            '''data_score = [score,wrong,per_correct,per_wrong,part_list,choice_correct,choice_correct,
                                                          choice_wrong,most_correct,most_wrong,no_ansRepeat] '''
                                            img_score,data_score = A4Bubble_ScoreModule.main(img_std,key_data,num_questions,num_choices,name_part,QEachPart_list)
                                            scoreStd_list.append(data_score[0])
                                            
                                            # ========== SAVE SCORE ========= #
                                            path_score = os.path.join(path_full,"{}_score{}.jpg".format(ID_Num,data_score[0]))
                                            #path_score = "C:/KMUTNB/Project KMUTNB/AnswerChecker/Student/A5/test/test_save/Answer_field/{}.jpg".format(ID_Num)
                                            cv2.imwrite(path_score,img_score[2])
                                            
                                            # ========== SAVE REPEAT/NO ANSWER ========= #
                                            if len(data_score[10]) != 0:
                                                no_ansRepeat = str(std_name) +"   "+str(std_surname) +"      "+ str(ID_Num)+":"+str(data_score[10])
                                                no_ansRepeatList.append(no_ansRepeat)
                                            else:
                                                pass
                                            
                                            # ========== Display SCORE on GUI ========= #
                                            #imgScore_resize = cv2.resize(img_score[2],(400,500))
                                            #cv2.imshow("STUDENT SCORE",img_score[2])
                                            score_label = tk.Label(window1,bg="white")
                                            score_label.pack()
                                            score_label.place(x=760,y=30) #(400,300)  
                                            window1.after(1,display_Score)
                                            
                                            # ========================================= #
                                            
                                            # =================== DISPLAY TEXT IN BIG BOX  ===================#
                                            box_display2.insert(tk.END,"\n====== Student:{} ======\n".format(std_count))
                                            box_display2.insert(tk.END,"Name:{} {}\n".format(std_name,std_surname))
                                            box_display2.insert(tk.END,"ID:{}\n".format(ID_Num))
                                            box_display2.insert(tk.END,"--------------------------------")
                                            box_display2.insert(tk.END,"\n#### CALCULATE SCORE ####\n")
                                            box_display2.insert(tk.END,"Score:{}/{}".format(data_score[0],num_questions))
                                            box_display2.insert(tk.END,"\nCorrect: {} |Wrong: {} ".format(data_score[0],data_score[1]))
                                            box_display2.insert(tk.END,"\nCorrect[%]: {}% | Wrong[%]: {}%".format(data_score[2],data_score[3]))
                                            box_display2.insert(tk.END,"\n___________________________________________________\n")
                                            box_display2.see(tk.END)  # Scroll to the bottom
                                            # =================== SAVE SCORE ===================== #
                                            part_list1 = data_score[4]
                                            for n in range(len(name_part)):
                                                index_std = df.index[df_pos]
                                                df.loc[index_std,name_part[n]] = int(part_list1[name_part[n]][1])
                                                if n >= len(name_part)-1:
                                                    df.loc[index_std,"Correct"] = data_score[0]
                                                    df.loc[index_std,"False"] = data_score[1]
                                                    df.loc[index_std,"Correct %"] = data_score[2]
                                                    df.loc[index_std,"False %"] = data_score[3]
                                                    df.loc[index_std,"Total"] = str(data_score[0])+"/"+str(num_questions)
                                            
                                            name_ch = ["A","B","C","D","E"]
                                            for indC,nameC in enumerate(name_ch):
                                                df.loc[index_std,nameC] = data_score[5][indC]
                                                    
                                            wrong_ID = 0       
                                            std_count += 1
                                            reject_cond = 0
                                            reject_paper = 0
                                            next_paper()
                                            
                                        else:
                                            # If Student ID Not Found 
                                            if wrong_ID <= 10:
                                                 capture_count = 0
                                                 wrong_ID +=1
                                                 print("!!!! WRONG STUDENT ID !!!!")
                                                 print("----->",ID_Num,": NOT MATCH TO ID STUDENT","<-----")
                                                 print("----- SCREENSHOT AGAIN ------")
                                                 box_display2.insert(tk.END,"\n!!!! WRONG STUDENT ID !!!!\n")
                                                 box_display2.insert(tk.END,"\n----->",ID_Num,": NOT MATCH TO ID STUDENT","<-----\n")
                                                 box_display2.see(tk.END)  # Scroll to the bottom
                                            else:
                                                 print("!!!! ID NOT FOUND !!!!")
                                                 print("======= REJECT PAPER [ID] =======")
                                                 box_display2.insert(tk.END,"\n!!!! ID NOT FOUND !!!!\n")
                                                 box_display2.see(tk.END)  # Scroll to the bottom
                                                 path_reject = os.path.join(path_reject,"REJECT_ID{}.jpg".format(rejectID_count))
                                                 cv2.imwrite(path_reject,warp_page1)
                                                 rejectID_count += 1 
                                                 wrong_ID = 0
                                                 next_paper()
                                                 
                                            
                                    elif len(ID_Std) == 13 and ID_Std[0] != 0 and key_count == 1 and repeat_ID == 1:
                                            #capture_count = 0
                                            print("!!!! THIS ID IS ALREADY CHECK !!!!")       
                                            next_paper()
                                            print("=======> LODING NEW PAPER <=======")
                                            
                                    elif len(ID_Std) == 13 and ID_Std[0] != 0 and key_count == 0: 
                                        print("!!!! WE CANNOT CALCULATE WITHOUT KEY !!!!")
                                        print("----> PLEASE INSERT KEY FIRST <----")
                                        print("___________________________________________________")
                                        next_paper()
                                        box_display2.insert(tk.END,"!!!! WE CANNOT CALCULATE WITHOUT KEY !!!!\n")
                                        box_display2.insert(tk.END,"----> PLEASE INSERT KEY FIRST <----\n")
                                        box_display2.see(tk.END)  # Scroll to the bottom
                                        
                                    else: 
                                        # If Student ID Not Found 
                                        if wrong_ID <= 10:
                                             capture_count = 0
                                             wrong_ID +=1
                                             print("!!!! WRONG STUDENT ID !!!!")
                                             print("----->",ID_Num,": NOT MATCH TO ID STUDENT","<-----")
                                             print("----- SCREENSHOT AGAIN ------")
                                             box_display2.insert(tk.END,"\n!!!! WRONG STUDENT ID !!!!\n")
                                             box_display2.insert(tk.END,"\n----->",ID_Num,": NOT MATCH TO ID STUDENT","<-----\n")
                                             box_display2.see(tk.END)  # Scroll to the bottom
                                        else:
                                             print("!!!! ID NOT FOUND !!!!")
                                             print("======= REJECT PAPER [ID] =======")
                                             box_display2.insert(tk.END,"\n!!!! ID NOT FOUND !!!!\n")
                                             box_display2.see(tk.END)  # Scroll to the bottom
                                             path_reject = os.path.join(path_reject,"REJECT_ID{}.jpg".format(rejectID_count))
                                             cv2.imwrite(path_reject,warp_page1)
                                             rejectID_count += 1 
                                             wrong_ID = 0
                                             next_paper()
                                        
                                        
                                else:
                                    # SCREENSHOT AGAIN len(Area)-> ANS,ID
                                    reject_paper += 1
                                    if reject_paper > 10: 
                                        print("======= REJECT PAPER [AREA] =======")
                                        path_reject = os.path.join(path_reject,"REJECT_AREA{}.jpg".format(rejectArea_count))
                                        cv2.imwrite(path_reject,warp_page1)
                                        rejectArea_count += 1 
                                        
                                        box_display2.insert(tk.END,"\n======= REJECT PAPER [AREA] =======\n")
                                        box_display2.see(tk.END)  # Scroll to the bottom
                                        next_paper()
                                        print("=======> LODING NEW PAPER <=======")
                                        box_display2.insert(tk.END,"\n=======> LODING NEW PAPER <=======\n")
                                        box_display2.see(tk.END)  # Scroll to the bottom
                                        reject_paper = 0
                                    else:
                                        # SCREENSHOT AGAIN len(Area)-> ANS,ID
                                        print("!!!! SCREENSHOT AGAIN(AREA) !!!!")
                                        box_display2.insert(tk.END,"\n!!!! SCREENSHOT AGAIN !!!!\n")
                                        box_display2.see(tk.END)  # Scroll to the bottom
                                        capture_count = 0
                                    
                            else:
                                # SCREENSHOT AGAIN len(EDGE) 
                                reject_paper += 1 
                                if reject_paper > 10: 
                                    print("======= REJECT PAPER[EDGE] =======")
                                    path_reject = os.path.join(path_reject,"REJECT_EDGE{}.jpg".format(rejectEdge_count))
                                    cv2.imwrite(path_reject,warp_page1)
                                    rejectEdge_count += 1 
                                    box_display2.insert(tk.END,"\n======= REJECT PAPER [EDGE] =======\n")
                                    box_display2.see(tk.END)  # Scroll to the bottom
                                    next_paper()
                                    print("=======> LODING NEW PAPER <=======")
                                    reject_paper = 0
                                else:
                                    # SCREENSHOT AGAIN len(SORT_EDGE)
                                    print("!!!! SCREENSHOT AGAIN(EDGE) !!!!")
                                    capture_count = 0
                                    
                        else:
                            #cap count, pass count
                            print("", end =".")
                            #print("Wait For Camera Focus -> ", end =".")
                            
                    else:
                        # len quest and len std id 
                        print("", end =".")
                        #pass
                        
                
                # =================== A5 PAPER BUBBLE ===================== #
                elif form_paper == 2: #A5 Bubble
                    areaQ_Check,obj_contourQuest_Check = contour_area(warp_ans_Check[3],contour_quest_Check,a1_Q,a2_Q) #200,500
                    areaID_Check,obj_contourID_Check = contour_area(warp_ID_Check[3],contour_ID_Check,a1_ID,a2_ID)
                    
                    #cv2.imshow("Th IDField",warp_ID_Check[2])
                    # cv2.imshow("Th AnswerField",warp_ans_Check[2])
                    
                    # cv2.imshow("Contour IDField",warp_ID_Check[3])
                    # cv2.imshow("Contour AnswerField",warp_ans_Check[3])
                    
                    #if len(obj_contourQuest_Check) == 600 and len(obj_contourID_Check) == 170 and len(center_sortAll) == 19: 
                    if len(obj_contourQuest_Check) == 600 and len(obj_contourID_Check) >= 170 :
                        if capture_count == 0 and pass_count == 1:
                            screenshot =  cap.read()[1]  # Screenshot Image 
                            screenshot_roi = screenshot[roi_y1:roi_y2,roi_x1:roi_x2].copy()
                            screenshot_contour = screenshot_roi.copy()
                            #screenshot_contour = screenshot.copy()
                            #cv2.imshow("Screenshot CHECK",screenshot_roi)
                            capture_count = 1
                            
                            screenshot_gray = cv2.cvtColor(screenshot_roi,cv2.COLOR_BGR2GRAY)
                            screenshot_th = cv2.adaptiveThreshold(screenshot_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,5)
                            #cv2.imshow("Contour AnswerField",warp_ansField_contour)
                            contour_BE,heirarchy = cv2.findContours(screenshot_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                            area_list,obj_contour = contour_area(screenshot_contour,contour_BE,a1_BD ,a2_BD)
                            area_list,obj_contour = sort_area(obj_contour)
                            obj_center,obj_contour = moment_center(obj_contour_Check)
                            upside_down = check_upsidedown(obj_center)
                            edge_list1,length_list1 = edge_contour(obj_contour)
                            sort_edgeList1 = sort_edge(edge_list1)
                            
                            # ========= Check Edge ========= #
                            sum_ed = 0
                            for sort_ed1 in sort_edgeList1:
                                sum_ed += len(sort_ed1)
                                
                            if len(sort_edgeList1) == 5 and len(sort_edgeList1[0]) == 4 and sum_ed == 20:
                                '''warpID/ans -> [0] = RBG ,[1] = GrayScale, [2] = AdaptiveThreshold [3] = Contour Copy for draw'''
                                #warp_page1,warp_ID1,warp_ans1 = process_image(screenshot,sort_edgeList1,form_paper)
                                warp_page1,warp_ID1,warp_ans1 = process_image(screenshot_roi,sort_edgeList1,form_paper,upside_down)
                                areaRec,objRec_contour = contour_area(screenshot_contour,contours_rec,a1_rec,a2_rec) # 200,400 for detect rectangle
                                
                                # Split ID 
                                warp_ID_Cut1,center_mark1 = split_ID(warp_ID1[2],warp_ID1[3])
                                
                                # Split Ans
                                areaRec,objRec_contour = contour_area(screenshot_contour,contours_rec,a1_rec,a2_rec) # 200,400 for detect rectangle
                                warp_ans_QCut1 = warp_ans1[2].copy() #[2] = AdaptiveThreshold
                                warp_ans_QCut1,screenshot_contour,center_sortAll = split_ch(warp_ans_QCut1,warp_ans1[3],objRec_contour,form_paper)
                                
                                contour_quest,heirarchy = cv2.findContours(warp_ans_QCut1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                                contour_ID,heirarchy = cv2.findContours(warp_ID_Cut1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                                
                                #-----contour_area(img_contour,contours,a1,a2)-----#
                                areaQ,obj_contourQuest = contour_area(warp_ans1[3],contour_quest,a1_Q,a2_Q) #200,500
                                areaID,obj_contourID = contour_area(warp_ID1[3],contour_ID,a1_ID,a2_ID) #
                             
                                #cv2.imshow("TH SCORE",warp_ans_QCut1)
                                print("\n___________________________________________________")
                                print("****** CHECK PROPERTY SCREENSHOT ******")
                                print("--------------------------------")
                                print("No.contour Big,Quest,ID,Rec:",len(area_list),"|",len(areaQ),"|",len(areaID),"|",len(center_sortAll))
                                print("--------------------------------")
                                if len(obj_contourQuest) == 600 and len(obj_contourID) >= 170 : 
                                    # =================== ID PROCESSING ===================== #
                                    import StudentID_Module
                                    #global key_data,imgRes_ID,ID_label,studentID_list
                                    img_drawID = warp_ID1[0]
                                    img_id = warp_ID_Cut1
                                    imgRes_ID,ID_Std,ID_Num = StudentID_Module.ID_student(img_id,img_drawID)
                                    #studentID_list.append(ID_Num)
                                    print("NUM:",ID_Num)
                                    if ID_Num in studentID_list:
                                        repeat_ID = 1
                                    elif ID_Num not in studentID_list:
                                        repeat_ID = 0
                                
                                    if len(ID_Std) == 13 and ID_Std[0] == 0 and key_count == 0:
                                        print("##### KEY MODULE #####")
                                        import A5Bubble_KeyModule
                                        #path_Key = "D:/KMUTNB/Project KMUTNB/AnswerChecker/Student/A5/test/test_save/Key/std{}.jpg"
                                        #img_key = cv2.imread(path_Key)
                                        '''img_key -> [0] = RBG ,[1] = GrayScale, [2] = AdaptiveThreshold [3] = Contour Copy for draw'''
                                        img_key = [warp_ans1[0],warp_ans1[1],warp_ans_QCut1,warp_ans1[3]]
                                        key_markList,key_indexList,key_corner = A5Bubble_KeyModule.main(img_key,num_questions,num_choices)
                                        key_data = [key_markList,key_indexList,key_corner]
                                        key_count += 1   
                                        std_count += 1
                                        # ========== Display Key on GUI ========= #
                                        #imgKey_resize = cv2.resize(img_key[3],(400,500))
                                        #cv2.imshow("STUDENT SCORE",img_key[0])
                                        key_label = tk.Label(window1,bg="white")
                                        key_label.pack()
                                        key_label.place(x=765,y=30) #(400,300)  
                                        window1.after(1,display_Key)
                                        # =================== DISPLAY TEXT IN BIG BOX  ===================#
                                        box_display2.insert(tk.END,"##### KEY MODULE #####\n")
                                        box_display2.insert(tk.END,"----> FINISH KEY <----\n")
                                        box_display2.insert(tk.END,"***** THIS ID IS KEY WAIT FOR STUDENT PAPER *****\n")
                                        box_display2.see(tk.END)  # Scroll to the bottom
                                        next_paper()
                                        print("----> FINISH KEY <----")
                                        print("***** THIS ID IS KEY WAIT FOR STUDENT PAPER *****")
                                        print("\n___________________________________________________")
                                          
                                  
                                    elif len(ID_Std) == 13 and ID_Std[0] != 0 and key_count == 1 and repeat_ID == 0:
                                        df_mask = df["Student_ID"].values == ID_Num #find value that match with ID in df 
                                        df_pos = np.flatnonzero(df_mask) # get index(row) that equal ID 
                                        row_count = sum(1 for row in range(len(df)))
                                        row_range = np.arange(0,row_count)
                                        if df_pos in row_range:
                                            print("====== Student:{} ======".format(std_count))
                                            for std in range(len(df_pos)):
                                                std_name ,std_surname = df.iloc[df_pos[std],2],df.iloc[df_pos[std],3] # 1 is col "Name" | 2 is col "Surname"
                                                print("Name:",std_name,std_surname)
                                                print("ID:",ID_Num)
                                            print("--------------------------------")
                                            
                                            # ========== Display ID on GUI ========= #
                                            # ========== Display ID on GUI ========= #
                                            imgID_resize = cv2.resize(imgRes_ID, (350,220))
                                            #cv2.imshow("ID STUDENT",imgID_resize)
                                            ID_label = tk.Label(window1,bg="white")
                                            ID_label.pack()
                                            ID_label.place(x=420,y=300) #(400,300)
                                            #display_image()
                                            window1.after(1,display_ID)
                                            
                                            #global key_data
                                            # =================== CALCULATE SCORE ===================== #
                                            print("#### CALCULATE SCORE ####")
                                            import A5Bubble_scoreModule
                                            #global img_score,score_label
                                            studentID_list.append(ID_Num)
                                            #img_std = cv2.imread(path_answerField)
                                            img_std = [warp_ans1[0],warp_ans1[1],warp_ans_QCut1,warp_ans1[3]]
                                            '''data_score = [score,wrong,per_correct,per_wrong,part_list,choice_correct,choice_correct,
                                                          choice_wrong,most_correct,most_wrong,no_ansRepeat] '''
                                            img_score,data_score = A5Bubble_scoreModule.main(img_std,key_data,num_questions,num_choices,name_part,QEachPart_list)
                                            scoreStd_list.append(data_score[0])
                                            #cv2.imshow("STUDENT SCORE",img_score[2])
                                            
                                            # ========== SAVE SCORE ========= #
                                            path_score = os.path.join(path_full,"{}_score{}.jpg".format(ID_Num,data_score[0]))
                                            cv2.imwrite(path_score,img_score[2])
                                            
                                            # ========== SAVE REPEAT/NO ANSWER ========= #
                                            if len(data_score[10]) != 0:
                                                no_ansRepeat = str(std_name) +"   "+str(std_surname) +"      "+ str(ID_Num)+":"+str(data_score[10])
                                                no_ansRepeatList.append(no_ansRepeat)
                                            else:
                                                pass
                                            
                                            # ========== Display SCORE on GUI ========= #
                                            imgScore_resize = cv2.resize(img_score[2], (500,300))
                                            #cv2.imshow("STUDENT SCORE",imgScore_resize)
                                            score_label = tk.Label(window1,bg="white")
                                            score_label.pack()
                                            score_label.place(x=765,y=30) #(400,300)
                                            
                                            window1.after(1,display_Score)
                                            # ========================================= #
                                            
                                            # =================== DISPLAY TEXT IN BIG BOX  ===================#
                                            #text_score1 = 
                                            box_display2.insert(tk.END,"\n====== Student:{} ======\n".format(std_count))
                                            box_display2.insert(tk.END,"Name:{} {}\n".format(std_name,std_surname))
                                            box_display2.insert(tk.END,"ID:{}\n".format(ID_Num))
                                            box_display2.insert(tk.END,"--------------------------------")
                                            box_display2.insert(tk.END,"\n#### CALCULATE SCORE ####\n")
                                            box_display2.insert(tk.END,"Score:{}/{}".format(data_score[0],num_questions))
                                            box_display2.insert(tk.END,"\nCorrect: {} |Wrong: {} ".format(data_score[0],data_score[1]))
                                            box_display2.insert(tk.END,"\nCorrect[%]: {}% | Wrong[%]: {}%".format(data_score[2],data_score[3]))
                                            box_display2.see(tk.END)  # Scroll to the bottom
        
                                            # =================== SAVE SCORE ===================== #
                                            part_list1 = data_score[4]
                                            for n in range(len(name_part)):
                                                index_std = df.index[df_pos]
                                                df.loc[index_std,name_part[n]] = int(part_list1[name_part[n]][1])
                                                if n >= len(name_part)-1:
                                                    df.loc[index_std,"Correct"] = data_score[0]
                                                    df.loc[index_std,"False"] = data_score[1]
                                                    df.loc[index_std,"Correct %"] = data_score[2]
                                                    df.loc[index_std,"False %"] = data_score[3]
                                                    df.loc[index_std,"Total"] = str(data_score[0])+"/"+str(num_questions)
                                            
                                            name_ch = ["A","B","C","D","E"]
                                            for indC,nameC in enumerate(name_ch):
                                                df.loc[index_std,nameC] = data_score[5][indC]
                                                    
                                            wrong_ID = 0       
                                            std_count += 1
                                            reject_cond = 0
                                            reject_paper = 0
                                            next_paper()
                                            
                                        else:
                                            # If Student ID Not Found 
                                            if wrong_ID <= 10:
                                                 capture_count = 0
                                                 wrong_ID +=1
                                                 print("!!!! WRONG STUDENT ID !!!!")
                                                 print("----- SCREENSHOT AGAIN ------")
                                                 box_display2.insert(tk.END,"\n!!!! WRONG STUDENT ID !!!!\n")
                                                 box_display2.insert(tk.END,"\n----->",ID_Num,": NOT MATCH TO ID STUDENT","<-----\n")
                                                 box_display2.see(tk.END)  # Scroll to the bottom
                                            else:
                                                 print("!!!! ID NOT FOUND !!!!")
                                                 box_display2.insert(tk.END,"\n!!!! ID NOT FOUND !!!!\n")
                                                 box_display2.see(tk.END)  # Scroll to the bottom
                                                 path_reject = os.path.join(path_reject,"REJECT_ID{}.jpg".format(rejectID_count))
                                                 cv2.imwrite(path_reject,warp_page1)
                                                 rejectID_count += 1 
                                                 wrong_ID = 0
                                                 next_paper()
                                            
                                    elif len(ID_Std) == 13 and ID_Std[0] != 0 and key_count == 1 and repeat_ID == 1:
                                            #capture_count = 0
                                            print("!!!! THIS ID IS ALREADY CHECK !!!!")
                                            box_display2.insert(tk.END,"!!!! THIS ID IS ALREADY CHECK !!!!")
                                            box_display2.see(tk.END)  # Scroll to the bottom
                                            next_paper()
                                            print("=======> LODING NEW PAPER <=======")
                                    
                                    elif len(ID_Std) == 13 and ID_Std[0] != 0 and key_count == 0: 
                                        print("!!!! WE CANNOT CALCULATE WITHOUT KEY !!!!")
                                        print("----> PLEASE INSERT KEY FIRST <----")
                                        box_display2.insert(tk.END,"!!!! WE CANNOT CALCULATE WITHOUT KEY !!!!")
                                        box_display2.insert(tk.END,"----> PLEASE INSERT KEY FIRST <----")
                                        box_display2.see(tk.END)  # Scroll to the bottom
                                        print("___________________________________________________")
                                        
                                    else: 
                                        # If Student ID Not Found 
                                        if wrong_ID <= 10:
                                             capture_count = 0
                                             wrong_ID +=1
                                             print("!!!! WRONG STUDENT ID !!!!")
                                             print("----->",ID_Num,": NOT MATCH TO ID STUDENT","<-----")
                                             print("----- SCREENSHOT AGAIN ------")
                                             box_display2.insert(tk.END,"\n!!!! WRONG STUDENT ID !!!!\n")
                                             box_display2.insert(tk.END,"\n----->",ID_Num,": NOT MATCH TO ID STUDENT","<-----\n")
                                             box_display2.see(tk.END)  # Scroll to the bottom
                                        
                                        else:
                                             print("!!!! ID NOT FOUND !!!!")
                                             print("======= REJECT PAPER [ID] =======")
                                             box_display2.insert(tk.END,"\n!!!! ID NOT FOUND !!!!\n")
                                             box_display2.see(tk.END)  # Scroll to the bottom
                                             path_reject = os.path.join(path_reject,"REJECT_ID{}.jpg".format(rejectID_count))
                                             cv2.imwrite(path_reject,warp_page1)
                                             rejectID_count += 1 
                                             wrong_ID = 0
                                             next_paper()   
                                            
                                else:
                                    # SCREENSHOT AGAIN len(Area)-> ANS,ID
                                    reject_paper += 1
                                    if reject_paper > 10: 
                                        print("======= REJECT PAPER [AREA] =======")
                                        box_display2.insert(tk.END,"\n======= REJECT PAPER [AREA] =======\n")
                                        
                                        path_reject = os.path.join(path_reject,"REJECT_Area{}.jpg".format(rejectArea_count))
                                        cv2.imwrite(path_reject,warp_page1)
                                        rejectArea_count += 1 
                                        next_paper()
                                        print("=======> LODING NEW PAPER <=======")
                                        box_display2.insert(tk.END,"\n=======> LODING NEW PAPER <=======\n")
                                        box_display2.see(tk.END)  # Scroll to the bottom
                                        reject_paper = 0
                                    else:
                                        # SCREENSHOT AGAIN len(Area)-> ANS,ID
                                        print("!!!! SCREENSHOT AGAIN(AREA) !!!!")
                                        box_display2.insert(tk.END,"\n!!!! SCREENSHOT AGAIN(AREA) !!!!\n")
                                        box_display2.see(tk.END)  # Scroll to the bottom
                                        capture_count = 0
                            else:
                                # SCREENSHOT AGAIN len(EDGE) 
                                reject_paper += 1 
                                if reject_paper > 10: 
                                    print("======= REJECT PAPER [EDGE] =======")
                                    box_display2.insert(tk.END,"\n======= REJECT PAPER [EDGE] =======\n")
                                    path_reject = os.path.join(path_reject,"REJECT_Edge{}.jpg".format(rejectEdge_count))
                                    cv2.imwrite(path_reject,warp_page1)
                                    rejectEdge_count += 1 
                                    next_paper()
                                    print("=======> LODING NEW PAPER <=======")
                                    box_display2.insert(tk.END,"\n=======> LODING NEW PAPER <=======\n")
                                    box_display2.see(tk.END)  # Scroll to the bottom
                                    reject_paper = 0
                                else:
                                    # SCREENSHOT AGAIN len(SORT_EDGE)
                                    print("!!!! SCREENSHOT AGAIN(EDGE) !!!!")
                                    capture_count = 0   
                        else:
                            #cap count, pass count
                            print("", end =".")
                            #print("Wait For Camera Focus -> ", end =".")
                            
                    else:
                        # len quest and len std id 
                        print("", end =".")
                        #pass
                
            else:
                # EDGE CHECK
                print("", end =".")
def finish():
    global df,video,path_full,code_year,code_semester,code_section,no_ansRepeatList,path_set
    
    # =================== VISUALIZE DATA ===================== #
    plot_data(form_paper,scoreStd_list)
    
    # =================== SAVE REPEAT/NO ANSWER ===================== #
    if len(no_ansRepeatList) != 0:
        path_noRepeat = os.path.join(path_full,"{}_REPEAT_NO_ANSWER.txt".format(code_year))
        with open(path_noRepeat,"w",encoding='utf-8-sig') as f:
            f.write("##### THIS IS SHOW ID STUDENT WHO HAS REPEAT ANSWER OR NO ANSWER #####\n")
            f.write("NO    NAME    SURNAME      ID STUDENT    NO. Question\n")
            for indL,line in enumerate(no_ansRepeatList,1):
                line = str(indL)+"    " + line 
                f.write(line)
                f.write("\n")
    # =================== SAVE CSV FILE ===================== #
    path_csv = os.path.join(path_full,"{}_{}_{}_sec{}_{}.csv".format(code_semester,code_year,code_subject,code_section,path_set))
    #path_csv = "C:/KMUTNB/Project KMUTNB/AnswerChecker/Student/A4/test/test_save/csv/Student_Data1.csv"
    df.to_csv(path_csv,index = True, encoding='utf-8-sig')
    print("\n=============== SAVE TO CSV FILE ===============")
    
    # =================== SAVE CSV FILE ===================== #
    
    cap.release() 
    cv2.destroyAllWindows()
    # VideoCamera_contour.after_cancel()
    VideoCamera_contour.place_forget()
    #score_display.place_forget()
    window1.destroy()
    #video = None
    
if __name__ == "__main__":
    window1 = tk.Tk()
    window1.title("AUTOMATIC EXAM CHECKER") #named window gui
    # ============ CONFIGURATION GUI ============ #
    app_width,app_height = 1200,800
    screen_width = window1.winfo_screenwidth()
    screen_height = window1.winfo_screenheight()
    # Find Top-Left (x,y)
    global TL_x,TL_y
    TL_x = int((screen_width/2)-(app_width/2))
    TL_y = int((screen_height/2)-(app_height/2))
    window1.geometry(f'{app_width}x{app_height}+{int(TL_x)}+{int(TL_y)-50}') # "+" means top-left of your GUI
    #window1.geometry("1200x800") # size of window gui
    
    #============== Display Video/Image in GUI ===============#
    VideoCamera_contour = tk.Label(window1, bg = "White")
    VideoCamera_contour.place(x=420,y=30)
  
    checkbox_state1 = tk.IntVar() # store checkbox state 
    checkbox_set1 = tk.Checkbutton(window1, text="SET1 [0-200]",variable=checkbox_state1, command= Def_checkbox_set1,font=('Lato',10,"bold"))
    checkbox_set1.pack()
    #checkbox_set1.select()
    checkbox_set1.place(x=10,y=200)
    
    checkbox_state2 = tk.IntVar() # store checkbox state 
    checkbox_set2 = tk.Checkbutton(window1, text="SET2 [201-400]",variable=checkbox_state2, command= Def_checkbox_set2,font=("Arial",10,"bold"))
    checkbox_set2.pack()
    #checkbox_set2.select()
    checkbox_set2.place(x=145,y=200)
    

    L_year = tk.Label(text="ACADEMIC YEAR").place(x=10, y=225) #label
    E_year = tk.Entry(window1,width=12,font=10) #entry first box
    E_year.place(x=20,y=250)
    #B_year = tk.Button(window1, text="OK",bg='#007fff', font=2,command = ok_year).place(x=95,y=165)
    
    L_semester = tk.Label(text="SEMESTER").place(x=150, y=225) #label
    E_semester = tk.Entry(window1,width=6,font=10) #entry first box
    E_semester.place(x=150,y=250)
    B_Semester = tk.Button(window1, text="OK",bg='#007fff', font=2,command = ok_semester).place(x=225,y=245)
    
    L_subject = tk.Label(text="SUBJECT CODE").place(x=10, y=275) #label
    E_subject = tk.Entry(window1,width=12,font=10) #entry first box
    E_subject.place(x=20,y=300)
    #B_subject = tk.Button(window1, text="OK",bg='#007fff', font=2,command = ok_subject).place(x=95,y=235)
    
    L_section = tk.Label(text="SECTION").place(x=150, y=275) #label
    E_section = tk.Entry(window1,width=7,font=10) #entry first box
    E_section.place(x=150,y=300)
    B_section = tk.Button(window1, text="OK",bg='#007fff', font=2,command = ok_section).place(x=225,y=295)
    
    L1 = tk.Label(text="NO.QUESTION").place(x=10, y=325) #label
    #Label(text="[A5]->1-120 | [A4]->1-200").place(x=50, y=240) #label
    E_numQuest = tk.Entry(window1,width=7,font=5) #entry first box
    E_numQuest.place(x=20,y=350)
    #Label(text=int(E1.get())).place(x=350, y=240)
    B1 = tk.Button(window1, text="OK",bg='#007fff', font=2,command = ok_numQuest).place(x=95,y=345)
    
    L2 = tk.Label(text="NO.PART").place(x=150, y=325)
    E_numPart = tk.Entry(window1,width=7,font=5)
    E_numPart.place(x=150,y=350)
    B2 = tk.Button(window1, text="OK",bg='#007fff', font=2,command = ok_numPart).place(x=225,y=345)
    #label_list,entry_list = ok2(E2.get())
    
    bt1 = tk.Button(window1,text="START",fg="white",bg='blue',height= 2, width=10,command = run).place(x=20, y=730)
    bt2 = tk.Button(window1,text="STOP",fg="white",bg='red',height= 2, width=10,command= stop).place(x=115, y=730)
    bt_clear = tk.Button(window1,text="CLEAR",bg='orange',height= 2, width=10,command=Clear_part)#  font=5
    bt_clear.place(x=210, y=730)
    bt_finish = tk.Button(window1,text="FINISH",fg="white",bg='green',height= 2, width=10,command= finish).place(x=1070, y=730)
    
    #bt3 = tk.Button(window1,text="CAMERA ON",fg="white",bg='blue',height= 2, width=10,command= video_steam).place(x=200, y=620)
    #bt4 = tk.Button(window1,text="CAMERA OFF",fg="white",bg='blue',height= 2, width=10,command= video_off).place(x=300, y=620)
    
    bt_auto = tk.Button(window1,text="AUTO",fg="white",bg='purple',height= 2, width=10,command=auto)
    bt_auto.place(x=220, y=20)
    bt_manual = tk.Button(window1,text="MANUAL",fg="white",bg='orange',height= 2, width=10,command=manual)
    bt_manual.place(x=320, y=20)
    bt_file = tk.Button(window1,text="SELECT CSV FILE",fg="white",bg='green',height= 2, width=23,command=select_file)
    bt_file.place(x=20, y=20)
    bt5 = tk.Button(window1,text="A4",fg="white",bg='#007fff',height= 2, width=10,command=A4)
    bt5.place(x=20, y=80)
    bt6 = tk.Button(window1,text="A5",fg="white",bg='#007fff',height= 2, width=10,command=A5)
    bt6.place(x=120, y=80)
    bt7 = tk.Button(window1,text="ABCD",fg="white",bg='#007fff',height= 2, width=10,command= ABCD)
    bt7.place(x=20, y=140)
    bt8 = tk.Button(window1,text="ABCDE",fg="white",bg='#007fff',height= 2, width=10,command= ABCDE)
    bt8.place(x=120, y=140)
    
    
    scrollbar_box = tk.Scrollbar(window1,orient=VERTICAL)
    scrollbar_box.pack(side=RIGHT,fill=Y)

    L_prop = tk.Label(text="#====== PROPERTY CHECK ======#").place(x=20, y=545)
    box_display = tk.Text(window1,height=7,width=27,bg="gray",yscrollcommand=scrollbar_box.set)#,yscrollcommand=scrollbar_box
    box_display.place(x=20, y=570)
    
    
    #box_display.pack()
    #scrollbar_box.config(command=box_display.yview)
    
    box_display2 = tk.Text(window1,height=10,width=60,bg="White",yscrollcommand=scrollbar_box.set)#,yscrollcommand=scrollbar_box
    box_display2.place(x=420, y=570)
    
    #global pass_count,paper_count,key_count,capture_count,std_count
    #global studentID_list,scoreStd_list
    std_count,repeat_ID,wrong_ID  = 0,0,0
    pass_count ,paper_count,key_count,capture_count = 0,0,0,0 # paper_count -> count normal
    studentID_list,scoreStd_list,no_ansRepeatList = [],[],[]
    reject_paper, reject_cond ,rejectID_count,rejectArea_count,rejectEdge_count = 0,0,0,0,0
    
    window1.mainloop() # must do this
    
    # Get the execution time 
    end_time = time.time()
    elapsed_time = np.round((end_time-start_time),decimals=4)
    print("Execution time:",elapsed_time,"sec")
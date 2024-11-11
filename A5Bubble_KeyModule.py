# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:39:03 2022

@author: Worra
"""

import cv2 
import numpy as np 
import time

start_time = time.time()


def contour_area(img_contour,contours,a1,a2):
    area_list = []
    obj_contour = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > a1 and area < a2:
            area_obj = [area,cnt]
            #print(area)
            area_list.append(area)
            obj_contour.append(cnt)
            cv2.drawContours(img_contour,cnt,-1,(0,255,0),2)
            
    #print("No. in list contour:",len(obj_contour))
    
    return area_list,obj_contour


def corner_center(img_contour,obj_contour):
    corner_list = []
    centerRec_list = []
    CornCen_list = []
    i = 1
    for obj in obj_contour:
        rect = cv2.minAreaRect(obj)
        (x,y),(w,h),angle = rect
        #print(angle)
        centerRec_list.append(np.array((x,y),dtype = "int32"))
        #cv2.circle(img_contour, (int(x),int(y)), 2, (255,0,0), -1)
        
        corner = cv2.boxPoints(rect)# Get 4 corners
        #corner = np.int0(corner)
        corner = np.intp(corner)
        corner_list.append(corner)
        CornCen_temp = [int(x),int(y),corner]
        CornCen_list.append(CornCen_temp)
        #cv2.putText(img_contour,"{}".format(i),(int(x),int(y)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
        i +=1
        
    #return corner_list,centerRec_list,CornCen_list
    return CornCen_list



def sort_CenCorn(CornCen_list,img_contour):
    #sort_CornCenList = CornCen_list.copy()
    global CenCorn_sortY,CenCorn_sortX,sort_CornCenList
    # SORT Y [50 rows]
    CenCorn_sortY = sorted(CornCen_list,key=lambda y:y[1])
    
    
    # SORT X By divide with 20 [4 cols x 5 choice]
    row_cencorn = []
    sort_CornCenList = []
    #sortXY_temp = []
    #num_ele = len(CornCen_list)
    
    for ind1,cencorn1 in enumerate(CenCorn_sortY,1):
        row_cencorn.append(cencorn1)
        if ind1 % 30 == 0:
            CenCorn_sortX = sorted(row_cencorn,key=lambda x:x[0])
            #sortXY_temp.append(CenCorn_sortX)
            sort_CornCenList.append(CenCorn_sortX)
            row_cencorn.clear()
           
    # AFTER SORT 
    count = 0
    for sort_cencorn1 in sort_CornCenList: #[[x1,y1,corner1],[x2,y2,corner2]...[]]
        for ind2,sort_cencorn2 in enumerate(sort_cencorn1): #[x1,y1,corner1]
            x,y = sort_cencorn2[0],sort_cencorn2[1]
            cv2.circle(img_contour, (int(x),int(y)), 2, (255,0,0), -1)
            cv2.putText(img_contour,"{}".format(count),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)
            count +=1
            
    return sort_CornCenList

def warp_answerSmall(warp_img,corner_list,w,h):
    pts1 = np.float32([corner_list[0],corner_list[1],corner_list[2],corner_list[3]])
    #pts2 = np.float32([[0,900],[0,0],[800,0],[800,900]])
    pts2 = np.float32([[0,h],[0,0],[w,0],[w,h]])#normal
    #pts2 = np.float32([[w,0],[0,0],[0,h],[w,h]]) #Key01
    #pts2 = np.float32([[w,0],[w,h],[0,h],[0,0]])#1
    #pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    warp_res = cv2.warpPerspective(warp_img,matrix,(w,h))
    
    return warp_res

def sort_question(warp_ansField_contour,sort_CornCenList):
    sort_cornerC = []
    choice_list = []
    sort_cornerQ = []
    question_list = []
     
    index = 1
    for sort_corner in sort_CornCenList:
        for ind1 in range(len(sort_corner)):# 20 element 
            c = sort_corner[ind1][2]
            # use corner point to get array for image of each choice(200x200)
            #choice = warp_answer(warp_resTh1, c,200,200)
            choice = warp_answerSmall(warp_ansField_contour, c,200,200)
            choice_list.append(choice)
            sort_cornerC.append(c)
            if index % 5 == 0:
                #print("------question {}-----".format(index))
                corn_temp = sort_cornerC.copy()  # must have temp if not when we clear list it will gone all 
                sort_cornerQ.append(corn_temp)
                
                qust_temp = choice_list.copy()  # must have temp if not when we clear list it will gone all 
                question_list.append(qust_temp)
                
                sort_cornerC.clear()
                choice_list.clear()
            index +=1
            
    # Sort Question Zone
    zone1,zone2,zone3,zone4,zone5,zone6 = [],[],[],[],[],[]
    zone1_corn,zone2_corn,zone3_corn,zone4_corn,zone5_corn,zone6_corn = [],[],[],[],[],[]
    
    for index1 in range(len(question_list)):   
        if index1 %6 == 0:
            zone1.append(question_list[index1])
            zone1_corn.append(sort_cornerQ[index1])
            
        elif index1 %6 == 1:
            zone2.append(question_list[index1])
            zone2_corn.append(sort_cornerQ[index1])
           
        elif index1 %6 == 2:
            zone3.append(question_list[index1])
            zone3_corn.append(sort_cornerQ[index1])
            
        elif index1 %6 == 3:
            zone4.append(question_list[index1])
            zone4_corn.append(sort_cornerQ[index1])
            
        elif index1 %6 == 4:
            zone5.append(question_list[index1])
            zone5_corn.append(sort_cornerQ[index1])    
            
        elif index1 %6 == 5:
            zone6.append(question_list[index1])
            zone6_corn.append(sort_cornerQ[index1])
    
    # Sort Question 
    questSort_list = []
    questCorn_list = []
    zone_list = [zone1,zone2,zone3,zone4,zone5,zone6]
    zoneCorn_list = [zone1_corn,zone2_corn,zone3_corn,zone4_corn,zone5_corn,zone6_corn]
    for z1 in range(len(zone_list)):
        zone_num = len(zone_list[z1])
        for z2 in range(zone_num): 
            questSort_list.append(zone_list[z1][z2])
            questCorn_list.append(zoneCorn_list[z1][z2])
            
    return questSort_list,questCorn_list

def quest_pixel(question_list,num_quest,num_choice):
    pixel_list = []
    pixel_temp1 = []
    pixel_questList = []
    for nq in range(num_quest):#120
        for nc in range(num_choice):#5,4
            #cv2.imshow("q{}c{},".format(nq,nc),question_list[nq][nc])
            totalPixel = cv2.countNonZero(question_list[nq][nc])
            pixel_list.append(totalPixel)
            pixel_temp1.append(totalPixel)#clear this
            
        # out loop choice 
        pixel_temp2 = pixel_temp1.copy() # copy for not being clear when clear temp1
        pixel_questList.append(pixel_temp2) # store each 5 choice 
        pixel_temp1.clear() # clear temp1 list
    
    # find max non-pixel in each quest
    pixel_maxList = []
    for nq1 in pixel_questList:
        pixel_max = max(nq1)
        pixel_maxList.append(pixel_max)
    
    # Calculate threshold pass pixel by avg, +-error
    percent_PN = 0.2
    pixel_avg = int(sum(pixel_maxList)/len(pixel_maxList))
    pixel_avgP = int(pixel_avg*(1.1+percent_PN))
    pixel_avgN = int(pixel_avg*(1.0-percent_PN)) #0.98
    
    qst_markList = []
    qst_mark = []
    pass_pixel = []
    percen_pixelList = []
    percen_temp1 = []
    
    
    for nq2 in range(num_quest):#120
        for nCh2 in range(num_choice): # nq2 len = num_quest | nCh len = num_choice
          #cv2.imshow("q{}c{},".format(nq,nc),question_list[nq][nc])
          nonZero_pixel = pixel_questList[nq2][nCh2]
          percen_pixel = np.round((pixel_questList[nq2][nCh2] / pixel_avg),decimals=2)
          percen_temp1.append(percen_pixel)
          if nonZero_pixel >= pixel_avgN and nonZero_pixel <= pixel_avgP: 
              qst_mark.append(1)
              pass_pixel.append(nonZero_pixel)
              
          else:
              qst_mark.append(0)
        
        # Out loop choice
        # Store mark in quest List
        qst_markTemp = qst_mark.copy() # append copy list instead of real list 
        qst_markList.append(qst_markTemp) # must have temp if not when we clear list it will gone all
        qst_mark.clear() # clear real list 
        if qst_markList[nq2] == [0,0,0,0,0] or qst_markList[nq2] == [0,0,0,0]:
            #print("!!!! Error a Pixel Part Warning ----")
            print(nq2+1,": All not pass ")
            
        # Store percen in percen_pixel List
        percen_temp2 = percen_temp1.copy() # Prevent pass by reference[list original will change too]
        percen_pixelList.append(percen_temp2)
        percen_temp1.clear()
    
    pixel_statList = [pixel_avg,pixel_avgP,pixel_avgN,percen_pixelList]
    
    return pixel_questList,qst_markList,pass_pixel,pixel_statList
            
def key_answer(key_markList,questCorn_list):
    key_indexList = []
    #key_choice = []
    key_corner = []
    
    for k in range(len(key_markList)):
        if key_markList[k] == [1,0,0,0] or key_markList[k] == [1,0,0,0,0]:
            key_index = 0
            #key_choice.append(questSort_list[k][key_index])
            key_corner.append(questCorn_list[k][key_index])
            key_indexList.append(key_index)
        elif key_markList[k] == [0,1,0,0] or key_markList[k] == [0,1,0,0,0] :
            key_index = 1
            #key_choice.append(questSort_list[k][key_index])
            key_corner.append(questCorn_list[k][key_index])
            key_indexList.append(key_index)
        elif key_markList[k] == [0,0,1,0] or key_markList[k] == [0,0,1,0,0] :
            key_index = 2
            #key_choice.append(questSort_list[k][key_index])
            key_corner.append(questCorn_list[k][key_index])
            key_indexList.append(key_index)
        elif key_markList[k] == [0,0,0,1] or key_markList[k] == [0,0,0,1,0] :
            key_index = 3
            #key_choice.append(questSort_list[k][key_index])
            key_corner.append(questCorn_list[k][key_index])
            key_indexList.append(key_index)
        elif key_markList[k] == [0,0,0,0,1]:
            key_index = 4
            #key_choice.append(questSort_list[k][key_index])
            key_corner.append(questCorn_list[k][key_index])
            key_indexList.append(key_index)
        else:
            #print(key_markList[k])
            print(k+1,":","Unknow Choice In KeyModule")

    return key_corner,key_indexList

def show_choice(img_key,corner_choice,rgb):
    for c1 in corner_choice:
        cv2.polylines(img_key,[c1],True,rgb,2)
        

def main(img_key,num_question,num_choices):
    print("-----> PROCESSING KEY <-----")
    '''img_key -> [0] = RBG ,[1] = GrayScale, [2] = AdaptiveThreshold [3] = Contour Copy for draw'''
    warp_ansField,warp_resTh1,warp_ansField_contour = img_key[0],img_key[2],img_key[3]

    # find contour each choice
    contours_quest,heirarchy = cv2.findContours(warp_resTh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    areaQ,obj_contourQuest = contour_area(warp_ansField_contour,contours_quest,600,1000) #200,500 
    #print("No. in list contourBig:",len(area_list))
    print("No. in list contourSmall:",len(areaQ))
    CornCen_listQ = corner_center(warp_ansField_contour,obj_contourQuest)
    sort_CornCenList = sort_CenCorn(CornCen_listQ,warp_ansField_contour)
    
    # Part Sort Choice and Question
    # Question(index) -> 5 choice = 1 question  | ALL 120 question 600 choices
    questSort_list,questCorn_list = sort_question(warp_resTh1,sort_CornCenList)
 
    # Part Score
    #show_question(question_list,num_quest,num_choice):
    #pixel_list,qst_markList,pass_pixel = show_question(questSort_list,20,5)
    #pixel_list,key_markList,pass_pixel = show_question(questSort_list,num_question,num_choices)
    pixel_questList,key_markList,pass_pixel,pixel_statList = quest_pixel(questSort_list,num_question,num_choices)
    
    key_corner,key_indexList = key_answer(key_markList,questCorn_list)
    
    img_ShowKey = warp_ansField.copy()     
    show_choice(img_ShowKey,key_corner,(0,255,0))
    #print("--------------------------------")
    
    return key_markList,key_indexList,key_corner
    
    

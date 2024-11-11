# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 00:03:32 2022

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
            #print(area)
            area_list.append(area)
            obj_contour.append(cnt)
            cv2.drawContours(img_contour,cnt,-1,(0,255,0),2)
            
    #print("No. in list contour:",len(obj_contour))
    
    return area_list,obj_contour

def edge_contour(obj_contour):
    edge_list = []
    length_list = []
    for index in range(len(obj_contour)):
        length_con = cv2.arcLength(obj_contour[index],True)
        edge = cv2.approxPolyDP(obj_contour[index],0.02*length_con,True)
        #print(edge)
        #print("--------------")
        edge_list.append(edge)
        length_list.append(length_con)
   
    
    return edge_list, length_list


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
        if ind1 % 20 == 0:
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
            cv2.putText(img_contour,"{}".format(count),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
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
    zone1,zone2,zone3,zone4 = [],[],[],[]
    zone1_corn,zone2_corn,zone3_corn,zone4_corn = [],[],[],[]
    
    for index1 in range(len(question_list)):   
        if index1 %4 == 0:
            zone1.append(question_list[index1])
            zone1_corn.append(sort_cornerQ[index1])
            
        elif index1 %4 == 1:
            zone2.append(question_list[index1])
            zone2_corn.append(sort_cornerQ[index1])
           
        elif index1 %4 == 2:
            zone3.append(question_list[index1])
            zone3_corn.append(sort_cornerQ[index1])
            
        elif index1 %4 == 3:
            zone4.append(question_list[index1])
            zone4_corn.append(sort_cornerQ[index1])
            
     
    
    # Sort Question 
    questSort_list = []
    questCorn_list = []
    zone_list = [zone1,zone2,zone3,zone4]
    zoneCorn_list = [zone1_corn,zone2_corn,zone3_corn,zone4_corn]
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

def cal_score(qst_markList,key_markList,questCorn_list):
    correct_list,wrong_list = [],[]
    index_correct,index_wrong = [],[]
    score_list = []
    no_ansRepeat = []
    score = 0 #correct
    
    # mark = [,,,,]
    for m in range(len(qst_markList)):
        #print(sum(qst_markList[m]))
        if sum(qst_markList[m]) == 1:
            if qst_markList[m] == key_markList[m]:
                #print(m+1,":","✓","--->",qst_markList[m],"=",key_markList[m])
                qst_index = qst_markList[m]
                mark_index = qst_index.index(1)
                index_correct.append(mark_index)
                correct_list.append(questCorn_list[m][mark_index])
                score_list.append(1)
                score += 1
            else:
                #print(m+1,":","✕","--->",qst_markList[m],"!=",key_markList[m])
                qst_index = qst_markList[m]
                mark_index =qst_index.index(1)
                index_wrong.append(mark_index)
                wrong_list.append(questCorn_list[m][mark_index])
                score_list.append(0)
                score +=0
                
        elif sum(qst_markList[m]) == 0:
            print(m+1,":","✕","--->",qst_markList[m],"!=",key_markList[m],"# No answer #")
            no_ansRepeat.append(m+1)
            score_list.append(0)
            score +=0
            
        else:
            print(m+1,":","✕","--->",qst_markList[m],"!=",key_markList[m],"# More than 1 #")
            qst_index = qst_markList[m]
            mark_index = qst_index.index(1)
            index_wrong.append(mark_index)
            no_ansRepeat.append(m+1)
            wrong_list.append(questCorn_list[m][mark_index])
            score_list.append(0)
            score +=0
            
    return score,correct_list,wrong_list,index_correct,index_wrong,score_list,no_ansRepeat

def score_choice(index_correct,index_wrong,key_indexList):
    a_c,b_c,c_c,d_c,e_c = 0,0,0,0,0
    #a_w,b_w,c_w,d_w,e_w = 0,0,0,0,0
    a_all,b_all,c_all,d_all,e_all = 0,0,0,0,0
    for index1 in index_correct:
        if index1 == 0:
            a_c +=1
        elif index1 == 1:
            b_c +=1
        elif index1 == 2:
            c_c +=1
        elif index1 == 3:
            d_c +=1
        elif index1 == 4:
            e_c +=1
            
    for index3 in key_indexList:
        if index3 == 0:
            a_all +=1 
        elif index3 == 1:
            b_all +=1 
        elif index3 == 2:
            c_all +=1 
        elif index3 == 3:
            d_all +=1 
        elif index3 == 4:
            e_all +=1 
    
    num_eachCh = [a_all,b_all,c_all,d_all,e_all]
    choice_correct = [a_c,b_c,c_c,d_c,e_c]
    choice_wrong = [a_all-a_c,b_all-b_c,c_all-c_c,d_all-d_c,e_all-e_c]
    #choice_wrong = [a_w,b_w,c_w,d_w,e_w]
    #num_choice = [a_c+a_w,b_c+b_w,c_c+c_w,d_c+d_w,e_c+e_w]
    
    return choice_correct,choice_wrong,num_eachCh 

def most_choice(choice_correct,choice_wrong):
    correct_max = max(choice_correct)
    wrong_max = max(choice_wrong) 
    most_c = choice_correct.index(correct_max) # index that most correct
    most_w = choice_wrong.index(wrong_max) # index that most wrong
    
    if most_c == 0:
        most_correct ="A"
    elif most_c == 1:
        most_correct ="B"
    elif most_c == 2:
        most_correct ="C"
    elif most_c == 3:
        most_correct ="D"
    elif most_c == 4:
        most_correct ="E"
    
    if most_w == 0:
        most_wrong ="A"
    elif most_w == 1:
        most_wrong ="B"
    elif most_w == 2:
        most_wrong ="C"
    elif most_w == 3:
        most_wrong ="D"
    elif most_w == 4:
        most_wrong ="E"
    
    return most_correct,most_wrong



def show_choice(img_key,corner_choice,rgb):
    for c1 in corner_choice:
        cv2.polylines(img_key,[c1],True,rgb,2)


def Part_score(name_part,QEachPart_list,score_list):
    part_list = {t2 : [] for t2 in name_part} #Dictionary can access by name only name_part["part1"]
    #print("part list:",part_list)
    #print("name part:",name_part)
    #print("QEachPart:",QEachPart_list)
    t3,t4 = 0,0
    for indt,part in enumerate(name_part): # access index in dictionary -> name_part["part2"]
        #print(indt,part,QEachPart_list[indt])
        t3 = t4+QEachPart_list[indt] 
        part_list[part].append(score_list[t4:t3])
        t4 = t4+QEachPart_list[indt]
        
        correct_teach = sum(part_list[part][0]) #access index with str for dict then access index with int for list
        wrong_teach =  QEachPart_list[indt] - correct_teach
        part_list[part].append(correct_teach) 
        part_list[part].append(wrong_teach)
        print(name_part[indt],"--->","✓:",correct_teach,"/",QEachPart_list[indt],"|","✕:",wrong_teach,"/",QEachPart_list[indt])
        
        # # csv file
        # part_pos = df.columns.get_loc(part)
        # #print(part_pos)
        # df.loc[df.index[df_pos],part] = str(correct_teach)+"/"+str(QEachPart_list[indt])
    
    return part_list
    
def main(img_std,key_data,num_questions,num_choices,name_part,QEachPart_list):
    ''' img_stdAns = warp_ansField '''
    warp_ansField,warp_resTh1,warp_ansField_contour = img_std[0],img_std[2],img_std[3]
    key_markList,key_indexList,key_corner = key_data[0],key_data[1],key_data[2]

    # find contour each choice
    contours_quest,heirarchy = cv2.findContours(warp_resTh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    areaQ,obj_contourQuest = contour_area(warp_ansField_contour,contours_quest,600,1000) 
    #print("No. in list contourSmall:",len(areaQ))
    CornCen_listQ = corner_center(warp_ansField_contour,obj_contourQuest)
    sort_CornCenList = sort_CenCorn(CornCen_listQ,warp_ansField_contour)
    
    # Part Sort Choice and Question
    # Question(index) -> 5 choice = 1 question  | ALL 120 question 600 choices
    questSort_list,questCorn_list = sort_question(warp_resTh1,sort_CornCenList)
    #num_key = len()
    
    # Part Score
    #show_question(question_list,num_quest,num_choice):
    #pixel_list,qst_markList,pass_pixel = show_question(questSort_list,20,5)
    pixel_questList,qst_markList,pass_pixel,pixel_statList = quest_pixel(questSort_list,num_questions,num_choices)
    #pixel_list,qst_markList,pass_pixel = show_question(questSort_list,num_question,num_choices)
    
    # print("AVG:",pixel_statList[0],"MAX:",pixel_statList[1],"MIN:",pixel_statList[2])
    # #print("Pixel:",pixel_questList)
    # for ind,p in enumerate(pixel_questList,1):
    #     print(ind,":",p)
    
    num_quest = len(key_markList)
    key_corner = []
    print()
    for k1 in range(num_quest):
        k2 = key_indexList[k1]
        #print(k1,k2)
        key = questCorn_list[k1][k2]
        key_corner.append(key)
        
    score,correct_list,wrong_list,index_correct,index_wrong,score_list,no_ansRepeat  = cal_score(qst_markList,key_markList,questCorn_list)
    wrong = num_quest - score  
    per_correct = (score/num_quest)*100
    per_wrong = (wrong/num_quest)*100
    choice_correct,choice_wrong,num_choice = score_choice(index_correct,index_wrong,key_indexList)
    ch = ["A","B","C","D","E"]
    
    for ind in range(len(choice_correct)):
        if num_choice[ind] == 0:
            num_choice[ind] = 1
            percen_c = 0
            percen_w = 0
        else:
            percen_c = np.round((choice_correct[ind] / num_choice[ind])*100,decimals =2)
            percen_w = np.round((choice_wrong[ind] / num_choice[ind])*100,decimals =2) 
        print(ch[ind],"|","✓:",choice_correct[ind],"/",num_choice[ind],"[",percen_c,"%","]")
        print(" ","|","✕:",choice_wrong[ind],"/",num_choice[ind],"[",percen_w,"%","]")
    
    print("--------------------------------")
    # Teacher Part
    # Teacher_list = teacher_score(teachName_list,QTeach_list,score_list)
    
    # Each Part
    part_list = Part_score(name_part,QEachPart_list,score_list)
    '''part_list -> [0]= detail score , [1] = correct , [2] = wrong '''
    print("--------------------------------")
    
    most_correct,most_wrong = most_choice(choice_correct,choice_wrong)
    print("Most correct:",most_correct,"|","Most wrong:",most_wrong)
    print("Score:",score,"/",num_quest)
    print("Correct:",score,"|","Wrong:",wrong)
    print("Correct[%]:",per_correct,"%","|","Wrong[%]:",per_wrong,"%")
    
    img_ShowKey = warp_ansField.copy()
    show_choice(img_ShowKey,key_corner,(0,0,255))
    show_choice(img_ShowKey,correct_list,(0,255,0)) # Correct 
    #A5Cross_score.show_choice(img_key,wrong_list,(0,0,255)) # Wrong
    
    # For Return
    img_score = [warp_ansField,warp_ansField_contour,img_ShowKey,pixel_statList]
    data_score = [score,wrong,per_correct,per_wrong,part_list,choice_correct,choice_correct,
                  choice_wrong,most_correct,most_wrong,no_ansRepeat]

    # Get the execution time 
    end_time = time.time()
    elapsed_time = np.round((end_time-start_time),decimals=4)
    print("Execution time:",elapsed_time,"sec")


    return img_score,data_score

   
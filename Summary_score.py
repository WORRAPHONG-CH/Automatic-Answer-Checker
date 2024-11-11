# -*- coding: utf-8 -*-
"""
Created on Sat May 20 16:15:02 2023

@author: Worra
"""

from tkinter import *
import tkinter as tk
from tkinter import filedialog

import pandas as pd 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os 

from scipy import stats
import statistics

def select_file1():
    global csv_pathFile1,directory_path1,df1
    # Create a dialog box
    csv_pathFile1 = filedialog.askopenfilename()
    directory_path1 = os.path.dirname(csv_pathFile1)
    print(csv_pathFile1)
    print(directory_path1)
    #directory_path = filedialog.askdirectory()
    text_select = "✓|SELECT FILE SET1"+ "\n"
    box_display.insert(tk.END,text_select)
    box_display.see(tk.END)  # Scroll to the bottom
    df1 = pd.read_csv(str(csv_pathFile1),index_col=False)
    
    # PREPROCESS 
    row_Vmiss,col_Vmiss = np.where(pd.isnull(df1))
    state_null1 = df1.isnull().any()
    num_null1 = df1.isnull().sum()
    
    
    df1 = df1.replace(' ',np.nan)
    # drop rows and cols 
    df1.dropna(inplace=True)
    df1 = df1.drop(df1.columns[0],axis=1)
    df1 = df1.drop('Total',axis=1)
    df1 = df1.reset_index(drop=True)
    # get index of specific column name 
    index_part1_1 = df1.columns.get_loc("Part1")
    
    df1.iloc[:,0:index_part1_1] = df1.iloc[:,0:index_part1_1].astype(str).astype("category")
    df1.iloc[:,index_part1_1:] = df1.iloc[:,index_part1_1:].astype(float)
    
    #print(row_Vmiss,col_Vmiss)
    #print(state_null1)
    print(num_null1)
    
def select_file2():
    global csv_pathFile2,directory_path2,df2
    # Create a dialog box
    csv_pathFile2 = filedialog.askopenfilename()
    directory_path2 = os.path.dirname(csv_pathFile2)
    print(csv_pathFile2)
    print(directory_path2)
    #directory_path = filedialog.askdirectory()
    text_select = "✓|SELECT FILE SET2"+ "\n"
    box_display.insert(tk.END,text_select)
    box_display.see(tk.END)  # Scroll to the bottom
    df2 = pd.read_csv(str(csv_pathFile2),index_col=False)
    
    # PREPROCESS 
    row_Vmiss,col_Vmiss = np.where(pd.isnull(df2))
    state_null2 = df2.isnull().any()
    num_null2 = df2.isnull().sum()
    
    
    df2 = df2.replace(' ',np.nan)
    # drop rows and cols 
    df2.dropna(inplace=True)
    df2 = df2.drop(df2.columns[0],axis=1)
    df2 = df2.drop('Total',axis=1)
    df2 = df2.reset_index(drop=True)
    # get index of specific column name 
    index_part1_2 = df2.columns.get_loc("Part1")
    
    df2.iloc[:,0:index_part1_2] = df2.iloc[:,0:index_part1_2].astype(str).astype("category")
    df2.iloc[:,index_part1_2:] = df2.iloc[:,index_part1_2:].astype(float)
    
    #print(row_Vmiss2,col_Vmiss2)
    #print(state_null2)
    print(num_null2)
    
def cal_all():
    global df1,df2,df3,ID_list1,ID_list2
    if len(df1) == len(df2):
        print("df1 row:{} col:{}".format(len(df1),len(df1.columns)))
        print("df2 row:{} col:{}".format(len(df2),len(df2.columns)))
        print("======= ROWS AND COLUMNS CHECK ======= ")
        box_display.insert(tk.END,"✓|ROWS AND COLUMNS CHECK\n")
        box_display.insert(tk.END,"  [CSV1 rows:{} cols:{}]\n".format(len(df1),len(df1.columns)))
        box_display.insert(tk.END,"  [CSV2 rows:{} cols:{}]\n".format(len(df2),len(df2.columns)))
        box_display.see(tk.END)  # Scroll to the bottom
        
        # Check ID
        ID_list1 = list(df1.iloc[:,0]) # index access all row in Student_ID columns
        ID_list2 = list(df2.iloc[:,0])
        
        equal_state = True
        for ind_ID in range(len(ID_list1)):
            if ID_list1[ind_ID] == ID_list2[ind_ID]:
                print("{}: {} == {}".format(ind_ID,ID_list1[ind_ID],ID_list2[ind_ID]))
                equal_state = True
                
            elif ID_list1[ind_ID] != ID_list2[ind_ID]:
                print("{}: {} != {}".format(ind_ID,ID_list1[ind_ID],ID_list2[ind_ID]))
                box_display.insert(tk.END,"{} != {}\n".format(ID_list1[ind_ID],ID_list2[ind_ID]))
                box_display.see(tk.END)
                equal_state = False
                
        if equal_state == True:
            # Preprocess df3 for add
            box_display.insert(tk.END,"✓|STUDENT ID CHECK\n")
            box_display.see(tk.END)
            df3 = df1.copy()
            original_col = 12 # [ID,Name,Surname,Correct,FALSE,Correct % , False %, A,B,C,D,E]
            num_part_df3 = len(df3.columns)-original_col #  12 -> all col except col part |start col is 3 
            index_part1 = df3.columns.get_loc("Part1")
            index_afterPart = index_part1 + num_part_df3 # for insert df2 col
            #print("df3 before:",index_part1,index_afterPart)
            
            # Preprocess df2 to add new col in df3
            indexDf2_part1 = df2.columns.get_loc("Part1")
            num_part_df2 = len(df2.columns)-original_col # num_part = last_part
            indexDf2_lastPart = df2.columns.get_loc("Part{}".format(num_part_df2))
            #print("index df2:",indexDf2_part1,num_part_df2)
            #print("index last part: ",indexDf2_lastPart)
            
            num_partList = np.arange(1,num_part_df2+1,1) # for use col name 
            #print("part List:",num_partList)
            count_ind = 0
            for ind_p in num_partList:
                new_colName = "Part{}".format(ind_p+num_part_df3)
                col_nameDf2 = "Part{}".format(ind_p)
                df3.insert(index_afterPart+count_ind,new_colName,df2.loc[:,col_nameDf2])
                count_ind +=1
            
            # Check index cols again 
            num_part_df3T2 = len(df3.columns)-original_col # Check num part again
            index_afterPart2 = index_part1 +  num_part_df3T2 # index after last part 
            #print(num_part_df3T2,index_afterPart2)
            col_name = list(df3.columns[index_afterPart2:])
            
            #Sum col after part cols 
            for ind_p,col_p in enumerate(col_name,0): #(start,stop,step)
                for row1 in range(len(df3)):
                    df3.loc[row1,col_p] = df1.loc[row1,col_p] + df2.loc[row1,col_p]
                    #print(row1,col_p,':',df3.iloc[row1,col_p])
                    
            # Calculate percentage only Correct % and False %
            for row2 in range(len(df3)):
                percent_correct = (df3.loc[row2,"Correct"]/(df3.loc[row2,"Correct"] + df3.loc[row2,"FALSE"]))*100
                percent_false = (df3.loc[row2,"FALSE"]/(df3.loc[row2,"Correct"] + df3.loc[row2,"FALSE"]))*100
                df3.loc[row2,"Correct %"] = np.round(percent_correct,decimals=2)
                df3.loc[row2,"False %"] = np.round(percent_false,decimals=2)
                
                #(df3.loc[0,"Correct"]/(df3.loc[0,"Correct"] + df3.loc[0,"FALSE"]))*100
                 
                      
            box_display.insert(tk.END,"✓|CALCULATE FINISH\n")
            box_display.see(tk.END)  # Scroll to the bottom
        
        else:
            box_display.insert(tk.END,"✕|STUDENT ID CHECK\n")
            
            
            box_display.insert(tk.END,"### CANNOT CALCULATE SCORE ### \n")
            box_display.insert(tk.END,"!!! RECHECK CSV FILE AGAIN !!!\n")
            box_display.see(tk.END)
        
    else:
        box_display.insert(tk.END,"✕|ROWS AND COLUMNS CHECK\n")
        box_display.insert(tk.END,"!!! PLEASE SELECT AGAIN !!!\n")
        box_display.see(tk.END)  # Scroll to the bottom
        

def finish():
    global df3,scoreStd_list,directory_path1,directory_path2,csv_pathFile1,csv_pathFile2
    scoreStd_list = list(df3.loc[:,"Correct"])
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
    list_series = [pd.Series(["NO. Student",len(scoreStd_list)],index=df3.columns[:2]),
                    pd.Series(["MAX",score_max],index=df3.columns[:2]),
                    pd.Series(["MIN",score_min],index=df3.columns[:2]),
                    pd.Series(["MEAN",score_mean],index=df3.columns[:2]),
                    pd.Series(["MEDIAN",score_median],index=df3.columns[:2]),
                    pd.Series(["MODE",score_mode],index=df3.columns[:2]),
                    pd.Series(["SD",score_sd],index=df3.columns[:2])]
    
    df3 = df3.append(list_series,ignore_index=True)
    
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

    # =================== VISUALIZE DATA ===================== #
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
    plt.subplot(122)
    plt.grid()
    plt.hist(scoreStd_list,bins=10,edgecolor='black')
    plt.title("RANGE SCORE")
    plt.xlabel("RANGE SCORE"),plt.ylabel("STUDENT")
    plt.show()
    
    # --------- SAVE FILE --------- #
    filename1_text = os.path.basename(csv_pathFile1) 
    filename2_text = os.path.basename(csv_pathFile2)
    
    filename1 = os.path.splitext(filename1_text)[0] #split file name form .csv
    filename2 = os.path.splitext(filename2_text)[0]
    
    
    plt.savefig(os.path.join(directory_path1,filename1 +"_SET1-2.png"))
    plt.savefig(os.path.join(directory_path2,filename2 +"_SET1-2.png"))
    path_csv1 = os.path.join(directory_path1,filename1 +"_SET1-2.csv")
    path_csv2 = os.path.join(directory_path2, filename2 +"_SET1-2.csv")
    df3.to_csv(path_csv1,index = False)
    df3.to_csv(path_csv2,index = False)
    print("\n=============== SAVE TO CSV FILE ===============")
    window1.destroy()
    
    
if __name__ == "__main__":
    window1 = tk.Tk()
    window1.title("SUMMMARY") #named window gui
    # ============ CONFIGURATION GUI ============ #
    app_width,app_height = 350,500
    screen_width = window1.winfo_screenwidth()
    screen_height = window1.winfo_screenheight()
    # Find Top-Left (x,y)
    #global TL_x,TL_y
    TL_x = int((screen_width/2)-(app_width/2))
    TL_y = int((screen_height/2)-(app_height/2))
    window1.geometry(f'{app_width}x{app_height}+{int(TL_x)}+{int(TL_y)-50}') # "+" means top-left of your GUI
    #window1.geometry("1200x800") # size of window gui
    
    bt_file = tk.Button(window1,text="SELECT CSV FILE SET1",fg="black",font=("Arial",8,"bold"),bg='orange',height= 2, width=30,command=select_file1)
    bt_file.place(x=40, y=200)
    
    bt_file2 = tk.Button(window1,text="SELECT CSV FILE SET2",fg="black",font=("Arial",8,"bold"),bg='purple',height= 2, width=30,command=select_file2)
    bt_file2.place(x=40, y=260)
    
    bt_cal = tk.Button(window1,text="CALCULATE",fg="black",font=("Arial",8,"bold"),bg='blue',height= 2, width=30,command=cal_all)
    bt_cal.place(x=40, y=320)
    
    bt_finish = tk.Button(window1,text="FINISH",fg="black",font=("Arial",8,"bold"),bg='green',height= 2, width=30,command=finish)
    bt_finish.place(x=40, y=380)
    
    scrollbar_box = tk.Scrollbar(window1,orient=VERTICAL)
    scrollbar_box.pack(side=RIGHT,fill=Y)
    
    L_prop = tk.Label(text="#====== PROPERTY CHECK ======#").place(x=40, y=0)
    box_display = tk.Text(window1,height=7,width=30,bg="white",yscrollcommand=scrollbar_box.set)#,yscrollcommand=scrollbar_box
    box_display.place(x=20, y=30)
    
    window1.mainloop() # must do this

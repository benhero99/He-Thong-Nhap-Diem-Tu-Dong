from distutils.command import bdist
from tkinter import *
from tkinter.font import BOLD, ITALIC
from tkinter.ttk import *
from tkinter import messagebox
from tkinter import filedialog
import tkinter
from PIL import Image, ImageTk
import PIL

import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

from helper import preprocess
from segment import wordSegmentation
from Preprocessing import imformation_crop, removeline, removecircle
from digit_model import build_digit_model
from word_model import build_word_model
from Excel import class_list,lexicon_search,writing_to_excel

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


############# INITIAL CONFIG ################
with open('data\scoreList.txt', "r", encoding="utf-8") as f:
    reader = f.read()
scoreDict = sorted(reader.split(' '))

def num_to_label(num,alphabets):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret

#### RESTORE MODEL ###


#Name model
with open('data/charList.txt', 'r', encoding='utf-8') as f:
    alphabets_word = f.read()

max_str_len_word = 15
word_model, word_model_CTC = build_word_model(alphabets = alphabets_word, max_str_len = max_str_len_word)
#word_model.summary()
word_model_dir = r'model\word_model\word_model_last_6.h5'
word_model.load_weights(word_model_dir)

## MSSV model
alphabets_digit = '0123456789'
max_str_len_digit = 10
digit_model, digit_model_CTC = build_digit_model(alphabets = alphabets_digit, max_str_len = max_str_len_digit)
#digit_model.summary()
digit_model_dir = r'model\digit_model\digit_model_last_2022-07-05.h5'
digit_model.load_weights(digit_model_dir)
###############################################################    


################# RECOGNITION ###################
def btnStart():
    global class_list_dir, cap, index_confirmed,diem_confirmed
    URL_path = txt_URL.get()
    #'http://192.168.1.242:8080/video'
    # URL_path = r'data/Class_list_constrained/giaythi50.mp4'
    # URL_path = 'doc\giaythi54.png'
    cap = cv2.VideoCapture(URL_path)
    diem_confirmed =[]
    index_confirmed =[]
    monitor = []
    f = 0
    save[0] = False

    name_list, MSSV_list, name_MSSV_list, Diem_list = class_list(class_list_dir)
    while(cap.isOpened()):
        f+=1
        if save [0] or quit [0]:
            break
        ret, img = cap.read()
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        window.update()

        if f != 5: #Cứ 5 frame thì mới nhận dạng một lần
            cv2.imshow('video', cv2.resize(img, None, fx=0.4, fy=0.4))
            continue
        
        f=0
        giaythi  = img.copy()
        (MSSV_crop, name_crop, diem_crop) = imformation_crop(giaythi)
        if name_crop == []:
            #print ('cannot extract in4')
            continue

        ############### NAME_RECOGNITION ##########################

        name_crop_copy = removeline(name_crop)
        result = wordSegmentation(name_crop_copy, kernelSize=21, sigma=11, theta=4, minArea=500)
        name_recognized = str()

        for line in result:
            if len(line):
                for (_, w) in enumerate(line):
                    (wordBox, wordImg) = w
                    wordImg = preprocess(wordImg, imgSize = (128, 32))
                    wordImg = np.array(wordImg).reshape(-1, 128, 32, 1)
                    pred = word_model.predict(wordImg)
                    
                    decoded = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                                greedy=False,
                                                beam_width=50,
                                                top_paths=1)[0][0])
                    
                    name_recognized += num_to_label(decoded[0], alphabets = alphabets_word) + ' ' 

        ############### MSSV_RECOGNITION #######################
        MSSV_crop_copy = removeline(MSSV_crop)
        MSSV_crop_copy = preprocess(MSSV_crop_copy,(128,32))
        MSSV_crop_copy = np.array(MSSV_crop_copy).reshape(-1, 128, 32, 1)
        pred_MSSV = digit_model.predict(MSSV_crop_copy)

        decoded_MSSV = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(pred_MSSV, input_length=np.ones(pred_MSSV.shape[0])*pred_MSSV.shape[1], 
                                                greedy=False,
                                                beam_width=5,
                                                top_paths=1)[0][0])

        MSSV_recognized = num_to_label(decoded_MSSV[0], alphabets = alphabets_digit) 

        name_MSSV_recognized = name_recognized.strip() + ' ' + MSSV_recognized.strip()        
        name_MSSV_index, name_MSSV_recognized, name_MSSV_dis = lexicon_search (name_MSSV_recognized, name_MSSV_list)
        
        if name_MSSV_dis > int(0.85*len(name_MSSV_recognized)):
            #print ('name_MSSV_dis',name_MSSV_dis,'\nlen' ,len(name_MSSV_recognized))
            continue
        if name_MSSV_index in index_confirmed:
            print ('Điểm số đã được cập nhật cho {}: {}'
                .format(name_MSSV_recognized, diem_confirmed[index_confirmed.index(name_MSSV_index)]))
            
            lbl_result.configure(text ='Điểm số đã được cập nhật cho {}: {}'
                .format(name_MSSV_recognized, diem_confirmed[index_confirmed.index(name_MSSV_index)] ))
            continue
        
        print ('\nTên_MSSV:',name_MSSV_recognized)

        ############### DIEM_RECOGNITION #######################

        diem_crop_copy = removecircle(diem_crop)
        diem_crop_copy = preprocess(diem_crop_copy, imgSize = (128, 32))
        diem_crop_copy = np.array(diem_crop_copy).reshape(-1, 128, 32, 1)
        pred_diem = digit_model.predict(diem_crop_copy)

        decoded_diem = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(pred_diem, input_length=np.ones(pred_diem.shape[0])*pred_diem.shape[1], 
                                                greedy=False,
                                                beam_width=5,
                                                top_paths=1)[0][0])

        diem_recognized = num_to_label(decoded_diem[0], alphabets = alphabets_digit) 
        _, diem_recognized,_ = lexicon_search (diem_recognized, scoreDict)

        if diem_recognized != '10':
            diem_recognized = diem_recognized[:1]+ '.' + diem_recognized[1:]
        diem_recognized = float(diem_recognized)    
        print ('Điểm số:',diem_recognized)

        # Kiểm tra nếu cứ 4 lần liên tiếp nhận dạng giống nhau thì sau này không cần cập nhật nữa
        index_diem =[name_MSSV_index,diem_recognized]
        if not monitor:
            monitor.append (index_diem)
        else:
            if index_diem == monitor[-1]:
                monitor.append(index_diem)
            else:
                monitor =[]  

        if len(monitor)==3:
            index_confirmed.append(monitor[0][0])
            diem_confirmed.append(monitor[0][1])
            monitor = []

        cv2.putText(img,str(diem_recognized)+'      ' +str(name_MSSV_recognized[-7:]), 
            (15,100), cv2.FONT_HERSHEY_DUPLEX, 3, (255,255,255),3)
        cv2.imshow('video', cv2.resize(img, None, fx=0.4, fy=0.4))
        cv2.waitKey(1)

    print ('\nindex confirmed',index_confirmed)
    print ('diem confirmed',diem_confirmed)
    for i,index in enumerate(index_confirmed):
        name_MSSV_index = index
        diem_recognized = diem_confirmed[i] 
        writing_to_excel (class_list_dir, name_MSSV_index + 2,diem_recognized)
    cap.release()
    cv2.destroyAllWindows()
    if quit[0]:
        print ('ĐÃ LƯU DỮ LIỆU VÀ THOÁT CHƯƠNG TRÌNH!')
        lbl_result.configure(text ='ĐÃ LƯU DỮ LIỆU VÀ THOÁT CHƯƠNG TRÌNH!')
        window.quit()
        window.destroy()
    return  


############################# GUI ####################################
def browseFiles():
    global class_list_dir
    class_list_dir = filedialog.askopenfilename(parent=window,
        initialdir = r"C:\KLTN_Project\data/",
                                          title = "Select a File",
                                          filetypes = (("Excel files",
                                                        "*.xlsx*"),
                                                       ("all files",
                                                        "*.*")))
    root_dir= r'C:\KLTN_Project/'
    class_list_dir = class_list_dir[len(root_dir):]
    label_class_list_dir.configure(text=class_list_dir)

def btnSave(save):
    save[0] = True   
    print ('ĐÃ LƯU DỮ LIỆU!')
    lbl_result.configure(text ='ĐÃ LƯU DỮ LIỆU!')


def btnQuit(quit): 
    global index_confirmed, diem_confirmed
    quit[0] = True 
    if (quit[0] == True) and (save[0] == True):
        print ('ĐÃ LƯU DỮ LIỆU VÀ THOÁT CHƯƠNG TRÌNH!')
        window.quit()
        window.destroy()

    

window = Tk()
window.title("                                             HỆ THỐNG NHẬP ĐIỂM TỪ ẢNH BÀI THI")
window.geometry("700x600")
window.configure(bg='lavender')
save = [False]
quit = [False]
# Logo
image1 = Image.open("images\spkt.png")
image1 = image1.resize((100, 100))
logo1 = ImageTk.PhotoImage(image1)

label1 = tkinter.Label(image=logo1)
label1.image = logo1
label1.grid(column=0, row=0)

image2 = Image.open("images\cce.png")
image2 = image2.resize((100, 100))
logo2 = ImageTk.PhotoImage(image2)

label2 = tkinter.Label(image=logo2)
label2.image = logo2
label2.place(x=600,y=0)

# Thêm label
lb = tkinter.Label(window, text="ĐẠI HỌC SƯ PHẠM KỸ THUẬT", fg="black",bg ="lavender", font=("Arial", 20, BOLD))
lb.place(x=115, y=0)

lb = tkinter.Label(window, text="TP HCM", fg="black",bg ="lavender",border = 1, font=("Arial", 20, BOLD))
lb.place(x=290, y=50)

lb = tkinter.Label(window, text="KHOA ĐIỆN - ĐIỆN TỬ", fg="black",bg ="lavender",border = 1, font=("Arial", 20, BOLD))
lb.place(x=180, y=90)

lb = tkinter.Label(window, text="HỆ THỐNG NHẬP ĐIỂM TỰ ĐỘNG", fg="black",bg ="lavender", font=("Arial", 20, ITALIC))
lb.place(x=90, y=160)

lb = tkinter.Label(window, text="TỪ ẢNH BÀI THI", fg="black",bg ="lavender", font=("Arial", 20, ITALIC))
lb.place(x=220, y=210)

lb = tkinter.Label(window, text="Nhập đường dẫn:", fg="black",bg ="lavender", font=("Arial", 15))
lb.place(x=50, y=300)

lb = tkinter.Label(window, text="Video/URL:", fg="black",bg ="lavender", font=("Arial", 15))
lb.place(x=50, y=350)
txt_URL = Entry(window, width=50)
txt_URL.place(x=230, y=355)

lb = tkinter.Label(window, text="Danh sách lớp:", fg="black",bg ="lavender", font=("Arial", 15))
lb.place(x=50, y=400)

label_class_list_dir = tkinter.Label(window, text="...", fg="black",bg ="lavender", font=("Arial", 15))
label_class_list_dir.place(x=340, y=405)

lbl_result =  tkinter.Label(window, text="", fg="black",bg ="lavender", font=("Arial", 10))
lbl_result.place(x=30, y=550)


# Thêm button

btnStart = Button(window, text="Bắt đầu", command=btnStart)
btnStart.place(x=100, y=480)

buttonQuit = Button(window, text="Thoát", command=lambda: btnQuit(quit))
buttonQuit.place(x=500, y=480)

buttonSave = Button(window, text="Lưu", command=lambda: btnSave(save))
buttonSave.place(x=300, y=480)

btnexplore = Button(window,
                        text = "Chọn tệp",
                        command = browseFiles)
btnexplore.place(x=230, y=405)


window.mainloop()
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
import tkinter
from PIL import Image, ImageTk

from helper import preprocess
from segment import wordSegmentation

from Preprocessing import imformation_crop, removeline, removecircle
from digit_model import build_digit_model
from word_model import build_word_model
from Excel import class_list,lexicon_search,writing_to_excel

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



#Đọc video
#http://192.168.1.126:8080/video
#data/video\giaythi1.mp4
#class_list_dir = 'data\Class_list.xlsx'


def Start_camera(URL_path,class_list_dir, window, cap ):
    global index_confirmed,diem_confirmed
    #cap = cv2.VideoCapture(URL_path)
    ############# Initial Config
    diem_confirmed =[]
    index_confirmed =[]
    monitor = []
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

    ############## RESTORE MODEL ##########

    #Name model
    with open('data/charList.txt', 'r', encoding='utf-8') as f:
        alphabets_word = f.read()

    max_str_len_word = 15
    word_model, word_model_CTC = build_word_model(alphabets = alphabets_word, max_str_len = max_str_len_word)
    #word_model.summary()
    word_model_dir = 'model\word_model\word_model_last_6.h5'
    word_model.load_weights(word_model_dir)

    ## MSSV model
    alphabets_digit = '0123456789'
    max_str_len_digit = 10
    digit_model, digit_model_CTC = build_digit_model(alphabets = alphabets_digit, max_str_len = max_str_len_digit)
    #digit_model.summary()
    digit_model_dir = 'model\digit_model\digit_model_last_2022-07-05.h5'
    digit_model.load_weights(digit_model_dir)

    name_list, MSSV_list, name_MSSV_list, Diem_list = class_list(class_list_dir)

    
    f = 0

    while(cap.isOpened()):
        f+=1
        ret, img = cap.read()
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        window.update()
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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

        # Kiểm tra nếu cứ 5 lần liên tiếp nhận dạng giống nhau thì sau này không cần cập nhật nữa
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    return index_confirmed,diem_confirmed
import tkinter as tk
import cv2
import os, time
import csv
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import sys
from sys import platform
import argparse
import json
import math
from json import load
from typing import Dict
from matplotlib.pyplot import show
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, ConvLSTM2D, Flatten, MaxPooling2D, Dropout,BatchNormalization, Activation, TimeDistributed , Conv2D, LSTM, SpatialDropout2D
from keras.utils import np_utils, to_categorical
from sklearn import preprocessing
from os import listdir
from keras.optimizers import RMSprop
from tensorflow import keras
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


window = tk.Tk()
window.title('Demo')
window.geometry("300x500+250+150")

# 標示文字
label = tk.Label(window, text = 'Step1_video', font = ('bold',20))
label.pack()

# 輸入欄位
entry = tk.Entry(window, width = 20) # 輸入欄位的寬度
entry.pack()


def captureVideo():
    for i in range(1):
        startTime=0.0
        end = 0.0
        cap = cv2.VideoCapture(0)
        cap.set(3,800)#寬
        cap.set(4,1200)#高
        sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # 為儲存視訊做準備
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fps=30
        out = cv2.VideoWriter( 'D:/realData/tmp//'+ entry.get()+'_demo.avi' , fourcc,fps,sz)

        startTime = time.time()
        print(type(startTime))
        print(startTime)

    while True:
        # 一幀一幀的獲取影象
        ret,frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 1)
            # 在幀上進行操作
            # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # 開始儲存視訊
            out.write(frame)
            # 顯示結果幀
            end = time.time()
            total = end - startTime
            cv2.imshow("frame", frame)
            print(total)
            cv2.waitKey(1)
            if total > 3:
                break
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break
    # 釋放攝像頭資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 按鈕
button = tk.Button(window, text = "Video", command = captureVideo, bd = 2)
button.pack()


def videoToFrame():
    path = 'D:/realData' # 路徑自己改
    files = listdir(path+'/tmp')
    for file in files:
        print(file)
        cap = cv2.VideoCapture(path+'/tmp/'+file)
        # length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        videoName = file

        currentFrame = 0
        if not os.path.isdir(path+'/tmp/'+ videoName):
            os.mkdir(path+'/tmpFrame/'+ videoName)
            print("----------start "+videoName+'----------')
            while(currentFrame < cap.get(cv2.CAP_PROP_FRAME_COUNT)):  # get frame 
                # path = 'D:/sl_data/train_data/'+ videoName
                # if not os.path.isdir(path):
                #     os.mkdir(path)
                ret, frame = cap.read()   # Capture frame-by-frame 
                # Saves image of the current frame in jpg file
                name = path+'/tmpFrame/'+ videoName +'/frame' + str(currentFrame) + '.jpg'
                print ('Creating...' + name)
                cv2.imwrite(name, frame)
                # To stop duplicate images
                currentFrame += 1
        else:
            print("------exist-------")
        print("----------end "+videoName+'----------')
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
            

label2 = tk.Label(window, text = 'Step_2 frame', bd = 10, font = ('bold',20))
label2.pack()

button2 = tk.Button(window, text = "Get", command = videoToFrame)
button2.pack()


def frameToJson():
    
    frame_path = "D:/realData/tmpFrame" # frame address (data_1~data_4)
    json_path = "D:/realData/tmpJson" # 儲存 json 檔的資料夾

    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

    files = listdir(frame_path)

    for f in files: # f：frame_filename
        # if i < 22: continue  # 這邊是如果用gpu跑有噴錯的話，可以先把"沒存完的手語"的那個資料夾整個刪掉，然後看是第幾個手語，再從該手語開始run
        fullPath = join(frame_path,f)
        fullFiles = listdir(fullPath)
        os.mkdir(json_path + "/" + f) #建立該手語的資料夾
        for frame in fullFiles:
            print(fullPath+'/'+frame)
            # print(fullPath)
            # print(i[:-4])
            try:
                # Flags
                parser = argparse.ArgumentParser()
                parser.add_argument("--image_path", default= fullPath+"/"+frame, help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
                args = parser.parse_known_args()

                # Custom Params (refer to include/openpose/flags.hpp for more parameters)
                params = dict()
                params["model_folder"] = "../../../models/"
                # params["face"] = True
                params["hand"] = True
                params['write_json'] = json_path + "/" + f #儲存json檔的參數，只要設定儲存路徑 (檔名要到 datum.name 那邊改 )

                opWrapper = op.WrapperPython()
                opWrapper.configure(params)
                opWrapper.start()

                # Process Image
                datum = op.Datum()
                datum.name = str(f) + "_" + frame[:-4] #更改 json 檔案名稱
                imageToProcess = cv2.imread(args[0].image_path)
                datum.cvInputData = imageToProcess
                opWrapper.emplaceAndPop([datum])

                cv2.waitKey(0)
            except Exception as e:
                print(e)
                sys.exit(-1)


label3 = tk.Label(window, text = 'Step_3 Skeleton', bd = 10, font = ('bold',20))
label3.pack()

button3 = tk.Button(window, text = "Get", command = frameToJson)
button3.pack()




def jsonToMatrix():

    frame_path = "D:/realData/tmpJson"
    npz_path = "D:/realData/tmpMatrix"

    if os.path.isdir(npz_path):
        pass
    else:
        os.mkdir(npz_path)
    files = listdir(frame_path)
    for file in files:
        jsons_path = frame_path + '/' + file
        jsons = listdir(jsons_path)
        # print(jsons_path)

        one_sl_list = [] # 每個手語
        idx = 0
        jsons.sort(key = lambda x: int(expression(x)))
        for sl_json in jsons:
            json_path = jsons_path + '/' + sl_json
            # print(json_path)
            one_sl_list.append([])
            
            # 加總身體和手部的關節點
            tmp_list = []
            pose_list = body_data_to_list(json_path, 'pose_keypoints_2d')            
            Lhand_list = data_to_list(json_path, 'hand_left_keypoints_2d')
            Rhand_list = data_to_list(json_path, 'hand_right_keypoints_2d')

            tmp_list = pose_list + Lhand_list + Rhand_list
            frame_corre_list = corre_matrix(tmp_list)

            # print(pose_corre_list.shape)

            one_sl_list[idx].append(frame_corre_list)

            idx+=1
        one_sl_list = np.array(one_sl_list)
        print(one_sl_list.shape)
        # print('one sl : ',one_sl_list.shape)
        # print('each sl pose matrix',one_sl_list[0][0].shape)
        # print('each sl Lhand matrix',one_sl_list[0][1].shape)
        # print('each sl Rhand matrix',one_sl_list[0][2].shape)
        np.savez_compressed(npz_path+'/'+file+'.npz', sl_arr=one_sl_list) # one sl one npz
        print(file,' done')

def expression(x):
        y = x.split('_')[2]
        print(x.split('_'))
        return y[5:]

def body_data_to_list(json_path,data_name):
    with open(json_path, 'r') as f:
        data = json.load(f)
        data = data['people'][0]
        keypoints_data = data[data_name]
        keypoints_list = []
        keypoints_list.append([])
        point = 0
        for i, keypoint in enumerate(keypoints_data):
            if ((i >= 0 and i <= 26) or (i >= 45 and i <= 56)):   #  判斷關節點
                
                # print(i)
                if i % 3 == 2:
                    keypoints_list.append([])
                elif i % 3 == 0:
                    keypoints_list[point//3].append(keypoint)
                elif i % 3 == 1:
                    keypoints_list[point//3].append(keypoint)  
                point = point+1
                    
        keypoints_list = keypoints_list[:-1]
    return keypoints_list

def data_to_list(json_path,data_name):
    with open(json_path, 'r') as f:
        data = json.load(f)
        data = data['people'][0]
        keypoints_data = data[data_name]
        keypoints_list = []
        keypoints_list.append([])

        for i, keypoint in enumerate(keypoints_data):
            if i % 3 == 2:
                keypoints_list.append([])
            elif i % 3 == 0:
                keypoints_list[i//3].append(keypoint)
            elif i % 3 == 1:
                keypoints_list[i//3].append(keypoint)

        keypoints_list = keypoints_list[:-1]
    return keypoints_list

def corre_matrix(keypoints_list):
    corre_list = []
    idx = 0
    for x in keypoints_list:
        corre_list.append([])
        for y in keypoints_list:
            if (x[0] == 0 or x[1] == 0 or y[0] == 0 or y[1] == 0):
                dis = 0
            else:
                dis = math.sqrt(((x[0]-y[0])**2) + ((x[1]-y[1])**2))
            corre_list[idx].append(dis)
        idx += 1
    return np.array(corre_list)


label4 = tk.Label(window, text = 'Step_4 Matrix', bd = 10, font = ('bold',20))
label4.pack()

button4 = tk.Button(window, text = "Get", command = jsonToMatrix)
button4.pack()


def result():
    path = 'D:/realData/tmpMatrix'
    files = listdir(path)
    max = 0
    x = []
    y = []
    x_test = []
    y_test = []

    for file in files:
        y_l = file.split('_')
        y_test.append(y_l[0])
        # print(y_l[0])
        i = np.load(path + '/' + file)
        i = i['sl_arr']
        i = i.reshape(len(i),55, 55,1)
        if len(i) > max:
            max = len(i)
            print(y_l[0],max)
        x_pad = np.pad(array=i, pad_width=((0,(90-len(i))),(0,0),(0,0),(0,0)), mode='constant', constant_values=0) # padding time = max 
        x_test.append(x_pad)


    label_dict = {0:'eat', 1:'goodbye', 2:'hello', 3:'helpme', 4:'morning', 5:'night', 6:'please', 7:'sleep', 8:'sorry', 9:'thanks', 10:'welcome'}
    labels=['eat', 'goodbye', 'hello', 'helpme', 'morning', 'night', 'please', 'sleep', 'sorry', 'thanks', 'welcome']


    le = preprocessing.LabelEncoder() 
    y_label_encoded = le.fit_transform(y_test)
    print(y_label_encoded)
    y_test_onehot = keras.utils.to_categorical(y_label_encoded)
    x_test = np.array(x_test)
    y_test_onehot = np.array(y_test_onehot)
    # y_test_OneHot = to_categorical(y_test)
    print('x_test:',x_test.shape)
    print('y_test:',y_test_onehot)



    model = tf.keras.models.load_model("D:/realData/model_1031.h5")
    # scores = model.evaluate(x_test,y_test_onehot)
    prediction = model.predict_classes(x_test)
    print(prediction)
    lab = label_dict[prediction[0]]
    # print(lab)
    result_label.configure(text=lab, bg='yellow')


label5 = tk.Label(window, text = 'Step_5 Result', bd = 10, font = ('bold',20))
label5.pack()

button5 = tk.Button(window, text = "Get", command = result)
button5.pack()


result_label = tk.Label(window, fg='red', font = ('bold',30))
result_label.pack()





window.mainloop()
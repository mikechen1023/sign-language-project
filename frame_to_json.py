import os
from os import listdir
from os.path import isfile, isdir, join
import sys
import cv2
import os
from sys import platform
import argparse
# import torch

'''
說明：
1. 路徑有兩個要更改
    - frame_path : 連到切好的 frame 資料夾
    - json_path：連到要儲存 json 檔的資料夾 (程式會自動建立一個資料夾，再把該手語每個 frame 的 json 檔存進去)

2. listdir 的格式好像是 binary 的，要注意一下

3. 把這個.py檔放到 openpose/build/examples/tutorial_python 裡面run

'''
frame_path = "D:/sl_data/data/data_1" # frame address (data_1~data_4)
json_path = "D:/sl_data/data/data_1_json" # 儲存 json 檔的資料夾

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

for i, f in enumerate(files): # f：frame_filename
    # if i < 22: continue  # 這邊是如果用gpu跑有噴錯的話，可以先把"沒存完的手語"的那個資料夾整個刪掉，然後看是第幾個手語，再從該手語開始run
    fullPath = frame_path + '/' + f
    fullFiles = listdir(fullPath)
    os.mkdir(json_path + "/" + f) #建立該手語的資料夾
    for frame in fullFiles:
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
            # params["bool"] = True

            # # Add others in path?
            # for i in range(0, len(args[1])):
            #     curr_item = args[1][i]
            #     if i != len(args[1])-1: next_item = args[1][i+1]
            #     else: next_item = "1"
            #     if "--" in curr_item and "--" in next_item:
            #         key = curr_item.replace('-','')
            #         if key not in params:  params[key] = "1"
            #     elif "--" in curr_item and "--" not in next_item:
            #         key = curr_item.replace('-','')
            #         if key not in params: params[key] = next_item

            # Construct it from system arguments
            # op.init_argv(args[1])
            # oppython = op.OpenposePython()

            # Starting OpenPose
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()

            # Process Image
            datum = op.Datum()
            datum.name = str(f) + "_" + frame[:-4] #更改 json 檔案名稱
            imageToProcess = cv2.imread(args[0].image_path)
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop([datum])

            # Display Image
            # print("Body keypoints: \n" + str(datum.poseKeypoints))
            # print("Face keypoints: \n" + str(datum.faceKeypoints))
            # print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
            # print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
            # cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)

            
            # cv2.imwrite(fullPath+"/"+i, datum.cvOutputData)
            cv2.waitKey(0)
        except Exception as e:
            print(e)
            sys.exit(-1)
    # torch.cuda.empty_cache()

from os import listdir
from os.path import isfile, isdir, join
from pytube import YouTube
import cv2 
import csv
import numpy as np
import os

path = "D:/ASL/train_data"
files = listdir(path)
fileName = []
q=0
w=0
for f in files:
    fullpath = join(path, f)
    fullfiles = listdir(fullpath)
    if(fullfiles == []):
        fileName.append(f)
        q+=1
        # print(f)
        # print(fullpath)

with open('D:/ASL/train.csv',encoding='utf-8') as csvfile:
    rows = csv.DictReader(csvfile)
    for row in rows:
        end = int(row['end'])
        start = int(row['start'])
        videoName = row['file'].strip()
        if(videoName in fileName) :
            videoPath = 'D:/ASL/train_video/'+row['file']+'.mp4'
            cap = cv2.VideoCapture(videoPath) 
            currentFrame = 0 
            print("----------start "+videoName+'----------')
            while(currentFrame < cap.get(cv2.CAP_PROP_FRAME_COUNT)):  # get frame 
                path = 'D:/ASL/train_data/'+ videoName
                if not os.path.isdir(path):
                    os.mkdir(path)
                ret, frame = cap.read()   # Capture frame-by-frame 

                # Saves image of the current frame in jpg file
                if currentFrame >= start and currentFrame <= end: 
                    name = 'D:/ASL/train_data/'+ videoName +'/frame' + str(currentFrame) + '.jpg'
                    print ('Creating...' + name)
                    cv2.imwrite(name, frame)

                    # To stop duplicate images
                currentFrame += 1
            print("----------end "+videoName+'----------')
            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()
            
                




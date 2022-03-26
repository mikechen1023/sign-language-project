from pytube import YouTube
import cv2 
import csv
import numpy as np
import os


# #載影片
# def download_video(vname,url):
#     try:
#         yt = YouTube(url)
#         video=yt.streams.filter(file_extension='mp4').first()
#         video.download(r'D:/ASL/train_video',filename=vname)
#         print("-----Download "+vname+" successful------")
#     except:
#         print("error "+vname + " ： " + url)




with open('D:/sl_data/train.csv',encoding='utf-8') as csvfile:
    rows = csv.DictReader(csvfile)
    for row in rows:
        # download_video(row['file'],row['url'])
        end = int(row['end'])
        start = int(row['start'])
        videoName = row['clean_text']+"_"+row['file']
        videoPath = 'D:/sl_data/train_video/'+row['file']+'.mp4'


        cap = cv2.VideoCapture(videoPath) 
        currentFrame = 0 
        path = 'D:/sl_data/train_data/'+ videoName
        if not os.path.isdir(path):
            os.mkdir(path)
            print("----------start "+videoName+'----------')
            while(currentFrame < cap.get(cv2.CAP_PROP_FRAME_COUNT)):  # get frame 
                # path = 'D:/sl_data/train_data/'+ videoName
                # if not os.path.isdir(path):
                #     os.mkdir(path)
                ret, frame = cap.read()   # Capture frame-by-frame 

                # Saves image of the current frame in jpg file
                if currentFrame >= start and currentFrame <= end: 
                    name = 'D:/sl_data/train_data/'+ videoName +'/frame' + str(currentFrame) + '.jpg'
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
        

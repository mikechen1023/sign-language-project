import json
import numpy as np
import math
import os
from os import listdir

'''
說明:
    1. 兩個路徑要改:
        - frame_path : 轉換好 json 的資料夾路徑
        - npz_path : 要儲存 npz 的路徑

    2. concatenated：
        - sl_data (包含之前的影片和code)
            - data (純資料)
                - data_1 (frame, 裡面一種手語一個資料夾)
                - data_1_json
                - data_1_npz (一個手語一個npz)

    3. matrix shape
        - (frames數量, 1, 67, 67)
'''

frame_path = "D:/sl_data/data/data_1"
npz_path = "D:/sl_data/data/data_1_npz"

def main():
    if os.path.isdir(npz_path):
        pass
    else:
        os.mkdir('D:/sl_data/data/data_1_npz')
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
            pose_list = data_to_list(json_path, 'pose_keypoints_2d')            
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
        y = x.split('_')[1]
        return y[5:]

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
            dis = math.sqrt(((x[0]-y[0])**2) + ((x[1]-y[1])**2))
            corre_list[idx].append(dis)
        idx += 1
    return np.array(corre_list)


if __name__ == "__main__":
    main()

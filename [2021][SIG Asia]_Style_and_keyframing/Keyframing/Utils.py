import numpy as np
import math as mt
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

def extract_txt(data_path):
    if os.path.exists(data_path + ".npy"):
        return np.load(data_path + '.npy', allow_pickle=True)[()]

    unity_list = ['Camera', 'Toric', 'Actor',
                  'Head', 'Neck', 'LeftArm', 'LeftForeArm', 'LeftHand',
                  'RightArm', 'RightForeArm', 'RightHand', 'Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot',
                  'RightUpLeg', 'RightLeg', 'RightFoot']

    Dict = dict()

    File = open(data_path + ".txt")
    text = File.read()
    text = text.split('\n')
    Length = int(len(text))

    for name in unity_list:
        Dict[name] = []

    for j in tqdm(range(Length)):
        data = text[j].split(' ')

        Name = data[0]
        if len(Name) > 9:
            Name = Name[9:]

        if Name in unity_list:
            n = len(data) - 1
            d = np.zeros(n, dtype="float32")
            for i in range(n):
                d[i] = eval(data[i + 1])
            Dict[Name].append(d)
    Data = {"Skelet": np.array(Dict.copy())}

    np.save(data_path, Data)

    return Data

def vect_dist(v):
    return np.linalg.norm(v)

def vect_cross(v1, v2):
    result = (v1 * v2).sum() / (vect_dist(v1) * vect_dist(v2))
    return result

def extract_prediction_feature(skelet, actor, returnpos=False):
    skelet = skelet[()]
    frame = len(skelet['Toric'])
    Data = np.zeros((frame, 7 + 2), dtype="float32")
    Camera = np.zeros((frame, 5), dtype="float32")
    pos = np.zeros((frame, 6), dtype="float32")
    idx = [2, 4]
    for i in range(frame):
        p1 = i * 2 + 0
        p2 = i * 2 + 1
        head1 = skelet['Head'][p1][idx] - skelet['Neck'][p1][idx]
        head2 = skelet['Head'][p2][idx] - skelet['Neck'][p2][idx]
        shoulder1 = skelet['LeftArm'][p1][idx] - skelet['RightArm'][p1][idx]
        shoulder2 = skelet['LeftArm'][p2][idx] - skelet['RightArm'][p2][idx]
        Line = skelet['Hips'][p1][idx] - skelet['Hips'][p2][idx]
        pos[i][:3] = skelet['Head'][p1][2:]
        pos[i][3:] = skelet['Head'][p2][2:]
        Data[i][0] = vect_dist(skelet["Hips"][p1][idx] - skelet["Hips"][p2][idx])
        Data[i][1] = vect_cross(head1, head2)
        Data[i][2] = vect_cross(shoulder1, shoulder2)
        Data[i][3] = vect_cross(shoulder1, Line)
        Data[i][4] = vect_cross(shoulder2, Line)
        Data[i][5] = vect_cross(head1, Line)
        Data[i][6] = vect_cross(head2, Line)

        Data[i][7 + actor] = 1

        for j in range(5):
            Camera[i][0] = skelet['Toric'][i][0]
            Camera[i][1] = skelet['Toric'][i][1]
            Camera[i][2] = skelet['Toric'][i][2:3].mean()
            Camera[i][3] = skelet['Toric'][i][4]
            Camera[i][4] = skelet['Toric'][i][5]

    if returnpos:
        return Data, Camera, pos

    return Data, Camera

def data_load(data_path, actor, returnpos=False):
    skelet = extract_txt(data_path)
    return extract_prediction_feature(skelet['Skelet'], actor, returnpos)

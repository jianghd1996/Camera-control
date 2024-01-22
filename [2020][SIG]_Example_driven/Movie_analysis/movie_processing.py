import torch
import os
import json
import numpy as np
import math as mt
import cv2
from Net import combined_CNN
from Net import Prediction

def load_ckpt(model, ckpt):
    # load checkpoint
    ckpt = torch.load(ckpt)
    checkpoint = ckpt['state_dict']
    new_state_dict = dict()

    for k, v in checkpoint.items():
        name = k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

def Match(v1, v2):
    # Judge if v1, v2 belong to the same characters
    v1 = v1.flatten()
    v2 = v2.flatten()
    skelet_num = len(v1)
    num = 0
    for i in range(skelet_num):
        if v1[i] != 0 and v2[i] != 0 and abs(v1[i] - v2[i]) < 0.05:
            num += 1
        if v1[i] != 0 and v2[i] != 0 and abs(v1[i]-v2[i]) > 0.05:
            num -= 1
    return num > 2

def far_skelet(v):
    # Judge if the character is far from the screen (delete if so)
    x = v[::2]
    y = v[1::2]
    x = x.ravel()[np.flatnonzero(x)]
    y = y.ravel()[np.flatnonzero(y)]
    return max(x)-min(x) < 0.1 or max(y) - min(y) < 0.1

def skeleton_processing(skeleton, height, width):
    # extract and match skeleton in movie clips
    # input skeleton of each frame in list
    # height, width is used to normalize coordinates to [0, 1]
    frame = len(skeleton)
    skelet = []
    select = []
    for i in range(frame):
        f_s = []
        s = []
        for v in skeleton[i]['people']:
            d = v['LCRKeypoint_2d']
            for i in range(len(d)):
                if i % 2 == 0:
                    d[i] /= width
                else:
                    d[i] /= height
            f_s.append(np.array(d))
            s.append(0)
        skelet.append(f_s)
        select.append(s)

    character_skelet = []
    for i in range(frame):
        for j in range(len(skelet[i])):
            if far_skelet(skelet[i][j]):
                select[i][j] = 1

    for i in range(frame):
        for j in range(len(skelet[i])):
            if select[i][j] == 0:
                character = []
                select[i][j] = 1
                character.append([i, skelet[i][j]])
                for k in range(i, frame):
                    for l in range(len(skelet[k])):
                        if select[k][l] == 0 and Match(character[-1][1], skelet[k][l]):
                            select[k][l] = 1
                            if k != character[-1][0]:
                                character.append([k, skelet[k][l]])
                    if k - character[-1][0] > 20:
                        break
                if len(character) > 30:
                    character_skelet.append(character)

    if len(character_skelet) < 2:
        return None
    A = 0
    B = 0
    MaxLength = 0
    for i in range(len(character_skelet)):
        for j in range(i+1, len(character_skelet)):
            length = min(character_skelet[i][-1][0], character_skelet[j][-1][0]) - max(character_skelet[i][0][0], character_skelet[j][0][0])
            if length > MaxLength:
                A, B, MaxLength = i, j, length

    L = max(character_skelet[A][0][0], character_skelet[B][0][0])
    R = min(character_skelet[A][-1][0], character_skelet[B][-1][0])

    idx = [A, B]

    skelet_num = 7

    candidate_skelet = np.zeros((2, frame, skelet_num*2))

    for t in range(2):
        for v in character_skelet[idx[t]]:
            candidate_skelet[t][v[0]] = v[1]

    for t in range(2):
        for j in range(skelet_num*2):
            l = character_skelet[idx[t]][0][0]
            r = character_skelet[idx[t]][-1][0]
            for i in range(l+1, r+1):
                if candidate_skelet[t][i][2+j%2] == 0:
                    candidate_skelet[t][i][2+j%2] = candidate_skelet[t][i-1][2+j%2]
                if candidate_skelet[t][i][j] == 0 and candidate_skelet[t][i-1][j] != 0:
                    delt = candidate_skelet[t][i-1][j] - candidate_skelet[t][i-1][2+j%2]
                    candidate_skelet[t][i][j] = candidate_skelet[t][i-1][2+j%2] + delt
    if candidate_skelet[0][L][0] < candidate_skelet[1][L][0]:
        return [candidate_skelet[1][L:R + 1], candidate_skelet[0][L:R + 1]], L, R

    return [candidate_skelet[0][L:R+1], candidate_skelet[1][L:R+1]], L, R

def draw(img, skelet, color):
    # used for visualize skeleton on image
    height, width = img.shape[1], img.shape[0]
    for i in range(7):
        x, y = int(skelet[i*2+0] * height), int(skelet[i*2+1] * width)
        cv2.circle(img, (x, y), 5, color, -1)
    link = [[0, 1], [1, 2], [1, 3], [1, 4], [4, 5], [4, 6]]
    for p, q in link:
        xp, yp = int(skelet[p*2+0] * height), int(skelet[p*2+1] * width)
        xq, yq = int(skelet[q*2+0] * height), int(skelet[q*2+1] * width)
        cv2.line(img, (xp, yp), (xq, yq), color, 3)

def visualize(video, skeleton, st, ed, save_name, size, fps=30):
    # visualize extracted skeletons in video from st to ed frame
    vidcap = cv2.VideoCapture(video)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, size)

    cnt = 0
    while (True):
        success, image = vidcap.read()
        if (success == False):
            break
        if cnt >= st:
            draw(image, skeleton[0][cnt-st], (0, 0, 255))
            draw(image, skeleton[1][cnt-st], (255, 0, 0))
            video_writer.write(image)
        if cnt >= ed:
            break
        cnt += 1
    video_writer.release()

def get_movie_skelet(movie_path):
    # extract movie skeletons from movies
    if os.path.exists("movie_skelet.npy"):
        return np.load("movie_skelet.npy", allow_pickle=True)
    category = os.listdir(movie_path)
    result = []
    for cate in category:
        skeleton_path = os.path.join(movie_path, cate, "LCRKeypoint")
        video_path = os.path.join(movie_path, cate, 'raw')
        index = os.listdir(skeleton_path)
        for idx in index:
            frame = os.listdir(os.path.join(skeleton_path, idx))
            frame.sort()
            skeleton = []
            for f in frame:
                file = open(os.path.join(skeleton_path, idx, f), 'r')
                data = json.load(file)
                skeleton.append(data)
            video = os.path.join(video_path, idx+'.mp4')
            vidcap = cv2.VideoCapture(video)
            success, image = vidcap.read()
            width, height = image.shape[1], image.shape[0]
            skeleton = skeleton_processing(skeleton, height, width)
            if skeleton != None:
                skeleton, st, ed = skeleton
                result.append({"name" : os.path.join(video_path, idx),
                               "skeleton" : skeleton,
                               "size" : (height, width),
                               "frame" : (st, ed)})
            else:
                print(os.path.join(video_path, idx), " Not enough skeleton")
    np.save("movie_skelet", result)
    return result

def extract_movie_feature(data, type_name, size):
    # used for estimation network, extract multiple types of skeletons for estimation
    fovy = 45.0
    height, width = size
    frame = len(data[0])
    skelet_num = 7
    seq_length = 4

    std_width, std_height, std_fovy = 971.0, 528.0, 45.0 / 180.0 * mt.pi
    std_fovx = 2 * mt.atan(std_width / std_height * mt.tan(std_fovy / 2))
    std_aspect_ratio = std_width / std_height
    fovy = fovy / 180.0 * mt.pi
    fovx = 2 * mt.atan(width / height * mt.tan(fovy / 2))
    aspect_ratio = width / height

    skelet_2d = np.zeros((frame, skelet_num*2*2), dtype="float32")
    for i in range(frame):
        for j in range(skelet_num):
            for p in range(2):
                skelet_2d[i][p * skelet_num * 2 + 2* j+0] = (data[p][i][2*j+0]-0.5) * mt.tan(fovx/2) / mt.tan(std_fovx/2)
                skelet_2d[i][p * skelet_num * 2 + 2 * j + 1] = (data[p][i][2 * j + 1] - 0.5) * mt.tan(fovx/2) / mt.tan(std_fovx/2) / aspect_ratio * std_aspect_ratio

        if (type_name == "no_pY_"):
            posY = (skelet_2d[i][1] + skelet_2d[i][2*skelet_num+1]) / 2
            for j in range(skelet_num):
                for p in range(2):
                    skelet_2d[i][p * skelet_num * 2 + 2 * j + 1] -= posY
        elif (type_name == "no_pX_"):
            posX = (skelet_2d[i][0] + skelet_2d[i][2*skelet_num+0]) / 2
            for j in range(skelet_num):
                for p in range(2):
                    skelet_2d[i][p * skelet_num * 2 + 2 * j + 0] -= posX
        elif (type_name == "no_all_"):
            posX = (skelet_2d[i][0] + skelet_2d[i][2*skelet_num+0]) / 2
            posY = (skelet_2d[i][1] + skelet_2d[i][2*skelet_num+1]) / 2
            for j in range(skelet_num):
                for p in range(2):
                    skelet_2d[i][p * skelet_num * 2 + 2 * j + 0] -= posX
                    skelet_2d[i][p * skelet_num * 2 + 2 * j + 1] -= posY

    Input_data = []
    for i in range(frame):
        S = np.zeros((2 * seq_length, skelet_num * 4), dtype="float32")
        for j in range(2 * seq_length):
            for k in range(skelet_num * 4):
                l = min(frame - 1, max(0, i - seq_length + j))
                S[j][k] = skelet_2d[l][k]
        Input_data.append(np.transpose(S))

    return Input_data

def get_movie_feature():
    # estimate toric and character features from skeletons
    if os.path.exists("movie_feature.npy"):
        return np.load("movie_feature.npy", allow_pickle=True)
    skeleton = get_movie_skelet("Movie_data")
    seq_length = 4
    model = combined_CNN(seq_length=2 * seq_length, channels=28)

    load_ckpt(model, "Estimation.tar")

    estimation_result = []

    for v in skeleton:
        data0 = extract_movie_feature(v["skeleton"], "raw", size = v["size"])
        data1 = extract_movie_feature(v["skeleton"], "no_pX", size = v["size"])
        data2 = extract_movie_feature(v["skeleton"], "no_pY", size = v["size"])
        data3 = extract_movie_feature(v["skeleton"], "no_all", size = v["size"])

        frame = len(data0)
        result = np.zeros((frame, 12), dtype="float32")
        for j in range(frame):
            result[j][0] = data0[j][0][4] * 2  # x0
            result[j][1] = data0[j][14][4] * 2  # x1
            result[j][2] = (-data0[j][1][4] * 2 + -data0[j][15][4] * 2) / 2# y0
        data0 = torch.tensor(data0)
        data1 = torch.tensor(data1)
        data2 = torch.tensor(data2)
        data3 = torch.tensor(data3)

        with torch.no_grad():
            output = model([data0, data1, data2, data3]).cpu().numpy()

        for j in range(frame):
            result[j][3:] = output[j][:]

        print(v["name"])

        estimation_result.append({"name" : v["name"],
                                  "frame" : v["frame"],
                                  "size": v["size"],
                                  "feature" : result})
    np.save("movie_feature", estimation_result)
    return estimation_result

def get_movie_latent():
    # get movie features from skeletons
    if os.path.exists("movie_latent.npy"):
        return np.load("movie_latent.npy", allow_pickle=True)
    movie_feature = get_movie_feature()
    model = Prediction(num_experts=9)
    load_ckpt(model, "Gating_prediction.tar")

    movie_latent = []
    for v in movie_feature:
        frame = len(v["feature"])
        data = np.zeros((frame, 9+5), dtype="float32")
        for i in range(frame):
            data[i][:7] = v["feature"][i][5:]
            data[i][9:] = v["feature"][i][:5]
        for actor in range(2):
            for i in range(frame):
                data[i][7+actor] = 1
                data[i][8-actor] = 0
            input = torch.tensor(data).unsqueeze(0)
            latent = model(input).detach().numpy()
            movie_latent.append({"latent" : latent[-1],
                                 "name" : v["name"],
                                 "frame" : v["frame"],
                                 "size" : v["size"],
                                 "actor" : actor})
    np.save("movie_latent", movie_latent)
    return movie_latent

if __name__ == "__main__":
    get_movie_latent()

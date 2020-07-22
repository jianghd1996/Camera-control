import numpy as np
import os
import math as mt
from tqdm import tqdm


def extract_txt(data_path):
    if os.path.exists(os.path.join(data_path, "toric_data.npy")):
        return np.load(os.path.join(data_path, "toric_data.npy"), allow_pickle=True, encoding="latin1")


    file_path = os.path.join(data_path, "data")

    Unity_list = ['Camera', 'Toric', 'Actor',
                  'Head', 'Neck', 'LeftArm', 'LeftForeArm', 'LeftHand',
                  'RightArm', 'RightForeArm', 'RightHand', 'Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot',
                  'RightUpLeg', 'RightLeg', 'RightFoot']

    Dict = dict()
    Data = []

    List = os.listdir(file_path)
    List.sort()

    for i in range(len(List)):
        path = os.path.join(file_path, List[i])
        print("loading "+path)
        with open(path, "r") as f:
            text = f.read()
            f = f.read()
            text = text.split('\n')
            Length = int(len(text))

            for name in Unity_list:
                Dict[name] = []

            frame = 0

            for j in range(Length):
                data = text[j].split(' ')
                Name = data[0]

                if Name == "Camera":
                    frame += 1
                # first several frames delete
                if frame < 5:
                    continue

                if Name in Unity_list:
                    n = len(data)-1
                    d = np.zeros(n, dtype="float32")
                    for k in range(n):
                        d[k] = eval(data[k+1])
                    if Name == "Toric" and n == 6:
                        d = np.array([d[0], d[1], (d[2]+d[3])/2, d[4], d[5]])
                    Dict[Name].append(d)

            Data.append({"Skelet" : np.array(Dict.copy())})

    np.save(os.path.join(data_path, "toric_data"), Data)

    return Data

def vect_dist(v):
    return np.linalg.norm(v)

def vect_cross(v1, v2):
    result = (v1 * v2).sum() / (vect_dist(v1)*vect_dist(v2))
    return result

def deal(data, type):
    new_data = np.array(data)
    frame = new_data.shape[0]
    skelet_num = new_data.shape[2] // 2
    if type in ["no_pX", "no_all"]:
        for i in range(frame):
            posX = (data[i][0][0] + data[i][1][0]) / 2
            for p in range(2):
                for j in range(skelet_num):
                    new_data[i][p][2*j+0] -= posX
    if type in ["no_pY", "no_all"]:
        for i in range(frame):
            posY = (data[i][0][1] + data[i][1][1]) / 2
            for p in range(2):
                for j in range(skelet_num):
                    new_data[i][p][2*j+1] -= posY
    return new_data

def generate(data, idx, seq_length):
    frame = data.shape[0]
    skelet_num = data.shape[2] // 2
    S = np.zeros((2*seq_length, skelet_num*4), dtype="float32")
    for j in range(2*seq_length):
        l = min(frame-1, max(0, idx-seq_length+j))
        S[j][:skelet_num*2] = data[l][0]
        S[j][skelet_num*2:] = data[l][1]
    return np.transpose(S)

def extract_estimation_feature(load_path, save_path, refresh=False):
    print(load_path, save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if os.path.exists(os.path.join(save_path, "toric_estimation_data.npy")):
        return

    data = extract_txt(load_path)

    need_unity = ["Head", "Neck", "LeftArm",
                  "RightArm", "Hips", "LeftUpLeg", "RightUpLeg"]
    skelet_num = len(need_unity)

    idx = [2, 4]

    width, height = 971.0, 528.0
    seq_length = 4

    Input_raw = []
    Input_no_pX = []
    Input_no_pY = []
    Input_no_all = []
    Label = []

    for d in tqdm(data):
        skelet = d["Skelet"][()]
        frame = len(skelet["Toric"])

        raw_data = np.zeros((frame, 2, 2*skelet_num), dtype="float32")
        for i in range(frame):
            for p in range(2):
                for j in range(skelet_num):
                    raw_data[i][p][2*j+0] = skelet[need_unity[j]][2*i+p][0] / width - 0.5
                    raw_data[i][p][2*j+1] = skelet[need_unity[j]][2*i+p][1] / height - 0.5

        no_pX = deal(raw_data, "no_pX")
        no_pY = deal(raw_data, "no_pY")
        no_all = deal(raw_data, "no_all")

        valid = np.zeros(frame)
        for i in range(frame):
            if max(raw_data[i].max(), no_pX[i].max(), no_pY.max(), no_all.max()) > 1 or min(raw_data[i].min(), no_pX[i].min(), no_pY[i].min(), no_all[i].min()) < -1:
                valid[i] = 1

        # label
        label = np.zeros((frame, 5+7), dtype="float32")
        for i in range(frame):
            if len(skelet["Toric"][i]) == 5:
                label[i][:5] = skelet["Toric"][i]
            else:
                label[i][:2] = skelet["Toric"][i][:2]
                label[i][3] = skelet["Toric"][i][2:4].mean()
                label[i][4:6] = skelet["Toric"][i][4:]

            p1 = i*2+0
            p2 = i*2+1
            head1 = skelet['Head'][p1][idx] - skelet['Neck'][p1][idx]
            head2 = skelet['Head'][p2][idx] - skelet['Neck'][p2][idx]
            shoulder1 = skelet['LeftArm'][p1][idx] - skelet['RightArm'][p1][idx]
            shoulder2 = skelet['LeftArm'][p2][idx] - skelet['RightArm'][p2][idx]
            Line = skelet['Hips'][p1][idx] - skelet['Hips'][p2][idx]

            label[i][5] = vect_dist(skelet["Hips"][p1][idx] - skelet['Hips'][p2][idx])
            label[i][6] = vect_cross(head1, head2)
            label[i][7] = vect_cross(shoulder1, shoulder2)
            label[i][8] = vect_cross(shoulder1, Line)
            label[i][9] = vect_cross(shoulder2, Line)
            label[i][10] = vect_cross(head1, Line)
            label[i][11] = vect_cross(head2, Line)

        for i in range(frame):
            L = max(i-seq_length, 0)
            R = min(i+seq_length, frame-1)
            if valid[L:R+1].max() == 0:
                Input_raw.append(generate(raw_data, i, seq_length))
                Input_no_pY.append(generate(no_pY, i, seq_length))
                Input_no_pX.append(generate(no_pX, i, seq_length))
                Input_no_all.append(generate(no_all, i, seq_length))
                Label.append(label[i])

    np.save(os.path.join(save_path, "toric_estimation_data"), [[Input_raw, Input_no_pY, Input_no_pX, Input_no_all], Label])


if __name__ == "__main__":
    path = [
        "../estimation_data/fov45_10degree/",
		"../estimation_data/new_data/",
		"../estimation_data/new_close_character/",
		"../estimation_data/fov45_sin_cos/",
		"../estimation_data/fov45_sin_cos_2/",
		"../estimation_data/fov45_parallel/",
		"../estimation_data/fov45_complement/",
		"../estimation_data/fov45_side/",
        # "../generality/",
        # "../generality_rotation/",
        # "../generality_rotation_2/",
        # "../generality_test/"
        ]

    for v in path:
        extract_estimation_feature(v, os.path.join("data", v.split("/")[-2]))

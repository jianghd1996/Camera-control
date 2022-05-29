import torch
import os
import numpy as np
from Net import Prediction, Shared_bottom
from tqdm import tqdm
import matplotlib.pyplot as plt
import math as mt

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

def extract_prediction_feature(skelet, actor, returnpose=False):
    skelet = skelet[()]
    frame = len(skelet['Head']) // 2
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

    if returnpose:
        return Data, pos

    return Data

def data_load(data_path, actor, returnpose=False):
    skelet = extract_txt(data_path)
    return extract_prediction_feature(skelet['Skelet'], actor, returnpose)

def extract_latent(style, shot, actor):
    variaty = 11

    style, p = style

    alpha = dict()
    alpha[(-75, 75)] = 1.0381
    alpha[(-75, 25)] = 0.7072
    alpha[(-50, 50)] = 0.7278
    alpha[(-25, 75)] = 0.7072
    alpha[(-50, 0)] = 0.3639
    alpha[(0, 50)] = 0.3639

    if style == "direct":
        data_path = 'data/gt/direct_{}_actor{}'.format(shot, actor)
        style_data, style_toric = data_load(data_path, actor, returncamera=True)
        style_seq = np.concatenate((style_data, style_toric), axis=1)

        # relative
    elif style == "relative":
        data_path = 'data/gt/relative_{}_actor{}'.format(shot, actor)
        style_data, style_toric = data_load(data_path, actor, returncamera=True)
        print(style_toric[0], style_toric[1])
        style_seq = np.concatenate((style_data, style_toric), axis=1)

        # relative
    elif style == "side":
        data_path = 'data/gt/direct_{}_actor{}'.format(shot, actor)

        style_data, style_toric = data_load(data_path, actor, returncamera=True)
        pA = int(round(style_toric[0][1] * 100))
        pB = int(round(style_toric[0][0] * 100))
        r = 2 * (mt.pi - alpha[(pA, pB)])
        frame = len(style_data)
        for i in range(frame):
            if actor == 0:
                style_toric[i][3] = r * (0.8 + (p - (variaty // 2)) * 0.02)
            else:
                style_toric[i][3] = r * (0.2 + (p - (variaty // 2)) * 0.02)
        style_seq = np.concatenate((style_data, style_toric), axis=1)

        # relative
    elif style == "sin":
        data_path = 'data/gt/direct_{}_actor{}'.format(shot, actor)

        style_data, style_toric = data_load(data_path, actor, returncamera=True)
        pA = int(round(style_toric[0][1] * 100))
        pB = int(round(style_toric[0][0] * 100))
        r = 2 * (mt.pi - alpha[(pA, pB)])
        frame = len(style_data)
        for i in range(frame):
            style_toric[i][3] = r / 2.0 + (1 + p * 0.1) * mt.sin(2 * mt.pi * i / (100 + p * 20))
        style_seq = np.concatenate((style_data, style_toric), axis=1)

    global_input = torch.tensor(style_seq[:]).unsqueeze(0)
    return model(global_input, 0).detach().cpu().numpy()

def smooth(seq):
    seq = np.transpose(seq)

    frame = len(seq[0])
    feature = len(seq)

    tot = 8
    while (tot > 0):
        tot -= 1
        for i in range(feature):
            data = np.array(seq[i])

            for j in range(frame):
                seq[i][j] = data[max(0, j - 10):min(frame, j + 10)].mean()

    seq = np.transpose(seq)

    return seq

def get_style_code(style, shot, actor):
    style, p = style
    data = np.load("style_weights.npy", allow_pickle=True)[()]
    return data[(style, p, shot, actor)]

if __name__ == "__main__":
    ### initialize model
    seq_length = 60
    num_experts = 9
    input_size = 1380
    output_size = 150
    model = Prediction(num_experts=num_experts,
                       input_size=input_size,
                       output_size=output_size)

    checkpoint = torch.load('different_shot_perfect.tar')
    checkpoint = checkpoint['state_dict']
    new_state_dict = dict()
    for k, v in checkpoint.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = model.eval()


    ###
    data, pos = data_load("D:/screenshot/scene", actor=1, returnpose=True)
    frame = len(data)

    result_camera = np.zeros((frame, 5), dtype="float32")
    control_weight = np.zeros((frame, num_experts), dtype="float32")
    result_camera[0][:] = [0.5, -0.5, 0, 4.0, 3.14]

    # manually set the style of each segments
    # N = 13
    # st =    [0,                  270,            450,            575,            600,            1050,           1150,           1350,           1450,               1750,           1800,           2000,          2150]
    # ed =    [270,                450,            575,            600,            1050,           1150,           1350,           1450,           1750,               1800,           2000,           2150,          2300]
    # actor = [1,                  0,              0,              1,              1,              1,              1,              0,              1,                  1,              1,              1,             1]
    # style = [['direct', 0],     ['side', 10],   ['side', 10],   ['side', 0],     ['side', 0],   ['side', 10],   ['sin', 0],     ['side', 5],   ['relative', 0],    ['side', 0],   ['side', 5],    ['direct', 0],   ['relative', 0]]
    # shot =  ['medium_right',     'medium_left',  'medium_left',  'close',       'medium_mid',   'medium_right', 'far_left',     'close',      'far_right',        'medium_mid',   'far_right',    'far_right',      'far_right']

    N = 1
    st = [0]
    ed = [250]
    actor = [1]
    style = [['direct', 0]]
    shot = ['medium_mid']

    ### load latent vector
    for i in tqdm(range(N)):
        style_weights = get_style_code(style[i], shot[i], actor[i])
        for j in range(st[i], ed[i]):
            control_weight[j][:] = style_weights

    control_weight = smooth(control_weight)
    pos = smooth(pos)

    print('generate sequence')
    for i in tqdm(range(frame)):
        cw = torch.tensor(control_weight[i])
        local_skelet = np.zeros((120, 9), dtype="float32")
        local_camera = np.zeros((60, 5), dtype="float32")
        for j in range(120):
            idx = min(max(i + j - 60, 0), frame - 1)
            local_skelet[j][:] = data[idx]
        for j in range(60):
            idx = max(i + j - 60, 0)
            local_camera[j][:] = result_camera[idx]
        local_input = np.concatenate((local_skelet.flatten(), local_camera.flatten()))
        global_input = torch.tensor(control_weight).unsqueeze(0)
        local_input = torch.tensor(local_input).unsqueeze(0)
        camera = model(cw, local_input).detach().numpy()

        result_camera[i][:] = camera[0][:5]

    if not os.path.exists('visual'):
        os.mkdir('visual')

    axis_label_font_size = 40
    legend_fontsize = 10
    title_fontsize = 35
    x_label = 'frame'
    y_label = 'value'
    result_camera = result_camera.transpose(1, 0)
    pos = pos.transpose(1, 0)

    X = np.arange(frame)

    param = ['pA', 'pB', 'pY', 'theta', 'phi']
    for i in range(5):
        fig = plt.figure(figsize=(12.5, 10))
        plt.xlabel(x_label, fontsize=axis_label_font_size)
        plt.ylabel(y_label, fontsize=axis_label_font_size)
        plt.tick_params(labelsize=25)
        plt.title(param[i], fontsize=title_fontsize)
        p_result, = plt.plot(X, result_camera[i], color='green', linewidth=4)

        '''
        plt.legend([p_std, p_result],
                   [
                       "ground truth",
                       "our result",
                   ],
                   loc="upper right",
                   fontsize=legend_fontsize,
                   )
        '''
        plt.savefig('visual/' + param[i] + '.png')

    '''
    for i in range(num_experts):
        fig = plt.figure(figsize=(12.5, 10))
        plt.xlabel(x_label, fontsize=axis_label_font_size)
        plt.ylabel(y_label, fontsize=axis_label_font_size)
        plt.tick_params(labelsize=25)
        plt.title('expert{}'.format(i), fontsize=title_fontsize)
        p_result, = plt.plot(X, control_weight[i], color='green', linewidth=4)
        plt.savefig('visual/' + 'expert{}'.format(i) + '.png')
    '''

    output = ''
    for i in range(frame):
        for j in range(6):
            output = output + '{} '.format(pos[j][i])
        for j in range(5):
            output = output + '{} '.format(result_camera[j][i])
        output = output + "\n"

    file = open('camera.txt', 'w')
    file.write(output)
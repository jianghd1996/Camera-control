import numpy as np
import math as mt
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

def Normalize(X):
    '''
        Input is N * M, N is samples, M is feature number
    '''
    N = len(X)
    M = len(X[0])

    Xmean, Xstd = X.mean(axis=0), X.std(axis=0)

    for i in range(M):
        Xstd[i] = 1

    X = (X - Xmean) / Xstd

    return X, Xmean, Xstd

def load_data(data_path, epoch):
    Length = len(data_path[0])

    for i in range(len(data_path)):
        Length = min(Length, len(data_path[i]))

    print(Length)

    Length = Length // 2

    category = []
    data = []
    label = []

    print('loading data ...')
    for i in tqdm(range(len(data_path))):
        st = Length * epoch
        ed = Length * (epoch+1)

        for j in range(st, ed):
            j = j % len(data_path[i])
            d = np.load(data_path[i][j], allow_pickle=True, encoding='latin1')[()]
            category += [i] * len(d['input'])
            data += d['input']
            label += d['output']

    return category, data, label

def extract_txt(data_path):
    if os.path.exists(data_path+"toric_data.npy"):
        return

    Rules = os.listdir(data_path)
    unity_list = ['Camera', 'Toric', 'Actor',
        'Head', 'Neck', 'LeftArm', 'LeftForeArm', 'LeftHand',
        'RightArm', 'RightForeArm', 'RightHand', 'Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 
        'RightUpLeg', 'RightLeg', 'RightFoot']

    Dict = dict()
    Data = []

    # for every rule
    for rl in Rules:
        List = os.listdir(data_path+rl)
        List.sort()

        # for every file
        for i in range(len(List)):
            print(data_path+rl+"/"+List[i])
            File = open(data_path+rl+"/"+List[i])
            text = File.read()
            text = text.split('\n')
            Length = int(len(text))

            for name in unity_list:
                Dict[name] = []

            cnt = 0

            for j in range(5, Length):
                data = text[j].split(' ')

                Name = data[0]

                if (Name == "Camera"):
                    cnt += 1
                if (cnt < 2):
                    continue

                if Name in unity_list:
                    n = len(data)-1
                    d = np.zeros(n, dtype="float32")
                    for i in range(n):
                        d[i] = eval(data[i+1])
                    Dict[Name].append(d)
            Data.append({"Skelet" : np.array(Dict.copy()), "Rule" : rl})

    np.save(data_path+"toric_data", Data)

def vect_dist(v):
    return np.linalg.norm(v)

def vect_cross(v1, v2):
    result = (v1 * v2).sum() / (vect_dist(v1)*vect_dist(v2))
    return result

def Generate_data(Data, Camera, seq_length, Input_data, Output_data):
    frame = len(Data)
    S_seq = np.zeros((frame, 9+5), dtype="float32")

    for j in range(frame):
        S_seq[j][:9] = Data[j]
        S_seq[j][9:] = Camera[j]

    for j in range(frame):
        S = np.zeros((2*seq_length, 9), dtype="float32")

        for k in range(2*seq_length):
            t = min(max(j-seq_length+k, 0), frame-1)
            S[k][:] = Data[t]

        P_trajectory = np.zeros((seq_length, 5), dtype="float32")
        Q_trajectory = np.zeros((30, 5), dtype="float32")

        P_trajectory[0] = Camera[j-1]
        Q_trajectory[0] = Camera[j]

        for k in range(seq_length):
            P_trajectory[k][:] = Camera[max((j-seq_length+k), 0)]

        for k in range(30):
            Q_trajectory[k][:] = Camera[min(j+k, frame-1)]

        Input_data.append([S_seq, np.concatenate((S.flatten(), P_trajectory.flatten()))])
        Output_data.append(Q_trajectory.flatten())

def extract_prediction_feature(data_path, seq_length, refresh=False, rule=None):
    '''
        The data format is a list, for each unit is a dataset with a rule
        The training can be limited in one rule, or multiple rule with another one-shot label
        Input feature :
            previous and future trajectory -- position, orientation, velocity
            category -- one shot feature
            previous camera -- position, velocity
        Output feature :
            future predict trajectory -- position, orientation, velocity
            current camera -- position, rotate, velocity
            current charactor -- position velocity, angular velocity
    '''
    print("loading "+data_path)
    
    if (os.path.exists(data_path+"toric_prediction_data.npy")):
        return extract_eval_data(data_path, seq_length)

    variaty = int(11)

    if (os.path.exists(data_path+"RNN_"+str(variaty)+"_data.npy")) and not refresh:
        data = np.load(data_path+"RNN_"+str(variaty)+"_data.npy", allow_pickle=True, encoding='latin1')[()]
        return data["input"], data["output"]

    extract_txt(data_path)

    data = np.load(data_path+"toric_data.npy", allow_pickle=True, encoding='latin1')

    Input_data = []
    Output_data = []

    idx = [2, 4]

    alpha = dict()
    alpha[(-75, 75)] = 1.0381
    alpha[(-75, 25)] = 0.7072
    alpha[(-50, 50)] = 0.7278
    alpha[(-25, 75)] = 0.7072
    alpha[(-50,  0)] = 0.3639
    alpha[( 0,  50)] = 0.3639

    for d in tqdm(data):
        if rule == None or d['Rule'] == rule:
            skelet = d["Skelet"][()]

            frame = len(skelet["Toric"])

            # Distance 1
            # Relative orientation 2
            # Character space Shoulder orientation 4
            # Camera 5            
            Data = np.zeros((frame, 7+2), dtype="float32")
            Camera = np.zeros((frame, 5), dtype="float32")

            cnt = 0

            # distance
            for i in range(frame-1):
                p1 = i*2+0
                p2 = i*2+1

                Data[i][cnt+0] = vect_dist(skelet['Hips'][p1][idx]-skelet['Hips'][p2][idx])
            cnt += 1

            # relative orientation between heads
            # relative orientation between shoulders
            for i in range(frame):
                p1 = i*2+0
                p2 = i*2+1
                head1 = skelet['Head'][p1][idx]-skelet['Neck'][p1][idx]
                head2 = skelet['Head'][p2][idx]-skelet['Neck'][p2][idx]
                shoulder1 = skelet['LeftArm'][p1][idx]-skelet['RightArm'][p1][idx]
                shoulder2 = skelet['LeftArm'][p2][idx]-skelet['RightArm'][p2][idx]
                Line = skelet['Hips'][p1][idx] - skelet['Hips'][p2][idx]

                Data[i][cnt+0] = vect_cross(head1, head2)
                Data[i][cnt+1] = vect_cross(shoulder1, shoulder2)
                Data[i][cnt+2] = vect_cross(shoulder1, Line)
                Data[i][cnt+3] = vect_cross(shoulder2, Line)
                Data[i][cnt+4] = vect_cross(head1, Line)
                Data[i][cnt+5] = vect_cross(head2, Line)
            cnt += 6

            # Actor
            for i in range(frame):
                Data[i][cnt+int(skelet['Actor'][i][0])] = 1
            cnt += 2

            # camera
            for i in range(frame):
                Camera[i][0] = skelet['Toric'][i][0]
                Camera[i][1] = skelet['Toric'][i][1]
                Camera[i][2] = (skelet['Toric'][i][2]+skelet['Toric'][i][3]) / 2
                Camera[i][3] = skelet['Toric'][i][4]
                Camera[i][4] = skelet['Toric'][i][5]

            '''
                input size:
                    (orientation + velocity + distance) * (past seq_length + predict seq_length)
                    previous camera = 5
                output size:
                    (orientation + velocity + distance) * (future seq_length)
                    current camera = 5
            '''

            if ('direct' in data_path):
                Generate_data(Data, Camera, seq_length, Input_data, Output_data)

            if ('relative' in data_path):
                Generate_data(Data, Camera, seq_length, Input_data, Output_data)

            if ('Dolly' in data_path):
                Generate_data(Data, Camera, seq_length, Input_data, Output_data)

            if ('sin' in data_path):
                pA = int(round(Camera[i][1] * 100))
                pB = int(round(Camera[i][0] * 100))
                r = 2 * (mt.pi - alpha[(pA, pB)])
                for p in range(variaty):
                    for i in range(frame):
                        Camera[i][3] = r / 2.0 + (1+p*0.1) * mt.sin(2 * mt.pi * i / (100+p*20))

                    Generate_data(Data, Camera, seq_length, Input_data, Output_data)

            if ('side_track' in data_path):
                pA = int(round(Camera[i][1] * 100))
                pB = int(round(Camera[i][0] * 100))
                r = 2 * (mt.pi - alpha[(pA, pB)])
                for p in range(variaty):
                    for i in range(frame):
                        if Data[i][7] == 1:
                            Camera[i][3] = r * (0.8 + (p-(variaty // 2)) * 0.02)
                        else:
                            Camera[i][3] = r * (0.2 + (p-(variaty // 2)) * 0.02)

                    Generate_data(Data, Camera, seq_length, Input_data, Output_data)


    Save_file = np.array({"input" : Input_data, "output" : Output_data})

    np.save(data_path+"RNN_"+str(variaty)+"_data", Save_file)

    return Input_data, Output_data

def extract_estimation_feature(data_path, seq_length, refresh=False):
    '''
        This part extract data for feature estimation
        Input is 2d skeleton screen position and output is 3d features and toric parameters
    '''

    # estimation > feature_data
    # input, output, length
    if os.path.exists(data_path+"feature_data.npy") and (not refresh):
        data = np.load(data_path+"feature_data.npy")
        return data[0], data[1], data[2], data[3]

    extract_txt(data_path)

    data = np.load(data_path+"toric_data.npy")

    Input_data = []
    Output_data = []
    Length = []
    Actor = []

    need_unity = ["Head", "Neck", "LeftArm", "LeftForeArm",
        "RightArm", "RightForeArm", "Hips", "LeftUpLeg", "RightUpLeg"]

    idx = [2, 4]

    for d in data:
        skelet = d["Skelet"][()]
        
        # to deal with velocity feature, need to reduce 1
        frame = len(skelet["Toric"])-1
        Length.append(frame)
        if len(skelet["Actor"]) != 0:
            Actor.append(skelet["Actor"][0])
        skelet_num = len(need_unity)
        skelet_2d = np.zeros((frame, skelet_num*2*2+6), dtype = "float32")
        for i in range(frame):
            for j in range(skelet_num):
                for p in range(2):
                    skelet_2d[i][p*skelet_num*2+2*j+0] = skelet[need_unity[j]][2*i+p][0] / 971.0
                    skelet_2d[i][p*skelet_num*2+2*j+1] = skelet[need_unity[j]][2*i+p][1] / 528.0
            for j in range(skelet_num*2*2):
                if (skelet_2d[i][j] < 0 or skelet_2d[i][j] > 1):
                    skelet_2d[i][j] = 0

            # explicitly describe the screen scale, screen shoulder orientation, etc.
            '''
            for p in range(2):
                p0 = 2*i+p
                X = (skelet["Neck"][p0][0] - skelet["Hips"][p0][0]) / 971.0
                Y = (skelet["Neck"][p0][1] - skelet["Hips"][p0][1]) / 528.0
                spine = np.array([X, Y], dtype="float32")
                X = (skelet["LeftArm"][p0][0] - skelet["RightArm"][p0][0]) / 971.0
                Y = (skelet["LeftArm"][p0][1] - skelet["RightArm"][p0][1]) / 528.0
                shoulder = np.array([X, Y], dtype="float32")

                skelet_2d[i][skelet_num*4+p*3+0] = vect_dist(spine)
                skelet_2d[i][skelet_num*4+p*3+1] = vect_dist(shoulder)
                skelet_2d[i][skelet_num*4+p*3+2] = vect_cross(spine, shoulder)
            '''

        toric_data = np.zeros((frame, 5+7), dtype= "float32")

        for i in range(frame):
            cnt = 0
            for j in range(5):
                toric_data[i][cnt+j] = skelet["Toric"][i][j]
            cnt += 5

            p1 = i*2+0
            p2 = i*2+1

            toric_data[i][cnt+0] = vect_dist(skelet['Hips'][p1][idx]-skelet['Hips'][p2][idx])
            cnt += 1
            head1 = skelet['Head'][p1][idx]-skelet['Neck'][p1][idx]
            head2 = skelet['Head'][p2][idx]-skelet['Neck'][p2][idx]
            shoulder1 = skelet['LeftArm'][p1][idx]-skelet['RightArm'][p1][idx]
            shoulder2 = skelet['LeftArm'][p2][idx]-skelet['RightArm'][p2][idx]
            Line = skelet['Hips'][p1][idx] - skelet['Hips'][p2][idx]
            toric_data[i][cnt+0] = vect_cross(head1, head2)
            toric_data[i][cnt+1] = vect_cross(shoulder1, shoulder2)
            toric_data[i][cnt+2] = vect_cross(shoulder1, Line)
            toric_data[i][cnt+3] = vect_cross(shoulder2, Line)
            toric_data[i][cnt+4] = vect_cross(head1, Line)
            toric_data[i][cnt+5] = vect_cross(head2, Line)
            cnt += 6

        for i in range(frame):
            S = np.zeros((2*seq_length, skelet_num*4), dtype="float32")
            for j in range(2*seq_length):
                for k in range(skelet_num*4):
                    l = min(frame-1, max(0, i-seq_length+j))
                    S[j][k] = skelet_2d[l][k]
            Input_data.append(np.transpose(S))
            Output_data.append(np.array(toric_data[i]))

    np.save(data_path+"feature_data", [Input_data, Output_data, Length, Actor])

    return Input_data, Output_data, Length, Actor

def extract_movie_feature(data_path, seq_length, refresh=False):
    '''
        This part extract data for feature estimation
        Input is 2d skeleton screen position and output is 3d features and toric parameters
    '''

    # estimation > feature_data
    # input, output, length
    if os.path.exists(data_path+"feature_data.npy") and (not refresh):
        data = np.load(data_path+"feature_data.npy")
        return data[0], data[1], data[2], data[3]

    data = np.load(data_path+"movie_data.npy")

    Input_data = []
    Output_data = []
    Length = []
    Actor = []

    need_unity = ["Head", "Neck", "LeftArm", "LeftForeArm",
        "RightArm", "RightForeArm", "Hips", "LeftUpLeg", "RightUpLeg"]

    data = data[()]['Skelet']

    for skelet in data:
        
        # to deal with velocity feature, need to reduce 1
        frame = len(skelet["Toric"])-1
        Length.append(frame)
        Actor.append([skelet["Actor"][0]])
        skelet_num = len(need_unity)
        skelet_2d = np.zeros((frame, skelet_num*2*2+6), dtype = "float32")
        for i in range(frame):
            for j in range(skelet_num):
                for p in range(2):
                    skelet_2d[i][p*skelet_num*2+2*j+0] = skelet[need_unity[j]][2*i+p][0]
                    skelet_2d[i][p*skelet_num*2+2*j+1] = skelet[need_unity[j]][2*i+p][1]
            for j in range(skelet_num*2*2):
                if (skelet_2d[i][j] < 0 or skelet_2d[i][j] > 1):
                    skelet_2d[i][j] = 0

        for i in range(frame):
            S = np.zeros((2*seq_length, skelet_num*4), dtype="float32")
            for j in range(2*seq_length):
                for k in range(skelet_num*4):
                    l = min(frame-1, max(0, i-seq_length+j))
                    S[j][k] = skelet_2d[l][k]
            Input_data.append(np.transpose(S))
            Output_data.append(np.zeros(12, dtype="float32"))

    np.save(data_path+"feature_data", [Input_data, Output_data, Length, Actor])

    return Input_data, Output_data, Length, Actor

def extract_eval_data(data_path, seq_length):
    data, actor = np.load(data_path+'toric_prediction_data.npy')
    movie_num = len(data)

    Input_data = []
    Output_data = []

    for i in range(movie_num):
        frame = len(data[i])
        st = seq_length
        ed = frame-seq_length-1
        N = 9
        M = seq_length

        print(data_path + ' %d main charactor'%i, actor[i][0])

        Data = np.zeros((frame, 9), dtype="float32")
        Camera = np.zeros((frame, 5), dtype="float32")

        for j in range(frame):
            Data[j][:7] = data[i][j][5:]
            Data[j][7+int(actor[i][0])] = 1

        for j in range(frame):
            Camera[j] = data[i][j][:5]

        Generate_data(Data, Camera, seq_length, Input_data, Output_data)

    return Input_data, Output_data

def Visualization(std_data, result_data, name_list, prefix="", save_dir="visual/"):
    print("Visualize standard and result data")
    feature_num = len(name_list)

    std_Length = min(len(std_data[0]), 1875)
    result_Length = min(len(result_data[0]), 1875)
    std_X = np.arange(std_Length)
    result_X = np.arange(result_Length)

    print(std_data.shape)
    print(result_data.shape)

    axis_label_fontsize = 40
    legend_fontsize = 30
    title_fontsize = 35
    x_label = 'frame'
    y_label = 'value'

    # y_min = [-1.5, -1.5, -0.5, 0, 2.9, 
    #             0, 0, 0.5, 0, -0.2, -0.8, -0.8, -0.8, -0.8]
    # y_max = [0.5, 1.5, 1.0, 6.0, 3.4, 
    #             0.2, 0.2, 2.0, 0.02, 0.4, 0.8, 0.8, 0.8, 0.8]

    for i in range(feature_num):
        fig = plt.figure(figsize=(12.5, 10))
        plt.xlabel(x_label, fontsize=axis_label_fontsize)
        plt.ylabel(y_label, fontsize=axis_label_fontsize)
        # plt.xlim(0, 2000) 
        # plt.ylim(y_min[i], y_max[i])
        plt.tick_params(labelsize=25)
        plt.title(name_list[i], fontsize=title_fontsize)
        p_std, = plt.plot(std_X[5:], std_data[i][5:std_Length], color='red', linewidth=4)
        p_result, = plt.plot(result_X[5:], result_data[i][5:result_Length], color='green', linewidth=4)
        # plt.legend([p_std, p_result],
        #     [
        #         "ground truth",
        #         "our result"
        #     ],
        #     loc="upper right",
        #     fontsize=legend_fontsize,
        #     )
        plt.savefig(save_dir+prefix+name_list[i]+'.png')

def Visualiza_Unity(data_path):
    data = np.load(data_path+'toric_prediction_data.npy')
    data = data[0]
    movie_num = len(data)

    output = ''
    for i in range(movie_num):
        frame = len(data[i])
        for j in range(frame):
            output = output + "%.2f\n%.2f\n%.2f\n%.2f\n%.2f\n"%(data[i][j][0], data[i][j][1], data[i][j][2], data[i][j][3], data[i][j][4])

    file = open(data_path+'unity.txt', 'w')
    file.write(output)
    file.close()
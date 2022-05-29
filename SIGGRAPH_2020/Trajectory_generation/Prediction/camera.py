import numpy as np
import os
import matplotlib.pyplot as plt

def smooth(seq):
    seq = np.array(seq)
    seq = np.transpose(seq)

    frame = len(seq[0])
    feature = len(seq)

    tot = 1
    while (tot > 0):
        tot -= 1
        for i in range(feature):
            data = np.array(seq[i])

            for j in range(frame):
                seq[i][j] = data[max(0, j - 10):min(frame, j + 10)].mean()

    seq = np.transpose(seq)

    return seq

file = open('scene.txt', 'r')
text = file.read().split('\n')
file.close()

frame = len(text)

camera = []

for i in range(frame):
    if len(text[i]) > 5:
        txt = text[i].split(' ')
    c = []
    for j in range(6):
        c.append(eval(txt[j]))
    if len(c) > 0:
        camera.append(c)

camera = np.array(camera)

for j in range(6):
    for i in range(1, frame):
        while (abs(camera[i][j]-camera[i-1][j]) > 180):
            if (camera[i][j] > camera[i-1][j]):
                camera[i][j] -= 360
            else:
                camera[i][j] += 360

smooth_camera = smooth(camera)

camera = camera.transpose(1, 0)
smooth_camera = smooth_camera.transpose(1, 0)

output = ''
for i in range(frame):
    st = ""
    for j in range(6):
        st += str(smooth_camera[j][i]) + " "

    output += st + "\n"

file = open('smooth_camera.txt', 'w')
file.write(output)
file.close()
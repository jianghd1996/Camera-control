import numpy as np
import os
import cv2
from movie_processing import get_movie_latent

if __name__ == "__main__":
    compare_video = os.path.join("synthetic_latent", "example_latent.txt")
    file = open(compare_video, 'r')
    text = file.read().split('\n')[0].split(' ')

    movie_latent = []
    for i in range(9):
        movie_latent.append(eval(text[i]))

    if not os.path.exists("movie_latent.npy"):
        get_movie_latent()

    data = np.load("movie_latent.npy", allow_pickle=True)

    Mindist = []

    for v in data:
        Mindist.append([sum((movie_latent - v["latent"]) ** 2), v["name"], v["actor"], v["frame"], v["size"]])

    Mindist.sort()

    for i in range(2):
        if not os.path.exists('movie_clip'):
            os.mkdir('movie_clip')
        vidcap = cv2.VideoCapture(Mindist[i][1]+".mp4")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        st, ed = Mindist[i][3]
        video_writer = cv2.VideoWriter(os.path.join("movie_clip", "{}_{}.avi".format(i, Mindist[i][2])), fourcc, 30, Mindist[i][4])
        cnt = 0
        while True:
            success, image = vidcap.read()
            if (success == False):
                break
            if cnt >= st:
                video_writer.write(image)
            if cnt >= ed:
                break
            cnt += 1
        video_writer.release()

        print(Mindist[i])

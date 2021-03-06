import numpy as np
import matplotlib.pyplot as plt
import pickle
import os, cv2

src = 'Drones Over Dolphin Stampede_QUT Classroom_copyright 2014 David Anderson'
video_inp = '../data/' + src + '.MP4'
img_out = '../data/img/' + src + '.jpg'

video_reader = cv2.VideoCapture(video_inp)
nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

print(frame_h, frame_w)

print('Number of frames to process: {}'.format(nb_frames))

for i in range(nb_frames):
    ret, image = video_reader.read()
    # cv2.imshow(str(i), image)
    if i % 25 != 0:
        continue

    cv2.imwrite('../data/img/' + src + "_" + str(i) + ".jpg", image)
    print('Processed frame: ', i)

video_reader.release()

import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import matplotlib.pyplot as plt
import cv2, os, time, csv
import numpy as np
import tensorflow as tf

import argparse
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def yolonetDetection(videoPath):
    # Set tensorflow backend
    keras.backend.tensorflow_backend.set_session(get_session())

    config_path = 'config.json'
    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    weights_path = os.path.join('snapshots', 'full_yolo_dolphin.h5')

    yolo.load_weights(weights_path)

    videoName = os.path.splitext(os.path.basename(videoPath))[0]
    if not os.path.exists(os.path.join('yolonet','results')):
        os.makedirs(os.path.join('yolonet','results'))

    video_out = os.path.join('yolonet', 'results', videoName + '-yolonet.mp4')
    csv_out = os.path.join('yolonet', 'results', videoName + '-yolonet.csv')
    # video_path = '/Users/bjanson/Documents/Backups/uni/EGH/EGH455/Labeling/data/sample_dolphins_1.mp4'
    video_reader = cv2.VideoCapture(video_path)

    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(video_out,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps,
                                   (frame_w, frame_h))

    csvOut = ['FrameNumber', 'PredictionString']

    for i in tqdm(range(nb_frames)):
        _, image = video_reader.read()

        boxes = yolo.predict(image)
        [image, frameDetections] = draw_boxes(i, image, boxes, config['model']['labels'])
        if frameDetections:
            csvOut.append(frameDetections)

        video_writer.write(np.uint8(image))

    with open(csv_out, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvOut)

    video_reader.release()
    video_writer.release()

    return [video_out, csv_out]


if __name__ == '__main__':
    video_path = '/Users/bjanson/Documents/Backups/uni/EGH/EGH455/Labeling/data/Surfer - Evans 2016.10.14 F2 (1).mp4'
    [video_out, csvOut] = yolonetDetection(video_path)
    # with open('results.csv', 'w', newline='') as csvFile:
    #     writer = csv.writer(csvFile)
    #     writer.writerows(csvOut)

    # csvFile.close()
    cv2.destroyAllWindows()
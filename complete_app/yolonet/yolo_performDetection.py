import keras

import cv2, os, time, csv
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.visualization import draw_box, draw_caption
from utils import draw_boxes
from frontend import YOLO
import json


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def yolonetDetection(videoPath, progress):
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

    print('Loading Yolonet model')
    weights_path = os.path.join('snapshots', 'full_yolo_dolphin.h5')
    yolo.load_weights(weights_path)

    videoName = os.path.splitext(os.path.basename(videoPath))[0]
    if not os.path.exists(os.path.join('yolonet','results')):
        os.makedirs(os.path.join('yolonet','results'))

    video_out = os.path.join('yolonet', 'results', videoName + '-yolonet.mp4')
    csv_out = os.path.join('yolonet', 'results', videoName + '-yolonet.csv')
    video_reader = cv2.VideoCapture(videoPath)

    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(video_out,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps,
                                   (frame_w, frame_h))

    csvOut = [['FrameNumber', 'PredictionString']]

    for i in range(nb_frames):
        print('Frame {}/{}'.format(i, nb_frames))
        _, image = video_reader.read()
        draw = image.copy()

        boxes = yolo.predict(draw)
        [draw, frameDetections] = draw_boxes(i, draw, boxes, config['model']['labels'])
        if frameDetections:
            csvOut.append(frameDetections)

        video_writer.write(np.uint8(draw))
        progress.setValue(i+1)

    with open(csv_out, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvOut)

    video_reader.release()
    video_writer.release()

    return [video_out, csv_out]


# Overlay bounding boxes and captions for imported csv and video
def overlayCSV(csvFile, videoFile, progress):
    detections = []

    with open(csvFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            detections.append(row)

    # Label names
    labelNames = ['shark', 'dolphin', 'surfer']

    # Open video and get info
    videoReader = cv2.VideoCapture(videoFile)
    noFrames = int(videoReader.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(videoReader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(videoReader.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(videoReader.get(cv2.CAP_PROP_FPS))
    videoName = os.path.splitext(os.path.basename(videoFile))[0]

    # Video writer for output video
    if not os.path.exists(os.path.join('yolonet', 'results')):
        os.makedirs(os.path.join('yolonet', 'results'))
    out = cv2.VideoWriter(os.path.join('yolonet', 'results', videoName + '-overlay.mp4'),
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # print('Performing detection on: {}'.format())
    print('Number of frames to process: {}'.format(noFrames))

    # progress = QtWidgets.QProgressDialog("Processing video ...", "Abort", 0, noFrames)
    # progress.setWindowModality(QtCore.Qt.WindowModal)
    # progress.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

    # loop through frames
    for i in range(0, noFrames, 1):
        print('Frame {}/{}'.format(i, noFrames))

        ret, frame = videoReader.read()

        # frame to draw bounding boxes on
        draw = frame.copy()

        # check if there are any detections for frame
        if i in [int(item[0]) for item in detections]:
            index = [int(item[0]) for item in detections].index(i)
            objects = detections[index][1].split()
            # loop through detections
            for y in range(int(len(objects) / 6)):
                objectIndex = y * 6
                caption = "{} {:.2f}".format(objects[objectIndex], float(objects[objectIndex + 1]))
                box = np.array([float(objects[objectIndex + 2]) * width, float(objects[objectIndex + 3]) * height,
                                float(objects[objectIndex + 4]) * width, float(objects[objectIndex + 5]) * height])
                b = box.astype(int)
                colour = label_color(labelNames.index(objects[objectIndex]))
                draw_box(draw, b, color=colour)
                draw_caption(draw, b, caption)

        out.write(draw)

        # set progress for progress bar
        progress.setValue(i + 1)

    videoReader.release()
    out.release()

    outPath = os.path.join('yolonet', 'results', videoName + '-overlay.mp4')
    return outPath

# if __name__ == '__main__':
#     video_path = '/Users/bjanson/Documents/Backups/uni/EGH/EGH455/Labeling/data/Surfer - Evans 2016.10.14 F2 (1).mp4'
#     [video_out, csvOut] = yolonetDetection(video_path)
#     # with open('results.csv', 'w', newline='') as csvFile:
#     #     writer = csv.writer(csvFile)
#     #     writer.writerows(csvOut)
#
#     # csvFile.close()
#     cv2.destroyAllWindows()

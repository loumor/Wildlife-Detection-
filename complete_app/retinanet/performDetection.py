import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from PyQt5 import QtWidgets, QtCore

import matplotlib.pyplot as plt
import cv2, os, time, csv, re
import numpy as np
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# perform detection and return video with bounding boxes plus results csv
def retinanetDetection(videoPath, progress):
    # Set tensorflow backend
    keras.backend.tensorflow_backend.set_session(get_session())

    # Load retinanet model
    print('Loading Retinanet model')
    modelPath = os.path.join('retinanet', 'snapshots', 'inference4.h5')
    model = models.load_model(modelPath, backbone_name='resnet50')

    # Label names
    labelNames = {0: 'shark', 1: 'dolphin', 2: 'surfer'}    

    # Open video and get info
    videoReader = cv2.VideoCapture(videoPath)
    noFrames = int(videoReader.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(videoReader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(videoReader.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(videoReader.get(cv2.CAP_PROP_FPS))    
    videoName = os.path.splitext(os.path.basename(videoPath))[0]

    # List for csv output
    csvOut = []
    csvOut.append(['FrameNumber', 'PredictionString'])

    # Video writer for output video    
    if not os.path.exists(os.path.join('retinanet','results')):
        os.makedirs(os.path.join('retinanet','results'))
    out = cv2.VideoWriter(os.path.join('retinanet','results',videoName+'-retinanet.mp4'),  cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # display progress    
    print('Number of frames to process: {}'.format(noFrames))
    # progress = QtWidgets.QProgressDialog("Processing video ...", "Abort", 0, noFrames)
    # progress.setWindowModality(QtCore.Qt.WindowModal)
    # progress.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

    # loop through frames
    for i in range(noFrames):
        print('Frame {}/{}'.format(i,noFrames))

        ret, frame = videoReader.read()
        
        # frame to draw bounding boxes on
        draw = frame.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        
        # prep frame for detection
        image = preprocess_image(frame)
        image, scale = resize_image(image)

        # perform detection and scale bounding box coordinates
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
                
        frameDetections = [] # holds csv line for a frame
        detections = '' # detections for this frame

        # loop through each detection
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # Confidence threshold
            if score < 0.7:
                break                                                     

            # filter out bad detections
            if label in [0, 1, 2]:            
                # Only add frame line if detections exist
                if not frameDetections:                    
                    frameDetections.append(i)

                # draw bounding box and caption
                colour = label_color(label)          
                b = box.astype(int)
                draw_box(draw, b, color=colour)
                caption = "{} {:.3f}".format(labelNames[label], score)
                draw_caption(draw, b, caption)
                
                # format detection result for csv
                detection = '{} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} '.format(labelNames[label], score, box[0]/width, box[1]/height, box[2]/width, box[3]/height)                
                detections = detections + detection                        

        # draw bounding box and caption
        out.write(draw)

        # append detections to frame csv line
        frameDetections.append(detections)

        # append frame detections to full csv list            
        if frameDetections != ['']:
            csvOut.append(frameDetections)

        # set progress for progress bar
        progress.setValue(i+1)

    videoReader.release()    
    out.release()

    # write csv file
    with open(os.path.join('retinanet','results',videoName+'-results.csv'), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvOut)
    csvFile.close()  

    cv2.destroyAllWindows()
    csvPath = os.path.join('retinanet','results',videoName+'-results.csv')
    outPath = os.path.join('retinanet','results',videoName+'-retinanet.mp4')
    return [outPath, csvPath]

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
    if not os.path.exists(os.path.join('retinanet','results')):
        os.makedirs(os.path.join('retinanet','results'))
    out = cv2.VideoWriter(os.path.join('retinanet','results',videoName+'-overlay.mp4'),  cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    #print('Performing detection on: {}'.format())
    print('Number of frames to process: {}'.format(noFrames))

    # progress = QtWidgets.QProgressDialog("Processing video ...", "Abort", 0, noFrames)
    # progress.setWindowModality(QtCore.Qt.WindowModal)
    # progress.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

    # loop through frames
    for i in range(0,noFrames,1):
        print('Frame {}/{}'.format(i,noFrames))

        ret, frame = videoReader.read()
        
        # frame to draw bounding boxes on
        draw = frame.copy()        

        # check if there are any detections for frame
        if i in [int(item[0]) for item in detections]:
            index = [int(item[0]) for item in detections].index(i)
            objects = detections[index][1].split()
            # loop through detections
            for y in range(int(len(objects)/6)):
                objectIndex = y*6
                caption = "{} {:.2f}".format(objects[objectIndex], float(objects[objectIndex+1]))
                box = np.array([float(objects[objectIndex+2])*width, float(objects[objectIndex+3])*height, float(objects[objectIndex+4])*width, float(objects[objectIndex+5])*height])
                b = box.astype(int)
                colour = label_color(labelNames.index(objects[objectIndex]))                            
                draw_box(draw, b, color=colour)                
                draw_caption(draw, b, caption)

        out.write(draw)

        # set progress for progress bar
        progress.setValue(i+1)
        
    videoReader.release()    
    out.release()
        
    outPath = os.path.join('retinanet','results',videoName+'-overlay.mp4')
    return outPath
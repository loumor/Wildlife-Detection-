import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from PyQt5 import QtWidgets, QtCore

import matplotlib.pyplot as plt
import cv2, os, time, csv
import numpy as np
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def retinanetDetection(videoPath):
    # Set tensorflow backend
    keras.backend.tensorflow_backend.set_session(get_session())

    # Load retinanet model
    print('Loading Retinanet model')
    modelPath = os.path.join('retinanet', 'snapshots', 'inference2.h5')
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

    # Video writer for output video    
    if not os.path.exists(os.path.join('retinanet','results')):
        os.makedirs(os.path.join('retinanet','results'))
    out = cv2.VideoWriter(os.path.join('retinanet','results',videoName+'-retinanet.mp4'),  cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    #print('Performing detection on: {}'.format())
    print('Number of frames to process: {}'.format(noFrames))

    progress = QtWidgets.QProgressDialog("Processing video ...", "Abort", 0, noFrames)
    progress.setWindowModality(QtCore.Qt.WindowModal)
    progress.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)


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
            if score < 0.5:
                break                                                     

            # filter out bad detections
            if label in [0, 1, 2]:            
                # Only add frame line if detections exist
                if not frameDetections:                    
                    frameDetections.append(i)

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

        progress.setValue(i)

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

if __name__ == '__main__':
    out, csvOut = retinanetDetection(r'C:\Users\Marc\Documents\egh455\Shark_whale - Evans 2016.11.18 F1 (2).MP4')
    out.release()
    with open('results.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvOut)

    csvFile.close()
    cv2.destroyAllWindows()
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

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
    modelPath = os.path.join('keras-retinanet-master', 'snapshots', 'inference.h5')
    model = models.load_model(modelPath, backbone_name='resnet50')

    # Label names
    labelNames = {0: 'shark', 1: 'dolphin', 2: 'surfer'}

    csvOut = []

    # Open video and get info
    videoReader = cv2.VideoCapture(videoPath)
    noFrames = int(videoReader.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(videoReader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(videoReader.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(videoReader.get(cv2.CAP_PROP_FPS))

    # Video writer for output video
    out = cv2.VideoWriter('retinanetProcessed.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

    #print('Performing detection on: {}'.format())
    print('Number of frames to process: {}'.format(noFrames))

    for i in range(noFrames):
        print('Frame {}/{}'.format(i,noFrames))
        ret, frame = videoReader.read()
        
        draw = frame.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        
        image = preprocess_image(frame)
        image, scale = resize_image(image)

        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale

        frameDetections = ''

        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # Confidence threshold
            # if score < 0.5:
            #     break                                                     

            if label in [0, 1, 2]:            
                if not frameDetections:
                    frameDetections = '{},'.format(i)

                colour = label_color(label)
            
                b = box.astype(int)
                draw_box(draw, b, color=colour)

                caption = "{} {:.3f}".format(labelNames[label], score)
                draw_caption(draw, b, caption)

                result = ' {} {} {} {} {} {}'.format(labelNames[label], score, box[0]/width, box[1]/height, box[2]/width, box[3]/height)
                frameDetections = frameDetections + result

        out.write(draw)

        if frameDetections:
            csvOut.append([frameDetections])

    videoReader.release()    
    out.release()
    with open('results.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow(csvOut)
    csvFile.close()
    # with open('results.csv', 'w', newline='') as csvFile2:
    #     writer = csv.writer(csvFile2)
    #     writer.writerows(csvOut)
    # csvFile2.close()
    cv2.destroyAllWindows()
    csvPath = 'results.csv'
    outPath = 'retinanetProcessed.mp4'
    return [outPath, csvPath]

if __name__ == '__main__':
    out, csvOut = retinanetDetection(r'C:\Users\Marc\Documents\egh455\Shark_whale - Evans 2016.11.18 F1 (2).MP4')
    out.release()
    with open('results.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvOut)

    csvFile.close()
    cv2.destroyAllWindows()
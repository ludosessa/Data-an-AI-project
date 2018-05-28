# The German Traffic Sign Detection Benchmark
#
# modification of the readTrafficSigns.py for the classification task (GTSRB)
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSDB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import csv
import os, sys
import numpy as np
import scipy.misc
from PIL import Image
from io import open
import random

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 

def readDetectSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''

    images = [] # images
    boxes = [] # corresponding labels
    labels = []
    
    filename = os.listdir(rootpath)


    for i in filename:
        path = os.path.join(rootpath,i)
        if path.endswith(".ppm"):
            images.append(plt.imread(path)) # add all the ppm files
            boxes.append([]);
            labels.append([]);
        else:
            continue
        
    #gtFile = open(rootpath + '/' + 'gt'+ '.csv') # annotations file

    with open(rootpath + '/' + 'gt'+ '.csv') as gtFile:
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file

        # loop over all lines in annotations file
        for row in gtReader:
            number = int(row[0][0:5]) # get the image number
            
            coords = np.array([row[1],row[2],row[3],row[4]])
            lab = row[5]
            boxes[number].append(coords) # the 8th column is the label
            labels[number].append(lab)
            
        gtFile.close()
    
    trainval_dataset = []
    
    for i in range(len(images)):
        img = np.asarray(images[i], dtype=np.float32)
        img = np.transpose(img,[2,0,1])
        bbox = boxes[i]
        if not bbox:
            bbox = np.empty(shape = (0,4))
        bbox = np.asarray(bbox, dtype=np.float32)
        label = labels[i]
        label = np.asarray(label, dtype=np.int32)
        trainval_dataset.append([img, bbox, label])
      
    return trainval_dataset


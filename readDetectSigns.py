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


# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 

def readDetectSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''

    images = [] # images
    labels = [] # corresponding labels
        
    
    filename = os.listdir(rootpath)


    for i in filename:
        path = os.path.join(rootpath,i)
        if path.endswith(".ppm"):
            images.append(plt.imread(path)) # add all the ppm files
            labels.append([]);
        else:
            continue
        
    #gtFile = open(rootpath + '/' + 'gt'+ '.csv') # annotations file

    with open(rootpath + '/' + 'gt'+ '.csv') as gtFile:
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file

        # loop over all lines in annotations file
        for row in gtReader:
            number = int(row[0][0:5]) # get the image number
            print(row[0])
            coords = np.array([row[1],row[2],row[3],row[4]])
            labels[number].append(coords) # the 8th column is the label

        gtFile.close()
    
    return images, labels

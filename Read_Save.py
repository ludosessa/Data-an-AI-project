
# coding: utf-8

# In[ ]:


# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import csv
import shutil
import scipy.misc 

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns_save(rootpathRead, rootpathSave):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images =[]
    labels=[]
    # loop over all 42 classes
    for c in range(0,43):
        prefixRead = rootpathRead + '/' + format(c, '05d') + '/' # subdirectory for class
        prefixSave = rootpathSave + '/' + format(c, '05d') + '/'
        gtFile = open(prefixRead + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        #gtReader.next() # skip header
        next(gtReader)
        # loop over all images in current annotations file
        for row in gtReader:
            name = row[0]
            images = plt.imread(prefixRead + name) # the 1th column is the filename
            images = scipy.misc.imresize(images, (47, 47, 3))
            plt.imsave(prefixSave + name, images)
            labels.append(row[7])
        gtFile.close()
        shutil.copyfile(prefixRead + 'GT-'+ format(c, '05d') + '.csv',prefixSave + 'GT-'+ format(c, '05d') + '.csv')
    return


# In[ ]:


trafficSigns_train = './data/train/Final_Training/Images'
trafficSigns_resized = './data/train/Final_Training/Resized'
readTrafficSigns_save(trafficSigns_train,trafficSigns_resized)


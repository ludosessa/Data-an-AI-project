
# coding: utf-8

# ## Project: Traffic Signs Recognition 

import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as utils
from readTrafficSigns import readTrafficSigns
import random
import math
import scipy.misc 
from conv_layers import ConvNet



def get_data_TrafficSigns():
    """
    Load the German Traffic Sign Recognition Benchmark dataset from disk and perform preprocessing to prepare
    it for the linear classifier.  
    """
    # Load the raw traffic signs data
    trafficSigns_train = './data/train/Final_Training/Images'
    X_train, y_train = readTrafficSigns(trafficSigns_train)
 
    #print (len(X_train))   #vector of dimension 39209 --> one element per image : for each image (element) we get a matrix for each pixel containing the RGBs
    #print (len(y_train))   #vector of dimension 39209 --> one element per image : for each image (element) we get the number of the class it belongs to 
    #print (y_train[0])
    #print (type(y_train))
     #for the moment, images are classed in order --> firt 210 images belong to class 0 (y=0), next ones belong to class 1 etc...

    ind = list()
    
    # subsample the data
    for i in range(0,43):
        
        index_value = list()
        for j in range (0,39209):
            if int(y_train[j]) == i:
                index_value.append(j)
        ind.append(random.sample(range(min(index_value),max(index_value)+1), math.floor(0.02*len(index_value))))
    
    ind2 = [item for sublist in ind for item in sublist]
    X_val = [X_train[i] for i in ind2]
    y_val = [y_train[i] for i in ind2]
    for ind2 in sorted(ind2, reverse=True):
        del X_train[ind2]
        del y_train[ind2]
   
    return X_train, y_train, X_val, y_val

# Invoke the above function to get our data.
X_train, y_train, X_val, y_val = get_data_TrafficSigns()
 


def resize(X, length, width):
    X_new = np.empty(shape=(0,3,length,width))
    for i in range(len(X)):
        image = X[i];
        image = scipy.misc.imresize(image, (length, width, 3))
        transp_image = image.transpose(2,0,1)
        y = np.expand_dims(transp_image, axis=0)
        X_new = np.concatenate((X_new,y), axis=0)

    return X_new  

n = 47

X_train = resize(X_train, n,n)
X_val = resize(X_val, n,n)
y_train = np.asarray(y_train, dtype=np.int64)
y_val = np.asarray(y_val, dtype=np.int64)

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)


X_train, y_train = torch.from_numpy(X_train).type(torch.cuda.FloatTensor), torch.from_numpy(y_train).type(torch.cuda.LongTensor)
X_val, y_val = torch.from_numpy(X_val).type(torch.cuda.FloatTensor), torch.from_numpy(y_val).type(torch.cuda.LongTensor)

traindataset = utils.TensorDataset(X_train, y_train)
trainloader = utils.DataLoader(traindataset, batch_size=64, shuffle=True)

valdataset = utils.TensorDataset(X_val, y_val)
valloader = utils.DataLoader(valdataset, batch_size=64, shuffle=True)
    


net = ConvNet()
net.cuda()
################################################################################
# TODO:                                                                        #
# Choose an Optimizer that will be used to minimize the loss function.         #
# Choose a critera that measures the loss                                      #
################################################################################

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.005)

epochs = 1
steps = 0
running_loss = 0
print_every = 10
for e in range(epochs):
    start = time.time()
    for images, labels in iter(trainloader):
        
        steps += 1
        ################################################################################
        # TODO:                                                                        #
        # Run the training process                                                     #
        #                                                                              #
        # HINT: Do not forget to transform the inputs and outputs into Variable        #
        # which pytorch uses.                                                          #
        ################################################################################
        
        #transofrm inputs and outputs into Variable 
        inputs, targets = Variable(images).cuda(), Variable(labels).cuda()
        
        #set gradient to zero
        optimizer.zero_grad()
        
        # forward pass
        out = net.forward(inputs).cuda()
        
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        
        loss = criterion(out, targets)
        
        ################################################################################
        # TODO:                                                                        #
        # Run the training process                                                     #
        #                                                                              #
        # HINT: Calculate the gradient and move one step further                       #
        ################################################################################
        
        loss.backward()
        optimizer.step()
        
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        
        running_loss += loss.data[0]
        
        if steps % print_every == 0:
            stop = time.time()
            # Test accuracy
            accuracy = 0
            for ii, (images, labels) in enumerate(valloader):
                
                ################################################################################
                # TODO:                                                                        #
                # Calculate the accuracy                                                       #
                ################################################################################
                
                out = net.predict(Variable(images).cuda())
                _, prediction = torch.max(out, 1)
                pred_y = prediction.data.numpy().squeeze()
                target_y = (labels.numpy()).data
                accuracy += sum(pred_y == target_y)/64
                
                ################################################################################
                #                              END OF YOUR CODE                                #
                ################################################################################
            
            print("Epoch: {}/{}..".format(e+1, epochs),
                  "Loss: {:.4f}..".format(running_loss/print_every),
                  "Test accuracy: {:.4f}..".format(accuracy/(ii+1)),
                  "{:.4f} s/batch".format((stop - start)/print_every)
                 )
            running_loss = 0
            start = time.time()



torch.save(net.state_dict(), 'project.pt')







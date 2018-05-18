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


    #print (len(X_train))   #vector of dimension 39209 --> one element per image : for each image (element) we get a matrix for each pixel containing the RGBs
    #print (len(y_train))   #vector of dimension 39209 --> one element per image : for each image (element) we get the number of the class it belongs to 
    #print (y_train[0])
    #print (type(y_train))
     #for the moment, images are classed in order --> firt 210 images belong to class 0 (y=0), next ones belong to class 1 etc...

# Load the raw traffic signs data
trafficSigns_train = './data/train/Final_Training/Images'
X, y = readTrafficSigns(trafficSigns_train)
 
X_resized = []
for i in range(len(X)):
    image_resized = scipy.misc.imresize(X[i], (47, 47))
    X_resized.append(image_resized)
    
ind = list()
# subsample the data
for i in range(0,43):
    index_value = list()
    for j in range (0,39209):
        if int(y[j]) == i:
            index_value.append(j)
    ind.append(random.sample(range(min(index_value),max(index_value)+1), math.floor(0.02*len(index_value))))

ind2 = [item for sublist in ind for item in sublist]
X_val = [X_resized[i] for i in ind2]
y_val = [y[i] for i in ind2]
print(len(X_val)) 
for ind2 in sorted(ind2, reverse=True):
    del X_resized[ind2]
    del y[ind2]
X_train = X_resized
y_train = y

X_train = np.asarray(X_train)
X_train = np.transpose(X_train,[0,3,1,2])

X_val = np.asarray(X_val)
X_val = np.transpose(X_val,[0,3,1,2])

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
   
#hyperparameters:

net = ConvNet()
net.cuda()
lr = 9.8*10**(-4)
reg = 1.3*10**(-4)
momentum = 0.95
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = lr, weight_decay = reg, momentum = momentum)
epochs = 10
steps = 0
running_loss = 0
print_every = 50
for e in range(epochs):
    #start = time.time()
    for images, labels in iter(trainloader):

        steps += 1

        #transofrm inputs and outputs into Variable 
        inputs, targets = Variable(images).cuda(), Variable(labels).cuda()

        #set gradient to zero
        optimizer.zero_grad()

        # forward pass
        out = net.forward(inputs).cuda()

        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        if steps % print_every == 0:
        #stop = time.time()
            # Test accuracy
            accuracy = 0
            for ii, (images, labels) in enumerate(valloader):

                out = net.predict(Variable(images).cuda())
                _, prediction = torch.max(out, 1)
                pred_y = prediction.data.cpu().numpy().squeeze()
                target_y = (labels.numpy()).data
                accuracy += sum(pred_y == target_y)/64

            print("Epoch: {}/{}..".format(e+1, epochs),
                  "Loss: {:.4f}..".format(running_loss/print_every),
                  "Test accuracy: {:.4f}..".format(accuracy/(ii+1)))

'''print("val_acc: {:.4f}..".format(accuracy/(len(valloader))),
      "lr: {:.6e}..".format(lr),
      "reg: {:.6e}..".format(reg),
      "momentum: {:.4f}..".format(momentum)
      )'''
#start = time.time()
#torch.save(net.state_dict(), 'project.pt')​



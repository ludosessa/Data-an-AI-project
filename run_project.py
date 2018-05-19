# import the necessary packages
import readTrafficSigns
import argparse
import numpy as np
import pickle
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import optim
from conv_layers import ConvNet
import scipy.misc 

#########################################################################
# TODO:                                                                 #
# This is used to input our test dataset to your model in order to      #
# calculate your accuracy                                               #
# Note: The input to the function is similar to the output of the method#
# "get_CIFAR10_data" found in the notebooks.                            #
#########################################################################

    
def predict(X):
    #########################################################################
    # TODO:                                                                 #
    # - 
    #your saved model                                               #
    # - Do the operation required to get the predictions                    #
    # - Return predictions in a numpy array                                 #
    # Note: For the predictions, you have to return the index of the max    #
    # value                                                                 #
    #########################################################################
    
    images = Variable(torch.from_numpy(X).type(torch.cuda.FloatTensor))
    
    CNN_project = ConvNet(n_input_channels=3, n_output=43)
    CNN_project.load_state_dict(torch.load('./project.pt'))
    output = CNN_project.predict(images.cuda())
    _, prediction = torch.max(output.data, 1)
    pred_y = prediction.data.cpu().numpy().squeeze()
   
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return pred_y
   

def main():
    trafficSigns_test = './data/test/Final_Test/Images'
    X, y = readTrafficSigns(trafficSigns_test)
    #X_resized = []
    #for i in range(len(X)):
        #image_resized = scipy.misc.imresize(X[i], (47, 47))
        #X_resized.append(image_resized)
    #X_test = X_resized
    X_test = np.asarray(X)
    #X_test = np.transpose(X_test,[0,3,1,2])
    y_test = np.asarray(y, dtype=np.int64)
    X_test, y_test = torch.from_numpy(X_test).type(torch.cuda.FloatTensor), torch.from_numpy(y_test).type(torch.cuda.LongTensor)
    prediction_project = predict(X_test)
    acc_project = sum(prediction_project == y_test)/len(X_test)
    print("Accuracy %s"%(acc_project))
    
#if __name__ == "__main__":
    #ap = argparse.ArgumentParser()
    #ap.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    #ap.add_argument("-t", "--test", required=True, help="path to test file")
    #ap.add_argument("-g", "--group", required=True, help="group number")
    #args = vars(ap.parse_args())
    #args.cuda = not args.no_cuda and torch.cuda.is_available()
    #main(args["test"],args["group"])
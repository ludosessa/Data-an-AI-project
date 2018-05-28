import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from readDetectSigns import readDetectSigns
import random
import math 
#from chainercv.datasets import voc_bbox_label_names
#from chainercv.visualizations import vis_bbox
import torch
#from torchnet.meter import AverageValueMeter, MovingAverageValueMeter
import torch.nn as nn
from torch.autograd import Variable

from model.faster_rcnn import faster_rcnn
from model.utils.transform_tools import image_normalize

detectSigns_train = './data_detection/train'
dataset = readDetectSigns(detectSigns_train)

detectSigns_test = './data_detection/test'
test_dataset = readDetectSigns(detectSigns_test) 

ind = list()
# subsample the data
ind.append(random.sample(range(0, 599), math.floor(0.02*600)))

ind2 = [item for sublist in ind for item in sublist]
val_dataset = [dataset[i] for i in ind2]
trainval_dataset = dataset[:]
for ind2 in sorted(ind2, reverse=True):
    del dataset[ind2]
train_dataset = dataset

print('Train shape: ', len(train_dataset))
print('Validation shape: ', len(val_dataset))
print('trainval shape: ', len(trainval_dataset))
print('test shape: ', len(test_dataset))

def adjust_learning_rate(optimizer, epoch, init_lr, lr_decay_factor=0.1, lr_decay_epoch=10):
    """Sets the learning rate to the initial LR decayed by lr_decay_factor every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch == 0:
        lr = init_lr * (lr_decay_factor ** (epoch // lr_decay_epoch))
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


model = faster_rcnn(43, backbone='vgg16')
if torch.cuda.is_available():
    model = model.cuda()

optimizer = model.get_optimizer(is_adam=False)
#avg_loss = AverageValueMeter()
#avg_loss = list()
#ma20_loss = MovingAverageValueMeter(windowsize=20)
#ma20_loss = list()
model.train()

for epoch in range(1):
    adjust_learning_rate(optimizer, epoch, 0.001, lr_decay_epoch=10)
    #for i in range(len(train_dataset)):
     #   img, bbox, label = train_dataset[i]
    for i in range(len(trainval_dataset)):
        img, bbox, label = trainval_dataset[i]
        img = img/255
        #m = nn.AvgPool2d(3, stride=2)
        #input = Variable(torch.from_numpy(img).type(torch.cuda.FloatTensor))
        #img = m(input)
        #img = img.data.cpu().numpy()
        loss = model.loss(img, bbox, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.cpu().data.numpy()[0]
        #avg_loss.append(loss_value)
        #ma20_loss.append(float(loss_value))
        #print('[epoch:{}]  [batch:{}/{}]  [sample_loss:{:.4f}] '.format(epoch, i, len(train_dataset), loss_value))
        #print('[epoch:{}]  [batch:{}/{}]  [sample_loss:{:.4f}] [avg_loss:{:.4f}] [ma20_loss:{:.4f}] '.format(epoch, i, len(train_dataset), loss_value, avg_loss.value()[0], ma20_loss.value()[0]))
        print('[epoch:{}]  [batch:{}/{}]  [sample_loss:{:.4f}] '.format(epoch, i, len(trainval_dataset), loss_value))



model.eval()
for i in range(len(test_dataset)):
    img, _, _ = test_dataset[i]
    imgx = img/255
    #m = nn.AvgPool2d(3, stride=2)
    #input = Variable(torch.from_numpy(imgx).type(torch.cuda.FloatTensor))
    #imgx = m(input)
    #imgx = imgx.data.cpu().numpy()
    bbox_out, class_out, prob_out = model.predict(imgx, prob_threshold=0.95)

    plt.show()
    fig = plt.gcf()
    fig.set_size_inches(11, 5)
    fig.savefig('val_'+str(i)+'.jpg', dpi=100)


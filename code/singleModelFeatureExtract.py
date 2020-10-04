# subramanian https://github.com/PacktPublishing/Deep-Learning-with-PyTorch/blob/master/Chapter03/Image%20Classification%20Dogs%20and%20Cats.ipynb
# modified by gs
# imports
import pandas as pd
from glob import glob
import os
from shutil import copyfile
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from numpy.random import permutation
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18,resnet34
from torchvision.models.inception import inception_v3
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
import copy

import pickle
#%matplotlib inline
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# gpu check
is_cuda = torch.cuda.is_available()

# data_dir = "./data"
# PATH to data set images
train_set = './train_dark'
val_set = './val_ensemble'

# utility functions 
def imshow(inp,cmap=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp,cmap)


# custom pytorch dataset class for the preconvoluted features and loader    
class FeaturesDataset(Dataset):
    
    def __init__(self,featlst,labellst):
        self.featlst = featlst
        self.labellst = labellst
        
    def __getitem__(self,index):
        return (self.featlst[index],self.labellst[index])
    
    def __len__(self):
        return len(self.labellst)
# fit function takes number of epochs to train for, pytorch model, dataloader, scheduler and phase
# returns loss and accuracy
def fit(epoch,model,data_loader,scheduler,phase='training'):
    # use the best performing model
    # val_acc_history = []
    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0

    if phase == 'training':
	    model.train()
	    scheduler.step()
    if phase == 'validation':
        model.eval()
        
    running_loss = 0.0
    running_correct = 0
    for batch_idx , (data,target) in enumerate(data_loader):
        if is_cuda:
            data,target = data.cuda(),target.cuda()
        data , target = Variable(data),Variable(target)
        if phase == 'training':
            optimizer.zero_grad() # zero the parameter gradients
        output = model(data)
        # from import
        loss = F.cross_entropy(output,target)
        # statistics
        running_loss += F.cross_entropy(output,target,reduction='none').data[0]
        preds = output.data.max(dim=1,keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        # backward + optimize only if in training phase
        if phase == 'training':
            loss.backward()
            optimizer.step()
    
    loss = running_loss/len(data_loader.dataset)
    # typecast to avoid formatting error
    accuracy = 100. * running_correct.double()/len(data_loader.dataset)
    # deep copy the model
    # if phase == 'validation' and accuracy > best_acc:
    #     best_acc = accuracy
    #     best_model_wts = copy.deepcopy(model.state_dict())
    # if phase == 'validation':
    #     val_acc_history.append(accuracy)
    # old formatting syntax
    # print('{phase} loss is {loss:5.2f} and {phase} accuracy is {accuracy:10.4f}'.format(phase=phase, loss=loss, running_correct=running_correct, accuracy=accuracy))
    print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    
    # load best model weights
    # model.load_state_dict(best_model_wts)
    # save model
    torch.save(model.state_dict(), train_set + 'fc.pth')
    return loss,accuracy

##################################3
# creating datasets
# note image size - inception requires 299x299 while other models require 224x224
data_transform = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# create the class train and val datasets
# set the PATH to the dataset folders here
train_dset = ImageFolder(train_set,transform=data_transform)
val_dset = ImageFolder(val_set,transform=data_transform)
# number of classes
classes=7

# show a tensor
# imshow(train_dset[150][0])

# create data loader to use for both sets - for calculating preconvoluted features, shuffle must be set to false
# to maintain the exact sequence of data
train_loader = DataLoader(train_dset,batch_size=32,shuffle=False,num_workers=3)
val_loader = DataLoader(val_dset,batch_size=32,shuffle=False,num_workers=3)

# initialize resnet34 model to use layers of the pretrained model
# this trained model used for feature extraction by removing last fc layer
my_resnet = resnet18(pretrained=True)

if is_cuda:
    my_resnet = my_resnet.cuda()
# if feat extracting, man set all params req_grad attribute to false before reshaping, then reinit layer params have 
# req_grad set to true by default
# feature extraction = true, not finetuning
for p in my_resnet.parameters():
	p.requires_grad=False
# resnet34 sequential model by discarding the last linear layer; this trained model used for feature extraction
my_resnet = nn.Sequential(*list(my_resnet.children())[:-1])

# new model from nn.sequential false set because feature extraction and not finetune or train from scratch
# calculate preconvoluted features
# this approach speeds training by avoiding redundant computation
#For training data

# Stores the preconvoluted labels of the train data
trn_labels = [] 

# Stores the pre convoluted features of the train data
trn_features = [] 


# Iterate through the train data and store the calculated preconvoluted features and the labels
for d,la in train_loader:
    o = my_resnet(Variable(d.cuda()))
    o = o.view(o.size(0),-1)
    trn_labels.extend(la)
    trn_features.extend(o.cpu().data)

# For validation data

# Iterate through the validation data and store the calculated features and the labels
val_labels = []
val_features = []
for d,la in val_loader:
    o = my_resnet(Variable(d.cuda()))
    o = o.view(o.size(0),-1)
    val_labels.extend(la)
    val_features.extend(o.cpu().data)



# initialize custom dataset for preconvoluted features train and validation
trn_feat_dset = FeaturesDataset(trn_features,trn_labels)
val_feat_dset = FeaturesDataset(val_features,val_labels)

# Create data loader for train and validation
trn_feat_loader = DataLoader(trn_feat_dset,batch_size=64,shuffle=True)
val_feat_loader = DataLoader(val_feat_dset,batch_size=64)

# simple linear model to map preconvoluted features to corresponding categories
# fc network to serve as final layer to output
class FullyConnectedModel(nn.Module):
    
    def __init__(self, in_size, out_size):
        super().__init__()
        self.fc = nn.Linear(in_size,out_size)
        # self.fc = nn.AdaptiveAvgPool1d(out_size)

    def forward(self,inp):
    	# out = self.fc(inp)
    	# very strange flattening of validation accuracy if dropout not used here
        out = self.fc(F.dropout(inp,training=self.training))
        return out
# size of final layer of resnet34 ??
fc_in_size = 8192

# initialize
fc = FullyConnectedModel(fc_in_size, classes)
if is_cuda:
    fc = fc.cuda()
# initialize
# optimizer = optim.Adam(fc.parameters(),lr=0.0001)
learning_rate = 0.001
optimizer = optim.SGD(fc.parameters(), lr=.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# train and validate
train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
for epoch in range(1,25):
    epoch_loss, epoch_accuracy = fit(epoch,fc,trn_feat_loader,exp_lr_scheduler,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,fc,val_feat_loader,exp_lr_scheduler,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

# plot the training and validation accuracy
plt.plot(range(1,len(train_accuracy)+1), train_accuracy,'bo',label = 'train accuracy')
plt.plot(range(1,len(val_accuracy)+1), val_accuracy,'r', label='val accuracy')
plt.legend()
plt.show()

# torch.save(my_resnet.state_dict(), train_set + '.pth')

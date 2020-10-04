# subramanian https://github.com/PacktPublishing/Deep-Learning-with-PyTorch/blob/master/Chapter08/ensembles/Ensemble_Dogs_and_cats.ipynb
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
from torchvision.models import resnet18,resnet34,densenet121
from torchvision.models.inception import inception_v3
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler


import pickle
#%matplotlib inline
# PATH to data set images
train_set = './train_ensemble'
val_set = './val_ensemble'

num_classes = 7
# gpu check
is_cuda = torch.cuda.is_available()

# utility functions
# show a tensor
def imshow(inp,cmap=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp,cmap)

# modified fit function for ensembling
def fit(epoch,model,data_loader,scheduler,phase='training',volatile=False):
    if phase == 'training':
        model.train()
        # decay learning rate
        scheduler.step()
    if phase == 'validation':
        model.eval()
        volatile=True
    running_loss = 0.0
    running_correct = 0
    #F = torch.tensor(3)
    for batch_idx , (data1,data2,data3,target) in enumerate(data_loader):
        if is_cuda:
            data1,data2,data3,target = data1.cuda(),data2.cuda(),data3.cuda(),target.cuda()
        data1,data2,data3,target = Variable(data1,volatile),Variable(data2,volatile),Variable(data3,volatile),Variable(target)
        if phase == 'training':
            optimizer.zero_grad() # zero the parameter gradients
        output = model(data1,data2,data3)
        loss = F.cross_entropy(output,target)
        
        running_loss += F.cross_entropy(output,target,reduction='none').data[0]
        preds = output.data.max(dim=1,keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()
    
    loss = running_loss/len(data_loader.dataset)
    accuracy = 100. * running_correct.double()/len(data_loader.dataset)
    
    # print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {accuracy:{10}.{4}}')    # print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {accuracy:{10}.{4}}')
    #print('{phase} loss is {loss:5.2f} and {phase} accuracy is {accuracy:10.4f}'.format(phase=phase, loss=loss, running_correct=running_correct, accuracy=accuracy))
    print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')

    return loss,accuracy
# custom dataset
class FeaturesDataset(Dataset):
    
    def __init__(self,featlst1,featlst2,featlst3,labellst):
        self.featlst1 = featlst1
        self.featlst2 = featlst2
        self.featlst3 = featlst3
        self.labellst = labellst
        
    def __getitem__(self,index):
        return (self.featlst1[index],self.featlst2[index],self.featlst3[index],self.labellst[index])
    
    def __len__(self):
        return len(self.labellst)

# helper for inception
class LayerActivations():
    features=[]
    
    def __init__(self,model):
        self.features = []
        self.hook = model.register_forward_hook(self.hook_fn)
    
    def hook_fn(self,module,input,output):
        #out = F.avg_pool2d(output, kernel_size=8)
        self.features.extend(output.view(output.size(0),-1).cpu().data)

    
    def remove(self):
        
        self.hook.remove()

##################################3
# creating datasets
# note image size
data_transform = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# create the class train and val datasets
train_dset = ImageFolder(train_set,transform=data_transform)
val_dset = ImageFolder(val_set,transform=data_transform)
classes=7

# show a tensor
# imshow(train_dset[150][0])

# create data loader to use for both sets
train_loader = DataLoader(train_dset,batch_size=32,shuffle=False,num_workers=3)
val_loader = DataLoader(val_dset,batch_size=32,shuffle=False,num_workers=3)

# initialize models
#Create ResNet model
resnet_dark = resnet18(pretrained=True)
num_ftrs = resnet_dark.fc.in_features
resnet_dark.fc = nn.Linear(num_ftrs, num_classes)
resnet_dark.load_state_dict(torch.load('./datatrain_dark18.pth'))

if is_cuda:
    resnet_dark = resnet_dark.cuda()

resnet_dark = nn.Sequential(*list(resnet_dark.children())[:-1])

# for p in resnet_dark.parameters():
#     p.requires_grad = False

#Create ResNet model
resnet_light = resnet18(pretrained=True)
num_ftrs = resnet_light.fc.in_features
resnet_light.fc = nn.Linear(num_ftrs, num_classes)
resnet_light.load_state_dict(torch.load('./datatrain_light18.pth'))

if is_cuda:
    resnet_light = resnet_light.cuda()

resnet_light = nn.Sequential(*list(resnet_light.children())[:-1])

# for p in resnet_light.parameters():
#     p.requires_grad = False

#Create ResNet model
resnet_reg = resnet18(pretrained=True)
num_ftrs = resnet_reg.fc.in_features
resnet_reg.fc = nn.Linear(num_ftrs, num_classes)
resnet_reg.load_state_dict(torch.load('./datatrain_reg18.pth'))

if is_cuda:
    resnet_reg = resnet_reg.cuda()

resnet_reg = nn.Sequential(*list(resnet_reg.children())[:-1])

# for p in resnet_reg.parameters():
#     p.requires_grad = False



# feature extraction 
# calculate preconvoluted features and labels to save time during training
### For ResNet dark

trn_labels = []
val_labels = []

trn_resnet_d_features = []
for d,la in train_loader:
    o = resnet_dark(Variable(d.cuda()))
    o = o.view(o.size(0),-1)
    trn_labels.extend(la)
    trn_resnet_d_features.extend(o.cpu().data)


val_resnet_d_features = []
for d,la in val_loader:
    o = resnet_dark(Variable(d.cuda()))
    o = o.view(o.size(0),-1)
    val_labels.extend(la)
    val_resnet_d_features.extend(o.cpu().data)

######### resnet light
trn_resnet_l_features = []
for d,la in train_loader:
    o = resnet_light(Variable(d.cuda()))
    o = o.view(o.size(0),-1)
    # trn_labels.extend(la)
    trn_resnet_l_features.extend(o.cpu().data)

val_resnet_l_features = []
for d,la in val_loader:
    o = resnet_light(Variable(d.cuda()))
    o = o.view(o.size(0),-1)
    # val_labels.extend(la)
    val_resnet_l_features.extend(o.cpu().data)

############### resnet reg
trn_resnet_r_features = []
for d,la in train_loader:
    o = resnet_reg(Variable(d.cuda()))
    o = o.view(o.size(0),-1)
    # trn_labels.extend(la)
    trn_resnet_r_features.extend(o.cpu().data)

val_resnet_r_features = []
for d,la in val_loader:
    o = resnet_reg(Variable(d.cuda()))
    o = o.view(o.size(0),-1)
    # val_labels.extend(la)
    val_resnet_r_features.extend(o.cpu().data)



# create ensemble train and val datasets
trn_feat_dset = FeaturesDataset(trn_resnet_l_features,trn_resnet_d_features,trn_resnet_r_features,trn_labels)
val_feat_dset = FeaturesDataset(val_resnet_l_features,val_resnet_d_features,val_resnet_r_features,val_labels)

# create dataloader to use for both datasets
trn_feat_loader = DataLoader(trn_feat_dset,batch_size=32,shuffle=True)
val_feat_loader = DataLoader(val_feat_dset,batch_size=32)

# create ensemble model of 3 linear layers with overfitting controlled by dropout
class EnsembleModel(nn.Module):
    
    def __init__(self,out_size,training=True):
        super().__init__()
        self.fc1 = nn.Linear(8192,512)
        self.fc2 = nn.Linear(8192,512)
        self.fc3 = nn.Linear(8192,512)
        self.fc4 = nn.Linear(512,out_size)

    def forward(self,inp1,inp2,inp3):
        out1 = self.fc1(F.dropout(inp1,training=self.training))
        out2 = self.fc2(F.dropout(inp2,training=self.training))
        out3 = self.fc3(F.dropout(inp3,training=self.training))
        out = out1 + out2 + out3
        out = self.fc4(F.dropout(out,training=self.training))
        return out

# send the model to gpu
em = EnsembleModel(7)
if is_cuda:
    em = em.cuda()

learning_rate = 0.001
# initialize the optimizer
# optimizer = optim.Adam(em.parameters(),lr=0.01)
optimizer = optim.SGD(em.parameters(), lr=.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

##########################################################################
# train the ensemble model
train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
for epoch in range(1,25):
    epoch_loss, epoch_accuracy = fit(epoch,em,trn_feat_loader,exp_lr_scheduler,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,em,val_feat_loader,exp_lr_scheduler,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

# plot the training and validation accuracy
plt.plot(range(1,len(train_accuracy)+1), train_accuracy,'bo',label = 'train accuracy')
plt.plot(range(1,len(val_accuracy)+1), val_accuracy,'r', label='val accuracy')
plt.legend()
plt.show()

# for epoch in range(1,25):
#     epoch_loss, epoch_accuracy = fit(epoch,em,trn_feat_loader,phase='training')
#     val_epoch_loss , val_epoch_accuracy = fit(epoch,em,val_feat_loader,phase='validation')
#     train_losses.append(epoch_loss)
#     train_accuracy.append(epoch_accuracy)
#     val_losses.append(val_epoch_loss)
#     val_accuracy.append(val_epoch_accuracy)

# for epoch in range(1,25):
#     epoch_loss, epoch_accuracy = fit(epoch,em,trn_feat_loader,phase='training')
#     val_epoch_loss , val_epoch_accuracy = fit(epoch,em,val_feat_loader,phase='validation')
#     train_losses.append(epoch_loss)
#     train_accuracy.append(epoch_accuracy)
#     val_losses.append(val_epoch_loss)
#     val_accuracy.append(val_epoch_accuracy)

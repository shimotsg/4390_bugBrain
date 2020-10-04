# Sasank Chilamkurthy https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py
# Nathan Inkawhich https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#alexnet

# modified by gs

#from __future__ import print_function 
#from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.functional as F
from torch.optim import lr_scheduler



# finetunes or feature extracts 

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "./data"

# Models to choose from [resnet18, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 7

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
num_epochs = 15

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

# data set selection
train_set = "train_dark"
val_set = "val_dark"
#################################################
# file handling
# path to images folder
# path = './all'
# # read all the files inside the folder
# files = glob(os.path.join(path,'*/*.jpg'))
# print(f'Total no of images {len(files)}')
# # get number of images
# no_of_images = len(files)

# # create shuffled index to get random sample for validation set
# shuffle = np.random.permutation(no_of_images)
# # create validation directory for holdign validation images
# os.mkdir(os.path.join(path,'valid'))

# # create directories with label names
# for t in ['train','valid']:
#     for folder in ['dog/','cat/']:
#         os.mkdir(os.path.join(path,t,folder))

# # copy subset of images into validatoin folder
# for i in shuffle[:2000]:
#     #shutil.copyfile(files[i],'../chapter3/dogsandcats/valid/')
#     folder = files[i].split('/')[-1].split('.')[0]
#     image = files[i].split('/')[-1]
#     os.rename(files[i],os.path.join(path,'valid',folder,image))

# # copy subset of images into training folder
# for i in shuffle[2000:]:
#     #shutil.copyfile(files[i],'../chapter3/dogsandcats/valid/')
#     folder = files[i].split('/')[-1].split('.')[0]
#     image = files[i].split('/')[-1]
#     os.rename(files[i],os.path.join(path,'train',folder,image))


#################################################
# helper functions
# takes Pytorch model, dictionary of dataloaders, loss function, optimizer, num epochs to run, inception flag
# returns best performing model
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [train_set, val_set]:
            if phase == train_set:
                model.train()  # Set model to training mode
                scheduler.step() # decay learning rate
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == train_set):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == train_set:
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == train_set:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == val_set and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == val_set:
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save the model state_dict
    torch.save(model.state_dict(), data_dir + train_set + '18.pth')
    # save general checkpoint model and optimizer state_dict
    # torch.save({
    #     'model.state_dict': model.state_dict(),
    #     'optimizer.state_dict': optimizer.state_dict()
    #     }, data_dir + train_set + '.tar')


    return model, val_acc_history

# set model parameters ie if feature extracting sets requires_grad to false for all params
# default requires_grad is true is for finetune or training from scratch
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# creates and initializes pytorch model, reshapes last fc layer to have same # of outputs as number of classes
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        # after loading model but before reshaping, if feature extracting, all param requires_grad set to False
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        # reshape final fc layer to have same number of outputs as number of classes
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size
# ######################################################################################################
# train and validate a single model

# Initialize the model for this run
model_train_this, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_train_this)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    train_set: transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    val_set: transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [train_set, val_set]}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in [train_set, val_set]}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_train_this = model_train_this.to(device)

# gather params to be optimized/updated in this run; finetuning => all params
# feature extracting => only initialized params ie params w/ requires_grad = True 
params_to_update = model_train_this.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_train_this.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_train_this.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_train_this = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# learning rate scheduler
learning_rate = 0.001
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_train_this, step_size=7, gamma=0.1)

# Setup the loss fxn
criterion_train_this = nn.CrossEntropyLoss()

# Train and evaluate
model_train_this, hist = train_model(model_train_this, dataloaders_dict, criterion_train_this, optimizer_train_this,exp_lr_scheduler, num_epochs=num_epochs, is_inception=(model_name=="inception"))
# #######################################################################################################
# train and validate ensembled model
# modified dataset
# modified dataloader
# modified train
# Data augmentation and normalization for training
# Just normalization for validation
# custom dataset to accomodate features generated from all 3 models
# class EnsembleFeaturesDataset(Dataset):
    
#     def __init__(self,featlst1,featlst2,featlst3,labellst):
#         self.featlst1 = featlst1
#         self.featlst2 = featlst2
#         self.featlst3 = featlst3
#         self.labellst = labellst
        
#     def __getitem__(self,index):
#         return (self.featlst1[index],self.featlst2[index],self.featlst3[index],self.labellst[index])
    
#     def __len__(self):
#         return len(self.labellst)

# # this is the problem causing mismatch does not work with 224 or 299 image size
# input_size = 256
# data_transforms = {
#     train_set: transforms.Compose([
#         transforms.RandomResizedCrop(input_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     val_set: transforms.Compose([
#         transforms.Resize(input_size),
#         transforms.CenterCrop(input_size),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

# print("Initializing Datasets and Dataloaders...")

# # Create training and validation datasets
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [train_set, val_set]}
# # Create training and validation dataloaders
# dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in [train_set, val_set]}
# # Detect if we have a GPU available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # 
# class EnsembleModel(nn.Module):
    
#     def __init__(self, out_size, training=True):
#         super().__init__()
#         self.resnet_dark = models.resnet34(pretrained=True)
#         num_ftrs = self.resnet_dark.fc.in_features
#         self.resnet_dark.fc = nn.Linear(num_ftrs, out_size)
#         self.resnet_dark.load_state_dict(torch.load('./datatrain_dark.pth'))
#         # handle varying input size does not work, causes sizemismatch 4096x1
#         # self.resnet_dark.avgpool = nn.AdaptiveAvgPool2d(1)
#         # turn layers of model into feature extractor
#         self.resnet_dark = nn.Sequential(*list(self.resnet_dark.children())[:-1])
#         for param in self.resnet_dark.parameters():
#             param.requires_grad = False
#         self.fc_d = nn.Linear(8192, 512)

#         self.resnet_light = models.resnet34(pretrained=True)
#         num_ftrs = self.resnet_light.fc.in_features
#         self.resnet_light.fc = nn.Linear(num_ftrs, out_size)
#         self.resnet_light.load_state_dict(torch.load('./datatrain_light.pth'))
#         # handle varying input size
#         # self.resnet_light.avgpool = nn.AdaptiveAvgPool2d(1)
#         # turn layers of model into feature extractor
#         self.resnet_light = nn.Sequential(*list(self.resnet_light.children())[:-1])
#         for param in self.resnet_light.parameters():
#             param.requires_grad = False
#         self.fc_l = nn.Linear(8192, 512)

#         self.resnet_reg = models.resnet34(pretrained=True)
#         num_ftrs = self.resnet_reg.fc.in_features
#         self.resnet_reg.fc = nn.Linear(num_ftrs, out_size)
#         self.resnet_reg.load_state_dict(torch.load('./datatrain_dark.pth'))
#         # handle varying input size
#         # self.resnet_reg.avgpool = nn.AdaptiveAvgPool2d(1)
#         # turn layers of model into feature extractor
#         self.resnet_reg = nn.Sequential(*list(self.resnet_reg.children())[:-1])
#         for param in self.resnet_reg.parameters():
#             param.requires_grad = False
#         self.fc_r = nn.Linear(8192, 512)

#         # self.resnet_avg = nn.AdaptiveAvgPool2d(out_size)
#         self.fc_f = nn.Linear(512, out_size)

#     def forward(self,inp):
#         # out1 = self.fc_d(F.dropout(self.resnet_dark(inp),training=self.training))
#         # out1 = self.fc_d(F.dropout(self.resnet_dark(inp),training=self.training))
#         # out1 = self.fc_d(F.dropout(self.resnet_dark(inp),training=self.training))
#         # out2 = self.fc_l(self.resnet_light(inp))
#         # out3 = self.fc_r(self.resnet_reg(inp))
#         out1 = self.resnet_dark(inp)
#         out2 = self.resnet_light(inp)
#         out3 = self.resnet_reg(inp)
        
#         out4 = self.fc_l(out2)
#         out5 = self.fc_d(out1)
#         out6 = self.fc_r(out3)

#         out = out4 + out5 + out6
#         #out = self.resnet_avg(F.dropout(out,training=self.training))
#         out = self.fc_f(F.dropout(out,training=self.training))

#         return out

# # train and validate ensemble model
# model_train_this = EnsembleModel(7)

# # send to GPU
# model_train_this.cuda()

# print(model_train_this)
# # initialize optimizer
# # gather params to be optimized/updated in this run; finetuning => all params
# # feature extracting => only initialized params ie params w/ requires_grad = True 
# params_to_update = model_train_this.parameters()
# print("Params to learn:")
# if feature_extract:
#     params_to_update = []
#     for name,param in model_train_this.named_parameters():
#         if param.requires_grad == True:
#             params_to_update.append(param)
#             print("\t",name)
# else:
#     for name,param in model_train_this.named_parameters():
#         if param.requires_grad == True:
#             print("\t",name)


# # Observe that all parameters are being optimized
# optimizer_train_this = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# # learning rate scheduler
# learning_rate = 0.001
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_train_this, step_size=7, gamma=0.1)

# # Setup the loss fxn
# criterion_train_this = nn.CrossEntropyLoss()

# # Train and evaluate
# model_train_this, hist = train_model(model_train_this, dataloaders_dict, criterion_train_this, optimizer_train_this,exp_lr_scheduler, num_epochs=num_epochs, is_inception=(model_name=="inception"))

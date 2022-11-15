#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision import models
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin

import matplotlib.pyplot as plt
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# In[ ]:


# # Plot some training images
# real_batch = next(iter(train_loader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


# In[ ]:


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)    


# In[ ]:


# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 10

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# In[ ]:


# ### Load existing Generator and Discriminator

# In[2]:



# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)

# Print the model
print(netG)


# In[ ]:


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)


# In[ ]:


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

gen_path="models/GANs/netG1.p"
netG.load_state_dict(torch.load(gen_path,map_location=device))

disc_path="models/GANs/netD1.p"
netD.load_state_dict(torch.load(disc_path,map_location=device))


# ### Generate GAN images

# In[3]:


b_size=2000
noise = torch.randn(b_size, nz, 1, 1, device=device)
# Generate fake image batch with G
fake = netG(noise)

# normalize the values
GAN_fakes=(fake-torch.min(fake))/(torch.max(fake)-torch.min(fake))


# ### Generate images using VAEs

# In[4]:


image_dic={}


# #### Generate images using Vanilla VAE model

# In[5]:


model_nm="VanillaVAE"
args_filename="configs/vae.yaml"
with open(args_filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        
model = vae_models[config['model_params']['name']](**config['model_params'])


chk_path="logs/"+model_nm+"/version_2/checkpoints/last.ckpt"

checkpoint = torch.load(chk_path,map_location=torch.device(device))
model=model.to(device)

for nm,params in model.named_parameters():
    keyy="model."+nm 
    params.data=checkpoint["state_dict"][keyy]
    
    
X_vals_enc_arr=np.load("logs/"+model_nm+"/enc/test_aug_enc.npy")
X_vals_enc_arr = torch.from_numpy(X_vals_enc_arr).float().to(device)
mid=X_vals_enc_arr.shape[1]//2

with torch.no_grad():
    
    mu=X_vals_enc_arr[:,:mid]
    log_var=X_vals_enc_arr[:,mid:]

    mu=torch.tensor(mu)
    log_var=torch.tensor(log_var)
    print(mu.shape,log_var.shape)
    z = model.reparameterize(mu, log_var)
    z=z.to(device)
    print("z is ",z.shape)
    
    images=model.decode(z)    
    
# normalize the values
Vanilla_images=(images-torch.min(images))/(torch.max(images)-torch.min(images))


image_dic["Vanilla_images"]=Vanilla_images


# #### Generate images using Conditional VAE model

# In[6]:


model_nm="ConditionalVAE"
args_filename="configs/cvae.yaml"
with open(args_filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        
model = vae_models[config['model_params']['name']](**config['model_params'])

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
chk_path="logs/"+model_nm+"/version_0/checkpoints/last.ckpt"

checkpoint = torch.load(chk_path,map_location=torch.device(device))


for nm,params in model.named_parameters():
    keyy="model."+nm 
    params.data=checkpoint["state_dict"][keyy]
    
    
X_vals_enc_arr=np.load("logs/"+model_nm+"/enc/test_aug_enc.npy")
X_vals_enc_arr = torch.from_numpy(X_vals_enc_arr).float().to(device)
mid=X_vals_enc_arr.shape[1]//2

with torch.no_grad():
    
    mu=X_vals_enc_arr[:,:128]
    log_var=X_vals_enc_arr[:,128:256]
    then_some=X_vals_enc_arr[:,256:]
    mu=torch.tensor(mu)
    log_var=torch.tensor(log_var)
    then_some=torch.tensor(then_some)
    z = model.reparameterize(mu, log_var)
    z = torch.cat([z, then_some], dim = 1)
    print(z.shape)
    images=model.decode(z)        
# normalize the values
ConditionalVAE_images=(images-torch.min(images))/(torch.max(images)-torch.min(images))
image_dic["ConditionalVAE_images"]=ConditionalVAE_images            


# #### Generate images using DFC VAE model

# In[7]:


model_nm="DFCVAE"
args_filename="configs/dfc_vae.yaml"
with open(args_filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        
model = vae_models[config['model_params']['name']](**config['model_params'])

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
chk_path="logs/"+model_nm+"/version_0/checkpoints/last.ckpt"

checkpoint = torch.load(chk_path,map_location=torch.device(device))


for nm,params in model.named_parameters():
    keyy="model."+nm 
    params.data=checkpoint["state_dict"][keyy]
    
    
X_vals_enc_arr=np.load("logs/"+model_nm+"/enc/test_aug_enc.npy")
X_vals_enc_arr = torch.from_numpy(X_vals_enc_arr).float().to(device)
mid=X_vals_enc_arr.shape[1]//2

with torch.no_grad():
    
    mu=X_vals_enc_arr[:,:mid]
    log_var=X_vals_enc_arr[:,mid:]
    mu=torch.tensor(mu)
    log_var=torch.tensor(log_var)
    print(mu.shape,log_var.shape)
    
    z = model.reparameterize(mu, log_var)    
    images=model.decode(z)  
    print(images.shape)
    
    
    
    
# normalize the values
DFCVAE_images=(images-torch.min(images))/(torch.max(images)-torch.min(images))
image_dic["DFCVAE_images"]=DFCVAE_images                


# #### Generate images using Beta VAE model

# In[8]:


model_nm="BetaVAE"
args_filename="configs/bbvae.yaml"

with open(args_filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        
model = vae_models[config['model_params']['name']](**config['model_params'])

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
chk_path="logs/"+model_nm+"/version_0/checkpoints/last.ckpt"

checkpoint = torch.load(chk_path,map_location=torch.device(device))


for nm,params in model.named_parameters():
    keyy="model."+nm 
    params.data=checkpoint["state_dict"][keyy]
    
    
X_vals_enc_arr=np.load("logs/"+model_nm+"/enc/test_aug_enc.npy")
X_vals_enc_arr = torch.from_numpy(X_vals_enc_arr).float().to(device)
mid=X_vals_enc_arr.shape[1]//2

with torch.no_grad():
    
    mu=X_vals_enc_arr[:,:mid]
    log_var=X_vals_enc_arr[:,mid:]

    mu=torch.tensor(mu)
    log_var=torch.tensor(log_var)
    print(mu.shape,log_var.shape)
    z = model.reparameterize(mu, log_var)    
    images=model.decode(z)    
    
    
# normalize the values
BetaVAE_images=(images-torch.min(images))/(torch.max(images)-torch.min(images))
image_dic["BetaVAE_images"]=BetaVAE_images                


# #### Generate using MIWAE model

# In[9]:


# causes memory problem


# In[10]:


# model_nm="MIWAE"
# args_filename="configs/miwae.yaml"
# with open(args_filename, 'r') as file:
#     try:
#         config = yaml.safe_load(file)
#     except yaml.YAMLError as exc:
#         print(exc)
        
# model = vae_models[config['model_params']['name']](**config['model_params'])

# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# chk_path="logs/"+model_nm+"/version_0/checkpoints/last.ckpt"

# checkpoint = torch.load(chk_path,map_location=torch.device(device))


# for nm,params in model.named_parameters():    
#     keyy="model."+nm 
#     params.data=checkpoint["state_dict"][keyy]
    
    
# X_vals_enc_arr=np.load("logs/"+model_nm+"/enc/test_aug_enc.npy")
# mid=X_vals_enc_arr.shape[1]//2

# with torch.no_grad():
#     print("Creating images")
    
#     mu=X_vals_enc_arr[:,:mid]
#     log_var=X_vals_enc_arr[:,mid:]
#     mu=torch.tensor(mu)
#     log_var=torch.tensor(log_var)
    
#     mu = mu.repeat(model.num_estimates, model.num_samples, 1, 1).permute(2, 0, 1, 3) # [B x M x S x D]
#     log_var = log_var.repeat(model.num_estimates, model.num_samples, 1, 1).permute(2, 0, 1, 3) # [B x M x S x D]
#     print(mu.shape,log_var.shape)
    
    
#     print(mu.shape,log_var.shape)
#     z = model.reparameterize(mu, log_var)    
#     print("Done reparam",z.shape)
#     images=model.decode(z)  
#     print("done decoding")
#     print(images.shape)
    
#     images=images[:, 0, 0, :]
#     print(images.shape)    
    
    
# # normalize the values
# MIWAE_images=(images-torch.min(images))/(torch.max(images)-torch.min(images))
# image_dic["MIWAE_images"]=MIWAE_images                    


# #### Generate using MSSIMVAE model

# In[11]:


model_nm="MSSIMVAE"
args_filename="configs/mssim_vae.yaml"
with open(args_filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        
model = vae_models[config['model_params']['name']](**config['model_params'])

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
chk_path="logs/"+model_nm+"/version_0/checkpoints/last.ckpt"

checkpoint = torch.load(chk_path,map_location=torch.device(device))


for nm,params in model.named_parameters():
    keyy="model."+nm 
    params.data=checkpoint["state_dict"][keyy]
    
    
X_vals_enc_arr=np.load("logs/"+model_nm+"/enc/test_aug_enc.npy")
X_vals_enc_arr = torch.from_numpy(X_vals_enc_arr).float().to(device)
mid=X_vals_enc_arr.shape[1]//2

with torch.no_grad():
    
    mu=X_vals_enc_arr[:,:mid]
    log_var=X_vals_enc_arr[:,mid:]
    mu=torch.tensor(mu)
    log_var=torch.tensor(log_var)
    
    print(mu.shape,log_var.shape)
    z = model.reparameterize(mu, log_var)    
    images=model.decode(z)  
    print(images.shape)
    
# normalize the values
MSSIMVAE_images=(images-torch.min(images))/(torch.max(images)-torch.min(images))
image_dic["MSSIMVAE_images"]=MSSIMVAE_images                        


# #### Generate using WAE_MMD model

# In[12]:


model_nm="WAE_MMD"
args_filename="configs/wae_mmd_imq.yaml"
with open(args_filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        
model = vae_models[config['model_params']['name']](**config['model_params'])

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
chk_path="logs/"+model_nm+"/version_0/checkpoints/last.ckpt"

checkpoint = torch.load(chk_path,map_location=torch.device(device))


for nm,params in model.named_parameters():
    keyy="model."+nm 
    params.data=checkpoint["state_dict"][keyy]
    
X_vals_enc_arr=np.load("logs/"+model_nm+"/enc/test_aug_enc.npy")    
X_vals_enc_arr = torch.from_numpy(X_vals_enc_arr).float().to(device)
with torch.no_grad():    
    images=model.decode(torch.tensor(X_vals_enc_arr).float())    
    print(images.shape)
    
# normalize the values
WAE_MMD_images=(images-torch.min(images))/(torch.max(images)-torch.min(images))
image_dic["WAE_MMD_images"]=WAE_MMD_images                            


# ### Load the classifiers

# In[13]:


model_dic={}


# #### AlexNet

# In[14]:


alex_model = models.alexnet(pretrained=True)
# Here the size of each output sample is set to 2.
alex_model.classifier[6] = nn.Linear(4096,2)
alex_model = alex_model.to(device)
PATH="../../AFaceDetector/models/s1.pt"
alex_model.load_state_dict(torch.load(PATH))
alex_model.eval()



model_dic["alexnet"]=alex_model


# #### VGGNet

# In[15]:


model_vgg=models.vgg16(pretrained=True)
model_vgg.classifier[6]=nn.Linear(4096,2)
model_vgg=model_vgg.to(device)
PATH="../../AFaceDetector/models/1_VGGnet.pt"
model_vgg.load_state_dict(torch.load(PATH,map_location=device))
model_vgg.eval()



model_dic["vggnet"]=model_vgg


# #### ResNet

# In[16]:


# model_Resnet = models.resnet18(pretrained=True)
# num_ftrs = model_Resnet.fc.in_features
# model_Resnet.fc = nn.Linear(num_ftrs, 2)
# model_Resnet = model_Resnet.to(device)



# PATH="../../AFaceDetector/models/1_Renset.pt"
# model_Resnet.load_state_dict(torch.load(PATH,map_location=device))
# model_Resnet.eval()



# model_dic["resnet"]=model_Resnet


# ### DenseNet

# In[23]:


model_denseNet = models.densenet121(pretrained=True)

# Here the size of each output sample is set to 2.
model_denseNet.classifier = nn.Linear(1024, 2)
model_denseNet = model_denseNet.to(device)

PATH="../../AFaceDetector/models/1_Denseset.pt"
model_denseNet.load_state_dict(torch.load(PATH,map_location=device))
model_denseNet.eval()


# densenet takes too long and kernel death
# model_dic["densenet"]=model_denseNet


# In[18]:


model_dic.keys()


# ### Result on GAN

# In[ ]:


labels = torch.full((b_size,), fake_label, dtype=torch.long, device=device)
# Classify all fake batch with D

for model_name,model in model_dic.items():
    print(model_name)
    outputs = model(GAN_fakes)
    _,preds=torch.max(outputs,1)
    running_corrects = torch.sum(preds == labels.data)
    acc=running_corrects/labels.shape[0]
    print("DCGAN",model_name , "Accuracy is ",acc)


# ### Result on VAEs

# #### Loop through dict

# In[1]:


for name,images_vae in image_dic.items():
#     print(name,images_vae.shape)
    b_size=images_vae.shape[0]
    labels = torch.full((b_size,), fake_label, dtype=torch.long, device=device)
    
    for model_name,model in model_dic.items():
        outputs = model(images_vae)
        _,preds=torch.max(outputs,1)
        running_corrects = torch.sum(preds == labels.data)
        acc=running_corrects/labels.shape[0]
        print(name,model_name,"Accuracy is ",acc)


# #### Vanilla VAE

# In[21]:


# labels = torch.full((b_size,), fake_label, dtype=torch.long, device=device)
# # Classify all fake batch with D
# outputs = alex_model(Vanilla_images)
# _,preds=torch.max(outputs,1)
# running_corrects = torch.sum(preds == labels.data)
# acc=running_corrects/labels.shape[0]
# print("Accuracy is ",acc)


# #### Conditional VAE

# In[22]:



# b_size=ConditionalVAE_images.shape[0]
# labels = torch.full((b_size,), fake_label, dtype=torch.long, device=device)
# # Classify all fake batch with D
# outputs = alex_model(ConditionalVAE_images)
# _,preds=torch.max(outputs,1)
# running_corrects = torch.sum(preds == labels.data)
# acc=running_corrects/labels.shape[0]
# print("Accuracy is ",acc)


# In[ ]:





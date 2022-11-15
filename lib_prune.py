
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

from copy import deepcopy

import numpy as np
import math
import random

import copy

def create_mask_from_mean_wt(model,mean_weight_description,prune_rate):
    mask_whole_model=[]
    for nm, params in model.named_parameters():
        if "weight" in nm and "bn" not in nm and "linear" not in nm:
#             print(nm)
            mask_layer=torch.ones(params.shape)    
            mean_wt_layer=mean_weight_description[nm]
            wts_this_layer=[]
            wts=mean_weight_description[nm]
            abs_var=torch.std(wts.flatten())
            threshold=abs_var*prune_rate
            
            these_wts=copy.deepcopy(params.data)
            these_wts=these_wts.flatten()
            mask_layer=mask_layer.flatten()
            
            for i in range(these_wts.shape[0]):
                if torch.abs(these_wts[i])<threshold:
                    mask_layer[i]=0
            mask_layer=torch.reshape(mask_layer,params.data.shape)
#             print(nm,params.shape,mask_layer.shape,abs_var,threshold)
            mask_whole_model.append(mask_layer)
            
    return mask_whole_model
            
    
def get_weighted_mean(state_dicts,keyy,importance_vector):
    sum_val=0
    for i in range(len(importance_vector)):
        importance=importance_vector[i]
        wt_vals=state_dicts[i][keyy]
        importance_wt_vals=importance*wt_vals
        sum_val+=importance_wt_vals
    return sum_val



def apply_mask_model(model,list_mask_whole_model,layer_to_prune=None):
    mask_layer_count=0
    for nm, params in model.named_parameters():        
        if "weight" in nm and "bn" not in nm and "linear" not in nm:
#             print(mask_layer_count,layer_to_prune)
            if layer_to_prune is not None:
                if mask_layer_count>layer_to_prune:
#                     print(mask_layer_count,layer_to_prune,"returning model")
                    return model
            
            
            mask_layer=list_mask_whole_model[mask_layer_count]
            with torch.no_grad():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#                 print("pruning layer",mask_layer_count)
                mask_layer=mask_layer.to(device)    
                params.data=params.data*mask_layer            
            mask_layer_count+=1
    return model


def nonzero(tensor):

    return np.sum(tensor != 0.0)


def model_size(model, as_bits=False):
    
    

    total_params = 0
    nonzero_params = 0
    for tensor in model.parameters():
        t = np.prod(tensor.shape)
        nz = nonzero(tensor.detach().cpu().numpy())
        if as_bits:
            bits = dtype2bits[tensor.dtype]
            t *= bits
            nz *= bits
        total_params += t
        nonzero_params += nz
    return int(total_params), int(nonzero_params)    
# Author: Tran Minh Quan
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#-----------------------------------------------------------------------
import os
import sys
import glob
import random
import shutil
import logging
import argparse
from collections import OrderedDict
from natsort import natsorted
import math
import cv2
import numpy as np
#-----------------------------------------------------------------------
# Using torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.backends.cudnn as cudnn

import torchvision
import torchvision.models as models

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

#-----------------------------------------------------------------------
# # Using tensorflow
# import tensorflow as tf

#-----------------------------------------------------------------------
# An efficient dataflow loading for training and testing
# import tensorpack.dataflow as df

from tensorpack import dataflow, imgaug
from tensorpack.dataflow import *
import albumentations as AB

from test_tube import Experiment
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

#-----------------------------------------------------------------------
# Global configuration
#
DEBUG = False 

#-----------------------------------------------------------------------
# Custom module here
#
from dataflow.PretrainedSNEMI import *
from optim.RAdam import RAdam
from losses.FocalLoss import FocalLoss

#-----------------------------------------------------------------------
# Pretrained from well-known models
#
MODEL_NAMES = sorted(
    name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__") and callable(torchvision.models.__dict__[name])
)
print(MODEL_NAMES)



class PretrainedSNEMIModule(pl.LightningModule):

    def __init__(self, hparams):
        super(PretrainedSNEMIModule, self).__init__()
        self.hparams = hparams # architecture is stored here
        print(self.hparams)
        pass

    def forward(self, x):
        estim = self.model(x) # Focal loss already has sigmoid
        return estim

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        pass

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        pass

    def configure_optimizers(self):
        optimizer = RAdam(self.model.parameters(), lr=self.hparams.lr, betas=(0.9, 0.99), weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_nb):
        pass

    def validation_step(self, batch, batch_nb):
        pass

    def validation_end(self, outputs):
        pass

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet121', choices=MODEL_NAMES,
                            help='model architecture: ' +
                                 ' | '.join(MODEL_NAMES) +
                                 ' (default: densenet121)')
        parser.add_argument('--epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=2020,
                            help='seed for initializing training. ')
        parser.add_argument('-b', '--batch', default=16, type=int,
                            help='mini-batch size (default: 16), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('-lr', '--learning_rate', default=0.01, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        return parser
        
def get_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--data_path', metavar='DIR', type=str, default="/u01/data/CXR/CheXpert-v1.0/", 
                               help='path to dataset')
    parent_parser.add_argument('--save_path', metavar='DIR', type=str, default="train_log_pytorch", 
                               help='path to save output')
    parent_parser.add_argument('--gpus', type=int, default=1,
                               help='how many gpus')
    parent_parser.add_argument('--pred', action='store_true',
                               help='run predict')
    parent_parser.add_argument('--eval', action='store_true', help='run offline evaluation instead of training')
    parent_parser.add_argument('--load', help='load model')
       
    parser = PretrainedSNEMIModule.add_model_specific_args(parent_parser)
    return parser.parse_args()

def eval_segmentation(model, **kwargs):
    pass

def main(hparams):
    model = PretrainedSNEMIModule(hparams)
    
    if hparams.seed is not None:
        random.seed(hparams.seed)
        np.random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    use_cuda = torch.cuda.is_available()
    xpu = torch.device("cuda:{}".format(hparams.gpus) if torch.cuda.is_available() else "cpu")

    if hparams.load is not None: # Load the checkpoint here
        chkpt = torch.load(hparams.load, map_location=xpu)
        model.load_state_dict(chkpt['state_dict'])
        print('Loaded from {}...'.format(hparams.load))

        if hparams.eval:
            pass
            sys.exit(0)
        elif hparams.pred:
            sys.exit(0)
    else:
        logger = TestTubeLogger(
            save_dir=os.path.join(hparams.save_path, str(hparams.arch), str(hparams.shape)),
            version=1  # An existing version with a saved checkpoint
        )
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join('checkpoint', str(hparams.arch), str(hparams.shape)), 
            save_best_only=False,
            verbose=True,
            monitor='valid/avg_loss', #TODO
            mode='min',
            prefix=''
        )
        trainer = pl.Trainer(
            default_save_path=os.path.join(hparams.save_path, str(hparams.arch), str(hparams.shape)),
            gpus=-1, #hparams.gpus,
            max_nb_epochs=hparams.epochs, 
            checkpoint_callback=checkpoint_callback, 
            early_stop_callback=None,
            # distributed_backend=hparams.distributed_backend,
            # use_amp=hparams.use_16bit
        )
        trainer.fit(model)


if __name__ == '__main__':
    main(get_args())


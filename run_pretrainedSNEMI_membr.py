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
# from losses.F1Loss import F1Loss
# from losses.FocalLoss import FocalLoss
from losses.DiceLoss import DiceLoss

# from models.fusionnet import Fusionnet
from models.uppnet import UPPNet

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
        self.criterion = DiceLoss() #F1Loss() #nn.modules.loss.L1Loss() #DiceLoss()
        self.model = UPPNet() #Fusionnet(input_nc=1, output_nc=1, ngf=64)

    def forward(self, x):
        estim = self.model(x) 
        estim = torch.tanh(estim)
        estim = (estim + 1.0) / 2.0
        return estim

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        ds_train = PretrainedSNEMI(folder=self.hparams.data_path,
            train_or_valid='train',
            size=40000,
            # resize=int(self.hparams.shape),
            debug=DEBUG
            )
        

        ag_train = [
            imgaug.RotationAndCropValid(max_deg=180, interp=cv2.INTER_NEAREST),
            imgaug.Flip(horiz=True, vert=False),
            imgaug.Flip(horiz=False, vert=True),
            imgaug.Transpose(),
            imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.2, 0.5), 
                    aspect_ratio_range=(0.5, 2.0),
                    interp=cv2.INTER_NEAREST, 
                    target_shape=self.hparams.shape),
            imgaug.ToFloat32(),
        ]

        ds_train.reset_state()
        # ds_train = AugmentImageComponent(ds_train, ag_train, 0)
        ds_train = AugmentImageComponent(ds_train, [imgaug.Albumentations(AB.RandomBrightnessContrast())], (0))
        ds_train = AugmentImageComponents(ds_train, ag_train, (0, 1))
        ds_train = MapData(ds_train, lambda dp: [dp[0], 255.0*(dp[1]>0)*(1-skimage.segmentation.find_boundaries(dp[1], mode='inner'))])
        ds_train = BatchData(ds_train, self.hparams.batch)
        ds_train = PrintData(ds_train)
        ds_train = MapData(ds_train, lambda dp: [torch.tensor(dp[0][:,np.newaxis,:,:]), torch.tensor(dp[1][:,np.newaxis,:,:]).float()])
        ds_train = MultiProcessRunner(ds_train, num_proc=32, num_prefetch=8)
        return ds_train

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        ds_valid = PretrainedSNEMI(folder=self.hparams.data_path,
            train_or_valid='train',
            size=100,
            # resize=int(self.hparams.shape),
            debug=DEBUG
            )
        

        ag_valid = [
            imgaug.RotationAndCropValid(max_deg=180, interp=cv2.INTER_NEAREST),
            imgaug.Flip(horiz=True, vert=False),
            imgaug.Flip(horiz=False, vert=True),
            imgaug.Transpose(),
            imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.2, 0.5), 
                    aspect_ratio_range=(0.5, 2.0),
                    interp=cv2.INTER_NEAREST, 
                    target_shape=self.hparams.shape),
            imgaug.ToFloat32(),
        ]

        ds_valid.reset_state()
        # ds_valid = AugmentImageComponent(ds_valid, ag_train, 0)
        ds_valid = AugmentImageComponents(ds_valid, ag_valid, (0, 1))
        ds_valid = MapData(ds_valid, lambda dp: [dp[0], 255.0*(dp[1]>0)*(1-skimage.segmentation.find_boundaries(dp[1], mode='inner'))])
        ds_valid = BatchData(ds_valid, self.hparams.batch)
        ds_valid = PrintData(ds_valid)
        ds_valid = MapData(ds_valid, lambda dp: [torch.tensor(dp[0][:,np.newaxis,:,:]), torch.tensor(dp[1][:,np.newaxis,:,:]).float()])
        ds_valid = MultiProcessRunner(ds_valid, num_proc=16, num_prefetch=4)
        return ds_valid

    def configure_optimizers(self):
        optimizer = RAdam(self.model.parameters(), lr=self.hparams.lr, betas=(0.9, 0.99), weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_nb):
        inputs, target = batch
        images = inputs / 255.0
        labels = target / 255.0
        output = self.forward(images * 2.0 - 1.0)

        loss = self.criterion(output, labels)
        # if batch_nb==0:
        #     imgs = torch.cat([images, labels, output], axis=-1)
        #     grid = torchvision.utils.make_grid(imgs, nrow=2)
        #     self.logger.experiment.add_image('train/stack', grid, 
        #                                      self.global_step, dataformats='CHW')
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        inputs, target = batch
        images = inputs / 255.0
        labels = target / 255.0
        output = self.forward(images * 2.0 - 1.0)

        loss = self.criterion(output, labels)
        # if batch_nb==0:
        #     imgs = torch.cat([images, labels, output], axis=-1)
        #     grid = torchvision.utils.make_grid(imgs, nrow=2)
        #     self.logger.experiment.add_image('valid/stack', grid, 
        #                                      self.global_step, dataformats='CHW')
        return {'valid/loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['valid/loss'] for x in outputs]).mean().cpu().numpy()
        self.logger.experiment.add_scalar('valid/loss', avg_loss, self.global_step)
        return {'valid/avg_loss': avg_loss}

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('-a', '--arch', metavar='ARCH', default='uppnet', #
                            help='model architecture: (default: uppnet)')
        parser.add_argument('--epochs', default=200, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=2020,
                            help='seed for initializing training. ')
        parser.add_argument('-b', '--batch', default=8, type=int,
                            help='mini-batch size (default: 8), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--shape', default=512, type=int, 
                            help='size of each image')
        parser.add_argument('-lr', '--learning_rate', default=2e-4, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        return parser
        
def get_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--data_path', metavar='DIR', type=str, default="/u01/data/SNEMI", 
                               help='path to dataset')
    parent_parser.add_argument('--save_path', metavar='DIR', type=str, default="train_log_pytorch", 
                               help='path to save output')
    parent_parser.add_argument('--gpus', type=int, default=1,
                               help='how many gpus')
    parent_parser.add_argument('--pred', action='store_true',
                               help='run predict')
    parent_parser.add_argument('--eval', action='store_true', help='run offline evaluation instead of training')
    parent_parser.add_argument('--load', help='load model')
    parent_parser.add_argument('--distributed_backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'))
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
            save_best_only=True,
            verbose=True,
            monitor='valid/avg_loss', #TODO
            mode='min',
            prefix=''
        )
        trainer = pl.Trainer(
            nb_sanity_val_steps=10, 
            default_save_path=os.path.join(hparams.save_path, str(hparams.arch), str(hparams.shape)),
            gpus=-1, #hparams.gpus,
            max_nb_epochs=hparams.epochs, 
            checkpoint_callback=checkpoint_callback, 
            early_stop_callback=None,
            distributed_backend=hparams.distributed_backend,
            # use_amp=hparams.use_16bit
        )
        trainer.fit(model)


if __name__ == '__main__':
    main(get_args())


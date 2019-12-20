import argparse
import glob2
import numpy as np
import pandas as pd
import os
import cv2
import skimage.io
# import tensorflow as tf
from natsort import natsorted

from tensorpack import *
from tensorpack.utils.argtools import shape2d
# from tensorpack.utils.viz import *
# from tensorpack.utils.gpu import get_nr_gpu

# from tensorpack.tfutils.summary import add_moving_summary
# from tensorpack.tfutils.varreplace import freeze_variables
# from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope


class PretrainedSNEMI(RNGDataFlow):
    """ Produce images read from a list of files as (h, w, c) arrays. """
    def __init__(self, folder, size=None, train_or_valid='train', channel=1, resize=None, debug=False, shuffle=False):
        super(PretrainedSNEMI, self).__init__()
        self.folder = folder
        self.is_train = True if train_or_valid=='train' else False
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        if resize is not None:
            resize = shape2d(resize)
        self.resize = resize
        self.shuffle = shuffle
        self._size = size
        
        self.images = []
        self.labels = []

        if self.is_train:
            self.imageDir = os.path.join(folder, 'trainA')
            self.labelDir = os.path.join(folder, 'trainB')
        else:
            self.imageDir = os.path.join(folder, 'validA')
            self.labelDir = os.path.join(folder, 'validB')

        imageFiles = natsorted (glob2.glob(self.imageDir + '/*.*'))
        labelFiles = natsorted (glob2.glob(self.labelDir + '/*.*'))
        
        print(imageFiles, labelFiles)

        for imageFile in imageFiles:
            image = skimage.io.imread (imageFile)
            self.images.append(image)
        for labelFile in labelFiles:
            label = skimage.io.imread (labelFile)
            self.labels.append(label)

        self.images = np.concatenate(self.images, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)



    def reset_state(self):
        self.rng = get_rng(self)   

    def __len__(self):
        return self._size

    def __iter__(self):
        # TODO
        indices = list(range(self.__len__()))
        if self.shuffle:
            self.rng.shuffle(indices)

        for _ in indices:
            idx = self.rng.randint(0, self.images.shape[0])

            image = self.images[idx].copy()
            label = self.labels[idx].copy()

            yield [image, label]

if __name__ == '__main__':
    ds = PretrainedSNEMI(folder='/home/drive_1TB/data_em_segmentation/SNEMI', 
        train_or_valid='train',
        resize=1024, 
        size=2000,
        )
    ds.reset_state()
    ds = PrintData(ds)
    ds = MultiProcessRunnerZMQ(ds, num_proc=8)
    ds = BatchData(ds, 16)
    TestDataSpeed(ds).start()
    
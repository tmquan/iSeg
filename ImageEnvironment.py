from natsort import natsorted
import os
import glob
import glob2
import cv2
import skimage.io
import skimage.measure
import sklearn
import sklearn.metrics
import numpy as np
from PIL import Image
# Using torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision
import gym

from environment.BaseEnvironment import BaseEnvironment
from models.uppnet import UPPNet

def prob2size(prob, size):
    # Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high)
    # prob from [0 to 1) -> size [0 to size)
    # prob2size(0.8, 512)     409
    # prob2size(1.0, 512)     511
    # prob2size(0.0, 512)     0
    return (int)(prob * (size - 1) + 0.5)

class ImageEnvironment(BaseEnvironment):
    def __init__(self, datadir=None, size=100, istrain=True, writer=None,
        ckpt=None, max_step_per_episode=200):
        assert datadir is not None
        self.size = size
        self.ckpt = ckpt
        self.istrain = istrain
        self._load_images(datadir=datadir)

        self.uppnet = UPPNet() #FusionnetModel(input_nc=2, output_nc=2, ngf=32)
        self._load_pretrained(self.ckpt)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,))
        self.global_step = 0
        self.global_episode = 0
        self.max_step_per_episode = max_step_per_episode
        self.writer = writer

        self.reset()

    def _load_pretrained(self, ckpt=None):

        # use_cuda = torch.cuda.is_available()
        xpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if ckpt:
            ckpt = torch.load(self.ckpt, map_location=xpu)

            # [k.replace('models.', '') for k,v in dict(ckpt['state_dict']).items()]
            odict = {}
            for k,v in dict(ckpt['state_dict']).items():
                odict[k.replace('model.', '')] = v

            # self.uppnet.load_state_dict(ckpt['state_dict'])
            self.uppnet.load_state_dict(odict)

            print('Loaded from {}...'.format(self.ckpt))

        else:
            return None


    def _load_images(self, datadir=None):
        # Manipulate the train and valid set
        if self.istrain:
            datadir = os.path.join(datadir, 'train')
        else:
            datadir = os.path.join(datadir, 'valid')

        # Read the image
        self.imagedir = os.path.join(datadir, 'images')
        self.imagefiles = natsorted(glob.glob(self.imagedir + '/*.*'))
        self.images = None
        for imagefile in self.imagefiles:
            image = skimage.io.imread(imagefile)
            self.images = image if self.images is None else np.concatenate([self.images, image], axis=0)

        # Read the label data
        self.labeldir = os.path.join(datadir, 'labels')
        self.labelfiles = natsorted(glob.glob(self.labeldir + '/*.*'))
        self.labels = None
        for labelfile in self.labelfiles:
            label = skimage.io.imread(labelfile)
            self.labels = label if self.labels is None else np.concatenate([self.labels, label], axis=0)
        self.size = len(self.imagefiles)

        # To list
        self.images = list(self.images)
        self.labels = list(self.labels)

    def reset(self):
        randidz = np.random.randint(len(self.images))
        self.image = self.images[randidz].copy()
        self.label = self.labels[randidz].copy()
        self.estim = np.zeros_like(self.image)
        self.shape = self.image.shape[0]

    def step(self, action):
        # Need to return obs, rwd, done, info 
        obs = None
        rwd = None
        done = None
        info = None
        print('Action is {}'.format(action))

        ww = prob2size(action[0], self.shape/2) + 1
        hh = prob2size(action[1], self.shape/2) + 1
        # ww = ww / 2
        # hh = hh / 2
        xx = prob2size(action[2], self.shape - ww)
        yy = prob2size(action[3], self.shape - hh)

        # self.estim = np.zeros_like(self.image)
        # self.estim[yy:yy+hh, xx:xx+ww] = self.image[yy:yy+hh, xx:xx+ww]
        self.alive = cv2.resize(self.image[yy:yy+hh, xx:xx+ww], (512, 512), cv2.INTER_LINEAR)

        # Run the deployment on alive ROI
        with torch.no_grad():
            self.proba = 255-self.uppnet.forward(torch.tensor(self.alive / 127.5 - 1.0).float().unsqueeze(0).unsqueeze(0)).detach().squeeze().numpy() #.astype(np.float32)
        self.proba = cv2.resize(self.proba, (ww, hh), cv2.INTER_NEAREST)
        self.estim[yy:yy+hh, xx:xx+ww] = self.proba.astype(np.uint8)

        cv2.imshow('step', np.concatenate([self.image, self.estim], axis=1))
        # cv2.waitKey(0)
        
        self.global_step += 1
        cv2.imwrite('step_{}.png'.format(str(self.global_step).zfill(3)), np.concatenate([self.image, self.estim], axis=1))
        return obs, rwd, done, info


if __name__ == '__main__':
    writer = SummaryWriter()
    env = ImageEnvironment(datadir='/home/tmquan/data/SNEMI', size=100,
                           ckpt='checkpoint/uppnet.ckpt',
                           writer=writer)

    # if args.mode == 'random':
    np.random.seed(2222)
    obs, rwd, done, info = env.step([0, 0, 0, 0])
    for _ in range(100):
        act = np.random.uniform(0, 1, 4)
        # print(act)
        obs, rwd, done, info = env.step(act)
        print(done)

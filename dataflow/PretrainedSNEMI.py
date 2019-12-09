import argparse
import glob2
import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from natsort import natsorted

from tensorpack import *
from tensorpack.utils.argtools import shape2d
from tensorpack.utils.viz import *
from tensorpack.utils.gpu import get_nr_gpu

from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.varreplace import freeze_variables
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

import tensorpack.tfutils.symbolic_functions as symbf

import tensorflow as tf

class PretrainedSNEMI(RNGDataFlow):
	""" Produce images read from a list of files as (h, w, c) arrays. """
    def __init__(self, **kwargs):
		super(PretrainedSNEMI, self).__init__()
		# TODO

	def reset_state(self):
        self.rng = get_rng(self)   

    def __len__(self):
    	# TODO
    	pass

    def __iter__(self):
    	# TODO
    	pass
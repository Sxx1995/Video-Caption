from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import opts
import random
import numpy as np
import collections
import os
import tensorflow as tf
from six.moves import cPickle
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import *
import sys
import time
import misc.utils as utils
from collections import OrderedDict

sys.path.append("misc/bleu")
sys.path.append("misc/cider")

opt = opts.parse_opt()

# CiderD_scorer = CiderD(df='corpus')

class LanguageModelCriterion(nn.Module):
    def __init__(self, opt):
        super(LanguageModelCriterion, self).__init__()
        self.opt = opt
        # Build loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        return loss
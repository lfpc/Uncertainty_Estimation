import os
import torch
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from torch import nn
import os
import wandb
import NN_models as models
import NN_utils as utils
import NN_utils.train_and_eval as TE
import torch_data
import uncertainty as unc
from uncertainty import metrics

REPOSITORY_PATH = r'/home/luis-felipe/UncEst'
DATA_PATH = os.path.join(REPOSITORY_PATH,'data')
#CORRUPTED_DATA_PATH = os.path.join(DATA_PATH,'corrupted')

PATH_MODELS = os.path.join(REPOSITORY_PATH,'torch_models')
PATH_TRAINER = os.path.join(PATH_MODELS,'trainer')

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
os.environ["WANDB_SILENT"] = "True"
wandb.login()




    
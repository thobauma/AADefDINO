import os
from pathlib import Path
import getpass
import numpy as np
import pandas as pd
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchattacks
from torchattacks import *
from tqdm import tqdm
import random
import sys


# allow imports when running script from within project dir
[sys.path.append(i) for i in ['.', '..']]

# local
# from src.helpers.helpers import get_random_indexes, get_random_classes
from src.model.dino_model import get_dino, ViTWrapper
from src.model.data import *
from src.model.train import *

# seed
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_PATH = Path('/','cluster', 'scratch', 'thobauma', 'dl_data')
MAX_PATH = Path('/','cluster', 'scratch', 'mmathys', 'dl_data')

LOG_BASE_PATH = Path(MAX_PATH, 'logs')

# DamageNet
DN_PATH = Path(DATA_PATH, 'damageNet')
DN_LABEL_PATH = Path(DN_PATH, 'val_damagenet.txt')
DN_IMAGES_PATH = Path(DN_PATH, 'images')

# Image Net
ORI_PATH = Path(DATA_PATH, 'ori_data')
CLASS_SUBSET_PATH = Path(ORI_PATH, 'class_subset.npy')

VAL_PATH = Path(ORI_PATH, 'validation')
VAL_IMAGES_PATH = Path(VAL_PATH,'images')
VAL_LABEL_PATH = Path(VAL_PATH, 'correct_labels.txt')

TRAIN_PATH = Path(ORI_PATH, 'train')
TRAIN_IMAGES_PATH = Path(TRAIN_PATH,'images')
TRAIN_LABEL_PATH = Path(TRAIN_PATH, 'correct_labels.txt')


# If CLASS_SUBSET is specified, INDEX_SUBSET will be ignored. Set CLASS_SUBSET=None if you want to use indexes.
# INDEX_SUBSET = get_random_indexes(number_of_images = 50000, n_samples=1000)
# CLASS_SUBSET = get_random_classes(number_of_classes = 10, min_rand_class = 1, max_rand_class = 1001)


CLASS_SUBSET = np.load(CLASS_SUBSET_PATH)
CLASS_SUBSET = CLASS_SUBSET[:4]


NUM_WORKERS= 0
PIN_MEMORY=True

BATCH_SIZE = 100

DEVICE = 'cuda'



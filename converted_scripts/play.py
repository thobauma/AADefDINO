# %%
import os
from pathlib import Path
import getpass
import numpy as np
import pandas as pd
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import sys
import tarfile

# allow imports when running script from within project dir
[sys.path.append(i) for i in ['.', '..']]

# local
from src.helpers.helpers import get_random_indexes, get_random_classes
from src.helpers.argparser import parser
from src.model.dino_model import get_dino, ViTWrapper
from src.model.data import *
rng = np.random.default_rng(42)
# %%

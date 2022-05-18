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
import torchvision
from tqdm import tqdm
import random
import sys

# allow imports when running script from within project dir
[sys.path.append(i) for i in ['.', '..']]

# local
# from src.helpers.helpers import get_random_indexes, get_random_classes
from src.model.dino_model import get_dino, ViTWrapper
from src.helpers.argparser import parser
from src.model.data import *
from src.model.train import *


# seed
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


# # Import DINO
# Official repo: https://github.com/facebookresearch/dino
class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=9):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == "__main__":
# %%
    args = parser.parse_args()
    TRAIN_PATH = args.filtered_data/'train'
    VALIDATION_PATH = args.filtered_data/'validation'


    # Remember to set the correct transformation
    train_dataset = AdvTrainingImageDataset(TRAIN_PATH/'images', TRAIN_PATH/'labels.csv', ORIGINAL_TRANSFORM)
    val_dataset = AdvTrainingImageDataset(VALIDATION_PATH/'images', VALIDATION_PATH/'labels.csv', ORIGINAL_TRANSFORM)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=False)

    print(f'''train:      {len(train_dataset)}\nvalidation:  {len(val_dataset)}''')

    model, base_linear_classifier = get_dino(args=args)


    # Logging path
    LOG_PATH = Path(args.log_dir, "base_lin_clf")


    # Init model each time
    classifier = LinearClassifier(base_linear_classifier.linear.in_features).cuda()
    vits = ViTWrapper(model, classifier)
    
# %%
    # Train
    loggers = train(model, 
            classifier,
            train_loader,
            val_loader, 
            LOG_PATH, 
            epochs=10,
            adversarial_attack=None,
            show_image=False,
            args=args
            )

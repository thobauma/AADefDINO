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
ORI_PATH = Path(DATA_PATH, 'ori')
CLASS_SUBSET_PATH = Path(ORI_PATH, 'class_subset.npy')

VAL_PATH = Path(ORI_PATH, 'validation')
VAL_IMAGES_PATH = Path(VAL_PATH,'images')
VAL_LABEL_PATH = Path(VAL_PATH, 'labels.csv')

TRAIN_PATH = Path(ORI_PATH, 'train')
TRAIN_IMAGES_PATH = Path(TRAIN_PATH,'images')
TRAIN_LABEL_PATH = Path(TRAIN_PATH, 'labels.csv')

# If CLASS_SUBSET is specified, INDEX_SUBSET will be ignored. Set CLASS_SUBSET=None if you want to use indexes.
# INDEX_SUBSET = get_random_indexes(number_of_images = 50000, n_samples=1000)
# CLASS_SUBSET = get_random_classes(number_of_classes = 10, min_rand_class = 1, max_rand_class = 1001)


CLASS_SUBSET = np.load(CLASS_SUBSET_PATH)
CLASS_SUBSET = CLASS_SUBSET[:10]


NUM_WORKERS= 0
PIN_MEMORY=True

BATCH_SIZE = 16

DEVICE = 'cuda'

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit([i for i in CLASS_SUBSET])

# Remember to set the correct transformation

train_dataset = AdvTrainingImageDataset(TRAIN_IMAGES_PATH, TRAIN_LABEL_PATH, ADVERSARIAL_TRAINING_TRANSFORM, CLASS_SUBSET, index_subset=None, label_encoder=label_encoder)
val_dataset = AdvTrainingImageDataset(VAL_IMAGES_PATH, VAL_LABEL_PATH, ORIGINAL_TRANSFORM, CLASS_SUBSET, index_subset=None, label_encoder=label_encoder)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)

print(f'train:      {len(train_dataset)}\nvalidation:  {len(val_dataset)}')

model, base_linear_classifier = get_dino()

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000, hidden_size=512):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        
        self.linear1 = nn.Linear(dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.linear3 = nn.Linear(int(hidden_size / 2), int(hidden_size / 4))
        self.linear4 = nn.Linear(int(hidden_size / 4), num_labels)
#         self.linear.weight.data.normal_(mean=0.0, std=0.01)
#         self.linear.bias.data.zero_()
#         self.linear2.weight.data.normal_(mean=0.0, std=0.01)
#         self.linear2.bias.data.zero_()
#         self.initialize()

        self.relu = nn.ReLU()
    
    def initialize(self):
        nn.init.normal_(self.linear1.weight, mean=0, std=1.0)
        nn.init.normal_(self.linear1.bias, mean=0, std=1.0)
        nn.init.normal_(self.linear2.weight, mean=0, std=1.0)
        nn.init.normal_(self.linear2.bias, mean=0, std=1.0)
        nn.init.normal_(self.linear3.weight, mean=0, std=1.0)
        nn.init.normal_(self.linear3.bias, mean=0, std=1.0)
        nn.init.normal_(self.linear4.weight, mean=0, std=1.0)
        nn.init.normal_(self.linear4.bias, mean=0, std=1.0)
        # nn.init.xavier_uniform(self.linear.weight.data)
        # self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        
        return self.linear4(x)

attacks = [
    (dict(eps=0.001, alpha=2/255, steps=10), '001'),
    (dict(eps=0.003, alpha=2/255, steps=10), '003'),
    (dict(eps=0.007, alpha=2/255, steps=10), '007'),
    (dict(eps=0.01, alpha=4/255, steps=10), '01'),
    (dict(eps=0.03, alpha=6/255, steps=10), '03'),
    (dict(eps=0.1, alpha=6/255, steps=10), '1'),
    (dict(eps=0.3, alpha=6/255, steps=10), '3')
]

if __name__ == "__main__":
    for attack, name in attacks:
        # Logging path
        LOG_PATH = Path(LOG_BASE_PATH, 'report', name)
        os.makedirs(LOG_PATH, exist_ok=True)
    
        # Init model each time
        pgd_classifier = LinearClassifier(base_linear_classifier.linear.in_features, num_labels=len(CLASS_SUBSET), hidden_size=2048).cuda()
        vits = ViTWrapper(model, pgd_classifier)
        
        train_attack = PGD(vits, eps=attack['eps'], alpha=attack['alpha'], steps=attack['steps'])
    
    # Train
        loggers = train(model, 
                pgd_classifier,
                train_loader,
                val_loader, 
                LOG_PATH, 
                epochs=5,
                adversarial_attack=train_attack,
                show_image=False
               )

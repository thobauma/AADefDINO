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
from src.helpers.argparser import parser
from src.model.data import *
from src.model.train import *


# seed
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)



# from sklearn import preprocessing

# label_encoder = preprocessing.LabelEncoder()
# label_encoder.fit([i for i in CLASS_SUBSET])


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
    (dict(eps=0.001, alpha=(0.001*2)/3, steps=3), 'pgd_001_new'),
   (dict(eps=0.003, alpha=(0.003*2)/3, steps=3), 'pgd_003_new'),
#    (dict(eps=0.007, alpha=(0.007*2)/3, steps=3), 'pgd_007'),
#    (dict(eps=0.01, alpha=(0.01*2)/3, steps=3), 'pgd_01_test'),
#    (dict(eps=0.03, alpha=(0.03*2)/3, steps=3), 'pgd_03'),
#    (dict(eps=0.05, alpha=(0.05 * 2) / 3, steps=3), 'pgd_05'),
#    (dict(eps=0.1, alpha=(0.1 * 2) / 3, steps=3), 'pgd_1'),
]

if __name__ == "__main__":

    args = parser.parse_args()
    TRAIN_PATH = args.filtered_data/'train'
    VALIDATION_PATH = args.filtered_data/'validation'

    # Remember to set the correct transformation
    train_dataset = AdvTrainingImageDataset(TRAIN_PATH/'images', TRAIN_PATH/'labels.csv', ORIGINAL_TRANSFORM, index_subset=None)
    val_dataset = AdvTrainingImageDataset(VALIDATION_PATH/'images', VALIDATION_PATH/'labels.csv', ORIGINAL_TRANSFORM, index_subset=None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=False)
    
    # Assert we included the classification head as argument
    assert args.head_path, "You should provide a path to the pretrained classification head"

    print(f'train:      {len(train_dataset)}\nvalidation:  {len(val_dataset)}')

    model, base_linear_classifier = get_dino(args=args)
    
    # Fixed head. Load from disk.
    base_linear_classifier.load_state_dict(torch.load(args.head_path)).cuda()
    
    # Build wrapper (backbone + head)
    vits = ViTWrapper(model, base_linear_classifier)

    for attack, name in attacks:
        # Logging path
        LOG_PATH = Path(args.log_dir, name)
    
        # Init model each time
        adversarial_classifier = LinearClassifier(base_linear_classifier.linear.in_features, num_labels=9, hidden_size=2048).cuda()
        
        train_attack = PGD(vits, eps=attack['eps'], alpha=attack['alpha'], steps=attack['steps'])
    
        # Train
        loggers = train(model, 
                adversarial_classifier,
                train_loader,
                val_loader, 
                LOG_PATH, 
                epochs=5,
                adversarial_attack=train_attack,
                show_image=False,
                args=args
               )

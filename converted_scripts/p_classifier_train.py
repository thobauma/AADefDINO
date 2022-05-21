from pathlib import Path
from collections import defaultdict

import numpy as np
import random
import sys
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch import nn

# allow imports when running script from within project dir
[sys.path.append(i) for i in ['.', '..']]

# local
from src.model.dino_model import get_dino
from src.model.train import *
from src.model.data import *
from src.helpers.helpers import create_paths
from src.helpers.argparser import parser

# seed
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)



# Linear Binary Classifier
class LinearBC(nn.Module):
    def __init__(self, input_shape):
        self.num_labels = 2
        super(LinearBC,self).__init__()
        self.fc1 = nn.Linear(input_shape,2)

    def forward(self, x):
        x = self.fc1(x)
        return x

def train_posthoc_classifier(adv_attacks, args):
    logger_dict = {}
    model, base_linear_classifier = get_dino(args=args)
    ORI_TRAIN_PATH = args.filtered_data/'train'
    ORI_VALIDATION_PATH = args.filtered_data/'validation'
    for name in adv_attacks:
        LOG_PATH = Path(args.log_dir, name+"_posthoc")
        ADV_DATA = Path(args.data_root, 'adv', name)
        print("#"*50 + f''' training linear classifier for {name} ''' + "#"*50)
        # loaders
        ori_train = ORI_TRAIN_PATH/'images'
        adv_train = Path(ADV_DATA,'train', 'images')
        print(f'''original images: {ori_train}''')
        print(f'''adversarial images: {adv_train}''')
        train_set = PosthocTrainDataset(ori_train, adv_train,ORI_TRAIN_PATH/'labels.csv', Path(ADV_DATA,'train','labels.csv'))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=True)
        

        ori_validation = ORI_VALIDATION_PATH/'images'
        adv_validation = Path(ADV_DATA, 'validation', 'images')
        print(f'''original images: {ori_validation}''')
        print(f'''adversarial images: {adv_validation}''')
        val_set = PosthocTrainDataset(ori_validation, adv_validation, ORI_VALIDATION_PATH/'labels.csv', Path(ADV_DATA,'validation','labels.csv'))
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=False)

        print(f'''train samples: {len(train_set)} ''')
        print(f'''val samples: {len(val_set)} \n''')

        # Initialise network
        classifier = LinearBC(base_linear_classifier.linear.in_features)
        criterion = nn.BCEWithLogitsLoss()
        classifier.cuda()
        optimizer = torch.optim.Adagrad(classifier.parameters(), lr=0.001, lr_decay=1e-08, weight_decay=0)
        logger_dict[name] = train(model=model, 
                                classifier=classifier, 
                                train_loader=train_loader, 
                                validation_loader=val_loader, 
                                log_dir=LOG_PATH,
                                tensor_dir=None, 
                                optimizer=optimizer, 
                                criterion=criterion, 
                                adversarial_attack=None, 
                                epochs=args.epochs, 
                                val_freq=args.val_freq, 
                                batch_size=args.batch_size, 
                                to_restore = {"epoch": 0, "best_acc": 0.}, 
                                n=args.n_last_blocks, 
                                avgpool_patchtokens=args.avgpool_patchtokens)

        print(f'''\n''')
    return logger_dict

if __name__ == "__main__":
    attacks = [
        # "pgd_0001",
        "pgd_003",
        "pgd_01",
        # "fgsm_0001",
        # "fgsm_003",
        # "fgsm_01",
        # "cw_50"
    ]
    args = parser.parse_args()

    loggers = train_posthoc_classifier(attacks, args)


from pathlib import Path
from collections import defaultdict
import pickle

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
from src.model.dino_model import get_dino, LinearBC
from src.model.train import *
from src.model.data import *
from src.helpers.helpers import create_paths
from src.helpers.argparser import parser

# seed
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)



if __name__ == '__main__':
    args = parser.parse_args()
    DATA = args.data_root
    POSTHOC_MATRIX_PATH = DATA/'posthoc_matrix'
    POSTHOC_MATRIX_PATH.mkdir(parents=True, exist_ok=True)

    model, base_linear_classifier = get_dino(args=args)
    ORI_TRAIN_PATH = args.filtered_data/'train'
    ORI_VALIDATION_PATH = args.filtered_data/'validation'
    logger_dict = defaultdict(dict)

    posthoc_models = [
        "pgd_0001",
        "pgd_003",
        "pgd_01"
    ]
    attack_datasets = [
        "pgd_0001",
        "pgd_003",
        "pgd_01",
        "fgsm_0001",
        "fgsm_003",
        "fgsm_01",
        "cw_50"
    ]

    for adv_classifier in posthoc_models:
        print("#"*50 + f''' forwardpass on {adv_classifier} classifier ''' + "#"*50)
        
        LOG_PATH = Path(args.log_dir, 'posthoc', adv_classifier)
        classifier = LinearBC(1536)
        classifier.cuda()

        to_restore={'epoch':1}

        utils.restart_from_checkpoint(
            Path(LOG_PATH, "checkpoint.pth.tar"),
            run_variables=to_restore,
            state_dict=classifier
        )
        
        for adv_data in attack_datasets:
            print("\n"+"-"*50 + f''' dataset {adv_data} ''' + "-"*50)
            ADV_DATA = Path(args.data_root, 'adv', adv_data)
            ori_validation = dataset_paths['ori']['posthoc']['validation']['images']
            adv_validation = dataset_paths[adv_data]['posthoc']['validation']['images']
            print(f'''original images: {ori_validation}''')
            print(f'''adversarial images: {adv_validation}''')
            val_set = PosthocTrainDataset(ori_validation, adv_validation, val_dfs[adv_data])
            val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=False)
            print(f'''val samples: {len(val_set)} \n''')
            logger_dict[adv_classifier][adv_data] =  validate_network(model=None, 
                                                    classifier=classifier, 
                                                    validation_loader=val_loader, 
                                                    criterion=nn.CrossEntropyLoss(), 
                                                    tensor_dir=None,
                                                    adversarial_attack=None,  
                                                    path_predictions=Path(POSTHOC_MATRIX_PATH, 'c_'+adv_classifier+'_d_'+adv_data+'.csv'),
                                                    show_image=False,
                                                    log_interval=10)

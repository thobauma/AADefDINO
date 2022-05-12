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


ADV_DATASETS = ['cw', 'fgsm_06', 'pgd_03']
DATASETS = ['ori', *ADV_DATASETS]



def prepare_data_df(adv_datasets, dataset_paths, posthoc_path):
    train_dfs = {}
    for ds in adv_datasets:
        train_dfs[ds] = pd.read_csv(Path(posthoc_path, ds, 'train', 'labels_merged.csv'))

    val_dfs = {}
    for ds in adv_datasets:
        val_dfs[ds] = pd.read_csv(Path(posthoc_path, ds, 'validation', 'labels_merged.csv'))

    # get adversarial tuples
    for name, df in train_dfs.items():
        df=df[df['true_labels']==df['ori_pred']]
        df=df[df['true_labels']!=df[name+'_pred']]
        df =df[['file', 'true_labels', 'ori_pred', name+'_pred']]
        train_dfs[name]=df
    return train_dfs, val_dfs






# Linear Binary Classifier
class LinearBC(nn.Module):
    def __init__(self, input_shape):
        self.num_labels = 2
        super(LinearBC,self).__init__()
        self.fc1 = nn.Linear(input_shape,2)

    def forward(self, x):
        x = self.fc1(x)
        return x


if __name__ == '__main__':
    args = parser.parse_args()
    DATA = args.data_root
    POSTHOC_MATRIX_PATH = DATA/'posthoc_matrix'
    POSTHOC_MATRIX_PATH.mkdir(parents=True, exist_ok=True)
    train_dfs, val_dfs = prepare_data_df(ADV_DATASETS, DATA_PATHS, args.posthoc_data)

    logger_dict = defaultdict(dict)

    

    for adv_classifier in ADV_DATASETS:
        print("#"*50 + f''' forwardpass on {adv_classifier} classifier ''' + "#"*50)
        
        log_dir = Path(POSTHOC_MODELS_PATH, adv_classifier)
        classifier = LinearBC(1536)
        classifier.cuda()

        to_restore={'epoch':3}

        utils.restart_from_checkpoint(
            Path(log_dir, "checkpoint.pth.tar"),
            run_variables=to_restore,
            state_dict=classifier
        )
        
        for adv_data in ADV_DATASETS:
            print("\n"+"-"*50 + f''' dataset {adv_data} ''' + "-"*50)
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

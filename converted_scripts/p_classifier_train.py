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

DATA_PATH = Path('..', 'data_dir')
MAX_PATH = DATA_PATH

BASE_ADV_PATH = Path(MAX_PATH, 'adversarial_data_tensors')
BASE_POSTHOC_PATH = Path(MAX_PATH, 'posthoc_tensors')
POSTHOC_MODELS_PATH = Path(MAX_PATH, 'posthoc_models')

ORI_PATH = Path(DATA_PATH, 'ori')
CLASS_SUBSET_PATH = Path(ORI_PATH, 'class_subset.npy')
CLASS_SUBSET = np.load(CLASS_SUBSET_PATH)

ADV_DATASETS = ['cw', 'fgsm_06', 'pgd_03']

DATASETS = [*ADV_DATASETS, 'ori']


# In[ ]:


DATA_PATHS = create_paths(data_name='ori',
                 datasets_paths=None,  
                 initial_base_path=DATA_PATH, 
                 posthoc_base_path=BASE_POSTHOC_PATH, 
                 train_str='train', 
                 val_str='validation')
for adv_ds in ADV_DATASETS:
    DATA_PATHS = create_paths(data_name=adv_ds,
                 datasets_paths=DATA_PATHS,  
                 initial_base_path=BASE_ADV_PATH, 
                 posthoc_base_path=BASE_POSTHOC_PATH, 
                 train_str='train', 
                 val_str='validation')






# In[ ]:


def prepare_data_df(adv_datasets, dataset_paths):
    train_dfs = {}
    for ds in adv_datasets:
        train_dfs[ds] = pd.read_csv(Path(BASE_POSTHOC_PATH, ds, 'train', 'labels_merged.csv'))

    val_dfs = {}
    for ds in adv_datasets:
        val_dfs[ds] = pd.read_csv(Path(BASE_POSTHOC_PATH, ds, 'validation', 'labels_merged.csv'))

    # get adversarial tuples
    for name, df in train_dfs.items():
        df=df[df['true_labels']==df['ori_pred']]
        df=df[df['true_labels']!=df[name+'_pred']]
        df =df[['file', 'true_labels', 'ori_pred', name+'_pred']]
        train_dfs[name]=df
    return train_dfs, val_dfs


# In[ ]:


train_dfs, val_dfs = prepare_data_df(ADV_DATASETS, DATA_PATHS)


# In[ ]:


# Linear Binary Classifier
class LinearBC(nn.Module):
    def __init__(self, input_shape):
        self.num_labels = 2
        super(LinearBC,self).__init__()
        self.fc1 = nn.Linear(input_shape,2)

    def forward(self, x):
        x = self.fc1(x)
        return x


# In[ ]:


def train_posthoc_classifier(adv_datasets, dataset_paths, epochs=EPOCHS):
    logger_dict = {}
    for ds in adv_datasets:
        print("#"*50 + f''' training linear classifier for {ds} ''' + "#"*50)
        # loaders
        ori_train = dataset_paths['ori']['posthoc']['train']['images']
        adv_train = dataset_paths[ds]['posthoc']['train']['images']
        print(f'''original images: {ori_train}''')
        print(f'''adversarial images: {adv_train}''')
        train_set = PosthocTrainDataset(ori_train, adv_train, train_dfs[ds])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=True)
        
        
        ori_validation = dataset_paths['ori']['posthoc']['validation']['images']
        adv_validation = dataset_paths[ds]['posthoc']['validation']['images']
        print(f'''original images: {ori_validation}''')
        print(f'''adversarial images: {adv_validation}''')
        val_set = PosthocTrainDataset(ori_validation, adv_validation, val_dfs[ds])
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=False)

        print(f'''train samples: {len(train_set)} ''')
        print(f'''val samples: {len(val_set)} \n''')

        # Initialise network
        classifier = LinearBC(1536)
        criterion = nn.CrossEntropyLoss()
        classifier.cuda()
        optimizer = torch.optim.Adagrad(classifier.parameters(), lr=0.001, lr_decay=1e-08, weight_decay=0)
        logger_dict[ds] = train(model=None, 
                                classifier=classifier, 
                                train_loader=train_loader, 
                                validation_loader=val_loader, 
                                log_dir=Path(POSTHOC_MODELS_PATH,ds),
                                tensor_dir=None, 
                                optimizer=optimizer, 
                                criterion=criterion, 
                                adversarial_attack=None, 
                                epochs=epochs, 
                                val_freq=1, 
                                batch_size=16,  
                                lr=0.001, 
                                to_restore = {"epoch": 0, "best_acc": 0.}, 
                                n=4, 
                                avgpool_patchtokens=False)

        print(f'''\n''')
    return logger_dict


# In[ ]:


loggers = train_posthoc_classifier(['cw'], DATA_PATHS, 5)


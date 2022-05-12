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
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

# allow imports when running script from within project dir
[sys.path.append(i) for i in ['.', '..']]

# local
# from src.helpers.helpers import get_random_indexes, get_random_classes
from src.model.dino_model import get_dino, ViTWrapper
from src.model.data import *
from src.model.train import *
from src.model.multihead_model import *
from src.helpers.helpers import create_paths
from src.helpers.argparser import parser

from torchattacks import *
from sklearn import preprocessing

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

LINEAR_CLASSIFIER_MODELS_PATH = Path(MAX_PATH, 'linear_classifier_models')

LINEAR_CLASSIFIER_EVAL_PATH = Path(MAX_PATH, 'linear_classifier_evaluation')
LINEAR_CLASSIFIER_EVAL_PATH.mkdir(parents=True, exist_ok=True)

MULTIHEAD_EVAL_PATH = Path(MAX_PATH, 'multihead_eval')
MULTIHEAD_EVAL_PATH.mkdir(parents=True, exist_ok=True)

ORI_PATH = Path(DATA_PATH, 'ori')
CLASS_SUBSET_PATH = Path(ORI_PATH, 'class_subset.npy')
CLASS_SUBSET = np.load(CLASS_SUBSET_PATH)

ADV_DATASETS = ['cw', 'fgsm_06', 'pgd_03']

DATASETS = [*ADV_DATASETS, 'ori']


# In[2]:


INDEX_SUBSET = None
NUM_WORKERS= 0
PIN_MEMORY=True


EPOCHS= 3



# In[3]:


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


# # Import DINO
# Official repo: https://github.com/facebookresearch/dino

# In[4]:


model, linear_classifier = get_dino()


# # Load data

# In[6]:


# Remember to set the correct transformation
# encoder
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit([i for i in CLASS_SUBSET])

loader_dict = defaultdict(dict)

for k, v in DATA_PATHS.items():
    if not k == "ori":
        print(k)
        adv_train_dataset = EnsembleDataset(v["init"]["train"]["images"], 
                                            v["init"]["train"]["label"])
        
        adv_val_dataset = EnsembleDataset(v["init"]["validation"]["images"], 
                                          v["init"]["validation"]["label"])

        loader_dict[k]["train"] = DataLoader(adv_train_dataset, 
                                             batch_size=args.batch_size, 
                                             num_workers=args.num_workers, 
                                             pin_memory=args.pin_memory, 
                                             shuffle=True)
        
        loader_dict[k]["validation"] = DataLoader(adv_val_dataset, 
                                             batch_size=args.batch_size, 
                                             num_workers=args.num_workers, 
                                             pin_memory=args.pin_memory, 
                                             shuffle=False)
    else:
        clean_train_dataset = ImageDataset(v["init"]["train"]["images"], 
                                           v["init"]["train"]["label"], 
                                           ORIGINAL_TRANSFORM,
                                           CLASS_SUBSET, 
                                           index_subset=None, 
                                           label_encoder=label_encoder)

        clean_val_dataset = ImageDataset(v["init"]["validation"]["images"], 
                                         v["init"]["validation"]["label"],
                                         ORIGINAL_TRANSFORM,
                                         CLASS_SUBSET, 
                                         index_subset=None, 
                                         label_encoder=label_encoder)
        
        loader_dict["ori"]["train"] = DataLoader(clean_train_dataset, 
                              batch_size=args.batch_size, 
                              num_workers=args.num_workers, 
                              pin_memory=args.pin_memory, 
                              shuffle=True)
        
        loader_dict["ori"]["validation"] = DataLoader(clean_val_dataset,
                              batch_size=args.batch_size, 
                              num_workers=args.num_workers, 
                              pin_memory=args.pin_memory, 
                              shuffle=False)


# In[7]:


version = 'last_hope'


# ## Classifier

# In[8]:


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
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


# In[9]:


# Linear Binary Classifier
class LinearBC(nn.Module):
    def __init__(self, input_shape):
        self.num_labels = 2
        super(LinearBC,self).__init__()
        self.fc1 = nn.Linear(input_shape,2)

    def forward(self, x):
        x = self.fc1(x)
        return x


# ## Train various classifiers on all adversarial datasets

# In[ ]:


for attack, loaders in loader_dict.items():
        
    # Initialise classifier
    adv_linear_classifier = LinearClassifier(linear_classifier.linear.in_features, 
                                         num_labels=len(CLASS_SUBSET))
    adv_linear_classifier = adv_linear_classifier.cuda()
    
    
    # train
    pstr = "#"*50 + f''' Training classifier for {attack} ''' + "#"*50
    print(len(pstr)*"#")
    print(pstr)
    print(len(pstr)*"#")
    
    loggers = train(model, 
                    adv_linear_classifier, 
                    loaders["train"], 
                    loaders["validation"], 
                    log_dir=Path(LINEAR_CLASSIFIER_MODELS_PATH, version, attack),
                    tensor_dir=None, 
                    optimizer=None, 
                    adversarial_attack=None,
                    criterion=nn.CrossEntropyLoss(),
                    epochs=1, 
                    val_freq=1, 
                    batch_size=args.batch_size,  
                    lr=0.001, 
                    to_restore = {"epoch": 0, "best_acc": 0.}, 
                    n=4, 
                    avgpool_patchtokens=False, 
                    show_image=False)
    
    # Save adversarial Classifier
    save_path = Path(LINEAR_CLASSIFIER_MODELS_PATH, version, attack)
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        
    save_file_log = f"log_{attack}.pt"
    torch.save(loggers, Path(save_path, save_file_log))
    
    print(f'''Finished Training classifier on {attack}''')


# ## Evaluation

# ### Evaluate on all adversarial datasets

# In[1]:


attacks = [x for x in loader_dict.keys()]

VERSION_EVAL_PATH = Path(LINEAR_CLASSIFIER_EVAL_PATH, version)
VERSION_EVAL_PATH.mkdir(parents=True, exist_ok=True)

logger_dict = defaultdict(dict)
for attack in attacks:
#    if attack == "ori":
        pstr = "#"*30 + f''' evaluating adv_classifier trained on {attack} ''' + "#"*30
        print(len(pstr)*"#")
        print(pstr)
        print(len(pstr)*"#")
        adv_classifier = LinearClassifier(linear_classifier.linear.in_features, 
                                 num_labels=len(CLASS_SUBSET))
        adv_classifier.to(args.device)
        

        # load from checkpoint
        log_dir = Path(LINEAR_CLASSIFIER_MODELS_PATH,version, attack)
        to_restore={'epoch': 1}
        utils.restart_from_checkpoint(
            Path(log_dir, "checkpoint.pth.tar"),
            run_variables=to_restore,
            state_dict=adv_classifier
        )

        for applied_attack in attacks:
            
            print(">"*5 + f''' {applied_attack} dataset: {len(loader_dict[applied_attack]["validation"].dataset)} ''')
            info, logger = validate_network(model, 
                                           adv_classifier, 
                                           loader_dict[applied_attack]["validation"], 
                                           criterion=nn.CrossEntropyLoss(),
                                           tensor_dir=None, 
                                           adversarial_attack=None, 
                                           n=4, 
                                           avgpool_patchtokens=False, 
                                           path_predictions=Path(VERSION_EVAL_PATH, 'c_'+attack+'_d_'+applied_attack+'.csv'),
                                           log_interval = 10)
            logger_dict[attack][applied_attack] = logger
            print('\n')


# ### Evaluate on full pipeline with post-hoc as multiplexer

# In[10]:


# Load clean_classifier
clean_classifier = LinearClassifier(linear_classifier.linear.in_features, 
                                    num_labels=len(CLASS_SUBSET))
clean_classifier.to(args.device)

clean_classifier.load_state_dict(torch.load(Path(DATA_PATH,'adversarial_data','adv_classifiers','25_classes','clean.pt')))
clean_classifier.cuda()


# In[11]:


# Load posthoc
attacks=["cw", "fgsm_06", "pgd_03"]

# Perform validation on clean dataset
for post_model in attacks:
    
    log_dir = Path(POSTHOC_MODELS_PATH, post_model)
    
    posthoc = LinearBC(1536)
    posthoc.cuda()
    to_restore={'epoch':3}
    utils.restart_from_checkpoint(
        Path(log_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=posthoc
    )
    
    for adv_model in attacks:
        adv_classifier = LinearClassifier(linear_classifier.linear.in_features, 
                                          num_labels=len(CLASS_SUBSET))
        adv_classifier.to(args.device)
        
        log_dir = Path(LINEAR_CLASSIFIER_MODELS_PATH, version, adv_model)
        to_restore={'epoch': 1}
        
        utils.restart_from_checkpoint(
            Path(log_dir, "checkpoint.pth.tar"),
            run_variables=to_restore,
            state_dict=adv_classifier
        )
        
        for attack, loaders in loader_dict.items():
            
            pstr = "#"*30 + f''' Validating Posthoc: {post_model} and adv_classifier: {adv_model} on {attack} ''' + "#"*30
            print(len(pstr)*"#")
            print(pstr)
            print(len(pstr)*"#")
            
            log_dict, logger = validate_multihead_network(model, 
                                                          posthoc,
                                                          adv_classifier,
                                                          clean_classifier,
                                                          loader_dict[attack]["validation"], 
                                                          tensor_dir=None, 
                                                          adversarial_attack=None, 
                                                          n=4, 
                                                          avgpool=False,
                                                          path_predictions=Path(MULTIHEAD_EVAL_PATH, 'labels_p_'+ post_model +'_c_'+adv_model+'_d_'+attack+'.csv')
                                                         )
            
            # Save adversarial Classifier
            save_path = Path(MULTIHEAD_EVAL_PATH, version)
            save_path.mkdir(parents=True, exist_ok=True)
            save_file_log = f"log_p_{post_model}_c_{adv_model}_d_{attack}.pt"
            torch.save(logger, Path(save_path, save_file_log))


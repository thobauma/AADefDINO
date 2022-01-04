import os
from pathlib import Path
import getpass
import numpy as np
import time
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import random
import sys

# allow imports when running script from within project dir
[sys.path.append(i) for i in ['.', '..']]

# local
from src.helpers.helpers import get_random_indexes, get_random_classes
from src.model.dino_model import get_dino, ViTWrapper
from src.model.data import *

# Custom imports
import torchattacks
from torchattacks import *
import torch.optim as optim
from torchvision import transforms as pth_transforms
from torchvision.utils import save_image



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


# seed
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


DATA_PATH = Path('..', 'data_dir')
MAX_PATH = DATA_PATH

# Image Net
ORI_PATH = Path(DATA_PATH, 'ori')
CLASS_SUBSET_PATH = Path(ORI_PATH, 'class_subset.npy')

TRAIN_PATH = Path(ORI_PATH, 'train')
TRAIN_IMAGES_PATH = Path(TRAIN_PATH,'images')
TRAIN_LABEL_PATH = Path(TRAIN_PATH, 'labels.csv')

VAL_PATH = Path(ORI_PATH, 'validation')
VAL_IMAGES_PATH = Path(VAL_PATH,'images')
VAL_LABEL_PATH = Path(VAL_PATH, 'labels.csv')


INDEX_SUBSET = None
NUM_WORKERS= 0
PIN_MEMORY=True
CLASS_SUBSET = np.load(CLASS_SUBSET_PATH)


from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit([i for i in CLASS_SUBSET])


BATCH_SIZE = 64

DEVICE = 'cuda'


model, dino_classifier = get_dino()

linear_classifier = LinearClassifier(dino_classifier.linear.in_features, 
                         num_labels=len(CLASS_SUBSET))

linear_classifier.load_state_dict(torch.load(Path(DATA_PATH,'adversarial_data','adv_classifiers','25_classes','clean.pt')))
linear_classifier.cuda()


model_wrap = ViTWrapper(model, linear_classifier, device=DEVICE, n_last_blocks=4, avgpool_patchtokens=False)
model_wrap = model_wrap.to(DEVICE)

attacks = [
    (PGD(model_wrap, eps=0.03, alpha=0.015, steps=20), 'pgd_03'),
    (CW(model_wrap, c=50, lr=0.0031, steps=30), 'cw'),
    (FGSM(model_wrap, eps=0.06), 'fgsm_06')
]

if __name__ == '__main__':
    
    for atk, name in attacks:
        

        STORE_PATH = Path(MAX_PATH, 'adversarial_data_tensors', name)

        train_dataset = AdvTrainingImageDataset(TRAIN_IMAGES_PATH, 
                                                TRAIN_LABEL_PATH, 
                                                ORIGINAL_TRANSFORM, 
                                                CLASS_SUBSET, 
                                                index_subset=None, 
                                                label_encoder=label_encoder)
        train_loader = DataLoader(train_dataset, 
                                  batch_size=BATCH_SIZE, 
                                  num_workers=NUM_WORKERS, 
                                  pin_memory=PIN_MEMORY, 
                                  shuffle=False)

        print("-"*70)
        print(atk)
        print('train set')

        STORE_LABEL_PATH = Path(STORE_PATH, 'train', 'labels.csv')
        STORE_LABEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        STORE_IMAGES_PATH = Path(STORE_PATH, 'train', 'images')
        STORE_IMAGES_PATH.mkdir(parents=True, exist_ok=True)
        true_labels = []
        adv_labels = []
        names = []

        correct = 0
        start = time.time()
        

        print(f'''\nsaving predictions to: {STORE_LABEL_PATH}''')
        print(f'''saving output tensors to: {STORE_IMAGES_PATH}''')

        for images, labels, img_names in tqdm(train_loader):
            images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            adv_images = atk(images, labels)

            with torch.no_grad():
                outputs = model_wrap(adv_images)

            _, pre = torch.max(outputs.data, 1)

            correct += (pre == labels).sum()


            for adv_img, img_name in zip(adv_images, img_names):
                torch.save(adv_img, Path(STORE_IMAGES_PATH, Path(img_name.split('.')[0])))
            
            true_labels.extend(labels.detach().cpu().tolist())
            adv_labels.extend(pre.detach().cpu().tolist())
            names.extend(img_names)
            
        del images
        del adv_images
        del labels
        torch.cuda.empty_cache()

        print('\nTotal elapsed time (sec): %.2f' % (time.time() - start))

        print('Accuracy against attack: %.2f %%' % (100 * float(correct) / len(train_loader.dataset)))
        print(f'''\n''')
        
        data_dict = {'file': names, 'true_labels':true_labels, name+'_pred':adv_labels}
        df = pd.DataFrame(data_dict)
        df['file'] = df['file'].str.split('.').str[0]
        df.to_csv(STORE_LABEL_PATH, sep=",", index=None)
        
        print('Validation set')
        
        val_dataset = AdvTrainingImageDataset(VAL_IMAGES_PATH, 
                                      VAL_LABEL_PATH, 
                                      ORIGINAL_TRANSFORM, 
                                      CLASS_SUBSET, 
                                      index_subset=None, 
                                      label_encoder=label_encoder)
        val_loader = DataLoader(val_dataset, 
                                batch_size=BATCH_SIZE, 
                                num_workers=NUM_WORKERS, 
                                pin_memory=PIN_MEMORY,
                                shuffle=False)
        
        
        STORE_LABEL_PATH = Path(STORE_PATH, 'validation', 'labels.csv')
        STORE_LABEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        STORE_IMAGES_PATH = Path(STORE_PATH, 'validation', 'images')
        STORE_IMAGES_PATH.mkdir(parents=True, exist_ok=True)
        
        correct = 0
        start = time.time()
        true_labels = []
        adv_labels = []
        names = []

        print(f'''saving predictions to: {STORE_LABEL_PATH}''')
        print(f'''saving output tensors to: {STORE_IMAGES_PATH}\n''')

        for images, labels, img_names in tqdm(val_loader):
            images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            adv_images = atk(images, labels)

            with torch.no_grad():
                outputs = model_wrap(adv_images)

            _, pre = torch.max(outputs.data, 1)

            correct += (pre == labels).sum()

            
            for adv_img, img_name in zip(adv_images, img_names):
                torch.save(adv_img, Path(STORE_IMAGES_PATH, Path(img_name.split('.')[0])))
                
            true_labels.extend(labels.detach().cpu().tolist())
            adv_labels.extend(pre.detach().cpu().tolist())
            names.extend(img_names)

        del images
        del adv_images
        del labels
        torch.cuda.empty_cache()
        
        print('Total elapsed time (sec): %.2f' % (time.time() - start))
        print('Accuracy against attack: %.2f %%' % (100 * float(correct) / len(val_loader.dataset)))

        data_dict = {'file': names, 'true_labels':true_labels, name+'_pred':adv_labels}
        df = pd.DataFrame(data_dict)
        df['file'] = df['file'].str.split('.').str[0]
        df.to_csv(STORE_LABEL_PATH, sep=",", index=None)
        print(f'''\n''')

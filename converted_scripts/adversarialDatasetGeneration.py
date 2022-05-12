import os
from pathlib import Path
import getpass
import numpy as np
import time
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import random
import sys

# allow imports when running script from within project dir
[sys.path.append(i) for i in ['.', '..']]

# local
from src.helpers.helpers import get_random_indexes, get_random_classes
from src.helpers.argparser import parser
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



# from sklearn import preprocessing

# label_encoder = preprocessing.LabelEncoder()
# label_encoder.fit([i for i in CLASS_SUBSET])





attacks = {'pgd': PGD(model_wrap, eps=0.03, alpha=0.015, steps=20),
            'cw': CW(model_wrap, c=50, lr=0.0031, steps=30),
            'fgsm': FGSM(model_wrap, eps=0.06)}


def advDatasetGeneration(args, attacks):
    TRAIN_PATH = args.filtered_data/'train'
    VALIDATION_PATH = args.filtered_data/'validation'

    model, dino_classifier = get_dino(args.arch)
    linear_classifier = LinearClassifier(dino_classifier.linear.in_features, 
                         num_labels=len(CLASS_SUBSET))
    linear_classifier.load_state_dict(torch.load(Path(args.pretrained_weights)))
    linear_classifier.to(args.device)
    model_wrap = ViTWrapper(model, linear_classifier, device=args.device, n_last_blocks=4, avgpool_patchtokens=False)
    model_wrap = model_wrap.to(args.device)
    for atk, name in attacks:
        STORE_PATH = Path(args.out_dir, 'adversarial_data_tensors', name)

        train_dataset = AdvTrainingImageDataset(TRAIN_PATH/'images', 
                                                TRAIN_PATH/'labels.csv', 
                                                ORIGINAL_TRANSFORM, 
                                                CLASS_SUBSET, 
                                                index_subset=None, 
                                                label_encoder=label_encoder)
        train_loader = DataLoader(train_dataset, 
                                  batch_size=args.batch_size, 
                                  num_workers=args.num_workers, 
                                  pin_memory=args.pin_memory, 
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
            if args.device=='cuda':
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
        

        print('\nTotal elapsed time (sec): %.2f' % (time.time() - start))

        print('Accuracy against attack: %.2f %%' % (100 * float(correct) / len(train_loader.dataset)))
        print(f'''\n''')
        
        data_dict = {'file': names, 'true_labels':true_labels, name+'_pred':adv_labels}
        df = pd.DataFrame(data_dict)
        df['file'] = df['file'].str.split('.').str[0]
        df.to_csv(STORE_LABEL_PATH, sep=",", index=None)
        
        print('Validation set')
        
        val_dataset = AdvTrainingImageDataset(VALIDATION_PATH/'images', 
                                      VALIDATION_PATH/'labels.csv', 
                                      ORIGINAL_TRANSFORM, 
                                      CLASS_SUBSET, 
                                      index_subset=None, 
                                      label_encoder=label_encoder)
        val_loader = DataLoader(val_dataset, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers, 
                                pin_memory=args.pin_memory,
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
            if args.device=='cuda':
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
        
        print('Total elapsed time (sec): %.2f' % (time.time() - start))
        print('Accuracy against attack: %.2f %%' % (100 * float(correct) / len(val_loader.dataset)))

        data_dict = {'file': names, 'true_labels':true_labels, name+'_pred':adv_labels}
        df = pd.DataFrame(data_dict)
        df['file'] = df['file'].str.split('.').str[0]
        df.to_csv(STORE_LABEL_PATH, sep=",", index=None)
        print(f'''\n''')



if __name__ == '__main__':
    
    parser.add_argument('--attack', default='all', type=str, help="""type of attack""")

    args = parser.parse_args()
    advDatasetGeneration(args, attacks)


    
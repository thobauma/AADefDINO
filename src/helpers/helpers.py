import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from pathlib import Path
from collections import defaultdict

def get_random_classes(number_of_classes: int = 50, min_rand_class: int = 1, max_rand_class: int = 1001, seed: int = 42):
    np.random.seed(seed)
    return np.random.randint(low=min_rand_class, high=max_rand_class, size=(number_of_classes,))

def get_random_indexes(number_of_images: int = 50000, n_samples: int=1000, seed: int = 42):
    np.random.seed(seed)
    return np.random.choice(50000, n_samples, replace=False)

def imshow(img):
    npimg = img.cpu().numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.axis('off')
    plt.show()
    
    
    
def create_paths(data_name: str='ori',
                 datasets_paths: defaultdict=None,  
                 initial_base_path: Union[str, Path]='',
                 posthoc_base_path: Union[str, Path]='', 
                 train_str: Union[str, Path]='train', 
                 val_str:Union[str, Path]='validataion'):
    if datasets_paths is None:
        datasets_paths = defaultdict(lambda: defaultdict(
                                     lambda: defaultdict(
                                     lambda: defaultdict())))
    if initial_base_path != '':
        if train_str  != '':
            datasets_paths[data_name]['init'][train_str]['images']=Path(initial_base_path, data_name, train_str, 'images')
            datasets_paths[data_name]['init'][train_str]['label']=Path(initial_base_path, data_name, train_str, 'labels.csv')
        if val_str  != '':
            datasets_paths[data_name]['init'][val_str]['images']=Path(initial_base_path, data_name, val_str, 'images')
            datasets_paths[data_name]['init'][val_str]['label']=Path(initial_base_path, data_name, val_str, 'labels.csv')
            
    if posthoc_base_path  != '':
        if train_str  != '':
            datasets_paths[data_name]['posthoc'][train_str]['images']=Path(posthoc_base_path, data_name, train_str, 'images')
            datasets_paths[data_name]['posthoc'][train_str]['label']=Path(posthoc_base_path, data_name, train_str, 'labels.csv')
        if val_str  != '':
            datasets_paths[data_name]['posthoc'][val_str]['images']=Path(posthoc_base_path, data_name, val_str, 'images')
            datasets_paths[data_name]['posthoc'][val_str]['label']=Path(posthoc_base_path, data_name, val_str, 'labels.csv')
    return datasets_paths



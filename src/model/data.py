# Custom loader
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import traceback
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms as pth_transforms
from typing import List, Callable
from sklearn import preprocessing
import os

from src.model.forward_pass import forward_pass

##### DEFINE PRESET TRANSFORMS #####


# adversarial dataset creation, normalization happens in forwardpass
ORIGINAL_TRANSFORM = pth_transforms.Compose([
                                                pth_transforms.Resize(256, interpolation=3),
                                                pth_transforms.CenterCrop(224),
                                                pth_transforms.ToTensor(),
                                            ])


ADVERSARIAL_TRAINING_TRANSFORM = pth_transforms.Compose([
                                                pth_transforms.RandomResizedCrop(224),
                                                pth_transforms.RandomHorizontalFlip(),
                                                pth_transforms.ToTensor(),
                                            ])

# Use with adversarial dataset
ONLY_NORMALIZE_TRANSFORM = pth_transforms.Compose([
                                            pth_transforms.ToTensor(),
                                            ])


class ImageDataset(Dataset):
  def __init__(self, 
               img_folder: str, 
               labels_file_name: str, 
               transform: callable, 
               class_subset: List[int] = None, 
               index_subset: List[int] = None, 
               label_encoder=None):
    super().__init__()
    self.transform=transform
    self.img_folder=img_folder
    self.data = self.create_df(labels_file_name)
    self.class_subset = class_subset
    if self.class_subset is None:
      if index_subset is not None:
          self.data_subset = self.data.iloc[index_subset]
      else:
        self.data_subset = self.data
    else:
        if label_encoder is None:
            self.le = preprocessing.LabelEncoder()
            self.le.fit([i for i in class_subset])
        else:
            self.le = label_encoder
            
        self.data_subset = self.data[self.data['label'].isin(self.class_subset)] 
        trans_labels = self.le.transform(self.data_subset['label'])
        self.data_subset = self.data_subset.rename(columns={'label': 'original_label'})
        self.data_subset['label'] = trans_labels
  
  def create_df(self, labels_file_name: str):
    df = pd.read_csv(labels_file_name, sep=" ", header=None)
    df.columns=['file', 'label']
    return df
    
  def __len__(self):
    return len(self.data_subset)
  
  def __getitem__(self, index):
    filename = self.data_subset['file'].iloc[index]
    img = Image.open(Path(self.img_folder,filename))
    img = img.convert('RGB')

    img=self.transform(img)
    target=self.data_subset['label'].iloc[index]

    return img, target, filename


class AdvTrainingImageDataset(Dataset):
  def __init__(self, 
               img_folder: str, 
               labels_file_name: str, 
               transform: callable, 
               class_subset: List[int] = None, 
               index_subset: List[int] = None, 
               label_encoder=None):
    super().__init__()
    # MAP CLASSES TO [0, NUM_CLASSES]
    self.transform=transform
    self.img_folder=img_folder
    self.data = self.create_df(labels_file_name)
    self.class_subset = class_subset
    if self.class_subset is None:
      if index_subset is not None:
          self.data_subset = self.data.iloc[index_subset]
      else:
        self.data_subset = self.data
    else:
        if label_encoder is None:
            self.le = preprocessing.LabelEncoder()
            self.le.fit([i for i in class_subset])
        else:
            self.le = label_encoder
            
        self.data_subset = self.data[self.data['label'].isin(self.class_subset)] 
        trans_labels = self.le.transform(self.data_subset['label'])
        self.data_subset = self.data_subset.rename(columns={'label': 'original_label'})
        self.data_subset['label'] = trans_labels

  def create_df(self, labels_file_name: str):
    df = pd.read_csv(labels_file_name, sep=" ", header=None)
    df.columns=['file', 'label']
    return df
    
  def __len__(self):
    return len(self.data_subset)
  
  def __getitem__(self, index):
    filename = self.data_subset['file'].iloc[index]
    img = Image.open(os.path.join(self.img_folder,filename))
    img = img.convert('RGB')

    img=self.transform(img)
    target=self.data_subset['label'].iloc[index]

    return img, target, filename

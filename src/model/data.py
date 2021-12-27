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

class ImageDataset(Dataset):
  def __init__(self, img_folder: str, file_name: str, transform: callable, class_subset: List[int] = None, index_subset: List[int] = None):
    super().__init__()
    self.transform=transform
    self.img_folder=img_folder
    self.data = self.create_df(file_name)
    self.class_subset = class_subset
    if self.class_subset is None:
      if index_subset is not None:
          self.data_subset = self.data.iloc[index_subset]
      else:
        self.data_subset = self.data
    else:
      self.data_subset = self.data[self.data['label'].isin(self.class_subset)] 
  
  def create_df(self, file_name: str):
    df = pd.read_csv(file_name, sep=" ", header=None)
    df.columns=['file', 'label']
    return df
    
  def __len__(self):
    return len(self.data_subset)
  
  def __getitem__(self, index):
    img = Image.open(Path(self.img_folder,self.data_subset['file'].iloc[index]))
    img = img.convert('RGB')

    img=self.transform(img)
    target=self.data_subset['label'].iloc[index]

    return img,target


class AdvTrainingImageDataset(Dataset):
  def __init__(self, img_folder: str, file_name: str, transform: callable, class_subset: List[int] = None, index_subset: List[int] = None):
    super().__init__()
    # MAP CLASSES TO [0, NUM_CLASSES]
    self.transform=transform
    self.img_folder=img_folder
    self.data = self.create_df(file_name)
    self.class_subset = class_subset
    if self.class_subset is None:
      if index_subset is not None:
          self.data_subset = self.data.iloc[index_subset]
      else:
        self.data_subset = self.data
    else:
      self.le = preprocessing.LabelEncoder()
      self.le.fit([i for i in class_subset])
      self.data_subset = self.data[self.data['label'].isin(self.class_subset)] 
      trans_labels = self.le.transform(self.data_subset['label'])
      self.data_subset = self.data_subset.rename(columns={'label': 'original_label'})
      self.data_subset['label'] = trans_labels

  def create_df(self, file_name: str):
    df = pd.read_csv(file_name, sep=" ", header=None)
    df.columns=['file', 'label']
    return df
    
  def __len__(self):
    return len(self.data_subset)
  
  def __getitem__(self, index):
    img = Image.open(os.path.join(self.img_folder,self.data_subset['file'].iloc[index]))
    img = img.convert('RGB')

    img=self.transform(img)
    target=self.data_subset['label'].iloc[index]

    return img,target,self.data_subset['file'].iloc[index]


def create_loader(IMAGES_PATH, LABEL_PATH, INDEX_SUBSET=None, CLASS_SUBSET=None, BATCH_SIZE=8, num_workers=0, pin_memory=True, is_adv_training=False, transform=None):
    # Create loader
    # Taken from official repo: https://github.com/facebookresearch/dino/blob/main/eval_linear.py
    
    if not is_adv_training:
        loader_transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]) if transform is None else transform

        org_dataset = ImageDataset(img_folder = IMAGES_PATH,
                                   file_name = LABEL_PATH,
                                   transform=loader_transform,
                                   index_subset=INDEX_SUBSET,
                                   class_subset=CLASS_SUBSET)
    elif is_adv_training:
        loader_transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
        ]) if transform is None else transform
            
        org_dataset = AdvTrainingImageDataset(img_folder = IMAGES_PATH,
                           file_name = LABEL_PATH,
                           transform=loader_transform,
                           class_subset=CLASS_SUBSET,
                            index_subset=INDEX_SUBSET)

    org_loader = torch.utils.data.DataLoader(
        org_dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return org_loader


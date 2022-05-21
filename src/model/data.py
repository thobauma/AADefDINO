# Custom loader
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import traceback
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms as pth_transforms
from typing import List, Callable, Union
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
               img_folder: Union[Path, str], 
               labels_file_name: Union[Path, str], 
               transform: Callable, 
               class_subset: Union[List[int],None] = None, 
               index_subset: Union[List[int],None] = None, 
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




class PosthocForwardDataset(Dataset):
  def __init__(self, 
               img_folder: Union[str, Path], 
               labels_file_name: Union[str, Path], 
               class_subset: Union[List[int],None] = None, 
               index_subset: Union[List[int],None] = None):
    super().__init__()
    # MAP CLASSES TO [0, NUM_CLASSES]
    self.img_folder=img_folder
    self.data = self.create_df(labels_file_name)
    self.class_subset = class_subset
    self.index_subset = index_subset
    if self.class_subset is None:
      if index_subset is not None:
          self.data_subset = self.data.iloc[index_subset]
      else:
        self.data_subset = self.data
    else:   
        self.data_subset = self.data[self.data['true_labels'].isin(self.class_subset)]

  def create_df(self, labels_file_name: str):
    df = pd.read_csv(labels_file_name)
    return df
    
  def __len__(self):
    return len(self.data_subset)
  
  def __getitem__(self, index):
    filename = self.data_subset['file'].iloc[index]
    img = torch.load(Path(self.img_folder, filename)).cpu()
    target=self.data_subset['true_labels'].iloc[index]

    return img, target, filename


class PosthocTrainDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 or_img_folder, 
                 adv_img_folder, 
                 or_df_path,
                 adv_df_path,
                 transform=ORIGINAL_TRANSFORM):
        super().__init__()
        self.transform=transform
        self.or_img_folder = or_img_folder
        self.adv_img_folder = adv_img_folder
        self.or_df = pd.read_csv(or_df_path, sep=",", index_col=0)
        self.adv_df = pd.read_csv(adv_df_path, sep=",", index_col=0)
        self.transform = transform
        
    def __len__(self):
        return len(self.or_df) + len(self.adv_df)
    
    def __getitem__(self, index):  
        if index >= len(self.or_df):
            index = index - len(self.or_df)
            filename = self.adv_df['image'].iloc[index]
            label = torch.tensor([1., 0.])
            payload = torch.load(Path(self.adv_img_folder, filename)).cpu()

            return payload, label, filename
        else:
            filename = self.or_df['image'].iloc[index]
            label = torch.tensor([0., 1.])
            img = Image.open(Path(self.or_img_folder, filename))
            img = img.convert('RGB')
            filename= filename.split('.')[0]
            img=self.transform(img)

            return img, label, filename


class EnsembleDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 img_folder, 
                 df_path):
        super().__init__()
        self.img_folder = img_folder
        self.data = self.create_df(df_path)
    
    def create_df(self, labels_file_name: str):
        df = pd.read_csv(labels_file_name)
        return df
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):            
        filename = self.data['file'].iloc[index]
        filename = filename.split('.')[0]
        payload = torch.load(Path(self.img_folder, filename)).cpu()
        target=self.data['true_labels'].iloc[index]
        return payload, target, filename

    
class AdvTrainingImageDataset(Dataset):
    def __init__(self, 
                   img_folder: str, 
                   labels_file_name: str, 
                   transform: Callable,
                   index_subset: Union[List[int],None] = None):
        super().__init__()
        self.transform=transform
        self.img_folder=img_folder
        self.data = self.create_df(labels_file_name)
        self.index_subset=index_subset
        self.prepare_data()

    def prepare_data(self):
        if self.index_subset is not None:
            data_subset = self.data.iloc[index_subset]
        else:
            data_subset = self.data
        self.data = data_subset


    def create_df(self, labels_file_name: str):
        df = pd.read_csv(labels_file_name, sep=",", index_col=0)
        return df
    
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, index):
        filename = self.data['image'].iloc[index]
        img = Image.open(os.path.join(self.img_folder, filename))
        img = img.convert('RGB')
        filename= filename.split('.')[0]
        img=self.transform(img)
        red_target=self.data['reduced_label'].iloc[index]

        return img, red_target, filename

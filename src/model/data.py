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
ORIGINAL_TRANSFORM = pth_transforms.Compose([
                                                pth_transforms.Resize(256, interpolation=3),
                                                pth_transforms.CenterCrop(224),
                                                pth_transforms.ToTensor(),
                                                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                            ])

NO_NORM_TRANSFORM = pth_transforms.Compose([
                                                pth_transforms.Resize(256, interpolation=3),
                                                pth_transforms.CenterCrop(224),
                                                pth_transforms.ToTensor(),
                                            ])

TO_TENSOR_TRANSFORM = pth_transforms.Compose([pth_transforms.ToTensor()])


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
    filename = self.data_subset['file'].iloc[index]
    img = Image.open(Path(self.img_folder,filename))
    img = img.convert('RGB')

    img=self.transform(img)
    target=self.data_subset['label'].iloc[index]


    return img, target, filename



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
    filename = self.data_subset['file'].iloc[index]
    img = Image.open(os.path.join(self.img_folder,filename))
    img = img.convert('RGB')

    img=self.transform(img)
    target=self.data_subset['label'].iloc[index]

    return img, target, filename


def create_loader(IMAGES_PATH, LABEL_PATH, INDEX_SUBSET=None, CLASS_SUBSET=None, BATCH_SIZE=8, num_workers=0, pin_memory=True, remove_normalization=False, transform=None):
    # Create loader
    # Taken from official repo: https://github.com/facebookresearch/dino/blob/main/eval_linear.py
    
    if not remove_normalization:
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
    elif remove_normalization:
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
        shuffle=False,
    )
    
    return org_loader


# A Python generator for adversarial samples.
# Takes original and adversarial loaders, a model, classifier and yields
# a pair of original and adversarial samples based on the definition above.
def adv_dataset(org_loader, adv_loader, model, linear_classifier, n=4, device="cuda"):
  linear_classifier.eval()
  model.eval()
  for org, adv in zip(org_loader, adv_loader):
    # parse the original sample
    org_inp, org_tar, org_img_name = org
    org_inp = org_inp.to(device)
    org_tar = org_tar.to(device)

    # parse the adversarial sample
    adv_inp, adv_tar, adv_img_name = adv
    adv_inp = adv_inp.to(device)
    adv_tar = adv_tar.to(device)

    # forward pass original and adversarial sample
    org_pred = forward_pass(org_inp, model, linear_classifier, n)
    adv_pred = forward_pass(adv_inp, model, linear_classifier, n)
    
    # label, original image predicted class, adversarial image predicted class
    for y, org_y, adv_y, org_x, adv_x, org_name, adv_name in zip(org_tar, org_pred, adv_pred, org_inp, adv_inp, org_img_name, adv_img_name):
      # yield a new tuple based if the conditions match. skip otherwise.
      org_correct = y == org_y
      adv_correct = y == adv_y
      
      org_name = org_name.replace('.JPEG', '')
      adv_name = adv_name.replace('.png', '')
      adv_name = adv_name.replace('.JPEG', '')
        
      org_num = int(org_name.split("_")[-1])
      adv_num = int(adv_name.split("_")[-1])
      assert org_num == adv_num, f"Numbers are not matching: org={org_name}, adv={adv_name}"

      if org_correct and not adv_correct:
        yield org_name, org_x, 0 # original => 0
        yield adv_name, adv_x, 1 # adversarial => 1

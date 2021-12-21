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


def create_loader(IMAGES_PATH, LABEL_PATH, INDEX_SUBSET=None, BATCH_SIZE=8, num_workers=0, pin_memory=True):
    # Create loader
    # Taken from official repo: https://github.com/facebookresearch/dino/blob/main/eval_linear.py
    loader_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor()
    ])

    org_dataset = ImageDataset(img_folder = IMAGES_PATH,
                               file_name = LABEL_PATH,
                               transform=loader_transform,
                               index_subset=INDEX_SUBSET)

    org_loader = torch.utils.data.DataLoader(
        org_dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return org_loader
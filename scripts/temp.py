# %%
import os
from pathlib import Path
import getpass
import numpy as np
import pandas as pd
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import sys
import tarfile
rng = np.random.default_rng(42)
# %%
PATH_t7 = Path('/Volumes/T7','AADefDINO','imageNet')
file_PATH = Path(__file__).resolve()
home_PATH = file_PATH.parents[1]
data_PATH = Path(home_PATH, 'data_dir')
ori_PATH = Path(data_PATH, 'ori')
train_PATH = Path(ori_PATH, 'train')
validation_PATH = Path(ori_PATH, 'validation')
CLASS_SUBSET_PATH = Path(ori_PATH, "class_subset.npy")
DEV_PATH = Path(data_PATH, 'dev')
DEV_TRAIN = Path(DEV_PATH, 'train')
DEV_TRAIN_IMG = Path(DEV_TRAIN, 'img')
DEV_TEST = Path(DEV_PATH, 'test')
DEV_TEST_IMG = Path(DEV_TEST, 'img')
DEV_VALIDATION = Path(DEV_PATH, 'validation')
DEV_VALIDATION_IMG = Path(DEV_VALIDATION, 'img')
CLASS_SUBSET = np.load(CLASS_SUBSET_PATH)
# %%
def sample_files(label_file, size=10, class_subset=CLASS_SUBSET, random_state=42):
    if not isinstance(label_file, pd.DataFrame):
        data = pd.read_csv(label_file, sep=" ", names=['file', 'label'])
    else:
        data = label_file
    sampled_data = []
    for label  in class_subset:
        sampled_data.append(data[data['label']==label].sample(size, random_state=random_state))
    return pd.concat(sampled_data)

# %%
if __name__ == "__main__":
# %%
    validation_sampled = sample_files(validation_PATH / "labels.csv", size=5, class_subset=CLASS_SUBSET, random_state=np.random.RandomState(seed=42))
    train_test_sampled = sample_files(train_PATH / "labels.csv", size=25, class_subset=CLASS_SUBSET, random_state=np.random.RandomState(seed=42))
    train_sampled = sample_files(train_test_sampled, size=20, class_subset=CLASS_SUBSET, random_state=np.random.RandomState(seed=42))
    test_sampled = train_test_sampled.drop(train_sampled.index)
    validation_sampled.to_csv(DEV_VALIDATION / 'labels.csv', sep=" ", index=False)
    train_sampled.to_csv(DEV_TRAIN / 'labels.csv', sep=" ", index=False)
    test_sampled.to_csv(DEV_TEST / 'labels.csv', sep=" ", index=False)

# %%
    imgnet_val_tar = tarfile.open(PATH_t7 / "ILSVRC2012_img_val.tar")
    imgnet_val_members = imgnet_val_tar.getmembers()
    members_subset = []
    vsv = validation_sampled['file'].values
    for member in imgnet_val_members:    
        if member.name in vsv:
            members_subset.append(member)
    imgnet_val_tar.extractall(DEV_VALIDATION_IMG, members_subset)

# %%
    imgnet_train_tar = tarfile.open(PATH_t7 / "ILSVRC2012_img_train.tar")
    imgnet_train_members = imgnet_train_tar.getmembers()

# %%
    tar_21 = tarfile.open(DEV_PATH/"dummy" / "n01608432.tar")
    tar_21_members = tar_21.getmembers()
# %%
    member_subset = []
    tsv = train_sampled['file'].values
    for member in tar_21_members:
        if member.name in tsv:
            members_subset.append(member)
    
# %%
    trainsv = train_sampled['file'].values
    testsv = test_sampled['file'].values
    with tarfile.open(PATH_t7 / "ILSVRC2012_img_train.tar") as imgnet_train_tar:
        for item in imgnet_train_tar:
            m = imgnet_train_tar.extractfile(item)
            with tarfile.open(m) as nested:
                m_members = nested.getmembers()
                train_sub = []
                test_sub = []
                for member in m_members:    
                    if member.name in trainsv:
                        train_sub.append(member)
                    if member.name in testsv:
                        test_sub.append(member)
                print(f'''number of train: {len(train_sub)}, should be {len(trainsv)}''')
                print(f'''number of test: {len(test_sub)}, should be {len(testsv)}''')
                nested.extractall(DEV_TRAIN_IMG, train_sub)
                nested.extractall(DEV_TEST_IMG, test_sub)



# %%
    trainsv = train_sampled['file'].values
    testsv = test_sampled['file'].values
    imgnet_train_tar = tarfile.open(PATH_t7 / "ILSVRC2012_img_train.tar")
    for parent_member in imgnet_train_tar.getmembers():
        imgnet_train_tar.extract(parent_member, DEV_PATH/"dummy")
        n_p = Path(DEV_PATH/"dummy"/parent_member.name)
        print(n_p)
        with tarfile.open( n_p) as nested:
            m_members = nested.getmembers()
            train_sub = []
            test_sub = []
            for member in m_members:    
                if member.name in trainsv:
                    train_sub.append(member)
                if member.name in testsv:
                    test_sub.append(member)
            print(f'''number of train: {len(train_sub)}, should be {len(trainsv)}''')
            print(f'''number of test: {len(test_sub)}, should be {len(testsv)}''')
            nested.extractall(DEV_TRAIN_IMG, train_sub)
            nested.extractall(DEV_TEST_IMG, test_sub)
        n_p.unlink()


# %%

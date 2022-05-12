# In[17]:
import os
from pathlib import Path
import getpass
import numpy as np
import time
import torch
from torch import nn
from tqdm import tqdm
import random
import sys
from torch.utils.data import DataLoader
from PIL import Image
import torch
from torchvision import transforms

# Load ViT
from pytorch_pretrained_vit import ViT

# allow imports when running script from within project dir
[sys.path.append(i) for i in ['.', '..']]

# local
from src.helpers.helpers import get_random_indexes, get_random_classes
from src.helpers.argparser import parser
from src.model.dino_model import get_dino, ViTWrapper
from src.model.train import validate_network
from src.model.data import *

# Custom imports
import torchattacks
from torchattacks import *
import torch.optim as optim
from torchvision import transforms as pth_transforms
import argparse

# seed
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)






INDEX_SUBSET = get_random_indexes(n_samples=3000) # Randomly sample data




if __name__ == '__main__':

    args = parser.parse_args()

    # # Evaluate DINO ViT
    model, linear_classifier = get_dino(args=args)


    ORI_PATH = args.filtered_data
    org_dataset = ImageDataset(ORI_PATH/'images', ORI_PATH/'labels.csv', ORIGINAL_TRANSFORM, class_subset=None, index_subset=INDEX_SUBSET)
    org_loader = DataLoader(org_dataset, batch_size=args.batch_size, num_workers=args.number_workers, pin_memory=args.pin_memory, shuffle=True)


    # ## Wrap model
    model_wrap = ViTWrapper(model, linear_classifier, device=args.device, n_last_blocks=1, avgpool_patchtokens=True)
    model_wrap= model_wrap.to(args.device)

    #### ViT/S-16
    #model_wrap = ViTWrapper(model, linear_classifier, device=args.device, n_last_blocks=4, avgpool_patchtokens=False)
    #model_wrap= model_wrap.to(args.device)


    # ## Compute clean accuracy
    clean_correct = 0
    total = len(org_loader.dataset)
    start = time.time()

    with torch.no_grad():
        for images, labels, _ in tqdm(org_loader):

            cuda_images = images.to(args.device)
            clean_outputs = model_wrap(cuda_images)
            labels = labels.to(args.device)

            _, pre_clean = torch.max(clean_outputs.data, 1)

            clean_correct += (pre_clean == labels).sum()

    print('Total elapsed time (sec): %.2f' % (time.time() - start))
    print('Clean accuracy: %.2f %%' % (100 * float(clean_correct) / total))

    if args.device=='cuda':
        torch.cuda.empty_cache()


    # ## Attack model
    # We use TorchAttack library. See: https://github.com/Harry24k/adversarial-attacks-pytorch
    # Define attacks to be tested
    attacks = [
        #FGSM(model_wrap, eps=0.003),
        #FGSM(model_wrap, eps=0.03),
        #FGSM(model_wrap, eps=0.06),
        PGD(model_wrap, eps=0.003, alpha=0.003, steps=20),
        #PGD(model_wrap, eps=0.03, alpha=0.003, steps=20),
        #PGD(model_wrap, eps=0.06, alpha=0.003, steps=20),
        CW(model_wrap, c=10, lr=0.003, steps=30),
        #CW(model_wrap, c=50, lr=0.003, steps=30),
    ]
    total = len(org_loader.dataset)

    for atk in attacks:
        
        print("-"*70)
        print(atk)
        
        correct = 0
        clean_correct = 0
        start = time.time()
        
        for images, labels, _ in tqdm(org_loader):
            
            labels = labels.to(args.device)
            adv_images = atk(images, labels)
            
            with torch.no_grad():
                outputs = model_wrap(adv_images)

            _, pre = torch.max(outputs.data, 1)

            correct += (pre == labels).sum()

        print('Total elapsed time (sec): %.2f' % (time.time() - start))
        print('Accuracy against attack: %.2f %%' % (100 * float(correct) / total))
    
    if args.device=='cuda':
        del images
        del labels
        del adv_images
        torch.cuda.empty_cache()

    # # Evaluate Original ViT
    if args.device=='cuda':
        del model, linear_classifier, model_wrap
        torch.cuda.empty_cache()


    # ## Load model
    # We use pretrained model from: https://github.com/lukemelas/PyTorch-Pretrained-ViT
    vit_model = ViT('B_16_imagenet1k', pretrained=True).to(args.device)
    vit_model.eval()

    # Define custom transform for this model
    transform = transforms.Compose([
        transforms.Resize((384, 384)), 
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])


    # ## Load data

    vit_dataset = ImageDataset(ORI_PATH/'images', ORI_PATH/'labels.csv', transform, class_subset=None, index_subset=INDEX_SUBSET)
    vit_loader = DataLoader(vit_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=True)


    # ## Compute clean accuracy

    clean_correct = 0
    total = len(vit_loader.dataset)
    start = time.time()

    with torch.no_grad():
        for images, labels, _ in tqdm(vit_loader):

            cuda_images = images.to(args.device)
            clean_outputs = vit_model(cuda_images)
            labels = labels.to(args.device)

            _, pre_clean = torch.max(clean_outputs.data, 1)

            clean_correct += (pre_clean == labels).sum()

    print('Total elapsed time (sec): %.2f' % (time.time() - start))
    print('Clean accuracy: %.2f %%' % (100 * float(clean_correct) / total))


    # ## Attack model
    # We use TorchAttack library. See: https://github.com/Harry24k/adversarial-attacks-pytorch
    attacks_vit = [
        #FGSM(vit_model, eps=0.003),
        #FGSM(vit_model, eps=0.03),
        #FGSM(vit_model, eps=0.06),
        PGD(vit_model, eps=0.003, alpha=0.003, steps=20),
        #PGD(vit_model, eps=0.03, alpha=0.003, steps=20),
        #PGD(vit_model, eps=0.06, alpha=0.003, steps=20),
        CW(vit_model, c=10, lr=0.003, steps=30),
        #CW(vit_model, c=50, lr=0.003, steps=30),
    ]

    total = len(vit_loader.dataset)

    for atk in attacks_vit:
        
        print("-"*70)
        print(atk)
        
        correct = 0
        clean_correct = 0
        start = time.time()
        
        for images, labels, _ in tqdm(vit_loader):
            labels = labels.to(args.device)
            adv_images = atk(images, labels)
            
            with torch.no_grad():
                outputs = vit_model(adv_images)

            _, pre = torch.max(outputs.data, 1)

            correct += (pre == labels).sum()

            del images
            del labels
            del adv_images
            torch.cuda.empty_cache()

        print('Total elapsed time (sec): %.2f' % (time.time() - start))
        print('Accuracy against attack: %.2f %%' % (100 * float(correct) / total))



    # # # Visualize images
    # # 
    # # In this section we visualize images used for the report.
    # Image.open(Path(DATA_PATH,'ori/validation/images/ILSVRC2012_val_00000013.JPEG'))



    # import matplotlib.pyplot as plt

    # def imshow(img, title):
    #     npimg = img.numpy()
    #     fig = plt.figure(figsize = (5, 15))
    #     plt.imshow(np.transpose(npimg,(1,2,0)))
    #     plt.title(title)
    #     plt.axis('off')
    #     plt.show()




    # vis_loader = create_loader(ORI_PATH/'images', ORI_PATH/'labels.csv', [12], None, args.batch_size, is_adv_training=True)
    # len(vis_loader.dataset)




    # for images, labels, img_names in tqdm(vis_loader):
    #     imshow(images[0].detach().cpu(), "")


    # for atk in attacks:
        
    #     print("-"*70)
    #     print(atk)

    #     correct = 0
    #     clean_correct = 0
    #     start = time.time()
        
    #     for images, labels, img_names in tqdm(vis_loader):
            
    #         labels = labels.to(args.device)
    #         adv_images = atk(images, labels)
            
    #         with torch.no_grad():
    #             outputs = model_wrap(adv_images)

    #         _, pre = torch.max(outputs.data, 1)

    #         correct += (pre == labels).sum()
            
    #         imshow(adv_images[0].detach().cpu(), "")
            
    #         print(f"Is adversarial? {correct==0}")


import os
from pathlib import Path
import getpass
import numpy as np
import time
import torch
import PIL
from torch import nn
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import random
import sys

# allow imports when running script from within project dir
[sys.path.append(i) for i in ['.', '..']]

# local
from src.helpers.argparser import parser
from src.model.dino_model import get_dino, ViTWrapper
from src.model.data import *

# Custom imports
import torchattacks
from torchattacks import *
import torch.optim as optim
from torchvision import transforms as pth_transforms
from torchvision.utils import save_image

from torchvision.models.resnet import ResNet, Bottleneck

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
    
class CustomResNet(ResNet):
    def __init__(self, classifier=None):
        super(CustomResNet, self).__init__(block=Bottleneck, layers=[3, 4, 6, 3])
        self.load_state_dict(torch.load("/cluster/scratch/jrando/resnet/resnet.pth"))
        del self.fc
        
        self.classifier = classifier
        
    def _forward_impl(self, x):
        # Normalize
        transform = pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        x = transform(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if self.classifier:
            x = self.classifier(x)

        return x


# seed
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# bsub -n 4 -R "rusage[mem=10240]" -R "rusage[ngpus_excl_p=1]" -W "24:00" "python3 converted_scripts/train_linear_classifier.py --arch vit_base --n_last_blocks 1 --avgpool_patchtokens True"



def advDatasetGeneration(args, attacks=None):
    TRAIN_PATH = args.filtered_data/'train'
    VALIDATION_PATH = args.filtered_data/'validation'
    
    linear_classifier = LinearClassifier(2048, 9)
    linear_classifier.load_state_dict(torch.load(Path(args.head_path))['state_dict'])
    linear_classifier.to(args.device)

    model = CustomResNet(classifier=linear_classifier).to(args.device)
    
    attacks = [
        #(PGD(model, eps=0.001, alpha=(0.001*2)/3, steps=3), "pgd_0001"),
        #(PGD(model, eps=0.03, alpha=(0.03*2)/3, steps=3),"pgd_003"),
        (PGD(model, eps=0.1, alpha=(0.1*2)/3, steps=3),"pgd_01"),
        #(FGSM(model, eps=0.001),"fgsm_0001"),
        #(FGSM(model, eps=0.03),"fgsm_003"),
        #(FGSM(model, eps=0.1),"fgsm_01"),
        #(CW(model, c=50),"cw_50")
    ]
    
    for atk, name in attacks:
        if args.out_dir is None:
            STORE_PATH = args.data_root
        else:
            STORE_PATH = args.out_dir
        
        STORE_PATH = Path(STORE_PATH, 'adv', name)
        
        print('Validation set')
        
        val_dataset = AdvTrainingImageDataset(VALIDATION_PATH/'images', 
                                      VALIDATION_PATH/'labels.csv', 
                                      ORIGINAL_TRANSFORM)
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
                outputs = model(adv_images)

            _, pre = torch.max(outputs.data, 1)

            correct += (pre == labels).sum()

            
            for adv_img, img_name in zip(adv_images, img_names):
                with open(Path(STORE_IMAGES_PATH, img_name), 'wb') as f:
                    torch.save(adv_img, f)
                
            true_labels.extend(labels.detach().cpu().tolist())
            adv_labels.extend(pre.detach().cpu().tolist())
            names.extend(img_names)
        
        print('Total elapsed time (sec): %.2f' % (time.time() - start))
        print('Accuracy against attack: %.2f %%' % (100 * float(correct) / len(val_loader.dataset)))

        data_dict = {'image': names, 'reduced_label':true_labels, name+'_pred':adv_labels}
        df = pd.DataFrame(data_dict)
        df['image'] = df['image'].str.split('.').str[0]
        df.to_csv(STORE_LABEL_PATH, sep=",")
        print(f'''\n''')



if __name__ == '__main__':
    
    args = parser.parse_args()
    advDatasetGeneration(args, attacks)
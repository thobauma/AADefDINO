from pathlib import Path
from collections import defaultdict

import numpy as np
import random
import sys
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch import nn

# allow imports when running script from within project dir
[sys.path.append(i) for i in ['.', '..']]

# local
from src.model.dino_model import get_dino
from src.model.train import *
from src.model.data import *
from src.helpers.helpers import create_paths
from src.helpers.argparser import parser

# seed
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)



# # Import DINO
# Official repo: https://github.com/facebookresearch/dino
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


# # Forward Pass

def posthoc_forward_pass(model, classifier, datasets, args):
    logger_dict = {}
    for ds in datasets:
        ds_init = args.data_root/ds
        ds_posthoc = args.posthoc_data/ds
        
        logger_dict[ds] = {}
        print("\n"+"#"*50 + f''' forwardpass for {ds} ''' + "#"*50)
        
        for tv in ['train', 'validation']:            
            data = ds_init/tv/'images'
            labels = ds_init/tv/'labels.csv'
            pred_label = ds_posthoc/tv/'labels.csv'
            print(f'''images: {data}\nlabel: {labels}\npred: {pred_label}''')
            
            if ds == 'ori':
                transform = ORIGINAL_TRANSFORM
                data_set = AdvTrainingImageDataset(img_folder=data, 
                                   labels_file_name=labels, 
                                   transform=transform, 
                                   class_subset=CLASS_SUBSET,
                                   index_subset=None,
                                   label_encoder=label_encoder)
            
            else:
                data_set = PosthocForwardDataset(img_folder=data, 
                                                 labels_file_name=labels,
                                                 index_subset=None, 
                                                 class_subset=None)
            
            data_loader = DataLoader(data_set, 
                                     batch_size=args.batch_size, 
                                     num_workers=args.num_workers, 
                                     pin_memory=args.pin_memory, 
                                     shuffle=False)
            
            print(f'''{ds}: {tv} {len(data_set)}''')
            logger_dict[ds][tv] = validate_network(model=model, 
                                                   classifier=classifier, 
                                                   validation_loader=data_loader, 
                                                   criterion=nn.CrossEntropyLoss(), 
                                                   tensor_dir=ds_posthoc/tv/'images',
                                                   adversarial_attack=None, 
                                                   n=4, 
                                                   avgpool_patchtokens=False, 
                                                   path_predictions=pred_label,
                                                   show_image=False
                                                   )
            
    return logger_dict

if __name__ == '__main__':

    ADV_DATASETS = ['cw', 'fgsm_06', 'pgd_03']
    DATASETS = ['ori', *ADV_DATASETS]
    print(DATASETS)

    parser.add_argument('--adv_data', default='', type=Path, help="""path to the adversarial data.""")
    args = parser.parse_args()

    

    model, dino_classifier = get_dino(args = args)

    linear_classifier = LinearClassifier(dino_classifier.linear.in_features, 
                            num_labels=len(CLASS_SUBSET))

    linear_classifier.load_state_dict(torch.load(args.pretrained_weights))
    if args.device == 'cuda':
        linear_classifier.cuda()


    from sklearn import preprocessing

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit([i for i in CLASS_SUBSET])

    logger_dict = posthoc_forward_pass(model,
                                    linear_classifier, 
                                    DATASETS, 
                                    args)
    


    from functools import reduce

    def merge_frames(frames, on_what=['file', 'true_labels'], how='left'):
        merged_df = reduce(lambda left, right:pd.merge(left, right, on=on_what, how=how,  suffixes=('', '_drop')), frames)
        merged_df.drop(merged_df.filter(regex='_drop$').columns.tolist(),axis=1, inplace=True)
        return merged_df

    def get_merged_labels(datasets=DATASETS, datasets_types=['train', 'validation'], datasets_paths=args.data_root, save_path=args.posthoc_data, get_df_dict = False):
        df_data_types = {}
        df_data = {}
        for tv in datasets_types:
            df_data[tv] = {}
            for ds in datasets:
                ds_posthoc = Path(datasets_paths, ds, 'posthoc')
                pred_label = ds_posthoc/tv/'labels.csv'
                df_data[tv][ds] = pd.read_csv(pred_label)
                df_data[tv][ds].rename(columns = {'pred_labels': ds+'_pred'}, inplace = True)
                if ds != 'ori':
                    df_data[tv][ds] = pd.merge(df_data[tv][ds], df_data[tv]['ori'], on=['file', 'true_labels'], how='left')
                df_data[tv][ds].to_csv(Path(save_path, ds, tv, 'labels_merged.csv'), sep=",", index=None)
            df_data_types[tv] = merge_frames(df_data[tv].values())
            if save_path is not None:
                df_data_types[tv].to_csv(Path(save_path,tv+'.csv'), sep=",", index=None)
        if get_df_dict:
            return df_data_types, df_data

        return df_data_types


    df_types, df_data = get_merged_labels(get_df_dict=True)
    # for name in ADV_DATASETS:   
    #     print(f'''\n{name}:''')
    #     for tv in ['train', 'validation']: 
    #         print(f'    {tv}:')
    #         df = df_data[tv][name]

    #         ldf = len(df)
    #         print(f'''        total data:             {ldf}''')
    #         print(f'''        correct pred:           {len(df[df['true_labels']==df['ori_pred']])},   {len(df[df['true_labels']==df['ori_pred']])/ldf}''')
    #         print(f'''        incorrect adv pred:     {len(df[df['true_labels']!=df[name+'_pred']])},   {len(df[df['true_labels']!=df[name+'_pred']])/ldf}''')
    #         df_f = df[df['true_labels']==df['ori_pred']]
    #         print(f'''        number adv tuples:      {len(df_f[df_f['true_labels']!=df_f[name+'_pred']])},   {len(df_f[df_f['true_labels']!=df_f[name+'_pred']])/ldf}''')






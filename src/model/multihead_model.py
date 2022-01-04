import json
from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm


import torch
from torch import nn

import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as pth_transforms

from dino import utils
from src.helpers.helpers import imshow

# Perform validation on clean dataset
def validate_multihead_network(model, 
                               posthoc,
                               adv_classifier,
                               clean_classifier,
                               validation_loader,
                               tensor_dir=None, 
                               adversarial_attack=None, 
                               n=4, 
                               avgpool=False,
                               path_predictions=None):
    """ Validates a classifier
        
        :param model: base model (frozen)
        :param posthoc: posthoc classifier for detecting adversarial samples
        :param adv_classifier: classifier trained on adversarial data
        :param clean_classifier: classifier trained on clean data
        :param validation_loader: dataloader of the validation dataset
        :param tensor_dir: if set saves the output of the model in the dir
        :param adversarial_attack: adversarial attack for adversarial training. Default: None -> the classifier is trained without adversarial perturbation.
        :param n: from DINO. Default: 4
        :param avgpool_patchtokens: from DINO. Default: False
        
    """
    # set all models to evaluation mode
    posthoc.eval()
    adv_classifier.eval()
    clean_classifier.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if 'num_labels' in dir(clean_classifier):
        num_labels = clean_classifier.num_labels
    else:
        num_labels = clean_classifier.module.num_labels
    
    if path_predictions is not None:
        print(f'''saving predictions to: {path_predictions}''')
        path_predictions.parent.mkdir(parents=True, exist_ok=True)
        names = []
        true_labels = []
        predicted_labels = []
        if adversarial_attack is not None:
            adv_predicted_labels = []
    
    # validation
    for inp, target, batch_names in metric_logger.log_every(validation_loader, 5, header):

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True) 

        # Normalize
        transform = pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        norminp = transform(inp)  
        
        # benign
        # forward
        with torch.no_grad():
            if 'get_intermediate_layers' in dir(model):
                intermediate_output = model.get_intermediate_layers(norminp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(norminp)
                
            # save output
            if tensor_dir is not None:
                save_output_batch(output, batch_names, tensor_dir)
            
            is_adv = posthoc(output).argmax(axis=1)
            
            use_cuda = torch.cuda.is_available()
            final_output = torch.empty(output.shape[0], clean_classifier.linear.out_features).to("cuda" if use_cuda else "cpu")
            
            for ind, sample in enumerate(is_adv):
                sample_in = output[ind].unsqueeze(0)
                if sample:
                    final_output[ind] = adv_classifier(sample_in)
                else:
                    final_output[ind] = clean_classifier(sample_in)

            loss = nn.CrossEntropyLoss()(final_output, target)
        
        if num_labels >= 5:
            acc1, acc5 = utils.accuracy(final_output, target, topk=(1, 5))
        else:
            acc1, = utils.accuracy(final_output, target, topk=(1,))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)    
        if path_predictions is not None:
            names.extend(batch_names)
            true_labels.extend(target.tolist())
            predicted_labels.extend(torch.argmax(output,-1).tolist())
            if adversarial_attack is not None:
                adv_predicted_labels.extend(torch.argmax(adv_output,-1).tolist())
    if num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
        if adversarial_attack is not None:
            print('* adv_Acc@1 {top1.global_avg:.3f} adv_Acc@5 {top5.global_avg:.3f} adv_loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.adv_acc1, top5=metric_logger.adv_acc5, losses=metric_logger.adv_loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
        if adversarial_attack is not None:
            print('* adv_Acc@1 {top1.global_avg:.3f} adv_loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.adv_acc1, losses=metric_logger.adv_loss))
    if path_predictions is not None:
        data_dict = {"file": names, "true_labels": true_labels, "pred_labels": predicted_labels}
        if adversarial_attack is not None:
            data_dict["adv_pred_labels"] =  adv_predicted_labels
        pd.DataFrame(data_dict).to_csv(path_predictions, sep=",", index=None)
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metric_logger
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


def train(model, classifier, train_loader, validation_loader, log_dir=None, tensor_dir=None, optimizer=None, adversarial_attack=None, epochs=5, val_freq=1, batch_size=16,  lr=0.001, to_restore = {"epoch": 0, "best_acc": 0.}, n=4, avgpool_patchtokens=False):
    """ Trains a classifier ontop of a base model. The input can be perturbed by selecting an adversarial attack.
        
        :param model: base model (frozen)
        :param classifier: classifier to train
        :param train_loader: dataloader of the train dataset
        :param validation_loader: dataloader of the validation dataset
        :param log_dir: path to the log directory.
        :param tensor_dir: if set saves the output of the model in the dir
        :param optimizer: optimizer for the training process. Default: None -> uses the SGD as defined by DINO.
        :param adversarial_attack: adversarial attack for adversarial training. Default: None -> the classifier is trained without adversarial perturbation.
        :param epochs: number of epochs to train the classifier on. Default: 5
        :param val_freq: frequency (in epochs) in which the classifier is validated.
        :param batch_size: batch_size for training and validation. Default: 16
        :param lr: the learning rate of the optimizer if the DINO optimizer is used. Default: 0.001
        :param to_restore:
        :param n: from DINO. Default: 4
        :param avgpool_patchtokens: from DINO. Default: False
        
    """
    model.cuda()
    model.eval()
    
    if optimizer is None:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr * (batch_size * utils.get_world_size()) / 256., # linear scaling rule
            momentum=0.9,
            weight_decay=0, # we do not apply weight decay
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)
    
    # Optionally resume from a checkpoint
    utils.restart_from_checkpoint(
        Path(log_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    
    # train loop
    loggers = {'train':[], 'validation':[]}
    for epoch in range(start_epoch, epochs):
        if 'set_epoch' in dir(train_loader.sampler):
            train_loader.sampler.set_epoch(epoch)

        # train epoch
        train_stats, metric_logger = train_epoch(model, classifier, optimizer, train_loader, tensor_dir, adversarial_attack, epoch, n, avgpool_patchtokens)
        loggers['train'].append(metric_logger)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        
        # validate
        if epoch % val_freq == 0 or epoch == epochs - 1:
            test_stats, metric_logger = validate_network(model, classifier, validation_loader, tensor_dir, adversarial_attack, n, avgpool_patchtokens)
            loggers['validation'].append(metric_logger)
            print(f"Accuracy at epoch {epoch} of the network on the {len(validation_loader.dataset)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
        # log
        if utils.is_main_process():
            with (Path(log_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, Path(log_dir, "checkpoint.pth.tar"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))
    return loggers
    
    


def train_epoch(model, classifier, optimizer, train_loader, tensor_dir=None, adversarial_attack=None, epoch=0, n=4, avgpool=False):
    """ Trains a classifier ontop of a base model. The input can be perturbed by selecting an adversarial attack.
        
        :param model: base model (frozen)
        :param classifier: classifier to train
        :param optimizer: optimizer for the training process.
        :param train_loader: dataloader of the train dataset
        :param tensor_dir: if set saves the output of the model in the dir

        :param adversarial_attack: adversarial attack for adversarial training. Default: None -> the classifier is trained without adversarial perturbation.
        :param epochs: The current epch
        :param n: from DINO. Default: 4
        :param avgpool_patchtokens: from DINO. Default: False
        
    """
    classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target, names) in metric_logger.log_every(train_loader, 20, header):
        
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # adversarial attack  
        if adversarial_attack is not None:
            inp = adversarial_attack(inp, target)
        
        # Normalize
        transform = pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        inp = transform(inp)
        
        # forward
        with torch.no_grad():
            if 'get_intermediate_layers' in dir(model):
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)

        # save output      
        if tensor_dir is not None and epoch == 0:
            save_output_batch(output, names, tensor_dir)
        
        output = classifier(output)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metric_logger


def validate_network(model, classifier, validation_loader, tensor_dir=None, adversarial_attack=None, n=4, avgpool=False):
    """ Validates a classifier
        
        :param model: base model (frozen)
        :param classifier: classifier to train
        :param validation_loader: dataloader of the validation dataset
        :param tensor_dir: if set saves the output of the model in the dir
        :param adversarial_attack: adversarial attack for adversarial training. Default: None -> the classifier is trained without adversarial perturbation.
        :param n: from DINO. Default: 4
        :param avgpool_patchtokens: from DINO. Default: False
        
    """
    model.eval()
    classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    if 'num_labels' in dir(classifier):
        num_labels = classifier.num_labels
    else:
        num_labels = classifier.module.num_labels
    for inp, target, names in metric_logger.log_every(validation_loader, 20, header):

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
                save_output_batch(output, names, tensor_dir)

            output = classifier(output)
            loss = nn.CrossEntropyLoss()(output, target)
        
        if num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, target, topk=(1,))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
        # adversarial attack
        if adversarial_attack is not None:
            inp = adversarial_attack(inp, target)

            # Normalize
            transform = pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            inp = transform(inp)  

            # forward
            with torch.no_grad():
                if 'get_intermediate_layers' in dir(model):
                    intermediate_output = model.get_intermediate_layers(inp, n)
                    output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                    if avgpool:
                        output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                        output = output.reshape(output.shape[0], -1)
                else:
                    output = model(inp)

                # save output
                if tensor_dir is not None:
                    save_output_batch(output, names, tensor_dir)

                output = classifier(output)
                adv_loss = nn.CrossEntropyLoss()(output, target)

            if num_labels >= 5:
                adv_acc1, adv_acc5 = utils.accuracy(output, target, topk=(1, 5))
            else:
                adv_acc1, = utils.accuracy(output, target, topk=(1,))

            batch_size = inp.shape[0]
            metric_logger.update(adv_loss=adv_loss.item())
            metric_logger.meters['adv_acc1'].update(adv_acc1.item(), n=batch_size)
            if num_labels >= 5:
                metric_logger.meters['adv_acc5'].update(adv_acc5.item(), n=batch_size)
                
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
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metric_logger



def save_output_batch(batch_out, batch_names, output_dir):
    for out, name in zip(batch_out, batch_names):
        torch.save(out, Path(output_dir,name.split('.')[0]+'.pt'))
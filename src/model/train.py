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

def train(model, 
          classifier, 
          train_loader, 
          validation_loader, 
          log_dir=None, 
          tensor_dir=None, 
          optimizer=None, 
          criterion=nn.CrossEntropyLoss(), 
          adversarial_attack=None, 
          epochs=None, 
          val_freq=1, 
          batch_size=16, 
          lr=0.001, 
          to_restore = {"epoch": 0, "best_acc": 0.}, 
          n=4, 
          avgpool_patchtokens=False, 
          show_image=False,
          args = None):

    """ Trains a classifier ontop of a base model. The input can be perturbed by selecting an adversarial attack.
        
        :param model: base model (frozen)
        :param classifier: classifier to train
        :param train_loader: loader of the train set
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
    if args is not None:
        avgpool_patchtokens = args.avgpool_patchtokens
        device = args.device
        num_labels = args.num_labels
        if log_dir is None:
            log_dir = args.log_dir
        if tensor_dir is None:
            tensor_dir = args.out_dir
        if epochs is None:
            epochs = args.epochs
        val_freq = args.val_freq
        lr = args.lr
        n = args.n_last_blocks
        batch_size = args.batch_size
    if model is not None:
        model.cuda()
        model.eval()
        
    classifier.cuda()
    if optimizer is None:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr * (batch_size * utils.get_world_size()) / 256., # linear scaling rule
            momentum=0.9,
            weight_decay=0, # we do not apply weight decay
        )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)
    
    # Optionally resume from a checkpoint
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
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
        train_stats, metric_logger = train_epoch(model=model, 
                                                 classifier=classifier, 
                                                 optimizer=optimizer, 
                                                 criterion=criterion,
                                                 train_loader=train_loader, 
                                                 tensor_dir=tensor_dir, 
                                                 adversarial_attack=adversarial_attack, 
                                                 epoch=epoch, 
                                                 n=n, 
                                                 avgpool_patchtokens=avgpool_patchtokens, 
                                                 show_image=show_image)
        loggers['train'].append(metric_logger)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        
        # validate
        if epoch % val_freq == 0 or epoch == epochs - 1:
            if tensor_dir is not None:
                tensor_dir_epoch = Path(tensor_dir, epoch)
                tensor_dir_epoch.mkdir(parents=True, exist_ok=True)
            else:
                tensor_dir_epoch = tensor_dir
            test_stats, metric_logger = validate_network(model=model, 
                                                         classifier=classifier, 
                                                         validation_loader=validation_loader, 
                                                         criterion=criterion,
                                                         tensor_dir=tensor_dir_epoch, 
                                                         adversarial_attack=adversarial_attack, 
                                                         n=n, 
                                                         avgpool_patchtokens=avgpool_patchtokens, 
                                                         show_image=show_image)
            loggers['validation'].append(metric_logger)
            print(f"Accuracy at epoch {epoch} of the network on the {len(validation_loader)} test images: {test_stats['acc1']:.1f}%")
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
    
    


def train_epoch(model, 
                classifier, 
                train_loader, 
                optimizer, 
                criterion=nn.CrossEntropyLoss(), 
                tensor_dir=None, 
                adversarial_attack=None, 
                epoch=0, 
                n=4, 
                avgpool_patchtokens=False, 
                show_image=False,
                log_interval=None):
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
    if log_interval is None:
        if len(train_loader)<20:
            log_interval = 1
        elif len(train_loader)<100:
            log_interval = 5
        else:
            log_interval = 20   
    
    for (inp, target, names) in metric_logger.log_every(train_loader, log_interval, header):
        
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # adversarial attack  
        if adversarial_attack is not None:
            inp = adversarial_attack(inp, target)
        
        if model is not None:
            # forward
            with torch.no_grad():
                model_output = model_forward(model, inp, n, avgpool_patchtokens)
                
            # save output      
            if tensor_dir is not None and epoch == 0:
                save_output_batch(model_output, names, tensor_dir)
            
            output = classifier(model_output)
            
        else:
            output = classifier(inp)
            
        # compute loss
        loss = criterion(output, target)

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
    
    if show_image:
        imshow(inp[0].detach())
        if len(inp)>14:
            imshow(inp[14].detach())
        imshow(inp[-1].detach())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metric_logger


def validate_network(model, 
                     classifier, 
                     validation_loader, 
                     criterion=nn.CrossEntropyLoss(), 
                     tensor_dir=None, 
                     adversarial_attack=None, 
                     n=4, 
                     avgpool_patchtokens=False, 
                     path_predictions=None, 
                     show_image=False,
                     log_interval=None):
    """ Validates a classifier ontop of an optional model with an optional 
        adversarial attack. 
        
        :param model: base model (frozen)
        :param classifier: classifier to train
        :param validation_loader: dataloader of the validation dataset
        :param criterion: The loss criterion. Default: CrossEntropyLoss
        :param tensor_dir: if set saves the output of the model in the dir.
        :param adversarial_attack: adversarial attack for adversarial training.
                                   Default: None -> the classifier is trained 
                                   without adversarial perturbation.
        :param n: from DINO. Default: 4
        :param avgpool_patchtokens: from DINO. Default: False
        :param path_predictions: If given, saves a csv file at path_predictions
                                 containing the filenames, the true label, the 
                                 prediction and if there is an adversarial 
                                 attack the adversarial prediction.
        :param show_image: shows the last couple images in the last batch.
    """
    if model is not None:
        model.eval()
    classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    if 'num_labels' in dir(classifier):
        num_labels = classifier.num_labels
    else:
        num_labels = classifier.module.num_labels
    if path_predictions is not None:
        print(f'''saving predictions to: {path_predictions}''')
        path_predictions.parent.mkdir(parents=True, exist_ok=True)
        names = []
        true_labels = []
        predicted_labels = []
        if adversarial_attack is not None:
            adv_predicted_labels = []
    if tensor_dir is not None:
        tensor_dir.mkdir(parents=True, exist_ok=True)
        if adversarial_attack is not None:
            adv_tensor_dir = Path(tensor_dir,'adv')
            adv_tensor_dir.mkdir(parents=True, exist_ok=True)
    
    if log_interval is None:
        if len(validation_loader)<20:
            log_interval = 1
        elif len(validation_loader)<100:
            log_interval = 5
        else:
            log_interval = 20
    for inp, target, batch_names in metric_logger.log_every(validation_loader, log_interval, header):

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True) 

        # benign
        # forward
        with torch.no_grad():
            if model is not None:
                model_output = model_forward(model, inp, n, avgpool_patchtokens)
                
                # save output
                if tensor_dir is not None:
                    save_output_batch(model_output, batch_names, tensor_dir)
                
                output = classifier(model_output)
            else:
                output = classifier(inp)

            loss = criterion(output, target)
        
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
            
            # forward
            with torch.no_grad():
                if model is not None:
                    model_output = model_forward(model, inp, n, avgpool_patchtokens)

                    # save output
                    if tensor_dir is not None:
                        save_output_batch(model_output, batch_names, adv_tensor_dir)
                    
                    adv_output = classifier(model_output)
                    
                else:
                    adv_output = classifier(inp)
                    
                adv_loss = criterion(adv_output, target)

            if num_labels >= 5:
                adv_acc1, adv_acc5 = utils.accuracy(adv_output, target, topk=(1, 5))
            else:
                adv_acc1, = utils.accuracy(adv_output, target, topk=(1,))

            batch_size = inp.shape[0]
            metric_logger.update(adv_loss=adv_loss.item())
            metric_logger.meters['adv_acc1'].update(adv_acc1.item(), n=batch_size)
            if num_labels >= 5:
                metric_logger.meters['adv_acc5'].update(adv_acc5.item(), n=batch_size)
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
    if show_image:
        imshow(inp[0].detach())
        if len(inp)>14:
            imshow(inp[14].detach())
        imshow(inp[-1].detach())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metric_logger



def save_output_batch(batch_out, batch_names, output_dir):
    """ Saves the output of a model into the output_dir.
        
        :param batch_out: The model output that will be saved.
        :param batch_names: The corresponding names.
        :param output_dir: The output directory in which the output will be 
        saved.
    """
    batch_out_cpu = batch_out.clone().detach()
    batch_out_cpu.cpu()
    for out, name in zip(batch_out_cpu, batch_names):
        out_path = Path(output_dir,name.split('.')[0]+'.pt')
        torch.save(out, out_path)



def model_forward(model, inp, n=4, avgpool_patchtokens=False):
    """ Performs a forward pass on a dino model.
        
        :param model: dino model (frozen)
        :param inp: the input for the model
        :param n: from DINO. Default: 4
        :param avgpool_patchtokens: from DINO. Default: False
    """
    
    # Normalize
    transform = pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    inp = transform(inp)  

    if 'get_intermediate_layers' in dir(model):
        intermediate_output = model.get_intermediate_layers(inp, n)
        model_output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        if avgpool_patchtokens:
            model_output = torch.cat((model_output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
            model_output = model_output.reshape(model_output.shape[0], -1)
    else:
        model_output = model(inp)
    return model_output
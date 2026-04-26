import os
import csv
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    model.train()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]

    for batch_idx, data in enumerate(tqdm(loader)):
        if data == None:
            continue
        data = device.dict_data_to_device(data)
        ret_dict = model(data)

        loss, loss_details = model.get_loss(ret_dict, data['label'])
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print(data['origin_info'])
            continue
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()

        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                f'\tEpoch: {epoch_idx}, Batch({batch_idx}/{len(loader)}) done. Loss: {loss.item():.2f}  lr:{clr[0]:.6f}'
            )
            recoder.print_log(
                "\t"
                + ", ".join([f"{k}: {v.item():.2f}" for k, v in loss_details.items()])
            )
    optimizer.scheduler.step()
    recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    return loss_value

def seq_eval(
    cfg, loader, model, device, mode, epoch, work_dir, recoder
):
    model.eval()
    sum = 0
    acc = 0
    for batch_idx, data in enumerate(tqdm(loader)):
        if data == None:
            continue
        data = device.dict_data_to_device(data)
        label = data['label'].to(device.device).squeeze().long()
        sum += label.size(0)
        with torch.no_grad():
            ret_dict = model(data)
            _, predicted = torch.max(ret_dict['result'], 1)
            acc += (predicted == label).sum().item()
    recoder.print_log(
        f'\t{mode} Epoch: {epoch}, Acc: {(acc*100/sum):.2f} %'
    )
    recoder.print_log(
        f'\t{mode} Epoch: {epoch}, Acc: {(acc*100/sum):.2f} %', path=f'{work_dir}{mode}.txt'
    )

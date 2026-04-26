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
    turn_sum = 0
    turn_acc = 0
    class_correct = {0: 0, 1: 0, 2: 0, 3: 0}
    class_total = {0: 0, 1: 0, 2: 0, 3: 0}
    class_names = {0: 'Off', 1: 'Left', 2: 'Right', 3: 'Both'}
    for batch_idx, data in enumerate(tqdm(loader)):
        data = device.dict_data_to_device(data)
        labels_turn = data['label']
        with torch.no_grad():
            ret_dict = model(data)
            B, num_classes = ret_dict['turn_result'].shape
            turn_sum += B
            _, predicted_turn = torch.max(ret_dict['turn_result'], 1)
            turn_acc += (predicted_turn == labels_turn).sum().item()
            for cls_id in range(4):
                label_mask = (labels_turn == cls_id)
                class_total[cls_id] += label_mask.sum().item()
                class_correct[cls_id] += ((predicted_turn == cls_id) & label_mask).sum().item()
    detail_acc_str = ""
    for i in range(4):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            detail_acc_str += f" {class_names[i]}: {acc:.2f}%, total: {class_total[i]}, correct: {class_correct[i]} |"
        else:
            detail_acc_str += f" {class_names[i]}: N/A |"
    recoder.print_log(
            f'\tEpoch: {epoch}, Turn acc: {(turn_acc*100/turn_sum):.2f}%'
            )
    recoder.print_log(f'\tDetails ->{detail_acc_str}')
    recoder.print_log(
            f'\tEpoch: {epoch}, Turn acc: {(turn_acc*100/turn_sum):.2f}%', path=f'{work_dir}test.txt'
        )
    recoder.print_log(f'\tDetails ->{detail_acc_str}', path=f'{work_dir}test.txt')


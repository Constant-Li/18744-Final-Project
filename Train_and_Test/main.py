import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import utils
import numpy as np
import modules
import torch
import torch.nn as nn
import datasets
import yaml
import pdb
import json
import faulthandler
faulthandler.enable()

from seq_scripts import seq_train, seq_eval
# from seq_scripts_video import seq_train, seq_eval
import light_network, video_network, TLD_video_network

class SLRProcessor(object):
    def __init__(self, arg):
        super().__init__()

        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.recoder = utils.Recorder(
            self.arg.work_dir, self.arg.print_log, self.arg.log_interval
        )
        self.dataset = {}
        self.data_loader = {}

        # self.load_dataset_info()
        self.model, self.optimizer = self.loading()
        self.best_dev_wer, self.best_test_wer = 1000, 1000
    
    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def loading(self):
        self.device.set_device()
        print("Loading model")
        model = self.build_module(self.arg.model_args)
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)
        model = self.model_to_device(model)
        print("Loading model finished.")
        self.load_data()
        return model, optimizer

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = torch.load(self.arg.load_checkpoints)

        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            print("Loading random seeds...")
            self.rng.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            optimizer.to(self.device.device)
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        self.recoder.print_log("Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}")
    
    def model_to_device(self, model):
        model = model.to(self.device.device)
        return model

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)['model_state_dict']
        # pdb.set_trace()
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        # pdb.set_trace()
        model.load_state_dict(state_dict, strict=False)
    
    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size
            if mode == "train"
            else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,  # if train_flag else 0
            collate_fn=self.feeder.collate_fn,
        )

    def build_module(self, args):
        model_class = getattr(light_network, self.arg.model)
        # model_class = getattr(video_network, self.arg.model)
        # model_class = getattr(TLD_video_network, self.arg.model)
        model = model_class(
            **args,
        )
        return model

    def load_data(self):
        print("Loading data")
        self.feeder = getattr(datasets, self.arg.feeder)
        if self.arg.dataset == 'ETR':
            dataset_list = zip(
                ["train", "val", "test"], [True, False, False]
            )
            for idx, (mode, train_flag) in enumerate(dataset_list):
                self.dataset[mode] = self.feeder(mode, transfer_label=self.arg.transfer_label)
                self.data_loader[mode] = self.build_dataloader(
                    self.dataset[mode], mode, train_flag
                )
            print("Loading data finished.")
        elif self.arg.dataset == 'TLD_YT':
            dataset_list = zip(
                ["train", "test"], [True, False]
            )
            for idx, (mode, train_flag) in enumerate(dataset_list):
                self.dataset[mode] = self.feeder(mode)
                self.data_loader[mode] = self.build_dataloader(
                    self.dataset[mode], mode, train_flag
                )
            print("Loading data finished.")
    
    # def load_dataset_info(self):
    #     with open(f"./configs/dataset_configs/{self.arg.dataset}.yaml", 'r') as f:
    #         self.arg.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    
    def judge_save_eval(self, epoch):
        # save_model = (epoch % self.arg.save_interval == 0) and (epoch >= 0.7 * self.arg.num_epoch)
        save_model = (epoch % self.arg.save_interval == 0)
        # eval_model = (epoch % self.arg.eval_interval == 0) and (epoch >= 0)
        eval_model = (epoch % self.arg.eval_interval == 0) and (epoch == 9)
        return save_model, eval_model

    def save_model(self, epoch, save_path):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
                'rng_state': self.rng.save_rng_state(),
            },
            save_path,
        )

    def train(self):
        self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        for epoch in range(
            self.arg.optimizer_args['start_epoch'], self.arg.num_epoch
        ):
            save_model, eval_model = self.judge_save_eval(epoch)
            # if self.arg.dataset == 'TLD_YT' and epoch > 0:
            #     self.dataset['train'] = self.feeder('train')
            #     self.data_loader['train'] = self.build_dataloader(self.dataset['train'], 'train', True)
            seq_train(self.data_loader['train'], self.model, self.optimizer, self.device,
                epoch, self.recoder, **self.arg.train_args
            )

            if eval_model:
                self.test('test', epoch)
                if self.arg.dataset == 'ETR':
                    self.test('val', epoch)
            
            if save_model:
                model_path = f'{self.arg.work_dir}cur_test_model.pt'
                self.save_model(epoch, model_path)

    def test(self, mode, epoch):
        wer = seq_eval(
                self.arg,
                self.data_loader[mode],
                self.model,
                self.device,
                mode,
                epoch,
                self.arg.work_dir,
                self.recoder,
            )
        return wer

    def start(self):
        if self.arg.phase == 'train':
            self.train()
        elif self.arg.phase == 'test':
            # if self.arg.load_weights is None and self.arg.load_checkpoints is None:
            #     raise ValueError('Please appoint --load-weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            self.test('test', 6667)
            # self.test('train', 6667)
            if self.arg.dataset == 'ETR':
                self.test('val', 6667)
            self.recoder.print_log('Evaluation Done.\n')

if __name__ == '__main__':
    sparser = utils.get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert k in key
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()

    main_processor = SLRProcessor(args)
    main_processor.start()
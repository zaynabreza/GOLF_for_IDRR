# -*- coding: utf-8 -*-
"""
Created on Jan 21 2023
@author: JIANG Yuxin
"""

import time
import torch
import argparse
import logging as lgg
import adapters
import transformers.utils.logging
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from datetime import datetime
import warnings
import numpy as np
import os
import random
import json
from training_adapter import train
from GOLF import Model
from utils import MyDataset, get_time_dif

warnings.filterwarnings("ignore")
transformers.utils.logging.set_verbosity_error()


def setlogging(level, filename):
    for handler in lgg.root.handlers[:]:
        lgg.root.removeHandler(handler)
    lgg.basicConfig(level=level,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%H:%M',
                    filename=filename,
                    filemode='w')
    logc = lgg.StreamHandler()
    logc.setLevel(level=lgg.DEBUG)
    logc.setFormatter(lgg.Formatter('%(message)s'))
    lgg.getLogger().addHandler(logc)


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# torch.use_deterministic_algorithms(True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchical Implicit Discourse Relation Recognition')
    parser.add_argument('--cuda', type=int, default=0, choices=[0, 1], help='choose a cuda: 0 or 1')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    ## File paths
    parser.add_argument('--dataset', type=str, default='DiscoGem', help='the file of data')
    parser.add_argument('--reference_labels', type=str, default='PDTB3', help='labels to use')

    ## model arguments
    parser.add_argument('--model_name_or_path', type=str, default='roberta-large', help='the name of pretrained model')
    parser.add_argument('--freeze_bert', action='store_true', default=False,
                        help='whether freeze the parameters of bert')
    parser.add_argument('--adapter_name', type=str, default="lora", help='which adapter to use')
    parser.add_argument('--use_adapters', action='store_true', default=True,
                        help='whether to use adapters or fine-tune the whole model')
    parser.add_argument('--model_ckpt', type=str, default="./PDTB3/Ji/saved_dict/roberta-large_25epochs.ckpt", help='which model ckpt')

    ## training arguments
    parser.add_argument('--pad_size', type=int, default=100, help='the max sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epoch', type=int, default=15, help='training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='warmup_ratio')
    parser.add_argument('--evaluate_steps', type=int, default=100, help='number of evaluate_steps')
    parser.add_argument('--require_improvement', type=int, default=10000, help='early stop steps')

    args = parser.parse_args()

    args.data_file = '{}/data/'.format(args.dataset)
    args.t = datetime.now().strftime('%B%d-%H:%M:%S')
    args.unique_name = str(args.t)


    args.save_folder = "{}/runs/{}_{}/".format(args.dataset,args.model_name_or_path,args.unique_name)
    args.log = args.save_folder + 'train.log'


    os.makedirs(args.save_folder,exist_ok=True)

    if args.reference_labels == "PDTB3":
        args.i2sec = [x.strip() for x in open('PDTB3/Ji/data/sec.txt').readlines()]
    else:
        args.i2sec = [x.strip() for x in open(args.data_file + 'sec.txt').readlines()]

    args.sec2i = dict((x, xid) for xid, x in enumerate(args.i2sec))
    args.n_sec = len(args.i2sec)
    args.label_num = args.n_sec  # total label num(top:4,second:11,conn:102)]

    # args.n_sec = 14
    # args.label_num = 14
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    args.config = AutoConfig.from_pretrained(args.model_name_or_path)


    args.device = torch.device('cuda:{0}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

    setlogging(lgg.DEBUG, args.log)
    seed_torch(args.seed)

    hyper_parameters = vars(args).copy()

    hyper_parameters['i2sec'] = ''
    hyper_parameters['sec2i'] = ''
    hyper_parameters['tokenizer'] = ''
    hyper_parameters['config'] = ''
    hyper_parameters['device'] = str(args.device)
    lgg.info(hyper_parameters)

    with open('{}/config.json'.format(args.save_folder), 'w') as fp:
        json.dump(hyper_parameters, fp,indent=4)

    start_time = time.time()
    lgg.info("Loading data...")

    train_dataset = MyDataset(args, args.data_file + 'train.txt')
    dev_dataset = MyDataset(args, args.data_file + 'dev.txt')
    test_dataset = MyDataset(args, args.data_file + 'test.txt')

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    time_dif = get_time_dif(start_time)
    lgg.info("Time usage: {}".format(time_dif))

    # train
    model = Model(args).to(args.device)
    # for i,param in model.named_parameters():
    #     print(i,param,param.requires_grad)
    #     break
    #
    ### loading PDTB pre-trained roberta weights
    checkpoint = torch.load(args.model_ckpt)
    model.load_state_dict(checkpoint)
    # for i,param in model.named_parameters():
    #     print(i,param,param.requires_grad)
    #     break
    print(args)
    if args.use_adapters:
        adapters.init(model.bert)
        model.bert.add_adapter(args.adapter_name, config=args.adapter_name)
        for i,param in model.named_parameters():
            if args.adapter_name in i:
                param.requires_grad = True
            else:
                param.requires_grad = False
        model.bert.active_adapters = args.adapter_name
        # for i, param in model.named_parameters():
        #     print(i,param.shape,param.requires_grad)

    elif args.freeze_bert: ## freeze bert
        for i,param in model.bert.named_parameters():
            param.requires_grad = False


    train(args, model, train_loader, dev_loader, test_loader)

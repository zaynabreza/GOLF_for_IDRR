# -*- coding: utf-8 -*-
"""
Created on Jan 21 2023
@author: JIANG Yuxin
"""

import numpy as np
import torch
import torch.nn.functional as F
import logging as lgg
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained_bert.optimization import BertAdam


def train(args, model, train_loader, dev_loader, test_loader):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.lr,
                         warmup=args.warmup_ratio,
                         t_total=len(train_loader) * args.epoch)

    # Initialize variables for tracking progress
    total_batch = 0
    dev_best_acc_sec = 0.0
    dev_best_f1_sec = 0.0

    last_improve = 0
    flag = False
    for epoch in range(args.epoch):
        start_time = time.time()
        lgg.info('Epoch [{}/{}]'.format(epoch + 1, args.epoch))
        for i, (x, mask, token_type, y1_sec, y2_sec, arg1_mask, arg2_mask) in enumerate(train_loader):
            model.train()
            logits_sec, loss = model(x, mask, y1_sec, arg1_mask, arg2_mask, train=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_batch += 1
            if total_batch % args.evaluate_steps == 0:
                print(total_batch)

                y_true_sec = y1_sec.data.cpu() # (batch)
                y_predit_sec = torch.max(logits_sec.data, 1)[1].cpu() # (batch)
                train_acc_sec = metrics.accuracy_score(y_true_sec, y_predit_sec)
                
                # evaluate
                loss_dev, acc_sec, f1_sec = evaluate(args, model, dev_loader)
                
                if (acc_sec + f1_sec) > (dev_best_acc_sec + dev_best_f1_sec):
                    dev_best_f1_sec = f1_sec
                    dev_best_acc_sec = acc_sec
                    torch.save(model.state_dict(), args.save_file + args.model_name_or_path.split('/')[-1] + '.ckpt')
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                lgg.info('SEC: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%}, Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%} Time: {6} {7}'.format(total_batch, loss.item(), train_acc_sec, loss_dev, acc_sec, f1_sec, time_dif, improve))

                if total_batch - last_improve > args.require_improvement:
                    # training stop
                    lgg.info("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
        if flag:
            break

        time_dif = get_time_dif(start_time)
        lgg.info("Train time usage: {}".format(time_dif))
        acc_sec_test, f1_sec_test = test(args, model, test_loader)

    dev_msg = 'dev_best_acc_sec: {0:>6.2%},  dev_best_f1_sec: {1:>6.2%}'
    lgg.info(dev_msg.format(dev_best_acc_sec, dev_best_f1_sec))



def test(args, model, test_loader):
    model.load_state_dict(torch.load(args.save_file + args.model_name_or_path.split('/')[-1] + '.ckpt'))
    model.eval()
    start_time = time.time()

    test_loss, acc_sec, f1_sec, report_sec, confusion_sec, predictions = evaluate(args, model, test_loader, test=True)


    time_dif = get_time_dif(start_time)
    lgg.info("Test time usage: {}".format(time_dif))

    msg = 'SEC: Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test F1: {2:>6.2%}'
    lgg.info(msg.format(test_loss, acc_sec, f1_sec))
    
    lgg.info(report_sec)

    np.savetxt(args.save_preds+args.model_name_or_path+'preds.csv', predictions, fmt='%d', delimiter=',')
    return acc_sec, f1_sec


def evaluate(args, model, data_loader, test=False):
    model.eval()
    loss_total = 0
   

    predict_all_sec = np.array([], dtype=int)
    labels1_all_sec = np.array([], dtype=int)
    labels2_all_sec = np.array([], dtype=int)


    with torch.no_grad():
        for i, (x, mask, token_type, y1_sec, y2_sec, arg1_mask, arg2_mask) in enumerate(data_loader):

            logits_sec = model(x, mask, y1_sec, arg1_mask, arg2_mask, train=False)
            
            
            loss_sec = F.cross_entropy(logits_sec, y1_sec)

            loss = loss_sec 
            loss_total += loss
            
          
            y_predit_sec = torch.max(logits_sec.data, 1)[1].cpu().numpy()
          

            y1_true_sec = y1_sec.data.cpu().numpy()
            y2_true_sec = y2_sec.data.cpu().numpy()
            labels1_all_sec = np.append(labels1_all_sec, y1_true_sec) # collect all sec true label
            labels2_all_sec = np.append(labels2_all_sec, y2_true_sec)
            predict_all_sec = np.append(predict_all_sec, y_predit_sec) # collect all sec predicted label



    predict_sense_sec = predict_all_sec
    gold_sense_sec = labels1_all_sec
    mask = (predict_sense_sec == labels2_all_sec)
    gold_sense_sec[mask] = labels2_all_sec[mask]


    # PDTB2.0 cutoff
    if test:
        cut_off = 1039
    else:
        cut_off = 1165
    
    # PDTB3.0 cutoff    
    # if test:
    #     cut_off = 1474
    # else:
    #     cut_off = 1653


    gold_sense_sec = gold_sense_sec[: cut_off]
    predict_sense_sec = predict_sense_sec[: cut_off]
    acc_sec = metrics.accuracy_score(gold_sense_sec, predict_sense_sec)
    f1_sec = metrics.f1_score(gold_sense_sec, predict_sense_sec, average='macro')

    if test:

        report_sec = metrics.classification_report(gold_sense_sec, predict_sense_sec, target_names=args.i2sec, digits=4)
        confusion_sec = metrics.confusion_matrix(gold_sense_sec, predict_sense_sec)
        predictions = np.vstack((gold_sense_sec, predict_sense_sec)).T

        return loss_total / len(data_loader), acc_sec, f1_sec, report_sec, confusion_sec, predictions
                
    return loss_total / len(data_loader), acc_sec, f1_sec

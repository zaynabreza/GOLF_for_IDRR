import torch
import numpy as np
from sklearn import metrics
import logging as lgg
import time
from utils import MyDataset, get_time_dif

def CoNLL_eval(args, model, test_loader,ckpt_path=None):
    # Load the checkpoint
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
    else: ### Zayanab's condition
        model.load_state_dict(torch.load(args.save_file + args.model_name_or_path.split('/')[-1] + '.ckpt'))

    model.eval()
    start_time = time.time()

    gold_labels1, gold_labels2, predictions = [], [], []
    with torch.no_grad():
        for i, (x, mask, token_type, y1_sec, y2_sec, arg1_mask, arg2_mask) in enumerate(test_loader):
            logits_sec = model(x, mask, y1_sec, arg1_mask, arg2_mask, train=False)
            y_predit_sec = torch.max(logits_sec.data, 1)[1].cpu().numpy()

            gold_labels1.extend(y1_sec.data.cpu().numpy())
            gold_labels2.extend(y2_sec.data.cpu().numpy())
            predictions.extend(y_predit_sec)

    gold_labels1 = np.array(gold_labels1)
    gold_labels2 = np.array(gold_labels2)
    predictions = np.array(predictions)

    # CoNLL method evaluation
    # Initialize gold_labels with -1 as placeholder
    gold_labels = -1 * np.ones_like(gold_labels1)  # Set initially to -1

    # If gold1 is valid, start with gold1
    valid_gold1_mask = (gold_labels1 != -1)
    gold_labels[valid_gold1_mask] = gold_labels1[valid_gold1_mask]

    # Create mask where predictions match Gold2 but not Gold1 and Gold2 is valid
    valid_gold2_mask = (gold_labels2 != -1)  # Ensure gold2 is valid
    mask_gold2 = (predictions == gold_labels2) & (predictions != gold_labels1) & valid_gold2_mask

    # Update gold_labels where mask_gold2 is True to prioritize Gold2
    gold_labels[mask_gold2] = gold_labels2[mask_gold2]

    # Filter out any remaining -1 values in gold_labels
    valid_labels_mask = (gold_labels != -1)
    gold_labels = gold_labels[valid_labels_mask]
    predictions = predictions[valid_labels_mask]

    acc_sec = metrics.accuracy_score(gold_labels, predictions)
    f1_sec = metrics.f1_score(gold_labels, predictions, average='macro')

    # Save predictions with class names
    class_names = args.i2sec
    with open(args.save_folder + 'conll_predictions.txt', 'w') as f:
            for gold1, gold2, pred in zip(gold_labels1, gold_labels2, predictions):
                gold1_str = class_names[gold1] if gold1 != -1 else "Unknown"
                gold2_str = class_names[gold2] if gold2 != -1 else "Unknown"
                pred_str = class_names[pred]
                f.write(f"Gold1: {gold1_str}, Gold2: {gold2_str}, Predicted: {pred_str}\n")

    time_dif = get_time_dif(start_time)
    lgg.info("Test time usage: {}".format(time_dif))
    msg = 'CoNLL Method: Test Acc: {0:>6.2%}, Test F1: {1:>6.2%}'
    lgg.info(msg.format(acc_sec, f1_sec))

    return acc_sec, f1_sec

def dup_eval(args, model, test_loader,ckpt_path=None):
    # Load the checkpoint
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
    else: ### Zayanab's condition
        model.load_state_dict(torch.load(args.save_file + args.model_name_or_path.split('/')[-1] + '.ckpt'))

    model.eval()
    start_time = time.time()

    gold_list, pred_list = [], []

    with torch.no_grad():
        for i, (x, mask, token_type, y1_sec, y2_sec, arg1_mask, arg2_mask) in enumerate(test_loader):
            logits_sec = model(x, mask, y1_sec, arg1_mask, arg2_mask, train=False)
            y_predit_sec = torch.max(logits_sec.data, 1)[1].cpu().numpy()

            gold_labels1 = y1_sec.data.cpu().numpy()
            gold_labels2 = y2_sec.data.cpu().numpy()
            predictions = y_predit_sec

            for gold1, gold2, pred in zip(gold_labels1, gold_labels2, predictions):
                if pred == gold1 or pred == gold2:
                        if gold1 != -1:
                            gold_list.append(gold1)
                            pred_list.append(gold1)
                        if gold2 != -1:
                            gold_list.append(gold2)
                            pred_list.append(gold2)
                else:
                    if gold1 != -1:
                        gold_list.append(gold1)
                        pred_list.append(pred)
                    if gold2 != -1:
                        gold_list.append(gold2)
                        pred_list.append(pred)

    # Convert lists to numpy arrays for metrics calculation
    gold_labels = np.array(gold_list)
    predictions = np.array(pred_list)

    # Calculate Accuracy: TP+=1 if predicted is one of the gold labels
    acc_sec = metrics.accuracy_score(gold_labels, predictions)

    # Calculate Macro F1: Average of F1 scores for each label
    f1_sec = metrics.f1_score(gold_labels, predictions, average='macro')

    # Save predictions with class names
    class_names = args.i2sec
    with open(args.save_folder + 'dup_predictions.txt', 'w') as f:
            for gold1, gold2, pred in zip(gold_labels1, gold_labels2, predictions):
                gold1_str = class_names[gold1] if gold1 != -1 else "Unknown"
                gold2_str = class_names[gold2] if gold2 != -1 else "Unknown"
                pred_str = class_names[pred]
                f.write(f"Gold1: {gold1_str}, Gold2: {gold2_str}, Predicted: {pred_str}\n")


    time_dif = get_time_dif(start_time)
    lgg.info("Test time usage: {}".format(time_dif))
    msg = 'Duplicate Method: Test Acc: {0:>6.2%}, Test F1: {1:>6.2%}'
    lgg.info(msg.format(acc_sec, f1_sec))

    return acc_sec, f1_sec


def expand_eval(args, model, test_loader,ckpt_path=None):
    # Load the checkpoint
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
    else: ### Zayanab's condition
        model.load_state_dict(torch.load(args.save_file + args.model_name_or_path.split('/')[-1] + '.ckpt'))
    model.eval()
    start_time = time.time()

    gold_list, pred_list = [], []

    with torch.no_grad():
        for i, (x, mask, token_type, y1_sec, y2_sec, arg1_mask, arg2_mask) in enumerate(test_loader):
            logits_sec = model(x, mask, y1_sec, arg1_mask, arg2_mask, train=False)
            y_predit_sec = torch.max(logits_sec.data, 1)[1].cpu().numpy()

            gold_labels1 = y1_sec.data.cpu().numpy()
            gold_labels2 = y2_sec.data.cpu().numpy()
            predictions = y_predit_sec

            for gold1, gold2, pred in zip(gold_labels1, gold_labels2, predictions):
                if gold1 != -1:
                    gold_list.append(gold1)
                    pred_list.append(pred)
                if gold2 != -1:
                    gold_list.append(gold2)
                    pred_list.append(pred)

    # Convert lists to numpy arrays for metrics calculation
    gold_labels = np.array(gold_list)
    predictions = np.array(pred_list)

    # Calculate Accuracy: TP+=1 if predicted is one of the gold labels
    acc_sec = metrics.accuracy_score(gold_labels, predictions)

    # Calculate Macro F1: Average of F1 scores for each label
    f1_sec = metrics.f1_score(gold_labels, predictions, average='macro')

    # Save predictions with class names
    class_names = args.i2sec
    with open(args.save_folder + 'expand_predictions.txt', 'w') as f:
            for gold1, gold2, pred in zip(gold_labels1, gold_labels2, predictions):
                gold1_str = class_names[gold1] if gold1 != -1 else "Unknown"
                gold2_str = class_names[gold2] if gold2 != -1 else "Unknown"
                pred_str = class_names[pred]
                f.write(f"Gold1: {gold1_str}, Gold2: {gold2_str}, Predicted: {pred_str}\n")

    time_dif = get_time_dif(start_time)
    lgg.info("Test time usage: {}".format(time_dif))
    msg = 'Expand Method: Test Acc: {0:>6.2%}, Test F1: {1:>6.2%}'
    lgg.info(msg.format(acc_sec, f1_sec))

    return acc_sec, f1_sec


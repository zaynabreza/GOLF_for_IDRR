# -*- coding: utf-8 -*-
"""
Created on Jan 21 2023
@author: JIANG Yuxin
"""

import torch
from tqdm import tqdm
import time
from torch.utils.data import Dataset
PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'


def get_time_dif(start_time):
    """ """
    end_time = time.time()
    time_dif = end_time - start_time
    return time_dif
    # return timedelta(seconds=int(round(time_dif)))
    
    
class MyDataset(Dataset):
    
    def __init__(self, args, path):
        content = self.load_dataset(args, path)
        self.len = len(content)
        self.device = args.device
        
        self.x, self.mask, self.token_type, \
        self.y1_sec, self.y2_sec,\
        self.arg1_mask, self.arg2_mask = self._to_tensor(content)


    def __getitem__(self, index):
       return self.x[index], \
               self.mask[index], self.token_type[index], \
               self.y1_sec[index], self.y2_sec[index], \
               self.arg1_mask[index], self.arg2_mask[index]


    def __len__(self):             
        return self.len


    def load_dataset(self, args, path):
        contents = []

        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                labels1, labels2, arg1, arg2 = [_.strip() for _ in lin.split('|||')]
                labels1, labels2 = eval(labels1), eval(labels2)
                label_sec1 = args.sec2i[labels1[1]] if labels1[1] is not None else -1
                label_sec2 = args.sec2i[labels2[1]] if labels2[1] is not None else -1


                # arg1_token = args.tokenizer.tokenize(arg1)
                # arg2_token = args.tokenizer.tokenize(arg2)
                # token = [CLS] + arg1_token + [SEP] + arg2_token + [SEP]

                # token_type_ids = [0] * (len(arg1_token) + 2) + [1] * (len(arg2_token) + 1)
                # arg1_mask = [1] * (len(arg1_token) + 2)
                # arg2_mask = [0] * (len(arg1_token) + 2) + [1] * (len(arg2_token) + 1)

                input = args.tokenizer.encode_plus(arg1, arg2, add_special_tokens=True, 
                                                   max_length=args.pad_size, truncation=True, 
                                                   padding='max_length', return_tensors="pt")

                input_ids = input['input_ids'][0]
                attention_mask = input['attention_mask'][0]
                token_type_ids = input.get('token_type_ids')
                if token_type_ids is not None:
                    token_type_ids = token_type_ids[0]
                else:
                    token_type_ids = torch.zeros_like(input_ids)

                # Correctly identifying the argument masks
                sep_indices = (input_ids == args.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                arg1_mask = torch.zeros_like(input_ids)
                arg2_mask = torch.zeros_like(input_ids)
                if len(sep_indices) > 1:
                    arg1_mask[:sep_indices[0] + 1] = 1
                    arg2_mask[sep_indices[0] + 1:sep_indices[1] + 1] = 1

                contents.append((input_ids, attention_mask, token_type_ids,
                                 label_sec1, label_sec2,
                                 arg1_mask, arg2_mask))
        return contents

    
    
    def _to_tensor(self, datas):
        x = torch.stack([_[0] for _ in datas]).to(self.device)
        mask = torch.stack([_[1] for _ in datas]).to(self.device)
        token_type = torch.stack([_[2] for _ in datas]).to(self.device)

        y1_sec = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        y2_sec = torch.LongTensor([_[4] for _ in datas]).to(self.device)

        arg1_mask = torch.stack([_[5] for _ in datas]).to(self.device)
        arg2_mask = torch.stack([_[6] for _ in datas]).to(self.device)

        return x, mask, token_type, y1_sec, y2_sec, arg1_mask, arg2_mask
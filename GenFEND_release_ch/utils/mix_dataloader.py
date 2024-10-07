import os
import torch
import numpy as np
import random
import time
import math
import re
import jieba
import tqdm
import json
import pandas as pd
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
from torch.utils.data import TensorDataset, DataLoader, Dataset
from gensim.models.keyedvectors import KeyedVectors, Vocab
from torch.nn.utils.rnn import pad_sequence

Dmeta_model = SentenceTransformer('./pretrained_model/Dmeta-embedding') #your Dmeta-embedding file path

class Weibo21Dataset(Dataset):
    def __init__(self, path_to_file, emb_type):
        self.data = json.load(open(path_to_file, 'r'))
        directory, file_name = os.path.split(path_to_file)

        gen_cmt_dir = './data/role_virtual_comments'
        self.gen_data = json.load(open(os.path.join(gen_cmt_dir, file_name), 'r'))
        
        self.male_index = [i for i in range(0, 15)]
        self.female_index = [i for i in range(15, 30)]
        self.age_17_index = [i for i in range(0, 3)] + [i for i in range(15, 18)]
        self.age_18_29_index = [i for i in range(3, 6)] + [i for i in range(18, 21)]
        self.age_30_49_index = [i for i in range(6, 9)] + [i for i in range(21, 24)]
        self.age_50_64_index = [i for i in range(9, 12)] + [i for i in range (24, 27)]
        self.age_65_index = [i for i in range(12, 15)] + [i for i in range(27, 30)]
        self.edu_high_index = [i for i in range(0, 28, 3)]
        self.edu_mid_index = [i for i in range(1, 29, 3)]
        self.edu_low_index = [i for i in range(2, 30, 3)]

        if emb_type == 'Dmeta':
            self.cmt_emb_type = 'Dmeta'
            self.cmt_emb_dir = '/Dmeta-embedding-comments-feature'
        if 'train' in path_to_file:
            self.comment_feature_file_prefix = directory + self.cmt_emb_dir + '/train/train_comments_feature_'
            self.gen_comment_feature_file_prefix = gen_cmt_dir + self.cmt_emb_dir + '/train/train_comments_feature_'
        elif 'val' in path_to_file:
            self.comment_feature_file_prefix = directory + self.cmt_emb_dir + '/val/val_comments_feature_'
            self.gen_comment_feature_file_prefix = gen_cmt_dir + self.cmt_emb_dir + '/val/val_comments_feature_'
        elif 'test' in path_to_file:
            self.comment_feature_file_prefix = directory + self.cmt_emb_dir + '/test/test_comments_feature_'
            self.gen_comment_feature_file_prefix = gen_cmt_dir + self.cmt_emb_dir + '/test/test_comments_feature_'
        else:
            raise ValueError('Unknown file name')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        flag = True
        item = self.data[index]
        file_index = item['file_index']
        from_index = item['from_index']
        to_index = item['to_index']
        group_comment_feature = np.load(self.comment_feature_file_prefix + str(file_index) + '.npy')
        comment_feature = group_comment_feature[from_index: to_index]
        if(from_index == to_index):
            flag = False
        # if flag:
        #     new_to_index = min(from_index + 8, to_index)
        #     comment_feature = group_comment_feature[from_index: new_to_index]
        
        gen_item = self.gen_data[index]
        gen_file_index = gen_item['file_index']
        gen_from_index = gen_item['from_index']
        gen_to_index = gen_item['to_index']
        gen_group_comment_feature = np.float32(np.load(self.gen_comment_feature_file_prefix + str(gen_file_index) + '.npy'))
        gen_comment_feature = gen_group_comment_feature[gen_from_index: gen_to_index]
        male_cmt_feature = gen_comment_feature[self.male_index]
        female_cmt_feature = gen_comment_feature[self.female_index]
        
        new_item = {'content': item['content'], 'comment_fea': comment_feature, 
                    'flag': flag, 'cmt_emb_type': self.cmt_emb_type,
                    'cmt_fea': gen_comment_feature,
                    'male_cmt_fea': male_cmt_feature, 'female_cmt_fea': female_cmt_feature,
                    'age1_cmt_fea': gen_comment_feature[self.age_17_index], 'age2_cmt_fea': gen_comment_feature[self.age_18_29_index], 'age3_cmt_fea': gen_comment_feature[self.age_30_49_index], 'age4_cmt_fea': gen_comment_feature[self.age_50_64_index], 'age5_cmt_fea': gen_comment_feature[self.age_65_index],
                    'edu1_cmt_fea': gen_comment_feature[self.edu_high_index], 'edu2_cmt_fea': gen_comment_feature[self.edu_mid_index], 'edu3_cmt_fea': gen_comment_feature[self.edu_low_index],
                    'label': item['label']}
        return new_item

def word2input(texts, max_len):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    token_ids = []
    token_ids.append(
            tokenizer.encode(texts, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    # for i, text in enumerate(texts):
    #     token_ids.append(
    #         tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
    #                          truncation=True))
    # token_ids = torch.tensor(token_ids)
    token_ids = torch.tensor(token_ids).squeeze()
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks


def my_collate(batch):
    max_comments_num = max(item['comment_fea'].shape[0] for item in batch)
    new_batch = dict()
    content_token_ids_batch = []
    content_masks_batch = []
    label_batch = []
    fea_batch = []
    mask_batch = torch.zeros((len(batch), max_comments_num))

    cmt_fea_batch = []
    male_cmt_fea_batch = []
    female_cmt_fea_batch = []
    age1_cmt_fea_batch = []
    age2_cmt_fea_batch = []
    age3_cmt_fea_batch = []
    age4_cmt_fea_batch = []
    age5_cmt_fea_batch = []
    edu1_cmt_fea_batch = []
    edu2_cmt_fea_batch = []
    edu3_cmt_fea_batch = []

    for i in range(len(batch)):
        if batch[i]['cmt_emb_type'] == 'Dmeta':
            empty_cmt_fea = Dmeta_model.encode("")
        empty_cmt_fea = np.expand_dims(empty_cmt_fea, axis=0)
        content_token_ids, content_masks = word2input(batch[i]['content'], max_len = 170)
        content_token_ids_batch.append(content_token_ids)
        content_masks_batch.append(content_masks)
        label_batch.append(batch[i]['label'])
        if(batch[i]['flag']):
            fea_batch.append(torch.tensor(batch[i]['comment_fea']))
            comment_num_actual = batch[i]['comment_fea'].shape[0]
        else:
            fea_batch.append(torch.tensor(empty_cmt_fea))
            comment_num_actual = 1
        mask_batch[i, : comment_num_actual] = 1

        cmt_fea_batch.append(batch[i]['cmt_fea'])
        male_cmt_fea_batch.append(batch[i]['male_cmt_fea'])
        female_cmt_fea_batch.append(batch[i]['female_cmt_fea'])
        age1_cmt_fea_batch.append(batch[i]['age1_cmt_fea'])
        age2_cmt_fea_batch.append(batch[i]['age2_cmt_fea'])
        age3_cmt_fea_batch.append(batch[i]['age3_cmt_fea'])
        age4_cmt_fea_batch.append(batch[i]['age4_cmt_fea'])
        age5_cmt_fea_batch.append(batch[i]['age5_cmt_fea'])
        edu1_cmt_fea_batch.append(batch[i]['edu1_cmt_fea'])
        edu2_cmt_fea_batch.append(batch[i]['edu2_cmt_fea'])
        edu3_cmt_fea_batch.append(batch[i]['edu3_cmt_fea'])
    new_batch['content_token_ids'] = pad_sequence(content_token_ids_batch, batch_first = True)
    new_batch['content_masks'] = pad_sequence(content_masks_batch, batch_first=True)
    new_batch['comments_feature'] = pad_sequence(fea_batch, batch_first = True).to(torch.float32)
    new_batch['comments_masks'] = mask_batch
    new_batch['label'] = torch.tensor(label_batch)

    new_batch['cmt_fea'] = torch.tensor(cmt_fea_batch)
    new_batch['male_cmt_fea'] = torch.tensor(male_cmt_fea_batch)
    new_batch['female_cmt_fea'] = torch.tensor(female_cmt_fea_batch)
    new_batch['age1_cmt_fea'] = torch.tensor(age1_cmt_fea_batch)
    new_batch['age2_cmt_fea'] = torch.tensor(age2_cmt_fea_batch)
    new_batch['age3_cmt_fea'] = torch.tensor(age3_cmt_fea_batch)
    new_batch['age4_cmt_fea'] = torch.tensor(age4_cmt_fea_batch)
    new_batch['age5_cmt_fea'] = torch.tensor(age5_cmt_fea_batch)
    new_batch['edu1_cmt_fea'] = torch.tensor(edu1_cmt_fea_batch)
    new_batch['edu2_cmt_fea'] = torch.tensor(edu2_cmt_fea_batch)
    new_batch['edu3_cmt_fea'] = torch.tensor(edu3_cmt_fea_batch)
    return new_batch

def my_collate2(batch):
    new_batch = dict()
    content_token_ids_batch = []
    content_masks_batch = []
    label_batch = []
    fea_batch = []
    mask_batch = torch.zeros((len(batch), 1))

    cmt_fea_batch = []
    male_cmt_fea_batch = []
    female_cmt_fea_batch = []
    age1_cmt_fea_batch = []
    age2_cmt_fea_batch = []
    age3_cmt_fea_batch = []
    age4_cmt_fea_batch = []
    age5_cmt_fea_batch = []
    edu1_cmt_fea_batch = []
    edu2_cmt_fea_batch = []
    edu3_cmt_fea_batch = []
    for i in range(len(batch)):
        if batch[i]['cmt_emb_type'] == 'Dmeta':
            comments_empty_fea = Dmeta_model.encode("")
        comments_empty_fea = np.expand_dims(comments_empty_fea, axis = 0)
        content_token_ids, content_masks = word2input(batch[i]['content'], max_len = 170)
        content_token_ids_batch.append(content_token_ids)
        content_masks_batch.append(content_masks)
        label_batch.append(batch[i]['label'])
        fea_batch.append(torch.tensor(comments_empty_fea))
        comment_num_actual = 1
        mask_batch[i, : comment_num_actual] = 1

        cmt_fea_batch.append(batch[i]['cmt_fea'])
        male_cmt_fea_batch.append(batch[i]['male_cmt_fea'])
        female_cmt_fea_batch.append(batch[i]['female_cmt_fea'])
        age1_cmt_fea_batch.append(batch[i]['age1_cmt_fea'])
        age2_cmt_fea_batch.append(batch[i]['age2_cmt_fea'])
        age3_cmt_fea_batch.append(batch[i]['age3_cmt_fea'])
        age4_cmt_fea_batch.append(batch[i]['age4_cmt_fea'])
        age5_cmt_fea_batch.append(batch[i]['age5_cmt_fea'])
        edu1_cmt_fea_batch.append(batch[i]['edu1_cmt_fea'])
        edu2_cmt_fea_batch.append(batch[i]['edu2_cmt_fea'])
        edu3_cmt_fea_batch.append(batch[i]['edu3_cmt_fea'])

    new_batch['content_token_ids'] = pad_sequence(content_token_ids_batch, batch_first = True)
    new_batch['content_masks'] = pad_sequence(content_masks_batch, batch_first=True)
    new_batch['comments_feature'] = pad_sequence(fea_batch, batch_first = True).to(torch.float32)
    new_batch['comments_masks'] = mask_batch
    new_batch['label'] = torch.tensor(label_batch)

    new_batch['cmt_fea'] = torch.tensor(cmt_fea_batch)
    new_batch['male_cmt_fea'] = torch.tensor(male_cmt_fea_batch)
    new_batch['female_cmt_fea'] = torch.tensor(female_cmt_fea_batch)
    new_batch['age1_cmt_fea'] = torch.tensor(age1_cmt_fea_batch)
    new_batch['age2_cmt_fea'] = torch.tensor(age2_cmt_fea_batch)
    new_batch['age3_cmt_fea'] = torch.tensor(age3_cmt_fea_batch)
    new_batch['age4_cmt_fea'] = torch.tensor(age4_cmt_fea_batch)
    new_batch['age5_cmt_fea'] = torch.tensor(age5_cmt_fea_batch)
    new_batch['edu1_cmt_fea'] = torch.tensor(edu1_cmt_fea_batch)
    new_batch['edu2_cmt_fea'] = torch.tensor(edu2_cmt_fea_batch)
    new_batch['edu3_cmt_fea'] = torch.tensor(edu3_cmt_fea_batch)
    return new_batch

def get_dataloader(path, batch_size, no_comment, emb_type, shuffle):
    dataset = Weibo21Dataset(path, emb_type)
    if no_comment:
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = my_collate2)
    else:
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = my_collate)
    return dataloader
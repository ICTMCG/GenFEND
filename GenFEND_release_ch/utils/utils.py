from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
import torch
from torch.nn.functional import softmax, kl_div


class Recorder():

    def __init__(self, early_stop):
        self.max = {'metric': 0}
        self.cur = {'metric': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_stop = early_stop
    
    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("currenr", self.cur)
        return self.judge()
    
    def judge(self):
        if self.cur['metric'] > self.max['metric']:
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_stop:
            return 'esc'
        else:
            return 'continue'
    
    def showfinal(self):
        print("Max", self.max)

def metrics(y_true, y_pred):
    all_metrics = {}

    all_metrics['auc'] = roc_auc_score(y_true, y_pred, average = 'macro')
    all_metrics['spauc'] = roc_auc_score(y_true, y_pred, average = 'macro', max_fpr = 0.1)
    y_pred = np.around(np.array(y_pred)).astype(int)
    all_metrics['metric'] = f1_score(y_true, y_pred, average = 'macro')
    all_metrics['f1_real'], all_metrics['f1_fake'] = f1_score(y_true, y_pred, average = None)
    all_metrics['recall'] = recall_score(y_true, y_pred, average = 'macro')
    all_metrics['recall_real'], all_metrics['recall_fake'] = recall_score(y_true, y_pred, average = None)
    all_metrics['precision'] = precision_score(y_true, y_pred, average = 'macro')
    all_metrics['precision_real'], all_metrics['precision_fake'] = precision_score(y_true, y_pred, average = None)
    all_metrics['acc'] = accuracy_score(y_true, y_pred)

    return all_metrics



def data2gpu(batch, use_cuda):
    if use_cuda:
        batch_data = {
            'content': batch['content_token_ids'].cuda(),
            'content_masks': batch['content_masks'].cuda(),
            'comments': batch['comments_feature'].cuda(),
            'comments_masks': batch['comments_masks'].cuda(),
            'label': batch['label'].cuda()        
            }
    else:
        batch_data = {
            'content': batch['content_token_ids'],
            'content_masks': batch['content_masks'],
            'comments': batch['comments_feature'],
            'comments_masks': batch['comments_masks'],
            'label': batch['label']
            }
    return batch_data


def data2gpu_mix(batch, use_cuda):
    if use_cuda:
        batch_data = {
            'content': batch['content_token_ids'].cuda(),
            'content_masks': batch['content_masks'].cuda(),
            'comments': batch['comments_feature'].cuda(),
            'comments_masks': batch['comments_masks'].cuda(),
            'label': batch['label'].cuda(), 

            'cmt_fea': batch['cmt_fea'].cuda(),
            'male_cmt_fea': batch['male_cmt_fea'].cuda(),
            'female_cmt_fea': batch['female_cmt_fea'].cuda(),
            'age1_cmt_fea': batch['age1_cmt_fea'].cuda(),
            'age2_cmt_fea': batch['age2_cmt_fea'].cuda(),
            'age3_cmt_fea': batch['age3_cmt_fea'].cuda(),
            'age4_cmt_fea': batch['age4_cmt_fea'].cuda(),
            'age5_cmt_fea': batch['age5_cmt_fea'].cuda(),
            'edu1_cmt_fea': batch['edu1_cmt_fea'].cuda(),
            'edu2_cmt_fea': batch['edu2_cmt_fea'].cuda(),
            'edu3_cmt_fea': batch['edu3_cmt_fea'].cuda(),
        }
    else:
        batch_data = {
            'content': batch['content_token_ids'],
            'content_masks': batch['content_masks'],
            'comments': batch['comments_feature'],
            'comments_masks': batch['comments_masks'],
            'label': batch['label'],

            'cmt_fea': batch['cmt_fea'],
            'male_cmt_fea': batch['male_cmt_fea'],
            'female_cmt_fea': batch['female_cmt_fea'],
            'age1_cmt_fea': batch['age1_cmt_fea'],
            'age2_cmt_fea': batch['age2_cmt_fea'],
            'age3_cmt_fea': batch['age3_cmt_fea'],
            'age4_cmt_fea': batch['age4_cmt_fea'],
            'age5_cmt_fea': batch['age5_cmt_fea'],
            'edu1_cmt_fea': batch['edu1_cmt_fea'],
            'edu2_cmt_fea': batch['edu2_cmt_fea'],
            'edu3_cmt_fea': batch['edu3_cmt_fea'],
        }
    return batch_data

class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0
    
    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1
    
    def item(self):
        return self.v

def kl_dist(fea1, fea2):
    batch_loss = torch.zeros((0)).cuda()
    norm_fea1 = softmax(fea1, dim = -1)
    norm_fea2 = softmax(fea2, dim = -1)
    for i in range(norm_fea1.size(0)):
        loss1 = kl_div(torch.log(norm_fea1[i]), norm_fea2[i], reduction = 'batchmean')
        batch_loss = torch.cat((batch_loss, loss1.unsqueeze(0)), dim = 0)
    return batch_loss
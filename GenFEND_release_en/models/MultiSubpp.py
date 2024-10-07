import os
import torch
import tqdm
import numpy as np
from sklearn.metrics import *
import json
from .layers import *
from transformers import BertModel
from utils.utils import kl_dist

class MultiSubppModel(torch.nn.Module):
    def __init__(self, cnt_emb_dim, cmt_emb_dim, dropout):
        super(MultiSubppModel, self).__init__()
        self.cnt_emb_dim = cnt_emb_dim
        self.cmt_emb_dim = cmt_emb_dim
        self.fea_dim = 768
        self.male_mlp = MLP(self.cmt_emb_dim, [self.fea_dim], dropout, output_layer = False)
        self.female_mlp = MLP(self.cmt_emb_dim, [self.fea_dim], dropout, output_layer = False)

        self.age1_mlp = MLP(self.cmt_emb_dim, [self.fea_dim], dropout, output_layer = False)
        self.age2_mlp = MLP(self.cmt_emb_dim, [self.fea_dim], dropout, output_layer = False)
        self.age3_mlp = MLP(self.cmt_emb_dim, [self.fea_dim], dropout, output_layer = False)
        self.age4_mlp = MLP(self.cmt_emb_dim, [self.fea_dim], dropout, output_layer = False)
        self.age5_mlp = MLP(self.cmt_emb_dim, [self.fea_dim], dropout, output_layer = False)

        self.edu1_mlp = MLP(self.cmt_emb_dim, [self.fea_dim], dropout, output_layer = False)
        self.edu2_mlp = MLP(self.cmt_emb_dim, [self.fea_dim], dropout, output_layer = False)
        self.edu3_mlp = MLP(self.cmt_emb_dim, [self.fea_dim], dropout, output_layer = False)

        self.view_att = Attention()
        self.gate = nn.Sequential(
            nn.Linear(self.cnt_emb_dim + 28, 384),
            nn.ReLU(),
            nn.Linear(384, 3),
            nn.Softmax(dim = 1)
        )
    
    def gender_expert(self, gender_feature_list):
        mean_gender_feature = [torch.mean(gender_feature, dim = 1) for gender_feature in gender_feature_list]
        gender_kl = []
        for i in range(len(gender_feature_list)):
            for j in range(i+1, len(gender_feature_list)):
                gender_kl.append(kl_dist(gender_feature_list[i], gender_feature_list[j]))
                gender_kl.append(kl_dist(gender_feature_list[j], gender_feature_list[i]))
        gender_kl = torch.stack(gender_kl, dim = 1)
        return mean_gender_feature, gender_kl
    
    def age_expert(self, age_feature_list):
        mean_age_feature = [torch.mean(age_feature, dim = 1) for age_feature in age_feature_list]
        age_kl = []
        for i in range(len(age_feature_list)):
            for j in range(i+1, len(age_feature_list)):
                age_kl.append(kl_dist(age_feature_list[i], age_feature_list[j]))
                age_kl.append(kl_dist(age_feature_list[j], age_feature_list[i]))
        age_kl = torch.stack(age_kl, dim = 1)
        return mean_age_feature, age_kl
    
    def edu_expert(self, edu_feature_list):
        mean_edu_feature = [torch.mean(edu_feature, dim = 1) for edu_feature in edu_feature_list]
        edu_kl = []
        for i in range(len(edu_feature_list)):
            for j in range(i+1, len(edu_feature_list)):
                edu_kl.append(kl_dist(edu_feature_list[i], edu_feature_list[j]))
                edu_kl.append(kl_dist(edu_feature_list[j], edu_feature_list[i]))
        edu_kl = torch.stack(edu_kl, dim = 1)
        return mean_edu_feature, edu_kl
    
    def forward(self, **kwargs):

        cnt_fea = kwargs['cnt_fea']

        male_cmt_emb = kwargs['male_cmt_fea']
        female_cmt_emb = kwargs['female_cmt_fea']
        age1_cmt_emb = kwargs['age1_cmt_fea']
        age2_cmt_emb = kwargs['age2_cmt_fea']
        age3_cmt_emb = kwargs['age3_cmt_fea']
        age4_cmt_emb = kwargs['age4_cmt_fea']
        age5_cmt_emb = kwargs['age5_cmt_fea']
        edu1_cmt_emb = kwargs['edu1_cmt_fea']
        edu2_cmt_emb = kwargs['edu2_cmt_fea']
        edu3_cmt_emb = kwargs['edu3_cmt_fea']

        male_cmt_fea = self.male_mlp(male_cmt_emb)
        female_cmt_fea = self.female_mlp(female_cmt_emb)
        age1_cmt_fea = self.age1_mlp(age1_cmt_emb)
        age2_cmt_fea = self.age2_mlp(age2_cmt_emb)
        age3_cmt_fea = self.age3_mlp(age3_cmt_emb)
        age4_cmt_fea = self.age4_mlp(age4_cmt_emb)
        age5_cmt_fea = self.age5_mlp(age5_cmt_emb)
        edu1_cmt_fea = self.edu1_mlp(edu1_cmt_emb)
        edu2_cmt_fea = self.edu2_mlp(edu2_cmt_emb)
        edu3_cmt_fea = self.edu3_mlp(edu3_cmt_emb)

        gender_feature_list = [male_cmt_fea, female_cmt_fea]
        age_feature_list = [age1_cmt_fea, age2_cmt_fea, age3_cmt_fea, age4_cmt_fea, age5_cmt_fea]
        edu_feature_list = [edu1_cmt_fea, edu2_cmt_fea, edu3_cmt_fea]

        mean_gender_fea_list, gender_kl = self.gender_expert(gender_feature_list)
        gender_cmt_fea = torch.cat([feature.unsqueeze(1) for feature in mean_gender_fea_list], dim = 1)
        gender_cmt_fea, gender_weight = self.view_att(cnt_fea.unsqueeze(1), gender_cmt_fea, gender_cmt_fea)

        mean_age_fea_list, age_kl = self.age_expert(age_feature_list)
        age_cmt_fea = torch.cat([feature.unsqueeze(1) for feature in mean_age_fea_list], dim = 1)
        age_cmt_fea, age_weight = self.view_att(cnt_fea.unsqueeze(1), age_cmt_fea, age_cmt_fea)

        mean_edu_fea_list, edu_kl = self.edu_expert(edu_feature_list)
        edu_cmt_fea = torch.cat([feature.unsqueeze(1) for feature in mean_edu_fea_list], dim = 1)
        edu_cmt_fea, edu_weight = self.view_att(cnt_fea.unsqueeze(1), edu_cmt_fea, edu_cmt_fea)

        kl_fea = torch.cat([gender_kl, age_kl, edu_kl], dim = 1)
        all_cmt_fea = torch.cat([gender_cmt_fea, age_cmt_fea, edu_cmt_fea], dim = 1)
        gate_value = self.gate(torch.cat([cnt_fea, kl_fea], dim = 1))
        cmt_fea = torch.bmm(gate_value.unsqueeze(1), all_cmt_fea).squeeze(1)

        return cmt_fea, gate_value

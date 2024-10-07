import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default = 'bert_genfend') 
parser.add_argument('--root_path', default='./data/Weibo21/') 
parser.add_argument('--epoch', type = int, default = 50)
parser.add_argument('--max_len', type = int, default = 170)
parser.add_argument('--early_stop', type = int, default = 5)
parser.add_argument('--batchsize', type = int, default = 64)
parser.add_argument('--seed', type = int, default = 2022)
parser.add_argument('--gpu', default = '1')
parser.add_argument('--cnt_emb_dim', type = int, default = 768)
parser.add_argument('--cmt_emb_dim', type = int, default = 768)
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--save_param_dir', default = './param_model/')
parser.add_argument('--results_dir', default = './results_analysis')
parser.add_argument('--no_comment', type = bool, default = False)
parser.add_argument('--threshold', type = float, default = 0.6)
parser.add_argument('--cmt_emb_type', type = str, default = 'bge') #bge

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from gridsearch import Run
import torch
import numpy as np
import random

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("model name: {}; batchsize: {}; gpu: {}".format(args.model_name, args.batchsize, args.gpu))

config = {
        'use_cuda': True,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'root_path': args.root_path,
        'weight_decay': 5e-5,
        'model':
            {
            'mlp': {'content_dims': [384], 'dims': [768, 384], 'dropout': 0.2}
            },
        'cnt_emb_dim': args.cnt_emb_dim,
        'cmt_emb_dim': args.cmt_emb_dim,
        'lr': args.lr,
        'epoch': args.epoch,
        'model_name': args.model_name,
        'seed': args.seed,
        'save_param_dir': args.save_param_dir,
        'results_dir': args.results_dir,
        'no_comment': args.no_comment,
        'threshold': args.threshold,
        'cmt_emb_type': args.cmt_emb_type,
        }

if __name__ == '__main__':
    Run(config = config
        ).main()

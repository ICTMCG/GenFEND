'''
You can adjust the '--from_index' and '--to_index' to control the range of data you want to encode.
'''
import json
import numpy as np
import tqdm
import argparse
from sentence_transformers import SentenceTransformer
import os

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type = str, default = './LLM-mis/') #'./GossipCop/' #'./role_virtual_comments/'
parser.add_argument('--file_name', type = str, default = 'train.json')
parser.add_argument('--from_index', type = int, default = 0)
parser.add_argument('--to_index', type = int, default = 10000)
parser.add_argument('--file_to_path_prefix', type = str, default = './data/LLM-mis/bge-large-en-v1.5/')
parser.add_argument('--gpu', type = str, default = '1')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
model = SentenceTransformer('../pretrained_model/bge-large-en-v1.5')

def create_embedding_file_path():
    file_to_path_prefix = args.file_to_path_prefix
    for fn in ['train/', 'val/', 'test/']:
        if not os.path.exists(file_to_path_prefix + fn):
            os.makedirs(file_to_path_prefix + fn)
    

def encode_comments(fn, fi, ti):
    file_path = args.file_path
    create_embedding_file_path()
    with open(file_path + fn, 'r', encoding = 'UTF-8') as f:
        data_items = json.load(f)
    f.close()
    all_comments_feature = np.empty([0, 1024])
    for index, data in enumerate(tqdm.tqdm(data_items)):
        if index < fi:
            continue
        if index >= ti:
            break
        if 'comments' in data.keys():
            # comments = [comment for comment in data['comments']]
            comments = [item['comment'] for item in data['comments']]
        elif "all_comments" in data.keys():
            comments = [item['content'] for item in data['all_comments']]
        else:
            raise NameError("Invalid Keys!")
        for comment in comments:
            comment_feature = model.encode(comment)
            comment_feature = np.expand_dims(comment_feature, axis = 0)
            all_comments_feature = np.append(all_comments_feature, comment_feature, axis = 0)
        if (index + 1) % 100 == 0 or (index + 1) == len(data_items):
            if 'train' in fn:
                np.save(args.file_to_path_prefix + 'train/train_comments_feature_' + str(index + 1) + '.npy', all_comments_feature)
            if ('val' in fn):
                np.save(args.file_to_path_prefix + 'val/val_comments_feature_' + str(index + 1) + '.npy', all_comments_feature)
            if ('test' in fn):
                np.save(args.file_to_path_prefix + 'test/test_comments_feature_' + str(index + 1) + '.npy', all_comments_feature)
            all_comments_feature = np.empty([0, 1024])

if __name__ == '__main__':
    file_name = args.file_name
    from_index = args.from_index
    to_index = args.to_index
    encode_comments(file_name, from_index, to_index)
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
parser.add_argument('--file_path', type = str, default = './Weibo21/') #'./role_virtual_comments/'
parser.add_argument('--file_name', type = str, default = 'train.json') # 'val_index.json' 'test_index.json'
parser.add_argument('--from_index', type = int, default = 0)
parser.add_argument('--to_index', type = int, default = 10000)
parser.add_argument('--gpu', type = str, default = '0')
args = parser.parse_args()
file_name = args.file_name
from_index = args.from_index
to_index = args.to_index

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
model = SentenceTransformer('../pretrained_model/Dmeta-embedding') #your Demta-embedding file path
def encode_comments(fn, fi, ti):
    file_path = args.file_path
    with open(file_path + fn, 'r', encoding = 'UTF-8') as f:
        data_items = json.load(f)
        all_comments_feature = np.empty([0, 768])
        for index, data in enumerate(tqdm.tqdm(data_items)):
            if index < fi:
                continue
            if index >= ti:
                break
            if 'comments' in data.keys():
                if len(data["comments"]) == 0:
                    comments = []
                    continue
                if isinstance(data['comments'][0], dict):
                    comments = [comment["comment"] for comment in data['comments']]
                else:
                    comments = [comment for comment in data['comments']]
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
                    np.save(file_path + 'Dmeta-embedding-comments-feature/train/train_comments_feature_' + str(index + 1) + '_test.npy', all_comments_feature)
                if ('val' in fn):
                    np.save(file_path + 'Dmeta-embedding-comments-feature/val/val_comments_feature_' + str(index + 1) + '_test.npy', all_comments_feature)
                if ('test' in fn):
                    np.save(file_path + 'Dmeta-embedding-comments-feature/test/test_comments_feature_' + str(index + 1) + '_test.npy', all_comments_feature)
                all_comments_feature = np.empty([0, 768])
    f.close()

if __name__ == "__main__":
    encode_comments(file_name, from_index, to_index)
import logging
import os
import json

from models.dEFENDMultiSppModel import Trainer as dEFENDGenFENDTrainer
from models.BERTMtiSppModel import Trainer as BERTGenFENDTrainer

def frange(x, y, jump):
  while x < y:
      x = round(x, 8)
      yield x
      x += jump

class Run():
    def __init__(self, config):
        self.config = config
    
    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict
    
    def main(self):
        train_param = {
            'lr': [self.config['lr']] * 1,
            # 'lr': [self.config['lr']] * 10,
            # 'lr': [0.00003, 0.00005, 0.00007, 0.0001, 0.0002, 0.0005, 0.0009, 0.001]
        }
        
        param = train_param
        best_param = []
        json_path = './logs/json/' + self.config['model_name'] + '.json'
        json_result = []
        for p, vs in param.items():
            # p: weight; lr; vs: value
            best_metric = {}
            best_metric['metric'] = 0
            best_v = vs[0]
            best_path = ""
            for i, v in enumerate(vs):
                self.config['lr'] = v
                print("lr: {}".format(self.config['lr']))
                
                if self.config['model_name'] == 'bert_genfend':
                    trainer = BERTGenFENDTrainer(self.config)
                if self.config['model_name'] == 'defend_genfend':
                    trainer = dEFENDGenFENDTrainer(self.config)
                metrics, best_model_path = trainer.train()
                json_result.append(metrics)
                if(metrics['metric'] > best_metric['metric']):
                    best_metric['metric'] = metrics['metric']
                    best_v = v
                    best_path = best_model_path
            best_param.append({p: best_v})
            print("best metric: ", best_metric)
            print("best model path:", best_path)
        with open(json_path, 'w') as file:
            json.dump(json_result, file, indent = 4, ensure_ascii = False)
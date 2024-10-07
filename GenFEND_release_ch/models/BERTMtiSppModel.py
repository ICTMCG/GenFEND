import os
import torch
import tqdm
import numpy as np
from sklearn.metrics import *
import json
from .layers import *
from .coattention import *
from .MultiSubpp import *
from transformers import BertModel
from utils.utils import data2gpu_mix, Averager, metrics, Recorder
from utils.mix_dataloader import get_dataloader

class BERTGenFENDModel(torch.nn.Module):
    def __init__(self, cnt_emb_dim, cmt_emb_dim, dropout):
        super(BERTGenFENDModel, self).__init__()
        self.cnt_emb_dim = cnt_emb_dim
        self.cmt_emb_dim = cmt_emb_dim
        self.dropout = dropout
        self.fea_dim = 768
        self.mlp_dims = [self.fea_dim, 384]
        self.mtispp_module = MultiSubppModel(self.cnt_emb_dim, self.cmt_emb_dim,dropout)

        self.cnt_bert = BertModel.from_pretrained('bert-base-chinese').requires_grad_(False)
        for name, param in self.cnt_bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.cnt_att = MaskAttention(self.cnt_emb_dim)

        self.mlp = MLP(self.fea_dim * 2, self.mlp_dims, dropout, output_layer = True)
    
    def forward(self, **kwargs):
        content = kwargs['content']
        content_masks = kwargs['content_masks']

        cnt_fea = self.cnt_bert(content, attention_mask = content_masks)[0]
        cnt_fea, _ = self.cnt_att(cnt_fea, content_masks)

        kwargs['cnt_fea'] = cnt_fea
        gen_cmt_fea, gate_value = self.mtispp_module(**kwargs)

        final_fea = torch.cat((cnt_fea, gen_cmt_fea), dim = -1)
        output = self.mlp(final_fea)
        return torch.sigmoid(output.squeeze(1)), gate_value

class Trainer():
    def __init__(self, config):
        self.config = config
        self.save_path = os.path.join(self.config['save_param_dir'], self.config['model_name'])
        if os.path.exists(self.save_path):
            self.save_param_dir = self.save_path
        else:
            self.save_param_dir = os.makedirs(self.save_path)
        
        self.results_analysis_path = os.path.join(self.config['results_dir'], self.config['model_name'])
        if os.path.exists(self.results_analysis_path):
            self.results_dir = self.results_analysis_path
        else:
            self.results_dir = os.makedirs(self.results_analysis_path)
    
    def train(self):
        self.model = BERTGenFENDModel(self.config['cnt_emb_dim'], self.config['cmt_emb_dim'], self.config['model']['mlp']['dropout'])
        if self.config['use_cuda']:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        recorder = Recorder(self.config['early_stop'])
        val_loader = get_dataloader(self.config['root_path'] + 'val_index.json', self.config['batchsize'], no_comment = self.config['no_comment'], emb_type = self.config['cmt_emb_type'], shuffle = False)
        best_metric = 0

        for epoch in range(self.config['epoch']):
            self.model.train()
            train_loader = get_dataloader(self.config['root_path'] + 'train_index.json', self.config['batchsize'], no_comment = self.config['no_comment'], emb_type = self.config['cmt_emb_type'], shuffle = True)
            train_data_iter = tqdm.tqdm(train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu_mix(batch, self.config['use_cuda'])
                label = batch_data['label']
                pred, w = self.model(**batch_data)
                loss = loss_fn(pred, label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
            print('Training Epoch: {}, Loss: {}'.format(epoch, avg_loss.item()))
        
            results = self.test(val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                best_metric = results['metric']
                torch.save(self.model.state_dict(),
                        os.path.join(self.save_path, 'parameter_bertgenfend_' + str(best_metric) + '.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'parameter_bertgenfend_' + str(best_metric) + '.pkl')))
        test_loader = get_dataloader(self.config['root_path'] + 'test_index.json', self.config['batchsize'], no_comment = self.config['no_comment'], emb_type = self.config['cmt_emb_type'], shuffle = False)
        test_results = self.test(test_loader)
        print("test results: ", test_results)
        return test_results, os.path.join(self.save_path, 'parameter_bertgenfend_' + str(best_metric) + '.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        # weights1 = []
        # weights2 = []
        # weights3 = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)

        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu_mix(batch, self.config['use_cuda'])
                batch_label = batch_data['label']
                batch_pred, w = self.model(**batch_data)
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())
                # weights1.extend(w[:,0].detach().cpu().numpy().tolist())
                # weights2.extend(w[:,1].detach().cpu().numpy().tolist())
                # weights3.extend(w[:,2].detach().cpu().numpy().tolist())
        
        
        # gate_values = []
        # for i in range(len(weights1)):
        #     gate_values.append({"value1": weights1[i], "value2": weights2[i], "value3": weights3[i]})
        # with open(os.path.join(self.results_analysis_path, 'gate_values_0.8283578966601537.json'), 'w', encoding = 'UTF-8') as f:
        #     json.dump(gate_values, f, ensure_ascii = False, indent = 4)
        # f.close()
        return metrics(label, pred)
        

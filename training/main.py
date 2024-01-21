
import torch
import random
import argparse
import torch.nn as nn
import torch.nn.functional as F
import sys
import copy
import torch.optim as optim
import numpy as np
import os
from time import time
from tqdm import tqdm
from training.evaluator import evaluate
from training.load_data import load_EOD_data,load_graph_relation_data
from training.Trans_Graph_ODE import Stock_Hyper_ODE
import pickle
from training.utils import Data
device = 'cuda'

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

def trr_loss_mse_rank(pred, base_price, ground_truth, mask, alpha, no_stocks):
    return_ratio = torch.div((pred- base_price), base_price)
    reg_loss = weighted_mse_loss(return_ratio, ground_truth, mask)
    all_ones = torch.ones(no_stocks,1).to(device)
    pre_pw_dif =  (torch.matmul(return_ratio, torch.transpose(all_ones, 0, 1))
                    - torch.matmul(all_ones, torch.transpose(return_ratio, 0, 1)))
    gt_pw_dif = (
            torch.matmul(all_ones, torch.transpose(ground_truth,0,1)) -
            torch.matmul(ground_truth, torch.transpose(all_ones, 0,1))
        )

    mask_pw = torch.matmul(mask, torch.transpose(mask, 0,1))
    rank_loss = torch.mean(
            F.relu(
                ((pre_pw_dif*gt_pw_dif)*mask_pw)))
    loss = reg_loss + alpha*rank_loss
    del mask_pw, gt_pw_dif, pre_pw_dif, all_ones
    return loss, reg_loss, rank_loss, return_ratio


class Stock_HyperODE:
    def __init__(self, data_path, market_name, tickers_fname, n_node,
                 parameters, steps=1, epochs=100, batch_size=None, flat=False, gpu=True, in_pro=False):

        seed = 123456789
        random.seed(seed)
        np.random.seed(seed)

        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        # load data
        self.tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                                     dtype=str, delimiter='\t', skip_header=False)
        self.train_data = pickle.load(open('../data/relation/NYSE_File.txt', 'rb'))
        self.n_node = n_node
        self.train_data = Data(self.train_data, shuffle=True, n_node=n_node)
        print('#tickers selected:', len(self.tickers))
        self.eod_data, self.mask_data, self.gt_data, self.price_data,self.moving_matrix= \
            load_EOD_data(data_path, market_name, self.tickers, steps)

        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        self.flat = flat
        self.inner_prod = in_pro
        if batch_size is None:
            self.batch_size = len(self.tickers)  ##always,
        else:
            self.batch_size = batch_size
        self.Stock_num= len(self.tickers)
        self.in_dim=64
        self.emb_size=64
        self.days=20
        self.valid_index = 756
        self.test_index = 1008
        self.trade_dates = self.mask_data.shape[1]
        self.fea_dim = 5

        self.gpu = gpu

    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.parameters['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        return self.eod_data[:, offset:offset + seq_len, :], \
               self.moving_matrix[offset:offset+seq_len,:,:],\
               np.expand_dims(mask_batch, axis=1), \
               np.expand_dims(
                   self.price_data[:, offset + seq_len - 1], axis=1
               ), \
               np.expand_dims(
                   self.gt_data[:, offset + seq_len + self.steps - 1], axis=1
               )

    def train(self):
        global df
        if self.gpu == True:
            device_name = '/gpu:0'
        else:
            device_name = '/cpu:0'
        print('device name:', device_name)
        #模型函数
        model = Stock_Hyper_ODE(self.train_data.adjacency,self.Stock_num,self.n_node).to(device)

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
        optimizer_hgat = optim.Adam(model.parameters(),
                                    lr=self.parameters['lr'],
                                    weight_decay=5e-4)

        inp = self.train_data.raw
        items, num_node = [], []
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
        matrix = np.zeros((len(items), len(items)))
        for i in range(len(items)):
            seq_a = set(items[i])
            seq_a.discard(0)
            for j in range(i + 1, len(items)):
                seq_b = set(items[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap)) / float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0] * len(items))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0 / degree)
        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)
        #best_test_gt = np.zeros(
        #    [len(self.tickers), self.test_index - self.valid_index],
        #    dtype=float
        #)
        best_test_perf={'mse':np.inf,'mrrt':0.0,'btl':0.0}
        best_test_loss=np.inf
        for i in range(self.epochs):
            t1 = time()
            print('epoch:',i,'/',self.epochs)
            np.random.shuffle(batch_offsets)
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0
            model.train()
            if(i>50):
                self.parameters['a']=0.2
                self.parameters['b'] = 0.8
                self.parameters['c'] = 0
            for j in tqdm(range(self.valid_index - self.parameters['seq'] - self.steps + 1)):
                emb_batch,moving_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    batch_offsets[j])
                optimizer_hgat.zero_grad()
                output,gaussian,kl_all,res_loss= model.forward(torch.FloatTensor(emb_batch).to(device),torch.FloatTensor(moving_batch).to(device),torch.FloatTensor(matrix).to(device),torch.FloatTensor(degree).to(device))
                cur_loss, cur_reg_loss, cur_rank_loss, curr_rr_train = trr_loss_mse_rank(output.reshape((1737, 1)),
                                                                                         torch.FloatTensor(
                                                                                             price_batch).to(device),
                                                                                         torch.FloatTensor(gt_batch).to(
                                                                                             device),
                                                                                         torch.FloatTensor(
                                                                                             mask_batch).to(device),
                                                                                         self.parameters['alpha'],
                                                                                         self.batch_size)
                train_loss=cur_loss*self.parameters['b']+self.parameters['a']*kl_all+self.parameters['c']*res_loss

                train_loss.backward()
                optimizer_hgat.step()

                tra_loss += cur_loss.detach().cpu().item()
                tra_reg_loss += cur_reg_loss.detach().cpu().item()
                tra_rank_loss += cur_rank_loss.detach().cpu().item()
            print('Train Loss:',
                  tra_loss / (self.test_index - self.parameters['seq'] - self.steps + 1),
                  tra_reg_loss / (self.test_index - self.parameters['seq'] - self.steps + 1),
                  tra_rank_loss / (self.test_index - self.parameters['seq'] - self.steps + 1))

            with torch.no_grad():
                # test on validation set
                cur_valid_pred = np.zeros(
                    [len(self.tickers), self.test_index - self.valid_index],
                    dtype=float
                )
                cur_valid_gt = np.zeros(
                    [len(self.tickers), self.test_index - self.valid_index],
                    dtype=float
                )
                cur_valid_mask = np.zeros(
                    [len(self.tickers), self.test_index - self.valid_index],
                    dtype=float
                )
                val_loss = 0.0
                val_reg_loss = 0.0
                val_rank_loss = 0.0
                model.eval()
                for cur_offset in range(
                        self.valid_index - self.parameters['seq'] - self.steps + 1,
                        self.test_index - self.parameters['seq'] - self.steps + 1
                ):
                    emb_batch,moving_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                        cur_offset)

                    #output_val = model(torch.FloatTensor(emb_batch).to(device), hyp_input)
                    output_val,gaussian_valid,kl_all_valid,res_loss_valid= model(torch.FloatTensor(emb_batch).to(device),torch.FloatTensor(moving_batch).to(device),torch.FloatTensor(matrix).to(device),torch.FloatTensor(degree).to(device))
                    cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = trr_loss_mse_rank(output_val,
                                                                                      torch.FloatTensor(price_batch).to(
                                                                                          device),
                                                                                      torch.FloatTensor(gt_batch).to(
                                                                                          device),
                                                                                      torch.FloatTensor(mask_batch).to(
                                                                                          device),
                                                                                      self.parameters['alpha'],
                                                                                      self.batch_size)

                    cur_rr = cur_rr.detach().cpu().numpy().reshape((1737, 1))
                    val_loss += cur_loss.detach().cpu().item()
                    val_reg_loss += cur_reg_loss.detach().cpu().item()
                    val_rank_loss += cur_rank_loss.detach().cpu().item()
                    cur_valid_pred[:, cur_offset - (self.valid_index -
                                                    self.parameters['seq'] -
                                                    self.steps + 1)] = \
                        copy.copy(cur_rr[:, 0])
                    cur_valid_gt[:, cur_offset - (self.valid_index -
                                                  self.parameters['seq'] -
                                                  self.steps + 1)] = \
                        copy.copy(gt_batch[:, 0])
                    cur_valid_mask[:, cur_offset - (self.valid_index -
                                                    self.parameters['seq'] -
                                                    self.steps + 1)] = \
                        copy.copy(mask_batch[:, 0])
                print('Valid MSE:',
                      val_loss / (self.test_index - self.valid_index),
                      val_reg_loss / (self.test_index - self.valid_index),
                      val_rank_loss / (self.test_index - self.valid_index))
                cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt,
                                          cur_valid_mask)
                print('\t Valid preformance:', 'sharpe5:', cur_valid_perf['sharpe5'], 'ndcg_score_top5:',
                      cur_valid_perf['ndcg_score_top5'])
                # test on testing set
                cur_test_pred = np.zeros(
                    [len(self.tickers), self.trade_dates - self.test_index],
                    dtype=float
                )
                cur_test_gt = np.zeros(
                    [len(self.tickers), self.trade_dates - self.test_index],
                    dtype=float
                )
                cur_test_mask = np.zeros(
                    [len(self.tickers), self.trade_dates - self.test_index],
                    dtype=float
                )

                test_loss = 0.0
                test_reg_loss = 0.0
                test_rank_loss = 0.0
                model.eval()
                for cur_offset in range(self.test_index - self.parameters['seq'] - self.steps + 1,
                                        self.trade_dates - self.parameters['seq'] - self.steps + 1):
                    emb_batch,moving_batch, mask_batch, price_batch, gt_batch = self.get_batch(cur_offset)


                    #output_test = model(torch.FloatTensor(emb_batch).to(device), hyp_input)
                    output_test,gaussian_test,kl_all_test,res_loss_test= model(torch.FloatTensor(emb_batch).to(device),torch.FloatTensor(moving_batch).to(device),torch.FloatTensor(matrix).to(device),torch.FloatTensor(degree).to(device))
                    cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = trr_loss_mse_rank(output_test,
                                                                                      torch.FloatTensor(price_batch).to(
                                                                                          device),
                                                                                      torch.FloatTensor(gt_batch).to(
                                                                                          device),
                                                                                      torch.FloatTensor(mask_batch).to(
                                                                                          device),
                                                                                      self.parameters['alpha'],
                                                                                      self.batch_size)

                    cur_rr = cur_rr.detach().cpu().numpy().reshape((1737, 1))
                    test_loss += cur_loss.detach().cpu().item()
                    test_reg_loss += cur_reg_loss.detach().cpu().item()
                    test_rank_loss += cur_rank_loss.detach().cpu().item()

                    cur_test_pred[:, cur_offset - (self.test_index -
                                                   self.parameters['seq'] -
                                                   self.steps + 1)] = \
                        copy.copy(cur_rr[:, 0])
                    cur_test_gt[:, cur_offset - (self.test_index -
                                                 self.parameters['seq'] -
                                                 self.steps + 1)] = \
                        copy.copy(gt_batch[:, 0])
                    cur_test_mask[:, cur_offset - (self.test_index -
                                                   self.parameters['seq'] -
                                                   self.steps + 1)] = \
                        copy.copy(mask_batch[:, 0])
                print('Test MSE:',
                      test_loss / (self.trade_dates - self.test_index),
                      test_reg_loss / (self.trade_dates - self.test_index),
                      test_rank_loss / (self.trade_dates - self.test_index))
                cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
                print('\t Test performance:', 'sharpe5:', cur_test_perf['sharpe5'], 'ndcg_score_top5:',
                      cur_test_perf['ndcg_score_top5'])
                np.set_printoptions(threshold=sys.maxsize)
                if test_loss/(self.test_index-self.valid_index)<best_test_loss:
                    best_test_loss=test_loss/(self.test_index-self.valid_index)
                    #best_test_gt=copy.copy(cur_test_gt)
                    best_test_perf=copy.copy(cur_test_perf)

                    for item, key in enumerate(best_test_perf):
                        if (key != 'gt' and key!='prediction'):
                            with open('/data/res.txt', 'a+') as f:
                                f.write(str(key) + ':' + str(best_test_perf[key]) + '\n')
                        if (key == 'gt'):
                            np.savetxt('/data/gt.csv', best_test_perf['gt'], fmt='%.6f',
                                       delimiter=',')
                        else:
                            np.savetxt('/data/pre.csv', best_test_perf['prediction'],
                                       fmt='%.6f',
                                       delimiter=',')
            print('\t best performance:', 'sharpe5:', best_test_perf['sharpe5'], 'ndcg_score_top5:',
                  best_test_perf['ndcg_score_top5'])
            #torch.save(model.state_dict(),'best.pth')





    def update_model(self, parameters):
        for name, value in parameters.items():
            self.parameters[name] = value
        return True



if __name__ == '__main__':
    desc = 'train a relational rank lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', help='path of EOD data',
                        default='../data/2013-01-01-1')
    parser.add_argument('-m', help='market name', default='NYSE')
    parser.add_argument('-t', help='fname for selected tickers')
    parser.add_argument('-l', default=20,
                        help='length of historical sequence for feature')
    parser.add_argument('-u', default=64,
                        help='number of hidden units in lstm')
    parser.add_argument('-s', default=1,
                        help='steps to make prediction')
    parser.add_argument('-r', default=0.0001,
                        help='learning rate')
    parser.add_argument('-a', default=1,
                        help='alpha, the weight of ranking loss')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
    parser.add_argument('-rn', '--rel_name', type=str,
                        default='sector_industry',
                        help='relation type: sector_industry or wikidata')
    parser.add_argument('-ip', '--inner_prod', type=int, default=0)
    parser.add_argument('-node', default=1737,help='n_node')
    args = parser.parse_args()

    if args.t is None:
        args.t = args.m + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    args.gpu = (args.gpu == 1)

    args.inner_prod = (args.inner_prod == 1)

    parameters = {'seq': int(args.l), 'unit': int(args.u), 'lr': float(args.r),
                  'alpha': float(args.a),'a':0,'b':1,'c':0}

    RR_LSTM = Stock_HyperODE(
        data_path=args.p,
        market_name=args.m,
        tickers_fname=args.t,
        n_node=args.node,
        parameters=parameters,
        steps=1, epochs=100, batch_size=None,
        in_pro=args.inner_prod
    )

    pred_all = RR_LSTM.train()
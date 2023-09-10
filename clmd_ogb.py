import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
# from core.encoders import *

# from torch_geometric.datasets import TUDataset
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin_ogb import Encoder
from evaluate_embedding import evaluate_embedding
# from model import *

from arguments import arg_parse
from torch_geometric.transforms import Constant
import pdb

# from data_loader import *
from data_loader_amp import *
from load_dataset_amp import *
import copy
from My_LA.embedding_evaluation import EmbeddingEvaluation
from My_LA.encoder import TUEncoder
from My_LA.encoder import TUEncoder_sd
from My_LA.learning import MModel
from My_LA.learning import MModel_sd
from My_LA.utils import initialize_edge_weight, initialize_node_features, set_tu_dataset_y_shape
from My_LA.LGA_learner import LGALearner
from torch_scatter import scatter
import sys
import os
import signal

from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scipy import interpolate
from scipy.spatial.distance import cdist

import time

def metric(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores.detach().cpu())
    precision, recall, _ = precision_recall_curve(labels, scores.detach().cpu())
    fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
    aupr = auc(recall, precision)
    # aupr = 0
    auroc = auc(fpr, tpr)
    return auroc, fpr95, aupr
 
def lof(x, k=5, th=True):
    if th == True:
        odist = torch.cdist(x, x) + 0.01
        # print(odist[1])
        inds_sort = torch.argsort(odist, dim=-1)
        kdn = inds_sort[:, 1: 1 + k] # neighbor-index n * k
        kdist, _ = torch.sort(odist, dim=-1)
        kdist = kdist[:, k].reshape(-1) # Kth-reach-dist n
        lrd = torch.zeros(x.shape[0])
        for index in range(x.shape[0]):
            rd = 0
            for i in kdn[index]:
                rd += max(odist[index][i], kdist[i])
            lrd[index] = len(kdn[index]) * 1.0 / rd
        score = torch.zeros(x.shape[0])
        for index in range(x.shape[0]):
            lrd_nei = torch.sum(lrd[kdn[index]])
            score[index] = lrd_nei / k / lrd[index]
    else:
        odist = cdist(x, x) + 0.01
        # print(odist[1])
        inds_sort = np.argsort(odist, axis=-1)
        kdn = inds_sort[:, 1: 1 + k] # neighbor-index n * k
        kdist = np.sort(odist, axis=-1)[:, k].reshape(-1) # Kth-reach-dist n
        lrd = np.zeros(x.shape[0])
        for index in range(x.shape[0]):
            rd = 0
            for i in kdn[index]:
                rd += max(odist[index][i], kdist[i])
            lrd[index] = len(kdn[index]) * 1.0 / rd
        score = np.zeros(x.shape[0])
        for index in range(x.shape[0]):
            lrd_nei = np.sum(lrd[kdn[index]])
            score[index] = lrd_nei / k / lrd[index]
    return score

def pca(X, device, th=True):
    if th == True:
        mean = torch.mean(X, dim=0)
        X = X - mean
        X = X / torch.std(X, dim=0)
        cov = 1 / X.shape[0] * (X.t() @ X)
        Lambda, V = [torch.real(i) for i in torch.linalg.eig(cov)]
        max_index = torch.argmax(Lambda)
        max_eigval = torch.max(Lambda)
        # print("Torch")
        # mean :no grad
        return V[:, max_index], mean
    else:
        X_ = X.cpu().detach().numpy()
        mean_ = np.mean(X_, axis=0)
        X_ -= mean_
        X_ /= np.std(X_, axis=0)
        cov = 1 / X_.shape[0] * (X_.transpose() @ X_)
        Lambda, V = np.linalg.eig(cov)
        max_index = np.argmax(Lambda)
        max_eigval = np.max(Lambda)
        # mean :no grad
        return torch.from_numpy(V[:, max_index]).to(device), torch.from_numpy(mean_).to(device)

def kurtoses(X):
    X = (X - torch.mean(X)) / torch.std(X)
    return torch.mean(torch.pow(X, 4.0))
    
def ssd(ftrain, ftest, clusters=1, th=True):
    if clusters == 1:
        return get_scores_one_cluster(ftrain, ftest, th)
    else:
        if th == True:
            ypred, cluster_centers = kmeans(X=ftrain, num_clusters=clusters, distance='euclidean')
        else:
            ypred = sklearn.cluster.KMeans(n_clusters=clusters).fit_predict(ftrain)
        return get_scores_multi_cluster(ftrain, ftest, ypred, th)
        

def get_scores_one_cluster(ftrain, ftest, th=True):
    if th == True:
        cov = lambda x: torch.cov(x.T)
        con_inv = torch.linalg.pinv(cov(ftrain))
        if ftest is not None:
            dtest = torch.sum(
                (ftest - torch.mean(ftrain, dim=0, keepdims=True))
                * (
                    torch.mm(
                        con_inv,
                        (ftest - torch.mean(ftrain, dim=0, keepdims=True)).T
                    )
                ).T,
                dim=-1,
            )
    else:
        cov = lambda x: np.cov(x.T)
        con_inv = np.linalg.pinv(cov(ftrain))
        dtest = np.sum(
                (ftest - np.mean(ftrain, axis=0, keepdims=True))
                * (
                    np.matmul(
                        con_inv,
                        (ftest - np.mean(ftrain, axis=0, keepdims=True)).T
                    )
                ).T,
                axis=-1,
            )
    return dtest

def get_scores_multi_cluster(ftrain, ftest, ypred, th=True):
    if th == True:
        xc = [ftrain[ypred == i] for i in torch.unique(ypred)]
        dtest = [
            torch.sum(
                (ftest - torch.mean(x, axis=0, keepdims=True))
                * (
                    torch.mm(
                        torch.linalg.pinv(torch.cov(x.T)),
                        (ftest - torch.mean(x, dim=0, keepdims=True)).T
                    )
                ).T,
                dim=-1,
            )
            for x in xc
        ]
        dtest, _ = torch.min(torch.vstack(dtest), dim=0)
    else:
        xc = [ftrain[ypred == i] for i in np.unique(ypred)]
        dtest = [
            np.sum(
                (ftest - np.mean(x, axis=0, keepdims=True))
                * (
                    np.matmul(
                        np.linalg.pinv(np.cov(x.T)),
                        (ftest - np.mean(x, dim=0, keepdims=True)).T
                    )
                ).T,
                dim=-1,
            )
            for x in xc
        ]
        dtest = torch.min(np.vstack(dtest), dim=0)
    return dtest
    

class GcnInfomax(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(GcnInfomax, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)

    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch, num_graphs):

    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)
    
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)

    mode='fd'
    measure='JSD'
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
 
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    
    return local_global_loss + PRIOR


class simclr(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(simclr, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))

    self.init_emb()

  def init_emb(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch, num_graphs, edge_weight=None):

    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch, edge_weight=edge_weight)
    
    y = self.proj_head(y)
    
    return y

  def loss_cal(self, x, x_aug):
    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1) 
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / (torch.einsum('i,j->ij', x_abs, x_aug_abs))
    sim_matrix = torch.exp((sim_matrix / T))
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()

    return loss

def cal_score(args, edge_logits, batch, eva=False):
    bias = 0.0001
    eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
    edge_score = torch.log(eps) - torch.log(1 - eps)
    edge_score = edge_score.to(device)
    edge_score = (edge_score + edge_logits)
    batch_aug_edge_weight = torch.sigmoid(edge_score).squeeze()
    if eva:
        return batch_aug_edge_weight
    row, col = batch.edge_index
    edge_batch = batch.batch[row]
    uni, edge_batch_num = edge_batch.unique(return_counts=True)
    sum_pe = scatter((1 - batch_aug_edge_weight), edge_batch, reduce="sum")

    reg = []
    for b_id in range(args.batch_size):
        if b_id in uni:
            num_edges = edge_batch_num[uni.tolist().index(b_id)]
            reg.append(sum_pe[b_id] / num_edges)
        else:
            pass
    reg = torch.stack(reg)
    reg = reg.mean()
    ratio = args.ratio #0.4
    ratio = reg / ratio

    batch_aug_edge_weight = batch_aug_edge_weight / ratio # edge weight generalization  
    return batch_aug_edge_weight

def run(args, model, dataloader, dataloader_valid, dataloader_eval):
    if args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")
    setup_seed(args.seed)
    node_test, g_emb_test, y_test = model.encoder.get_embeddings(dataloader_eval, device=device, is_batch=True)
    node_valid, g_emb_valid, y_valid = model.encoder.get_embeddings(dataloader_valid, device=device, is_batch=True)
    node_train, g_emb_train,y_train = model.encoder.get_embeddings(dataloader, device=device, is_batch=True)

    amp_learner = LGALearner(len(node_train[0][0])).to(device)
    amp_optimizer = torch.optim.Adam(amp_learner.parameters(), lr=args.amplr) # l2-loss 2e-3;   
    best_val = 0
    result_dic = {}
    for epoch in range(1, 30):
        model_loss_all = 0
        i = 0
        for data in dataloader:
            batch = data
            batch = batch.to(device)
            amp_learner.train()
            amp_learner.zero_grad()
            edge_logits = amp_learner(node_train[i], batch.edge_index)

            batch_aug_edge_weight = cal_score(args, edge_logits, batch)
            ood_edge_weight = batch_aug_edge_weight - 1

            x_amp = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs, edge_weight = batch_aug_edge_weight) #shape: 128 * 96   dim=96, num_grpah=128
            x_ood = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs, edge_weight = ood_edge_weight)

            bal1 = args.bal1
            bal2 = args.bal1
            score = ssd(x_amp, x_amp)
            score_ood = ssd(x_amp, x_ood)
            amp_loss = torch.mean(score) + bal1 / torch.mean(score_ood) + bal2 * torch.norm(ood_edge_weight, p=2)
            
            if amp_loss != amp_loss:
                os.getpid() # os.getpid()获取当前进程id
                os.getppid() # os.getppid()获取父进程id
                os.kill(os.getpid(), signal.SIGKILL)
            amp_loss.backward()
            amp_optimizer.step()
            model_loss_all += amp_loss
            i += 1
            
        print('LOSS:{:.3f}'.format(model_loss_all))
     
        with torch.no_grad():    
            emb_train_amp = []
            emb_test_amp = []
            emb_valid_amp = []
            emb_ood_amp = []
            score_final = []
            i = 0
            for data in dataloader:
                batch = data
                batch = batch.to(device)
                edge_logits = amp_learner(node_train[i], batch.edge_index)
                batch_aug_edge_weight = cal_score(args, edge_logits, batch, eva=True)

                x_amp_train = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs, edge_weight = batch_aug_edge_weight)

                emb_train_amp.append(x_amp_train)
                ood_edge_weight = batch_aug_edge_weight - 1
                x_ood = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs, edge_weight = ood_edge_weight)
              
                i += 1
            i = 0
            for data in dataloader_eval:    
                batch = data
                batch = batch.to(device)
                edge_logits = amp_learner(node_test[i], batch.edge_index)
                batch_aug_edge_weight = cal_score(args, edge_logits, batch, eva=True)
                x_amp_test = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs, edge_weight = batch_aug_edge_weight)
                emb_test_amp.append(x_amp_test)
           
                ood_edge_weight = batch_aug_edge_weight - 1
                x_ood = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs, edge_weight = ood_edge_weight)
                emb_ood_amp.append(x_ood)
                i += 1
            i = 0
            for data in dataloader_valid:    
                batch = data
                batch = batch.to(device)
                edge_logits = amp_learner(node_valid[i], batch.edge_index)
                batch_aug_edge_weight = cal_score(args, edge_logits, batch, eva=True)

                x_amp_valid = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs, edge_weight = batch_aug_edge_weight)
                emb_valid_amp.append(x_amp_valid)
                i += 1
            
            emb_train_amp = torch.cat(emb_train_amp, 0)
            emb_valid_amp = torch.cat(emb_valid_amp, 0)
            emb_test_amp = torch.cat(emb_test_amp, 0)
            emb_ood_amp = torch.cat(emb_ood_amp, 0)

            ood_labels = np.zeros(emb_test_amp.shape[0])
            ood_slices = int(0.5 * emb_test_amp.shape[0])
            ood_labels[ood_slices:] = 1
            ood_valid_labels = np.zeros(emb_valid_amp.shape[0])
            ood_valid_slices = int(0.5 * emb_valid_amp.shape[0])
            ood_valid_labels[ood_slices:] = 1
            score = ssd(emb_train_amp, emb_test_amp)
            score_valid = ssd(emb_train_amp, emb_valid_amp)
       
            auroc, fpr95, aupr = metric(ood_labels, score)
            auroc_valid, fpr95_valid, aupr_valid = metric(ood_valid_labels, score_valid)
            
            print('EPOCH: {}, TEST: AUC:{:.4f}, FPR95:{:.4f}, AUPR:{:.4f}'.format(epoch, float(auroc), float(fpr95), float(aupr)))
            print('EPOCH: {}, VALI: AUC:{:.4f}, FPR95:{:.4f}, AUPR:{:.4f}'.format(epoch, float(auroc_valid), float(fpr95_valid), float(aupr_valid)))
            if auroc_valid >= best_val:
                best_val = auroc_valid
                result_dic["testauc"], result_dic["testfpr"], result_dic["testaupr"] = float(auroc), float(fpr95), float(aupr)
                result_dic["valauc"], result_dic["valfpr"], result_dic["valaupr"] = float(auroc_valid), float(fpr95_valid), float(aupr_valid)
                result_dic['epoch'] = epoch

    

def amp_train(args, model, dataloader, dataloader_valid, dataloader_eval):

    run(args, model, dataloader, dataloader_valid, dataloader_eval)
    
import random

def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    
    args = arg_parse()
    setup_seed(args.seed)
    print(args)

    accuracies = {'val':[], 'test':[]}
    epochs = 30
    log_interval = 10
    # batch_size = 800
    batch_size = 512
    lr = args.lr
    DS = args.DS
    
    data_train, data_val, data_test, data_all, data_nontrain, dataset_num_features = get_ogb_dataset(args)
 
    if DS == "PTC_MR":
        dataset_num_features = 1
   
    if args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")
    
    model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    trained_state = torch.load(args.DS_pair + "best_model.pth")
    model.load_state_dict(trained_state['model'])
    optimizer.load_state_dict(trained_state['optim'])
    
    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    
    model.eval()
    emb, y = model.encoder.get_embeddings(data_test, device=device)
    emb_valid, y_valid = model.encoder.get_embeddings(data_val, device=device)
    emb_train,y_train = model.encoder.get_embeddings(data_train, device=device)
    test_save = emb
    train_save = emb_train
    
    ood_labels = np.zeros(emb.shape[0])
    ood_slices = int(0.5 * emb.shape[0])
    ood_labels[ood_slices:] = 1
   
    score = ssd(torch.from_numpy(emb_train).to(device), torch.from_numpy(emb).to(device))
   
    auroc, fpr95, aupr = metric(ood_labels, score)
    print('Original Embedding AUC:{:.4f}, FPR95:{:.4f}, AUPR:{:.4f}'.format(float(auroc), float(fpr95), float(aupr)))
    

    amp_train(args, model, data_train, data_val, data_test)
    os.getpid() # os.getpid()获取当前进程id
    os.getppid() # os.getppid()获取父进程id

    # kill process
    os.kill(os.getpid(), signal.SIGKILL)
   
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import numpy as np

class LightGCN(nn.Module):
    def __init__(self, g, args, edge_index, item_min, item_max):
        super(LightGCN, self).__init__()
        self.args = args
        self.reg_weight = args.reg_weight
        self.dataset = args.dataset
        self.gpu = args.gpu
        self.s_reg_weight = args.s_reg_weight

        self.embedding = nn.Embedding(num_embeddings=item_max + 1, embedding_dim=args.embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)


        init = args.init
        self.retrain = args.retrain
        self.mask = None

        self.g = torch.sigmoid
        self.s = nn.Parameter(init * torch.ones(1), requires_grad=True)  #global
        self.gk = 1
        if self.retrain:
            self.init_retrain()

        self.sparse_v = self.embedding.weight.data
        self.emb_save_path = args.emb_save_path
        self.retrain_emb_param = args.retrain_emb_param

    def init_retrain(self):
        sparse_emb = np.load(self.args.emb_save_path + "-" + "param" + str(self.retrain_emb_param) +'.npy')
        sparse_emb = torch.from_numpy(sparse_emb)
        mask = torch.abs(torch.sign(sparse_emb))

        init_emb = np.load(self.args.emb_save_path + "-" + "init" + '.npy')
        init_emb = torch.from_numpy(init_emb)

        init_emb = init_emb * mask
        self.embedding = torch.nn.Parameter(init_emb, requires_grad=True)
        self.mask = mask
        self.gk = 0
        if self.use_cuda:
            self.mask = self.mask.cuda(self.gpu)

    def get_emb(self):
        self.sparse_v = torch.sign(self.embedding.weight) * torch.relu(torch.abs(self.embedding.weight) - (self.g(self.s) * self.gk))
        if self.retrain:
            self.sparse_v = self.sparse_v * self.mask
        return self.sparse_v

    def forward(self, users, pos, neg):
        all_emb = self.get_emb()

        users_emb = all_emb[users]
        pos_emb = all_emb[pos]
        neg_emb = all_emb[neg]

        users_emb_ego = self.embedding(users)
        pos_emb_ego = self.embedding(pos)
        neg_emb_ego = self.embedding(neg)
        reg_loss = (1 / 2) * (users_emb_ego.norm(2).pow(2) +
                              pos_emb_ego.norm(2).pow(2) +
                              neg_emb_ego.norm(2).pow(2)
                              ) / float(len(users))

        reg_s_loss = (1 / 2) * (self.s.norm(2).pow(2)) / float(len(users))
        s_loss = reg_s_loss * self.s_reg_weight


        loss_emb = self.create_bpr_loss(users_emb, pos_emb, neg_emb)

        reg_loss = reg_loss * self.reg_weight
        loss = loss_emb + reg_loss + s_loss

        return loss, loss_emb,  reg_loss, s_loss

    def create_bpr_loss(self, users_emb, pos_emb, neg_emb):
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss_emb = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss_emb

    def calc_sparsity(self):
        base = self.embedding.weight.shape[0] * self.embedding.weight.shape[1]
        non_zero_values = torch.nonzero(self.sparse_v).size(0)
        percentage = 1 - (non_zero_values / base)
        return percentage, non_zero_values
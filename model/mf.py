import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

class LightGCN(nn.Module):
    def __init__(self, g, args, edge_index, item_min, item_max):
        super(LightGCN, self).__init__()
        self.reg_weight = args.reg_weight
        self.num_layer = args.num_layer
        self.dataset = args.dataset
        self.tau = args.tau
        self.lossf = args.lossf
        self.g = g
        self.gpu = args.gpu
        self.embedding = nn.Embedding(num_embeddings=item_max + 1, embedding_dim=args.embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.weight_mask_fixed = nn.Parameter(torch.ones_like(self.embedding.weight), requires_grad=False)

        self.weight_nonzero = (item_max+1) * args.embedding_size
        self.weight_nonzero_user = (item_min) * args.embedding_size
        self.weight_nonzero_item = (item_max+1 - item_min) * args.embedding_size

    def get_emb(self):
        embeddings_list = [self.embedding.weight * self.weight_mask_fixed]
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1).squeeze()
        return lightgcn_all_embeddings

    def forward(self, users, pos, neg):
        all_emb = self.get_emb()
        if self.lossf == "bpr":
            users_emb = all_emb[users]
            pos_emb = all_emb[pos]
            neg_emb = all_emb[neg]

            users_emb_ego = self.embedding(users)
            pos_emb_ego = self.embedding(pos)
            neg_emb_ego = self.embedding(neg)
            reg_loss = (1 / 2) * (users_emb_ego.norm(2).pow(2) +
                                  pos_emb_ego.norm(2).pow(2) +
                                  neg_emb_ego.norm(2).pow(2)) / float(len(users))
            loss_emb = self.create_bpr_loss(users_emb, pos_emb, neg_emb)

        elif self.lossf == "nce":
            users_emb = all_emb[users]
            pos_emb = all_emb[pos]

            users_emb_ego = self.embedding(users)
            pos_emb_ego = self.embedding(pos)
            reg_loss = (1 / 2) * (users_emb_ego.norm(2).pow(2) +
                                  pos_emb_ego.norm(2).pow(2)) / float(len(users))
            loss_emb = self.create_nce_loss(users_emb, pos_emb)

        reg_loss = reg_loss * self.reg_weight
        loss = loss_emb + reg_loss

        return loss, loss_emb,  reg_loss
    def create_bpr_loss(self, users_emb, pos_emb, neg_emb):
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss_emb = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss_emb

    def create_nce_loss(self, users, pos_items):
        normalize_batch_user_emb = torch.nn.functional.normalize(users, p=2, dim=1)
        normalize_batch_pos_item_emb = torch.nn.functional.normalize(pos_items, p=2, dim=1)
        pos_score = (normalize_batch_user_emb * normalize_batch_pos_item_emb).sum(dim=1)
        ttl_score = torch.sum(torch.exp(torch.mm(normalize_batch_user_emb, normalize_batch_pos_item_emb.T) / self.tau), dim=1)
        pos_score = torch.exp(pos_score / self.tau)
        loss_emb = -torch.sum(torch.log(pos_score / ttl_score + 1e-15))
        return loss_emb

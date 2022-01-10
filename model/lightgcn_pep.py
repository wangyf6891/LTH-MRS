import numpy as np
import dgl.function as fn
import torch
import torch.nn as nn

gcn_msg = fn.src_mul_edge('h', 'adj', 'm')  #e is norm,
gcn_reduce = fn.sum(msg='m', out='h')  # source add, edge update!!


class LightGCN(nn.Module):
    def __init__(self, g, args, edge_index, item_min, item_max):
        super(LightGCN, self).__init__()
        self.args = args
        self.reg_weight = args.reg_weight
        self.dataset = args.dataset
        self.gpu = args.gpu
        self.num_layer = args.num_layer
        self.g = g
        self.s_reg_weight = args.s_reg_weight

        self.embedding = nn.Embedding(num_embeddings=item_max + 1, embedding_dim=args.embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)


        init = args.init
        self.retrain = args.retrain
        self.mask = None

        self.sigmoid = torch.sigmoid
        self.s = nn.Parameter(init * torch.ones(1), requires_grad=True)  #global
        self.gk = 1

        self.sparse_v = self.embedding.weight.data
        self.emb_save_path = args.emb_save_path
        self.retrain_emb_param = args.retrain_emb_param

        if self.retrain:
            self.init_retrain()
        self.norm()


    def norm(self):  #from mask get adj edata
        # self.g = dgl.transform.remove_self_loop(self.g)  #remove
        # self.g = dgl.add_self_loop(self.g)  # add self loop
        self.g.edata["mask"] = torch.ones(self.g.num_edges()).unsqueeze(dim=1).to(torch.device(self.gpu))
        self.g.update_all(fn.copy_e('mask', 'm'), fn.sum('m', 'd'))
        self.g.ndata["d"] = self.g.ndata["d"] ** -0.5
        self.g.ndata["d"][self.g.ndata["d"] == float("inf")] = 0
        self.g.apply_edges(fn.u_mul_e('d', 'mask', 'adj'))
        self.g.apply_edges(fn.e_mul_v('adj', 'd', 'adj'))

    def init_retrain(self):
        sparse_emb = np.load(self.args.emb_save_path + "-" + "param" + str(self.retrain_emb_param) +'.npy')
        sparse_emb = torch.from_numpy(sparse_emb)
        mask = torch.abs(torch.sign(sparse_emb))

        init_emb = np.load(self.args.emb_save_path + "-" + "init" + '.npy')
        init_emb = torch.from_numpy(init_emb)

        init_emb = init_emb * mask
        self.embedding = nn.Embedding.from_pretrained(embeddings=init_emb, freeze=False)
        self.mask = mask
        self.gk = 0

        self.mask = self.mask.cuda(self.gpu)

    def get(self):
        self.sparse_v = torch.sign(self.embedding.weight) * torch.relu(torch.abs(self.embedding.weight) - (self.sigmoid(self.s) * self.gk))
        if self.retrain:
            self.sparse_v = self.sparse_v * self.mask
        return self.sparse_v

    def get_emb(self):
        sparse_v = self.get()
        embeddings_list = [sparse_v]
        self.g.ndata['h'] = embeddings_list[-1]

        for layer in range(self.num_layer):
            self.g.update_all(gcn_msg, gcn_reduce)
            embeddings_list.append(self.g.ndata['h'])

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        return lightgcn_all_embeddings


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
import torch
import argparse
import numpy as np
from utils.dataset import Train_Dataset, Test_Dataset
from torch.utils.tensorboard import SummaryWriter
from utils.data_util import dataset_info
from utils.data_util import test_batcher
import dgl
from utils.misc import AverageMeter


from model.lightgcn_pep import LightGCN    #change this term to from pep.mf_pep import LightGCN
# from model.mf_pep import LightGCN


import os
from body import test
import sys
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description="Counterfactual Generation via Adversarial Training.")
    parser.add_argument('--epoch', type=int, default=2000, help='The inner loop max epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='.')
    parser.add_argument('--dataset', type=str, default='yelp2018', help='One of [yelp2018, Kwai, TikTok]')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size.')
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--num_layer", type=int, default=3)

    parser.add_argument('--reg_weight', type=float, default=1e-4, help='regular')
    parser.add_argument('--s_reg_weight', type=float, default=1e-1, help='regular')


    parser.add_argument("--topks", type=int, default=[20])
    parser.add_argument("--early_stop", type=int, default=50)

    parser.add_argument("--gpu", default=0, type=int, nargs='+', help="GPU id to use.")
    parser.add_argument("--seed", default=2020, type=int, help="seed")

    ##pep##
    parser.add_argument("--emb_save_path", type=str, default="./pep/embedding/", help="path to pep_embedding")
    parser.add_argument("--candidate_per", type=int, default=[0.5, 0.75, 0.9])
    parser.add_argument("--retrain", type=bool, default=False)
    parser.add_argument("--retrain_emb_param", default=2227525, type=int, help="retrain_emb_param(need to get from board")
    parser.add_argument("--clip", default=100, type=int, help="grad clip")

    parser.add_argument("--init", default=-10, type=int, help="init_s")
    parser.add_argument("--max_steps", default=600, type=int, help="max_steps")
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dgl.random.seed(seed)

class Engine(object):
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def save_pruned_embedding(self, param, step_idx):
        max_candidate_p = max(self.args.candidate_p)

        if max_candidate_p == 0:
            print("Minimal target parameters achieved, stop pruning.")
            exit(0)
        else:
            if param <= max_candidate_p:
                embedding = self.model.get().detach().cpu().numpy()   #mf need to change to get_emb()
                emb_save_path = self.args.emb_save_path + "-" + "param" + str(param)
                emb_save_dir, _ = os.path.split(emb_save_path)
                if not os.path.exists(emb_save_dir):
                    os.makedirs(emb_save_dir)
                np.save(emb_save_path, embedding)
                max_idx = self.args.candidate_p.index(max(self.args.candidate_p))
                self.args.candidate_p[max_idx] = 0
                print("*" * 80)
                print("Reach the target parameter: {}, save embedding with size: {}".format(max_candidate_p, param))
                print("*" * 80)
                torch.save(model, emb_save_path)

            elif step_idx == 0:
                embedding = self.model.embedding.weight.detach().cpu().numpy()
                emb_save_path = self.args.emb_save_path + "-" + "init"
                emb_save_dir, _ = os.path.split(emb_save_path)
                if not os.path.exists(emb_save_dir):
                    os.makedirs(emb_save_dir)
                np.save(emb_save_path, embedding)
                print("*" * 80)
                print("Save the initial embedding table")
                print("*" * 80)

    def train(self, train_loader, optimizer, epoch_idx, step):
        self.model.train()
        avg_loss = AverageMeter()
        avg_re_loss = AverageMeter()
        avg_reg_loss = AverageMeter()
        avg_reg_s_loss = AverageMeter()
        avg_s = AverageMeter()

        max_steps = self.args.max_steps

        if step == 0:   # save the origion
            sparsity, params = self.model.calc_sparsity()
            self.save_pruned_embedding(params, step)

        for i, batch in enumerate(train_loader):
            step += 1
            user, item, item_j = batch
            bsz = len(user)
            loss, reconstruct_loss, reg_loss, s_loss = self.model(user.to(torch.device(args.gpu)),
                                                     item.to(torch.device(args.gpu)),
                                                     item_j.to(torch.device(args.gpu)))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)  # 梯度裁剪
            optimizer.step()


            if step % max_steps == 0:
                sparsity, params = self.model.calc_sparsity()
                if not self.args.retrain:
                    self.save_pruned_embedding(params, step)

                print('[Epoch {}|Step {}|Sparsity {:.4f}|Params {}]'.format(epoch_idx, step % len(train_loader),
                                                                            sparsity, params))

            avg_loss.update(loss.item(), bsz)
            avg_re_loss.update(reconstruct_loss.item(), bsz)
            avg_reg_loss.update(reg_loss.item(), bsz)
            avg_s.update(self.model.s.item(), bsz)
            avg_reg_s_loss.update(s_loss.item(), bsz)
        return step, avg_loss.avg, avg_re_loss.avg, avg_reg_loss.avg, avg_s.avg, avg_reg_s_loss.avg

    def retrain(self, train_loader, optimizer, args):
        self.model.train()
        avg_loss = AverageMeter()
        avg_re_loss = AverageMeter()
        avg_reg_loss = AverageMeter()
        avg_reg_s_loss = AverageMeter()
        avg_s = AverageMeter()

        for i, batch in enumerate(train_loader):
            user, item, item_j = batch
            bsz = len(user)
            loss, reconstruct_loss, reg_loss, s_loss = self.model(user.to(torch.device(args.gpu)),
                                                     item.to(torch.device(args.gpu)),
                                                     item_j.to(torch.device(args.gpu)))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)  # 梯度裁剪
            optimizer.step()

            avg_loss.update(loss.item(), bsz)
            avg_re_loss.update(reconstruct_loss.item(), bsz)
            avg_reg_loss.update(reg_loss.item(), bsz)
            avg_s.update(self.model.s.item(), bsz)
            avg_reg_s_loss.update(s_loss.item(), bsz)
        return avg_loss.avg, avg_re_loss.avg, avg_reg_loss.avg, avg_s.avg, avg_reg_s_loss.avg



if __name__ == '__main__':
    warnings.simplefilter("once", UserWarning)
    args = parse_args()
    arg = vars(args)
    set_seed(seed=2020)
    # args.gpu = args.gpu[0]
    args.emb_save_path = args.emb_save_path + args.dataset + "-dim-" + str(args.embedding_size)  # save teh rewind

    if not os.path.exists(args.emb_save_path):
        os.makedirs(args.emb_save_path)

    item_min, item_max, data, length, user_seq, edge_index,\
    group_u_num, group_i_num, user_group, item_group = dataset_info(args.dataset)

    base = args.embedding_size * (item_max + 1)
    args.candidate_p = [int((1-i)*base) for i in args.candidate_per]

    g = dgl.DGLGraph()
    g.add_nodes(item_max+1)
    g.add_edges(edge_index[0], edge_index[1])
    g.add_edges(edge_index[1], edge_index[0])

    train_dataset = Train_Dataset(data=data, length=length, item_min=item_min, item_max=item_max, user_seq=user_seq, )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=2, pin_memory=True)

    test_dataset = Test_Dataset(item_min, args)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=2, collate_fn=test_batcher(),)

    model = LightGCN(g.to(torch.device(args.gpu)), args=args, edge_index=edge_index, item_min=item_min, item_max=item_max).to(torch.device(args.gpu))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    eng = Engine(model, args)

    best_epoch = 0
    best_loss = {'loss': 0, 'emb': 0, 'reg': 0}
    max_beyond = 0
    args.start_epoch = 1
    best_result = {'precision': 0, 'recall': 0, 'ndcg': 0}

    s_max = float("-inf")
    max_beyond_s = 0

    step = 0
    for epoch in range(args.start_epoch, args.epoch + 1):
        train_dataset.__update__()

        if args.retrain:
            loss, re_loss, reg_loss,s, s_loss = eng.retrain(train_loader, optimizer, args)
            result = test(test_loader, model, item_min, user_seq, args)

            sys.stdout.write('\rEpoch[{}/{}],'
                             '<loss:{:.4f}, emb:{:.4f}, reg:{:.4f}, s_loss:{:.4f}>'
                             '<pre:{:.4f}, rec:{:.4f}, ndcg:{:.4f}>'
                             '<s:{:.4f}>'
                             '  [Best prec:{:.4f}, rec:{:.4f}, ndcg:{:.4f}, emb:{:.4f}, reg:{:.4f} epoch:{}]'.format(
                epoch, args.epoch, loss, re_loss, reg_loss, s_loss,
                result['precision'], result['recall'], result['ndcg'],
                s,
                best_result['precision'], best_result['recall'], best_result['ndcg'], best_loss['emb'],
                best_loss['reg'], best_epoch))

            sys.stdout.flush(),
            print("")
            # early stop
            if result['recall'] > best_result['recall']:
                best_result = result
                best_epoch = epoch
                best_loss['loss'] = loss
                best_loss['emb'] = re_loss
                best_loss['reg'] = reg_loss
                max_beyond = 0

            else:
                max_beyond += 1
            if max_beyond >= args.early_stop:
                break

        else:
            step, loss, re_loss, reg_loss, s, s_loss = eng.train(train_loader=train_loader, optimizer=optimizer, epoch_idx=epoch, step=step)
            result = test(test_loader, model, item_min, user_seq, args)

            sys.stdout.write('\rEpoch[{}/{}],'
                             '<loss:{:.4f}, emb:{:.4f}, reg:{:.4f}, s_loss:{:.4f}>'
                             '<pre:{:.4f}, rec:{:.4f}, ndcg:{:.4f}>'
                             '<s:{:.4f}>'
                             '  [Best prec:{:.4f}, rec:{:.4f}, ndcg:{:.4f}, emb:{:.4f}, reg:{:.4f} epoch:{}]'.format(
                epoch, args.epoch, loss, re_loss, reg_loss, s_loss,
                result['precision'], result['recall'], result['ndcg'],
                s,
                best_result['precision'], best_result['recall'], best_result['ndcg'], best_loss['emb'],
                best_loss['reg'], best_epoch))

            sys.stdout.flush(),
            print("")

            if s > s_max:
                s_max = s
                max_beyond_s = 0
            else:
                max_beyond_s += 1

            if max_beyond_s >= 5:
                eng.args.s_reg_weight = (1 + 0.1) * args.s_reg_weight
                max_beyond_s = 0

            if result['recall'] > best_result['recall']:
                best_result = result
                best_epoch = epoch
                best_loss['loss'] = loss
                best_loss['emb'] = re_loss
                best_loss['reg'] = reg_loss
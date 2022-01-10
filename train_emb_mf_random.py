import os
import sys
import torch
import copy
import warnings
import dgl
import psutil
from model.mf import LightGCN
from utils.dataset import Train_Dataset, Test_Dataset
from torch.utils.tensorboard import SummaryWriter
from utils.data_util import dataset_info
from utils.data_util import test_batcher
from utils.pruning_emb import random_pruning_emb
from body import train, test, set_seed, parse_args

def run_fix_mask_emb(g, args, imp_num, item_min, item_max, user_seq, edge_index, train_loader,
                     test_loader, group_u_num, group_i_num, user_group, item_group, path,
                     emb_percent, adj_percent):
    set_seed(seed=args.seed)
    path = path + "-imp-" + str(imp_num)

    model = LightGCN(g.to(torch.device(args.gpu)), args=args, edge_index=edge_index,
                     item_max=item_max, item_min=item_min).to(torch.device(args.gpu))

    print("-" * 100)
    random_pruning_emb(model, emb_percent[imp_num])    # pruning mask_fixed


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    rewind_weight = copy.deepcopy(model.state_dict())  # save the previous emb, and new mask

    best_epoch = 0
    best_loss = {'loss': 0, 'emb': 0, 'reg': 0}
    max_beyond = 0
    args.start_epoch = 1
    best_result = {'precision': 0, 'recall': 0, 'ndcg': 0}

    for epoch in range(args.start_epoch, args.epoch + 1):
        train_dataset.__update__()
        loss, re_loss, reg_loss = train(train_loader, model, optimizer, args)

        result = test(test_loader, model, item_min, user_seq, args)

        sys.stdout.write('\rF{}, Epoch[{}/{}],'
                         '<loss:{:.4f}, emb:{:.4f}, reg:{:.4f}>,'
                         '<pre:{:.4f}, rec:{:.4f}, ndcg:{:.4f}>'
                         '  [Best prec:{:.4f}, rec:{:.4f}, ndcg:{:.4f}, emb:{:.4f}, reg:{:.4f} epoch:{}]'.format(
            imp_num, epoch, args.epoch, loss, re_loss, reg_loss,
            result['precision'], result['recall'], result['ndcg'],
            best_result['precision'], best_result['recall'], best_result['ndcg'],
            best_loss['emb'], best_loss['reg'], best_epoch))

        sys.stdout.flush(),
        # early stop
        if result['recall'] > best_result['recall']:
            best_result = result
            best_epoch = epoch
            best_loss['loss'] = loss
            best_loss['emb'] = re_loss
            best_loss['reg'] = reg_loss

            torch.save(model, path)  # save the new g and best para

            max_beyond = 0
        else:
            max_beyond += 1
        if max_beyond >= args.early_stop:
            break

        torch.cuda.empty_cache()
    print("")


if __name__ == '__main__':
    warnings.simplefilter("once", UserWarning)
    # set parameters
    args = parse_args()
    arg = vars(args)
    print(arg)
    args.gpu = args.gpu[0]

    dir = "random_mf/ckpt/emb/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    file = "part-" + str(args.part) + "-" + args.dataset + "-reg-" + str(args.reg_weight) + \
           "-loss-" + args.lossf + "-stop-" + str(args.early_stop) + "-epoch-" + str(args.epoch) + "-" + \
           str(args.stop) + "-dim-" + str(args.embedding_size) + "-per-" + str(args.pruning_percent_wei) + \
           "-" + str(args.seed)
    path = dir+file

    # get the information of the dataset
    item_min, item_max, data, length, user_seq, edge_index, \
    group_u_num, group_i_num, user_group, item_group = dataset_info(args.dataset)

    g = dgl.DGLGraph()
    g.add_nodes(item_max + 1)
    g.add_edges(edge_index[0], edge_index[1])
    g.add_edges(edge_index[1], edge_index[0])

    train_dataset = Train_Dataset(data=data, length=length, item_min=item_min, item_max=item_max, user_seq=user_seq)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=2)

    test_dataset = Test_Dataset(item_min, args)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=2, collate_fn=test_batcher(), )

    mem = psutil.virtual_memory()
    print("before training", mem.used / 1024 ** 3)

    emb_percent = [(1 - (1 - args.pruning_percent_wei) ** (i)) for i in range(31)]
    adj_percent = [(1 - (1 - args.pruning_percent_adj) ** (i)) for i in range(31)]

    start = 1
    for imp in range(start, 31):
        run_fix_mask_emb(g, args, imp, item_min, item_max, user_seq, edge_index,
                                         train_loader, test_loader, group_u_num, group_i_num, user_group, item_group,
                                         path, emb_percent, adj_percent)
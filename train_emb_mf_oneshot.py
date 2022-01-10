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
from utils.print_details import emb_pruning_detail
from utils.data_util import test_batcher
from utils.pruning_emb import print_sparsity_emb, get_final_mask_epoch_emb, \
    get_final_mask_epoch_emb_user, get_final_mask_epoch_emb_item, print_sparsity_emb_user, \
    print_sparsity_emb_item, print_emb_mask_distribution
from body import train, test, set_seed, parse_args

def run_get_mask_emb(g, args, imp_num, rewind_weight_mask, item_min, item_max, user_seq, edge_index, train_loader,
                     test_loader, group_u_num, group_i_num, user_group, item_group, path):
    set_seed(seed=2020)

    path = path + "-imp-" + str(imp_num)

    model = LightGCN(g.to(torch.device(args.gpu)), args=args, edge_index=edge_index,
                     item_max=item_max, item_min=item_min).to(torch.device(args.gpu))

    # embedding pruning detail
    if rewind_weight_mask is not None:
        dict = {"weight_mask_fixed"}
        model_dict = model.state_dict()
        rewind_weight_mask = {k: v for k, v in rewind_weight_mask.items() if k in dict}
        model_dict.update(rewind_weight_mask)
        model.load_state_dict(model_dict)
        print("-" * 100)
        emb_pruning_detail(model.weight_mask_fixed, item_min, user_group, group_u_num, item_group, group_i_num)   #percent list

    if args.part == "user" or args.part == "item" or args.part == "user_item":  # pruning emb according item ? user ? or all
        spar = print_sparsity_emb_user(model, item_min)
        spar = print_sparsity_emb_item(model, item_min)
    else:
        spar = print_sparsity_emb(model)


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

        sys.stdout.write('\rM{}, Epoch[{}/{}],'
                         '<loss:{:.4f}, emb:{:.4f}, reg:{:.4f}>,'
                         '<pre:{:.4f}, rec:{:.4f}, ndcg:{:.4f}>'
                         '  [Best prec:{:.4f}, rec:{:.4f}, ndcg:{:.4f}, emb:{:.4f}, reg:{:.4f} epoch:{}]'.format(
            imp_num, epoch, args.epoch, loss, re_loss, reg_loss,
            result['precision'], result['recall'], result['ndcg'],
            best_result['precision'], best_result['recall'], best_result['ndcg'],
            best_loss['emb'], best_loss['reg'], best_epoch))

        sys.stdout.flush(),

        if result['recall'] > best_result['recall']:
            best_result = result
            best_epoch = epoch
            best_loss['loss'] = loss
            best_loss['emb'] = re_loss
            best_loss['reg'] = reg_loss
        # if fix the epoch num, then annote here
            torch.save(model, path)

            max_beyond = 0
        else:
            max_beyond += 1
        if max_beyond >= args.early_stop:
            break

    print("")
    model = torch.load(path)
    print_emb_mask_distribution(model.embedding.weight * model.weight_mask_fixed)   # print the embedding distribution of the best epoch

    torch.cuda.empty_cache()
    return rewind_weight


if __name__ == '__main__':
    warnings.simplefilter("once", UserWarning)
    args = parse_args()
    arg = vars(args)
    print(arg)
    args.gpu = args.gpu[0]


# /////////////////////////////////////////////////////////////////////////////
    # get the information of the dataset
    item_min, item_max, data, length, user_seq, edge_index, \
    group_u_num, group_i_num, user_group, item_group = dataset_info(args.dataset)

    dir = "oneshot_mf_lot/ckpt/emb/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    file = "part-" + str(args.part) + "-" + args.dataset + "-reg-" + str(args.reg_weight) + \
           "-loss-" + args.lossf + "-stop-" + str(args.early_stop) + "-epoch-" + str(args.epoch) + "-" + \
            str(args.stop) + "-dim-" + str(args.embedding_size) + "-per-" + str(args.pruning_percent_wei) # save teh rewind

    path = dir + file

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

    #from baseline
    weight = [(1 - (1 - args.pruning_percent_wei) ** (i)) for i in range(1, 51, 1)]
    for i in range(29, 51, 1):
        args.pruning_percent_wei = weight[i]
        if args.dataset == "yelp2018":
            model = torch.load("./mflot/ckpt/emb/part-join-reg-0.0001-loss-bpr-stop-100-epoch-1000-nonstop-dim-32-per-0.1-imp-1")
        elif args.dataset == "Kwai":
            model = torch.load(
                "./mflot/ckpt/emb/part-join-Kwai-reg-0.0001-loss-bpr-stop-100-epoch-1000-nonstop-dim-32-per-0.1-imp-1")
        elif args.dataset == "Tiktok":
            model = torch.load(
                "./mflot/ckpt/emb/part-join-Tiktok-reg-0.0001-loss-bpr-stop-100-epoch-1000-nonstop-dim-32-per-0.1-imp-1")

        rewind_weight = copy.deepcopy(model.state_dict())
        if args.part == "user":  # pruning emb according item ? user ? or all
            rewind_weight = get_final_mask_epoch_emb_user(model, rewind_weight, args, item_min)
            # ! save the best rewind_weight
        elif args.part == "item":
            rewind_weight = get_final_mask_epoch_emb_item(model, rewind_weight, args, item_min)
        elif args.part == "user_item":
            rewind_weight = get_final_mask_epoch_emb_user(model, rewind_weight, args, item_min)
            rewind_weight = get_final_mask_epoch_emb_item(model, rewind_weight, args, item_min)
        else:
            rewind_weight = get_final_mask_epoch_emb(model, rewind_weight, args)

        rewind_weight = run_get_mask_emb(g, args, i, rewind_weight, item_min, item_max, user_seq, edge_index,
                                        train_loader, test_loader, group_u_num, group_i_num,user_group,item_group,path)
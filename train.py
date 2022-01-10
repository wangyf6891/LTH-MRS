import os
import sys
import torch
import warnings
import dgl
from model.lightgcn_me import LightGCN
from torch.utils.tensorboard import SummaryWriter
from utils.data_util import dataset_info
from utils.data_util import test_batcher
from utils.dataset import Train_Dataset, Test_Dataset
from utils.pruning_emb import print_emb_mask_distribution
from body import train, test, set_seed, parse_args

if __name__ == '__main__':
    warnings.simplefilter("once", UserWarning)
    # set parameters
    args = parse_args()
    arg = vars(args)
    print(arg)
    args.gpu = args.gpu[0]


    set_seed(seed=2020)
    file = "part-" + str(args.part) + "-reg-" + str(
        args.reg_weight) + "-loss-" + args.lossf + "-stop-" + str(args.early_stop)
    dir = "lot/ckpt_model/baseline/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    model_path = dir + file

    item_min, item_max, data, length, user_seq, edge_index,\
    group_u_num, group_i_num, user_group, item_group = dataset_info(args.dataset)

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

    model = LightGCN(g.to(torch.device(args.gpu)), args=args, edge_index=edge_index, item_max=item_max).to(torch.device(args.gpu))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_epoch = 0
    best_loss = {'loss': 0, 'emb': 0, 'reg': 0}
    max_beyond = 0
    args.start_epoch = 1
    best_result = {'precision': 0, 'recall': 0, 'ndcg': 0}

    for epoch in range(args.start_epoch, args.epoch + 1):
        train_dataset.__update__()
        loss, re_loss, reg_loss = train(train_loader, model, optimizer, args)

        result = test(test_loader, model, item_min, user_seq, args)

        sys.stdout.write('\rEpoch[{}/{}],'
            '<loss:{:.4f}, emb:{:.4f}, reg:{:.4f}>,'
            '<pre:{:.4f}, rec:{:.4f}, ndcg:{:.4f}>'
            '  [Best prec:{:.4f}, rec:{:.4f}, ndcg:{:.4f}, emb:{:.4f}, reg:{:.4f} epoch:{}]'.format(
            epoch, args.epoch, loss, re_loss, reg_loss,
            result['precision'], result['recall'], result['ndcg'],
            best_result['precision'], best_result['recall'], best_result['ndcg'], best_loss['emb'], best_loss['reg'], best_epoch))

        # early stop
        sys.stdout.flush(),
        # early stop
        if result['recall'] > best_result['recall']:
            best_result = result
            best_epoch = epoch
            best_loss['loss'] = loss
            best_loss['emb'] = re_loss
            best_loss['reg'] = reg_loss
            max_beyond = 0
            torch.save(model, model_path)  # save the new g and best para
            print("")
            print("user:  ", end=" ")
            print_emb_mask_distribution(model.embedding.weight[:item_min])
            print("item:  ", end=" ")
            print_emb_mask_distribution(model.embedding.weight[item_min:])
        else:
            max_beyond += 1
        if max_beyond >= args.early_stop:
            break

    model = torch.load(model_path)

    torch.cuda.empty_cache()
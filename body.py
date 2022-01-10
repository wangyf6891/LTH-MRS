import torch
import argparse
import numpy as np
from utils.metrics import get_test
from utils.misc import AverageMeter
import dgl

def parse_args():
    parser = argparse.ArgumentParser(description="Counterfactual Generation via Adversarial Training.")
    parser.add_argument('--epoch', type=int, default=1000, help='The inner loop max epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='.')
    parser.add_argument('--dataset', type=str, default='yelp2018', help='One of [yelp2018, Kwai, TikTok]')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size.')
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument('--reg_weight', type=float, default=1e-4, help='regular')
    parser.add_argument('--tau', type=float, default='0.1', help='tau')
    parser.add_argument("--topks", type=int, default=[20])
    parser.add_argument("--early_stop", type=int, default=100)
    parser.add_argument("--tb_folder", type=str, default="./result/", help="path to tensorboard")
    parser.add_argument('--num_layer', type=int, default=3, help='light_gcn num layer.')

    parser.add_argument('--lossf', type=str, default='bpr', help='loss func')

    parser.add_argument('--s1', type=float, default=0.0001, help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--s2', type=float, default=0.0001, help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--pruning_percent_wei', type=float, default=0.1)
    parser.add_argument('--pruning_percent_adj', type=float, default=0.05)

    parser.add_argument("--gpu", default=0, type=int, nargs='+', help="GPU id to use.")
    parser.add_argument("--seed", default=2019, type=int, help="seed")

    parser.add_argument('--part', type=str, default='join', help='user, item, user_item, join')

    parser.add_argument('--stop', type=str, default='nonstop', help='stop, nonstop')
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dgl.random.seed(seed)


# according user
def group_test_user(test_loader, model, item_min, user_seq, args, group_u_num, group_i_num, user_group, item_group):
    model.eval()
    z = model.get_emb()
    item_emb = z[item_min:]

    result = []
    userid = torch.arange(0, item_min, 1)
    max_K = max(args.topks)
    for i in range(len(group_u_num)):
        user_id = userid[user_group == i]
        groundTrue = [user_seq[i] for i in user_id.tolist()]
        user_emb = z[user_id]
        ranking_score = user_emb.matmul(item_emb.t())

        for idx, user in enumerate(user_id):
            ranking_score[idx][np.array(user_seq[user.item()]) - item_min] = -np.inf

        _, rec = torch.topk(ranking_score, max_K)  # return the idx
        rec = rec.detach().cpu().numpy()

        res = []
        for k, v in get_test([rec], [groundTrue], args).items():
            res.append(v[0])

        result.append(res)  # list

    print("user_group_precision:", end=" ")
    print([result[i][0] for i in range(len(group_u_num))])

    print("user_group_recall:", end=" ")
    print([result[i][1] for i in range(len(group_u_num))])
    return result


# according item
def group_test_item(test_loader, model, item_min, user_seq, args, group_u_num, group_i_num, user_group,
                    item_group):  # ndcg is wrong ,recall and precision is correct
    model.eval()
    z = model.get_emb()
    item_emb = z[item_min:]

    rating_list = [[] for _ in range(len(group_i_num))]
    groundTrue_list = []

    max_K = max(args.topks)

    for batch in test_loader:
        user_id, groundTrue = batch
        user_emb = z[user_id]
        ranking_score = user_emb.matmul(item_emb.t())

        for idx, user in enumerate(user_id):
            ranking_score[idx][np.array(user_seq[user]) - item_min] = -np.inf

        _, rec = torch.topk(ranking_score, max_K)  # return the idx
        rec = rec.detach().cpu().numpy()
        rating = [[[] for _ in range(len(user_id))] for _ in range(len(group_i_num))]
        for i in range(len(user_id)):
            for r in rec[i]:
                # item_group[i - item_min]  belongs to ~ group
                rating[item_group[r]][i].append(r)

        for i in range(len(rating)):
            for j in range(len(rating[i])):
                rating[i][j].extend([[-1]] * (20 - len(rating[i][j])))
            rating_list[i].append(np.array(rating[i]))

        groundTrue_list.append(groundTrue)

    res = []
    for i in range(len(group_i_num)):
        result = []
        r = get_test(rating_list[i], groundTrue_list, args)
        for k, v in r.items():
            result.append(round(v[0], 6))
        res.append(result)

    print("item_group_precision:", end=" ")
    print([res[i][0] for i in range(len(group_i_num))])

    print("item_group_recall:", end=" ")
    print([res[i][1] for i in range(len(group_i_num))])

    # recall_whole = sum([res[i][1] for i in range(10)])
    # precision_whole = sum([res[i][0] for i in range(10)])
    return res


def test(test_loader, model, item_min, user_seq, args):
    model.eval()
    z = model.get_emb()

    item_emb = z[item_min:]

    rating_list = []
    groundTrue_list = []

    max_K = max(args.topks)
    # print("\n==> Testing...")
    for batch in test_loader:
        user_id, groundTrue = batch
        user_emb = z[user_id]
        ranking_score = user_emb.matmul(item_emb.t())

        for idx, user in enumerate(user_id):
            ranking_score[idx][np.array(user_seq[user]) - item_min] = -np.inf

        _, rec = torch.topk(ranking_score, max_K)  # return the idx
        rec = rec.detach().cpu().numpy()

        rating_list.append(rec)
        groundTrue_list.append(groundTrue)

    result = get_test(rating_list, groundTrue_list, args)
    for k, v in result.items():
        result[k] = v[0]

    return result


def train(train_loader, model, optimizer, args):
    model.train()
    avg_loss = AverageMeter()
    avg_re_loss = AverageMeter()
    avg_reg_loss = AverageMeter()
    for i, batch in enumerate(train_loader):
        # if i > 329:
        #     b = 0
        # if i == 10:
        #     break
        # # lightgcn nce
        if args.lossf == "nce":
            user, item = batch
            bsz = len(user)
            loss, reconstruct_loss, reg_loss = model(user.to(torch.device(args.gpu)),
                                                     item.to(torch.device(args.gpu)), None)

        # lightgcn bpr
        if args.lossf == "bpr":
            user, item, item_j = batch
            bsz = len(user)
            loss, reconstruct_loss, reg_loss = model(user.to(torch.device(args.gpu)), item.to(torch.device(args.gpu)),
                                                     item_j.to(torch.device(args.gpu)))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        avg_loss.update(loss.item(), bsz)
        avg_re_loss.update(reconstruct_loss.item(), bsz)
        avg_reg_loss.update(reg_loss.item(), bsz)
    return avg_loss.avg, avg_re_loss.avg, avg_reg_loss.avg
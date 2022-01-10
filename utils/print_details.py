def emb_pruning_detail(weight_mask_fixed, item_min, user_group, group_u_num, item_group, group_i_num):
    cut_per_u = []
    for i in range(len(group_u_num)):
        a = weight_mask_fixed[:item_min][user_group == i]
        cut_per_u.append(round((a.sum() / a.shape[0] / a.shape[1]).item(), 4))
    cut_per_i = []
    for i in range(len(group_i_num)):
        b = weight_mask_fixed[item_min:][item_group == i]
        cut_per_i.append(round((b.sum() / b.shape[0] / b.shape[1]).item(), 4))
    print("emb_cut_user_percent:", end="  ")
    print(cut_per_u)
    print("emb_cut_item_percent:", end="  ")
    print(cut_per_i)

def adj_pruning_detail(g, item_min, user_group, group_u_num, item_group, group_i_num):
    # adj pruning detail
    cut_per_u = []
    degs = g.out_degrees().float().clamp(min=1)
    for i in range(len(group_u_num)):
        cut_per_u.append(round((degs[:item_min][user_group == i].sum( ) /group_u_num[i]).item(), 4))
    cut_per_i = []
    for i in range(len(group_i_num)):
        cut_per_i.append(round((degs[item_min:][item_group == i].sum( ) /group_i_num[i]).item(), 4))
    print("adj_cut_user_percent:", end="  ")
    print(cut_per_u)
    print("adj_cut_item_percent:", end="  ")
    print(cut_per_i)
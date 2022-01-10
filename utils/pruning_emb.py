# pruning detail
import torch
import random

def subgradient_update_mask_emb(model, args):
    model.embedding.weight.grad.data.add_(args.s1 * torch.sign(model.embedding.weight.data))

def print_sparsity_emb(model):  #values is 1
    weight_nonzero = model.weight_nonzero
    weight_mask_nonzero = model.weight_mask_fixed.data.sum().item()
    emb_spar = weight_mask_nonzero * 100 / weight_nonzero
    print("Sparsity: Emb:[{:.2f}%]".format(emb_spar))
    return emb_spar

def print_sparsity_emb_user(model, item_min): #values is 1
    weight_nonzero = model.weight_nonzero_user
    weight_mask_nonzero = model.weight_mask_fixed.data[:item_min].sum().item()
    emb_spar = weight_mask_nonzero * 100 / weight_nonzero
    print("Sparsity: Emb_user:[{:.2f}%]".format(emb_spar))
    return emb_spar

def print_sparsity_emb_item(model, item_min): #values is 1
    weight_nonzero = model.weight_nonzero_item
    weight_mask_nonzero = model.weight_mask_fixed.data[item_min:].sum().item()
    emb_spar = weight_mask_nonzero * 100 / weight_nonzero
    print("Sparsity: Emb_item:[{:.2f}%]".format(emb_spar))
    return emb_spar

#/////////////////////////////////////////////////////////////////////////////

def get_each_mask(mask_weight_tensor, threshold):  #
    ones = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor)
    mask = torch.where(mask_weight_tensor.abs() > threshold, ones, zeros)
    return mask

def get_mask_distribution(mat, if_numpy=True):  # dedao fei 0 de xiangliang
    wei_mask_tensor = mat.flatten()
    nonzero = torch.abs(wei_mask_tensor) > 0
    wei_mask_tensor = wei_mask_tensor[nonzero]   # 13264 - 2708

    if if_numpy:
        return wei_mask_tensor.detach().cpu().numpy()
    else:
        return wei_mask_tensor.detach().cpu()
#/////////////////////////////////////////////////////////////////////////////

def get_final_mask_epoch_emb(model, rewind_weight, args):
    wei_percent = args.pruning_percent_wei
    mat = model.embedding.weight * model.weight_mask_fixed.data
    wei_mask = get_mask_distribution(mat, if_numpy=False)

    wei_total = wei_mask.shape[0]
    ### sort
    wei_y, wei_i = torch.sort(wei_mask.abs())
    ### get threshold
    wei_thre_index = int(wei_total * wei_percent)
    wei_thre = wei_y[wei_thre_index]

    rewind_weight['weight_mask_fixed'] = get_each_mask(mat, wei_thre)
    # wei_spar = rewind_weight['weight_mask_fixed'].sum() / model.weight_nonzero
    return rewind_weight

#/////////////////////////////////////////////////////////////////////////////

#  part user and item
def get_final_mask_epoch_emb_user(model, rewind_weight, args, item_min):
    wei_percent = args.pruning_percent_wei
    mat = model.embedding.weight[:item_min] * model.weight_mask_fixed.data[:item_min]
    wei_mask = get_mask_distribution(mat, if_numpy=False)

    wei_total = wei_mask.shape[0]
    ### sort
    wei_y, wei_i = torch.sort(wei_mask.abs())
    ### get threshold
    wei_thre_index = int(wei_total * wei_percent)
    wei_thre = wei_y[wei_thre_index]

    rewind_weight['weight_mask_fixed'][:item_min] = get_each_mask(mat, wei_thre)
    # wei_spar = rewind_weight['weight_mask_fixed'][:item_min].sum() / model.weight_nonzero
    return rewind_weight

#/////////////////////////////////////////////////////////////////////////////

def get_final_mask_epoch_emb_item(model, rewind_weight, args, item_min):
    wei_percent = args.pruning_percent_wei
    mat = model.embedding.weight[item_min:] * model.weight_mask_fixed.data[item_min:]
    wei_mask = get_mask_distribution(mat, if_numpy=False)

    wei_total = wei_mask.shape[0]
    ### sort
    wei_y, wei_i = torch.sort(wei_mask.abs())
    ### get threshold
    wei_thre_index = int(wei_total * wei_percent)
    wei_thre = wei_y[wei_thre_index]

    rewind_weight['weight_mask_fixed'][item_min:] = get_each_mask(mat, wei_thre)
    # wei_spar = rewind_weight['weight_mask_fixed'][item_min:].sum() / model.weight_nonzero
    return rewind_weight

#/////////////////////////////////////////////////////////////////////////////

def get_final_mask_epoch_per_IDemb(model, rewind_weight, args):
    wei_percent = args.pruning_percent_wei
    mat = model.embedding.weight * model.weight_mask_fixed.data
    for i in range(len(mat)):
        wei_mask_tensor = mat[i]
        nonzero = torch.abs(wei_mask_tensor) > 0
        wei_mask = wei_mask_tensor[nonzero].detach().cpu()  # 13264 - 2708
        wei_total = wei_mask.shape[0]
        ### sort
        wei_y, wei_i = torch.sort(wei_mask.abs())
        ### get threshold
        wei_thre_index = int(wei_total * wei_percent)
        wei_thre = wei_y[wei_thre_index]
        ones = torch.ones_like(wei_mask_tensor)
        zeros = torch.zeros_like(wei_mask_tensor)
        rewind_weight['weight_mask_fixed'][i] = torch.where(wei_mask_tensor.abs() > wei_thre, ones, zeros)
    return rewind_weight

#/////////////////////////////////////////////////////////////////////////////

def get_final_mask_epoch_emb_grad(model, grad, rewind_weight, args):
    wei_percent = args.pruning_percent_wei
    mat = grad * model.weight_mask_fixed.data
    wei_mask = get_mask_distribution(mat, if_numpy=False)

    wei_total = wei_mask.shape[0]
    ### sort
    wei_y, wei_i = torch.sort(wei_mask.abs())
    ### get threshold
    wei_thre_index = int(wei_total * wei_percent)
    wei_thre = wei_y[wei_thre_index]

    rewind_weight['weight_mask_fixed'] = get_each_mask(mat, wei_thre)
    return rewind_weight

#/////////////////////////////////////////////////////////////////////////////

def random_pruning_emb(model, wei_percent):
    wei_nonzero = model.embedding.weight.nonzero()
    wei_total = wei_nonzero.shape[0]

    wei_pruned_num = int(wei_total * wei_percent)
    wei_index = random.sample([i for i in range(wei_total)], wei_pruned_num)

    wei_pruned = wei_nonzero[wei_index].tolist()

    for i, j in wei_pruned:
        model.weight_mask_fixed[i][j] = 0

    emb_spar = (wei_total-wei_pruned_num) * 100 / wei_total
    print("Sparsity: Emb:[{:.2f}%]".format(emb_spar))


#/////////////////////////////////////////////////////////////////////////////

def print_emb_mask_distribution(mask):
    res = []
    mask_size = mask.shape[0] * mask.shape[1]
    num = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ones = torch.ones_like(mask)
    zeros = torch.zeros_like(mask)
    for i in range(len(num)-1):
        res.append(round(((torch.where((mask.abs() >= num[i]) & (mask.abs() < num[i+1]), ones, zeros).sum())/mask_size/100).item(), 4))
    res.append(1-sum(res))
    print("emb_mask_destribution: ", end=" ")
    print(res)




import torch
import torch.nn as nn
import math
import copy
import pdb


def zero_out_weights_for_tune_head(var,):
    for name, param in var.named_parameters():
        # 检查参数名称是否与 Cross Attention 层的权重匹配
        if ('cross_attn.proj' in name):
            # 将权重置为0
            with torch.no_grad():  # 确保不会在这个操作中跟踪梯度
                param.fill_(0.0)
    print("Proj layer of cross attention weights have been zeroed out.")



def zero_out_cross_attention_weights(model):
    for name, param in model.named_parameters():
        # 检查参数名称是否与 Cross Attention 层的权重匹配
        if ('cross_attn.proj' in name):
            # 将权重置为0
            with torch.no_grad():  # 确保不会在这个操作中跟踪梯度
                param.fill_(0.0)
    print("Proj layer of cross attention weights have been zeroed out.")


def apply_lvl_emb_and_pos_1LC(args,state_dict,patch_nums):
    '''适配256的weight到512'''
    print('reusing lvl_embed and pos_1LC for adaptation to higher resolutions...')
    C=state_dict['word_embed.weight'].shape[0]
    init_std = math.sqrt(1 / C / 3)

    if 'lvl_embed.weight' in state_dict.keys():
        # lvl_emb
        old_lvl_emb = state_dict['lvl_embed.weight']
        # # 这样初始化
        # new_lvl_emb=torch.cat([old_lvl_emb]+
        #                       ([old_lvl_emb[-1].unsqueeze(0)]*(len(patch_nums)-old_lvl_emb.shape[0])))

        # 还是这样初始化
        new_lvl_emb=torch.empty(len(patch_nums),C)#L,C
        nn.init.trunc_normal_(new_lvl_emb.data, mean=0, std=init_std)
        new_lvl_emb[:old_lvl_emb.shape[0]]=old_lvl_emb
        if not args==None:
            if args.saln==True:
                new_lvl_emb[old_lvl_emb.shape[0]::]=(old_lvl_emb[-1].unsqueeze(0).repeat(len(patch_nums)-old_lvl_emb.shape[0],1))
        state_dict['lvl_embed.weight'] = new_lvl_emb

    if 'pos_1LC' in state_dict.keys():
        # pos_1LC
        _,L,C=state_dict['pos_1LC'].shape
        new_pos_1LC=torch.empty(1,sum(pn**2 for pn in patch_nums),C)
        nn.init.trunc_normal_(new_pos_1LC.data, mean=0, std=init_std)
        new_pos_1LC[:,:L]=state_dict['pos_1LC']#不支持针对saln的修改
        state_dict['pos_1LC']=new_pos_1LC
    return state_dict
    # filtered_checkpoint = {k: v for k, v in checkpoint.items() if not (k.startswith('lvl_embed') or k.startswith('pos_1LC'))}


def apply_codebook(state_dict,vocab_size):
    '''适配256的weight到512'''
    print('reusing codebook for adaptation to higher resolutions...')


    if 'quantize.embedding.weight' in state_dict.keys():
        # pos_1LC
        V_old,Cvae=state_dict['quantize.embedding.weight'].shape
        new_embed=torch.empty(vocab_size, Cvae)
        nn.init.trunc_normal_(new_embed, mean=0, std=0.1)
        new_embed[:V_old,:]=state_dict['quantize.embedding.weight']#不支持针对saln的修改
        state_dict['quantize.embedding.weight']=new_embed
    
    if 'quantize.ema_vocab_hit_SV' in state_dict.keys():
        # pos_1LC
        num_scales,V_old=state_dict['quantize.ema_vocab_hit_SV'].shape
        new_ema_vocab=torch.empty(num_scales,vocab_size)
        nn.init.trunc_normal_(new_ema_vocab, mean=0.1, std=0.1)
        new_ema_vocab[:,:V_old]=state_dict['quantize.ema_vocab_hit_SV']#不支持针对saln的修改
        state_dict['quantize.ema_vocab_hit_SV']=new_ema_vocab
    return state_dict
    # filtered_checkpoint = {k: v for k, v in checkpoint.items() if not (k.startswith('lvl_embed') or k.startswith('pos_1LC'))}

def apply_quant_resi(state_dict,share_quant_resi):
    if share_quant_resi==14 and not 'quantize.quant_resi.qresi_ls.5.weight' in state_dict.keys():
        idx=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 3, 3, 3]
        new_state_dict=copy.deepcopy(state_dict)
        for i in range(len(idx)):
            new_state_dict['quantize.quant_resi.qresi_ls.%d.weight'%i]=state_dict['quantize.quant_resi.qresi_ls.%d.weight'%idx[i]]
            new_state_dict['quantize.quant_resi.qresi_ls.%d.bias'%i]=state_dict['quantize.quant_resi.qresi_ls.%d.bias'%idx[i]]
        return new_state_dict
    else:
        return state_dict


def apply_embedding(state_dict,num_scales):
    if 'quantize.embedding.weight' in state_dict.keys():
        for i in range(num_scales):
            state_dict['quantize.embedding.%d.weight'%i]=state_dict['quantize.embedding.weight']
        state_dict.pop('quantize.embedding.weight')
    return state_dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
# import matplotlib.pyplot as plt
import os
# import seaborn as sns
from models.helpers import DropPath, drop_path
from models.embed_rope import apply_rotary_emb


# this file only provides the 3 blocks used in VAR transformer
__all__ = ['FFN', 'AdaLNSelfAttn', 'AdaLNBeforeHead']


# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError: pass
# automatically import faster attention implementations
try: from xformers.ops import memory_efficient_attention
except ImportError: pass
try: from flash_attn import flash_attn_func              # qkv: BLHc, ret: BLHcq
except ImportError: pass
# try: from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
# except ImportError:

patch_nums =(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
begin_ends=[];cur=0
for i, pn in enumerate(patch_nums):
    begin_ends.append((cur, cur+pn ** 2))
    cur += pn ** 2

cross_attn_maps={}

def save_cross_attn(attn_map,save_name):#[H,L,77]
    # attn=torch.norm(attn_map,dim=0,keepdim=False)#[L,77]
    attn=torch.mean(attn_map,dim=0,keepdim=False)#[L,77]
    for idx in range(10):
        # Select the attention map for the current head
        attn_ = attn[:,idx]  # Shape: [L]

        fig,axes=plt.subplots(1,10,figsize=(45, 6))
        for scale,begin_end in enumerate(begin_ends):
            begin,end=begin_end
            attn_cur=attn_[begin:end].reshape(patch_nums[scale],patch_nums[scale])
            ax = axes[scale]  # 获取第 head_idx 个子图
            sns.heatmap(attn_cur.cpu().detach().numpy(), cmap='viridis', cbar=True, ax=ax)
            ax.set_title(f"scale_{scale}")

        # Set the title
        plt.title(f"{save_name}_{idx}-th_text")

        # Save the heatmap to a file
        filename = f"./oup_attn_maps/{save_name}_{idx}-th_text.png"
        plt.savefig(filename)
        print(filename)
        plt.close()


def save_attn_map(attn, layer_num, attn_type="cross",attn_mask=None):
    # Create the output directory if it doesn't exist
    os.makedirs("./oup_attn_maps/", exist_ok=True)
    attn_softmax = attn.softmax(dim=-1)

    # Normalize the softmax result to [0, 1] (min-max scaling)
    attn_min = attn_softmax.min(dim=-1, keepdim=True)[0]  # Get the minimum value along the -1 dimension
    attn_max = attn_softmax.max(dim=-1, keepdim=True)[0]  # Get the maximum value along the -1 dimension

    # Perform min-max normalization
    attn = (attn_softmax - attn_min) / (attn_max - attn_min)

    # Only proceed if the batch size is 1
    if attn.shape[0] != 1:
        attn=attn[0].unsqueeze(0)
        # raise ValueError("Batch size should be 1 for saving heatmaps.")

    # Squeeze the batch dimension
    attn = attn.squeeze(0)  # Shape: [H, L, L]

    if attn_type=='cross':
        pass
        # cross_attn_maps['%d'%layer_num]=attn
        # if layer_num==15:
        #     mean_attn=torch.mean(torch.stack([cross_attn_maps[key] for key in cross_attn_maps.keys()],dim=0),dim=0,keepdim=True)
        #     save_cross_attn(attn,f"layer_{layer_num}_{attn_type}")#[H,L,77]
    else:
        # Calculate the square root of the mean of squared attention across all heads
        sq_sum_attn_map = torch.sqrt(torch.mean(attn ** 2, dim=0)).add_(attn_mask[0,0]).cpu().detach().numpy()  # Shape: [L, L]

        # Plotting the heatmap
        ''' 
        plt.figure(figsize=(8, 6))
        sns.heatmap(sq_sum_attn_map, cmap='coolwarm', cbar=True, vmin=0, vmax=1)

        # Remove the axis ticks (both x and y axes)
        plt.gca().set_xticks([])  # Remove x-axis ticks
        plt.gca().set_yticks([])  # Remove y-axis ticks

        # Set the title
        plt.title(f"layer_{layer_num}_{attn_type}_square_sum_head")

        # Save the heatmap as SVG
        filename = f"./oup_attn_maps/layer_{layer_num}_{attn_type}_square_sum_head.png"
        plt.savefig(filename)
        print(filename)
        plt.close()
        '''

def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0,layer_id=0,type='self'):
    attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
    if attn_mask is not None: 
        attn.add_(attn_mask)#-inf在softmax后会变成0
        # save_attn_map(attn,layer_id,type,attn_mask)
    return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value#全-inf的softmax会变成nan


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., fused_if_available=True):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
    
    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, bias1=self.fc1.bias, bias2=self.fc2.bias,
                activation='gelu_approx', save_pre_act=self.training, return_residual=False, checkpoint_lvl=0,
                heuristic=0, process_group=None,
            ))
        else:
            return self.drop(self.fc2( self.act(self.fc1(x)) ))
    
    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'


class Attention(nn.Module):
    def __init__(
        self, block_idx, embed_dim=768, is_cross=False, in_dim_cross=77, num_heads=12,
        attn_drop=0., proj_drop=0., attn_l2_norm=False, flash_if_available=True,
        rotary_pos_emb=False
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim, self.embed_dim = block_idx, num_heads, embed_dim // num_heads, embed_dim  # =64
        self.attn_l2_norm = attn_l2_norm
        self.is_cross=is_cross
        self.rotary_pos_emb=rotary_pos_emb
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
            
        if is_cross:
            self.mat_kv = nn.Linear(in_dim_cross, embed_dim * 2, bias=False)
            self.mat_q=nn.Linear(embed_dim, embed_dim, bias=False)
        else:
            self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)

        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None and not self.is_cross
        # if self.rotary_pos_emb:
        #     self.freqs_cis=precompute_freqs_cis()
        
        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None
    
    def kv_caching(self, enable: bool): self.caching, self.cached_k, self.cached_v = enable, None, None
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, attn_bias=None, encoder_hidden_states=None, freqs_cis=None,layer_id=0):#freqs_cis:(L,C//head)
        B, L, C = x.shape
        if not self.is_cross:
            qkv = F.linear(input=x, weight=self.mat_qkv.weight, 
                           bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
                           ).view(B, L, 3, self.num_heads, self.head_dim)
                    # qkv: BL3Hc        #x是fp32，qkv会自动转为fp16
            using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
            if using_flash or self.using_xform:
                q, k, v = qkv.unbind(dim=2); dim_cat = 1   # q or k or v: BLHc, freqs_cis: BLc//2
                if self.rotary_pos_emb:
                    q,k=apply_rotary_emb(q.transpose(1,2),k.transpose(1,2),freqs_cis=freqs_cis)
                    q=q.transpose(1,2);k=k.transpose(1,2)
            else:
                q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); dim_cat = 2 #(b,l,3,num_h,h_dim)->(3,b,num_h,l,h_dim)->(b,h,l,c)
                if self.rotary_pos_emb:
                    q,k=apply_rotary_emb(q,k,freqs_cis=freqs_cis)
            # q or k or v: BHLc
        else:
            assert encoder_hidden_states is not None, "hidden states of text prompts needed to be input as 'encoder_hidden_states'"
            q=F.linear(input=x, weight=self.mat_q.weight, #对哦，这不同scale的大小都不一样，cross-attention能一样吗
                           bias=self.q_bias
                           ).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)#(b,h,77,c)
            kv=F.linear(input=encoder_hidden_states, weight=self.mat_kv.weight, 
                           bias=torch.cat((self.zero_k_bias, self.v_bias))
                           ).view(B, -1, 2, self.num_heads, self.head_dim)#(b,l,2,h,c)
            using_flash = self.using_flash and attn_bias is None and kv.dtype != torch.float32
            if using_flash or self.using_xform: 
                k, v = kv.unbind(dim=2); dim_cat = 1   # q or k or v: BLHc
                q=q.permute(0,2,1,3)#BHLc->BLHc
            else: k, v = kv.permute(2, 0, 3, 1, 4).unbind(dim=0); dim_cat = 2               # q or k or v: BHLc
        
        if self.attn_l2_norm:#qk norm
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if (using_flash or self.using_xform):
                scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            # q = normalize(q, dim=-1).mul(scale_mul.to(v.dtype)).to(v.dtype)
            # k = normalize(k, dim=-1).to(v.dtype)#我自己写的训出来是nan，很奇怪，懒得debug了，先这样吧
            q = F.normalize(q, dim=-1).mul(scale_mul).to(v.dtype)
            k = F.normalize(k, dim=-1).to(v.dtype)
        
        if self.caching:
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat); v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
        
        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash and (not self.is_cross):
            assert attn_bias is None and qkv.dtype != torch.float32
            oup = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
        elif self.using_xform:
            if not self.is_cross:
                oup = memory_efficient_attention(q, k, v, attn_bias=None if attn_bias is None else attn_bias.to(dtype=q.dtype).expand(B, self.num_heads, -1, -1), p=dropout_p, scale=self.scale).view(B, L, C)
            else:#(B,1,1,77)->(B,h,l1,l2)?
                oup = memory_efficient_attention(q, k, v, attn_bias=attn_bias.expand(B, self.num_heads, q.shape[1], -1), p=dropout_p, scale=self.scale).view(B, L, C)
        else:
            # oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p,
            #                 layer_id=layer_id,type='cross' if self.is_cross else 'self').transpose(1, 2).reshape(B, L, C)
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                oup=F.scaled_dot_product_attention(query=q,key=k,value=v,#b,h,l,c
                                                   attn_mask=attn_bias,
                                                   scale=self.scale,dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
        return self.proj_drop(self.proj(oup))#到这一步之前oup是fp16

    def extra_repr(self) -> str:
        return f'using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}'


class AdaLNSelfAttn(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
        num_heads, in_dim_cross=77, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
        flash_if_available=False, fused_if_available=True,
    ):
        super(AdaLNSelfAttn, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = Attention(block_idx=block_idx, embed_dim=embed_dim, 
                              num_heads=num_heads, attn_drop=attn_drop, 
                              proj_drop=drop,attn_l2_norm=attn_l2_norm,
                              flash_if_available=flash_if_available)
        self.cross_attn = Attention(block_idx=block_idx, embed_dim=embed_dim, 
                                    is_cross=True, in_dim_cross=in_dim_cross,
                              num_heads=num_heads, attn_drop=attn_drop, 
                              proj_drop=drop, attn_l2_norm=attn_l2_norm,
                              flash_if_available=flash_if_available)
        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=drop, fused_if_available=fused_if_available)
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
        
        self.fused_add_norm_fn = None
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, cond_BD, attn_bias, encoder_hidden_states,encoder_attention_mask):   # C: embed_dim, D: cond_dim
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        x = x + self.drop_path(self.attn( self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1), attn_bias=attn_bias ).mul_(gamma1))

        if encoder_hidden_states is not None:
            x = self.cross_attn(x,
                                encoder_hidden_states=encoder_hidden_states,
                                attn_bias=encoder_attention_mask)[0] + x

        x = x + self.drop_path(self.ffn( self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed when FusedMLP is used
        return x
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}'
    

class AttnBlock(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
        num_heads, in_dim_cross=1024, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
        enable_cross=True, cross_attn_ln=True,
        flash_if_available=False, fused_if_available=True,rotary_pos_emb=True
    ):
        super(AttnBlock, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = Attention(block_idx=block_idx, embed_dim=embed_dim, 
                              num_heads=num_heads, attn_drop=attn_drop, 
                              proj_drop=drop,attn_l2_norm=attn_l2_norm,
                              flash_if_available=flash_if_available,
                              rotary_pos_emb=rotary_pos_emb)
        if enable_cross==True:
            self.cross_attn = Attention(block_idx=block_idx, embed_dim=embed_dim, 
                                        is_cross=True, in_dim_cross=in_dim_cross,
                                num_heads=num_heads, attn_drop=attn_drop, 
                                proj_drop=drop, attn_l2_norm=attn_l2_norm,
                                flash_if_available=flash_if_available)
        else:
            print("no cross attention...")
        self.enable_cross=enable_cross
        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=drop, fused_if_available=fused_if_available)
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        self.cross_attn_ln = cross_attn_ln
        print("cross_attention_layernorm: ",self.cross_attn_ln)
        
        self.fused_add_norm_fn = None
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, cond_BD, attn_bias, encoder_hidden_states=None,encoder_attention_mask=None,freqs_cis=None,layer_id=0):   # C: embed_dim, D: cond_dim
        # freq_cis:(L,C//head)
        if self.shared_aln==True and cond_BD!=None:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
        else:
            gamma1=1; gamma2=1; scale1=0; scale2=0; shift1=0; shift2=0

        x = x + self.drop_path(
                    self.attn(self.ln_wo_grad(x).mul(scale1+1).add_(shift1), 
                              attn_bias=attn_bias, freqs_cis=freqs_cis,
                              layer_id=layer_id
                            ).mul_(gamma1))
        # x = x + self.drop_path(self.attn( self.ln_wo_grad(x), attn_bias=attn_bias, freqs_cis=freqs_cis))

        if (encoder_hidden_states is not None) and (self.enable_cross==True):
            if self.cross_attn_ln:
                x = self.cross_attn(self.ln_wo_grad(x),
                                    encoder_hidden_states=encoder_hidden_states,
                                    attn_bias=encoder_attention_mask,
                                    layer_id=layer_id)[0] + x
            else:
                x = self.cross_attn(x,
                                    encoder_hidden_states=encoder_hidden_states,
                                    attn_bias=encoder_attention_mask)[0] + x
        # x = x + self.drop_path(self.ffn( self.ln_wo_grad(x))) # this mul(gamma2) cannot be in-placed when FusedMLP is used
        x = x + self.drop_path(self.ffn( self.ln_wo_grad(x).mul(scale2+1).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed when FusedMLP is used
        return x
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}'


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2*C))
        self.zero_out_cross_weights()

    
    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        scale, shift = torch.chunk(self.ada_lin(cond_BD),2,dim=-1)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
        # return x_BLC.mul(scale.add(1)).add_(shift)
    

    def zero_out_cross_weights(self):
        for name, param in self.ada_lin.named_parameters():
            with torch.no_grad():  # 确保不会在这个操作中跟踪梯度
                param.fill_(0.0)
        print("Proj layer of cross attention weights have been zeroed out.")


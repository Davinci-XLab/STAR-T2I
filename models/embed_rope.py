import torch
import torch.nn as nn
from functools import partial
from PIL import Image
import numpy as np
import pdb

import torch.nn.functional as F

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    depth = freqs.shape[1]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_cis = []
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_cis.append(torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y))
                    
    return torch.cat(freqs_cis, dim=-1)


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0, normalize=32):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))#从1到1/theta的逐渐衰减的长为dim//4的序列
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y,normalize=normalize)#指定pos的位置
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)#每个pos每个通道
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    #单位长度的，以freqs_x作辐角的向量:dim层，第一层幅角是1～end_x,第二层幅角是(1～end_x)*freqs_x[1]
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)#一半channel表示x，一半表示y

# def init_t_xy(end_x: int, end_y: int):#如果要归一化的话，可以在这搞，但是不知道会不会对结果有影响
#     t = torch.arange(end_x * end_y, dtype=torch.float32)
#     t_x = (t % end_x).float()
#     t_y = torch.div(t, end_x, rounding_mode='floor').float()
#     return t_x, t_y


def init_t_xy(end_x: int, end_y: int, normalize=-1):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    if not normalize==-1 and t_x.shape[0]>1:
        t_x=t_x/t_x.max()*normalize
        t_y=t_y/t_y.max()*normalize
    return t_x, t_y


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):#L,C
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    # elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):#H,L,C
    #     shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[0], x.shape[-2], x.shape[-1]):#B,L,C
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
        shape[0]=x.shape[0]
        
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))#两个dim组成一个虚数
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)#拆成实部和虚部
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

if __name__ == '__main__':
    import torch.nn as nn
    dtype=torch.float16#torch.bfloat16
    embed_dim=512
    num_heads=embed_dim//64
    rope_theta=100
    pn_list=[16,24,32]
    values_list=[]
    for pn in pn_list:
        compute_cis = partial(compute_axial_cis, dim=64, theta=rope_theta, normalize=32)
        freqs_cis = compute_cis(end_x = pn, end_y = pn)#[pn*pn,dim//2]

        q=torch.ones([1,num_heads,pn**2,64]).to(dtype);k=torch.ones([1,num_heads,pn**2,64]).to(dtype)
        v=k=torch.ones([1,num_heads,pn**2,64]).to(dtype)
        q,k=apply_rotary_emb(q,k,freqs_cis=freqs_cis)#q,k:bhlc

        attn1 = q.float() @ k.float().transpose(-2, -1) # BHLc @ BHcL => BHLL
        attn1=attn1.to(dtype)
        print(attn1.max(),attn1.min())
        print(attn1[0,0,0].max(),attn1[0,0,0].min())

        attn1_oup=(attn1[0,0,0].reshape(pn,pn).float()).numpy()#.astype(np.uint8)#BHLc
        values_list.append(np.diag(attn1_oup))
        # Image.fromarray(attn1_oup).save('./dim12.png')
        # Image.fromarray(attn1_oup[0].reshape(pn,pn)).save('./02_attn.png')
    print(values_list)
    # pn=8
    # compute_cis = partial(compute_axial_cis, dim=64, theta=rope_theta)
    # freqs_cis = compute_cis(end_x = pn, end_y = pn)#[pn*pn,dim//2]

    # q=torch.ones([1,num_heads,pn**2,64]);k=torch.ones([1,num_heads,pn**2,64])
    # q,k=apply_rotary_emb(q,k,freqs_cis=freqs_cis)#q,k:bhlc

    # attn2 = q @ k.transpose(-2, -1) # BHLc @ BHcL => BHLL
    import matplotlib.pyplot as plt

    def plot_value_list(value_list):
        num_items = len(value_list)
        
        # 创建纵向排列的子图，num_items行，1列
        fig, axes = plt.subplots(num_items, 1, figsize=(6, 3 * num_items))

        # 处理只有一个子图的情况
        if num_items == 1:
            axes = [axes]

        # 遍历列表中的每个item
        for i, values in enumerate(value_list):
            x_max=len(values)
            axes[i].plot([idx+1 for idx in range(x_max-1)],values[1:], linewidth=5,color='darkblue', alpha=0.3)
            axes[i].plot([idx+1 for idx in range(x_max-1)],values[1:], linewidth=1,color='darkblue')
            axes[i].set_xlim(0, x_max)  # 设置 x 轴的范围
            # axes[i].set_title(f'Subplot {i + 1}',fontsize=14)
            # axes[i].set_ylabel('Values',fontsize=12)

            # 移除右边和上边的框
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].tick_params(axis='both', which='major', labelsize=12)
        # axes[-1].set_xlabel('X-axis',fontsize=12)

        plt.tight_layout()
        # plt.savefig('./models/rope2.png')
        plt.savefig('./models/rope2.png',transparent=True)


    plot_value_list(values_list)

import math
from functools import partial
from typing import Optional, Tuple, Union
from utils.mask_utils import Scheduler
import pdb

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.basic_var import AttnBlock,Attention
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2
from models.quant import VectorQuantizer2
from models.embed_rope import compute_axial_cis
import numpy as np
from utils.sample_subset import prob_subset_selection
    

def prepare_attn_mask(encoder_attention_mask):
    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = torch.where(encoder_attention_mask==1,0,-torch.inf)
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(1)#(B,1,1,77)(b,h,c,l)
    return encoder_attention_mask

class SharedAdaLin(nn.Linear):#1,L,C
    def forward(self, cond_BD):
        B,L,C_=cond_BD.shape
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(B, L, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        enable_cross=True,
        in_dim_cross=1024,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
        noise_sampling=False,
        rotary_pos_emb=True,
        absolute_lvl_emb=True,
        rope_theta=100.0,rope_norm=32,
        drop_scale_length=None,
        enable_logit_norm=True,
        enable_adaptive_norm=True,
        train_mode='head_only',
        sample_from_idx=9
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        self.rotary_pos_emb=rotary_pos_emb
        self.absolute_lvl_emb=absolute_lvl_emb
        self.shared_aln=shared_aln
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.drop_scale_length=drop_scale_length
        if drop_scale_length==None:
            print('no drop, using full self-attention for training...')
        else:
            print('force self-attention map to size ',drop_scale_length)
            self.drop_start_idx=13
            self.drop_start=self.begin_ends[self.drop_start_idx][0]
            self.num_tokens_to_drop=self.L-self.drop_scale_length

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)#每个class的embed：torch.Size([1001, 1024])
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))#第一层token的初始偏置：torch.Size([1, 1, 1024])，影响不大，仍能产生合理结果
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        if not self.rotary_pos_emb:
            print('using absolute positional encoding...')
        # 3. absolute position embedding
            pos_1LC = []
            for i, pn in enumerate(self.patch_nums):
                pe = torch.empty(1, pn*pn, self.C)
                nn.init.trunc_normal_(pe, mean=0, std=init_std)
                pos_1LC.append(pe)
            pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C：所有层的token的偏置：torch.Size([1, 680, 1024]),去掉会对模型性能有很大影响
            assert tuple(pos_1LC.shape) == (1, self.L, self.C)
            self.pos_1LC = nn.Parameter(pos_1LC)
            self.freqs_cis=None

        else:
            # -----------RoPE----------------------
            # RoPE axiel（TODO:他们还有一个mixed的版本性能更好，似乎是rope+ape插值）
            print('using rotary positional encoding...')
            self.freqs_cis=[]
            self.rope_norm=rope_norm
            self.compute_cis = partial(compute_axial_cis, dim=embed_dim//num_heads, theta=rope_theta, normalize=self.rope_norm)
            for i, pn in enumerate(self.patch_nums):
                freqs_cis = self.compute_cis(end_x = pn, end_y = pn)
                self.freqs_cis.append(freqs_cis)
            self.freqs_cis=torch.cat(self.freqs_cis,dim=0).to(dist.get_device())#(L,C//h)
            # ---------------------------------------

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        self.lvl_1L = dT[:, 0].contiguous().to(dist.get_device())
        
        if self.absolute_lvl_emb:
            print('using absolute level embeding (lvl_embed)...')
            # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
            self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
            nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
            # self.lvl_1LC=torch.cat(
            #                         [self.lvl_embed[i].unsqueeze(0).repeat(patch_nums[i]**2,1).unsqueeze(0) for i in range(len(patch_nums))]
            #                        ,dim=1)#(1,L,C)
        if self.shared_aln:
            print("using shared_adaln...")
            self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
            self.lvl_embed_proj = nn.Linear(self.C*2, self.C)
            self.lvl_embed_adaln = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C))
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AttnBlock(
                cond_dim=self.D, shared_aln=shared_aln, in_dim_cross=in_dim_cross,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                enable_cross=enable_cross,attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
                rotary_pos_emb=rotary_pos_emb
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 这种情况下attn map是正常的
        self.attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L).contiguous().to(dist.get_device())
        print('using casual attention...')
        
        # 6. classifier head
        # self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head_logits = nn.Linear(self.C, self.V)
        self.encoder_proj=nn.Linear(in_dim_cross,embed_dim)
        self.noise_sampling=noise_sampling
        print('if using noise sampling: ',noise_sampling)

        
        # True for 1024 only
        self.enable_logit_norm=enable_logit_norm
        self.enable_adaptive_norm=enable_adaptive_norm
        if self.enable_logit_norm:
            print('enable norm in getting logits...')
            self.logit_norm = norm_layer(embed_dim, elementwise_affine=False)
        if self.enable_adaptive_norm:
            print('enable adaptive norm in getting logits...')
            self.word_embed_head=nn.Linear(self.Cvae, self.C)
            encoder_depth=3
            self.feat_extract_blocks = nn.ModuleList([
                AttnBlock(
                    cond_dim=self.D, shared_aln=shared_aln, in_dim_cross=in_dim_cross,
                    block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                    enable_cross=False,attn_l2_norm=attn_l2_norm,
                    flash_if_available=flash_if_available, fused_if_available=fused_if_available,
                    rotary_pos_emb=rotary_pos_emb
                )
                for block_idx in range(encoder_depth)
            ])
            self.from_idx=sample_from_idx
            self.bg_last,_=self.begin_ends[self.from_idx];_,self.ed_last=self.begin_ends[-1]
            # self.adaptive_norm = AdaLNBeforeHead(embed_dim, self.D, norm_layer=norm_layer)
            length_=self.ed_last-self.bg_last
            # self.attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L).contiguous().to(dist.get_device())
            self.attn_mask=torch.where(d[:,self.bg_last-1:self.ed_last]==dT[...,self.bg_last-1:self.ed_last],0,-torch.inf).reshape(1, 1, length_+1, length_+1).contiguous().to(dist.get_device())
            self.attn_mask[:,:,0]=0;self.attn_mask[...,0]=0
            self.pos_start_last = nn.Parameter(torch.empty(1, self.first_l, embed_dim))
            nn.init.trunc_normal_(self.pos_start_last.data, mean=0, std=init_std)
            self.logit_norm = norm_layer(embed_dim, elementwise_affine=False)

            self.mask_scheduler=Scheduler()
            self.head_logits2 = nn.Linear(self.C, self.V)
            self.encoder_proj2=nn.Linear(in_dim_cross,embed_dim)
            self.head_proj=nn.Linear(2*embed_dim,embed_dim)
            self.feat_drop_enabled=False
            self.stage_2_faster=True
            self.drop_thresh=0.8
            self.lvl_embed_2 = nn.Embedding(len(self.patch_nums), self.C)
            nn.init.trunc_normal_(self.lvl_embed_2.weight.data, mean=0, std=init_std)

        else:
            pass

        self.train_mode=train_mode
        if self.train_mode=='head_only':
            print('train_mode: head_only')
            [p.requires_grad_(False) for p in self.parameters()]
            [p.requires_grad_(False) for p in self.head_logits.parameters()]
            [p.requires_grad_(True) for p in self.lvl_embed_2.parameters()]

            [p.requires_grad_(True) for p in self.word_embed_head.parameters()]
            [p.requires_grad_(True) for p in self.feat_extract_blocks.parameters()]
            self.pos_start_last.requires_grad_(True)
            [p.requires_grad_(True) for p in self.head_logits2.parameters()]
            [p.requires_grad_(True) for p in self.encoder_proj2.parameters()]
            [p.requires_grad_(True) for p in self.head_proj.parameters()]
            self.init_weights(init_adaln=0.5, init_adaln_gamma=5e-5, init_head=0.02, init_std=-1)
        else:
            print('train_mode: all')


# feature=x[:,step,None],prev_emb=prev_emb
    def sample(self,feature,prev_emb=None,attn_bias=None,freqs_cis=None):
        # featre:B,L,1920; embed_real:vqvae的latent feat做embed后的结果
        # 现在需要想的是怎么把logits的sample转成casual的，即之前的tokens的logit会对新的token造成影响
        # 得到所有的logits后，是一个BL*4096的tensor，每个通道表示对应index的logits
        # 两个tokens的概率分布越接近，那么他们对应的交叉熵越小
        # 假设先选择一部分tokens预测，得到index再做个embed，可以转回feature，这个feature再和新的token做attention
        # attention值拿来做adaln的scale和shift，可以学一个条件概率
        if not prev_emb==None:
            feat_=torch.cat([feature,prev_emb],dim=-1)
            feat_=self.head_proj(feat_)
            AttnBlock.forward
            for block in self.feat_extract_blocks:
                # x = b(x=x,
                #     encoder_hidden_states=encoder_hidden_states,
                #     encoder_attention_mask=encoder_attention_mask,
                #     cond_BD=cond_lvl_emb_cur, 
                #     attn_bias=None,
                #     freqs_cis=freqs_cis_cur,
                #     layer_id=i)#当前scale的pn*pn个freq_cis
                feat_ = block(x=feat_,cond_BD=None,attn_bias=attn_bias,freqs_cis=freqs_cis)
            # attn_feat=attn_feat#B,L,1920
        return self.head_logits2(self.logit_norm(feat_.float())).float()
            # return self.head_logits(feature.float()).float()


    def from_logit2emb(self,logits_BlV,t,rng,top_k,top_p,B):#单个token的sample    
        logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]    #(B,l,4096)是每个token和vqvae codebook的4096个token的相似程度

        idx_Bl,conf = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1,return_conf=True)#和哪个token最相似，对应的离散token的idx
        h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae=32，是再把确定的离散token的idx获得一个embed
        return h_BChw,conf

    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        '''mode:train或是inference'''
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual

        if self.enable_logit_norm:
            logits_feature=self.logit_norm(h.float())
        else:
            logits_feature=h.float()

        return self.head_logits(logits_feature).float()



    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        encoder_hidden_states=None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_pool_feat=None,
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
        w_mask=False,
        sample_version='new'
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        encoder_attention_mask=prepare_attn_mask(
            encoder_attention_mask=encoder_attention_mask)#(2*B,77)->(2*B,1,1,77)
        # [0,0,...,-10000,-10000,...]

        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        # if label_B is None:
        #     label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        # elif isinstance(label_B, int):
        #     label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        # TODO：不用imagenet的class，而是用clip的pooling embedding做第一个scale的生成
        # 但这样可能会影响生成的多样性，可以考虑第一个scale加随机噪声，用cross attn给第一个scale加引导信息（但这样可能不能给足够的引导，还是容易出现图文无关）
        # 或许第一个scale可以从pooling层加一个噪声偏置开始
        if not self.noise_sampling:
            sos = self.encoder_proj(encoder_pool_feat)#self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))# (2*B,1024)
        else:
            sos = torch.randn([2*B,self.D])

        if not self.rotary_pos_emb:
            lvl_pos = self.pos_1LC    #lvl_pos是level的pos emb和每层每个token的emb的和（所有的pos emb)
            #实际上是引导模型在不同位置/timestep去关注不同channel
        else:
            lvl_pos=0

        if self.absolute_lvl_emb:
            lvl_pos=lvl_pos + self.lvl_embed(self.lvl_1L)
            cond_lvl_emb=None
        if self.shared_aln:
            cond_lvl_emb=self.lvl_embed_proj(
                    torch.cat([self.lvl_embed(self.lvl_1L),
                               self.lvl_embed(torch.full(self.lvl_1L.shape,self.lvl_1L[0,-1],device=self.lvl_1L.device))],dim=-1))
            cond_lvl_emb = self.lvl_embed_adaln(cond_lvl_emb)#cond_BD是加各种pos emb之前的class_emb

        if (not self.rotary_pos_emb) or (self.absolute_lvl_emb):
            next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        else:
            next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1)
        #pos_start是初始token的pos emb
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])#第一个token有随机性
        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            t = cfg * ratio#不同scale的cfg不一样？
            # last_L = cur_L
            cur_L += pn*pn  #当前层的patch数
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            freqs_cis_cur=self.freqs_cis[cur_L-pn*pn:cur_L,:] if not self.freqs_cis==None else None
            cond_lvl_emb_cur=cond_lvl_emb[:,cur_L-pn*pn:cur_L,...] if not cond_lvl_emb==None else None
            x = next_token_map
            AttnBlock.forward
            for i, b in enumerate(self.blocks):
                x = b(x=x,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    cond_BD=cond_lvl_emb_cur, 
                    attn_bias=None,
                    freqs_cis=freqs_cis_cur,
                    layer_id=i)#当前scale的pn*pn个freq_cis
            # pdb.set_trace()
            if w_mask and si>=self.from_idx:
                emb_list=[]
                embed_Cvae=None
                h_BChw=torch.zeros((B,pn*pn,self.Cvae),device=x.device)
                mask=torch.zeros((B,pn*pn),device=x.device)

                if sample_version=='1024':
                    step_thresh=5;si_thresh=13#step<5；scale index<13的，直接用var输出的tokens
                    self.mask_scheduler.step=15 if si>12 else 8
                else:
                    step_thresh=10000;si_thresh=-1
                    self.mask_scheduler.step=8

                self.mask_scheduler._create_scheduler(patch_size=pn)
                for step in range(self.mask_scheduler.step):
                    _,t_maskratio=self.mask_scheduler.get_mask(step,x[...,0])
                    if step<step_thresh and si<si_thresh:
                        logits_BlV = self.get_logits(x)
                        embed_Cvae,conf_Bl=self.from_logit2emb(logits_BlV,t=t,rng=rng,top_k=top_k,top_p=top_p,B=B)
                        # conf_Bl = torch.rand_like(conf_Bl)#根据conf_bl产生mask的，所以conf_bl随机就是mask随机
                        tresh_conf, indice_mask = torch.topk(conf_Bl.view(B, -1), k=t_maskratio, dim=-1)
                        mask_0=mask.clone().detach()
                        for i_mask, ind_mask in enumerate(indice_mask):
                            mask[i_mask, ind_mask] = 1
                        h_BChw += embed_Cvae*(mask[...,None]-mask_0[...,None])
                    else:
                        # pdb.set_trace
                        # _,t_maskratio=self.mask_scheduler.get_mask(step,x[...,0])
                        # logits_BlV = self.get_logits(x)
                        text_pool_feat=self.encoder_proj2(encoder_pool_feat.unsqueeze(1))+self.pos_start_last
                        cur_feature=torch.cat([text_pool_feat,self.word_embed_head(h_BChw*mask[...,None]).repeat(2, 1, 1)\
                                            +self.lvl_embed_2(self.lvl_1L[:,cur_L-pn*pn:cur_L])],dim=1)

                        logits_BlV=self.sample(feature=cur_feature,#x:2B,1,embed；当前token的feature，来自var
                                            prev_emb=torch.cat([text_pool_feat,x],dim=1),
                                            attn_bias=None,
                                            freqs_cis=torch.cat([self.freqs_cis[0,None],freqs_cis_cur],dim=0))#2B,1,32；上一个token的emb，来自自回归模型
                        logits_BlV=logits_BlV[:,1:]
                        embed_Cvae,conf_Bl=self.from_logit2emb(logits_BlV,t=cfg,rng=rng,top_k=top_k,top_p=top_p,B=B)
                    # conf_Bl = torch.rand_like(conf_Bl)#根据conf_bl产生mask的，所以conf_bl随机就是mask随机
                    conf_Bl = torch.rand_like(conf_Bl) if step<5 else conf_Bl#根据conf_bl产生mask的，所以conf_bl随机就是mask随机
                    conf_Bl=conf_Bl*(1-mask[...,None])
                    tresh_conf, indice_mask = torch.topk(conf_Bl.view(B, -1), k=t_maskratio, dim=-1)
                    # update the mask
                    mask_0=mask.clone().detach()
                    for i_mask, ind_mask in enumerate(indice_mask):
                        mask[i_mask, ind_mask] = 1
                    tresh_conf = tresh_conf[:, -1]
                    h_BChw += embed_Cvae*(mask[...,None]-mask_0[...,None])


            else:
                # print('x max & min val:  ',x.max(),x.min(),x.mean(),si,pn)
                logits_BlV = self.get_logits(x)
                # print('logits:  ',logits_BlV.max(),logits_BlV.min())
            
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]    #(B,l,4096)是每个token和vqvae codebook的4096个token的相似程度
                
                if not more_smooth:#gumbel_softmax_with_rng这个是可微的，但sample_with_top_k_top_p_这个不行
                    if sample_version=='old':
                        if si<12:
                            top_k=top_k if si<9 else 300
                        else:top_k=100
                        top_p=0.8
                    else:
                        pass
                    idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)#和哪个token最相似，对应的离散token的idx
                    h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae=32，是再把确定的离散token的idx获得一个embed
                else:
                    gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                    # 根据logit得到class
                    h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
                # print('h_BChw:  ',h_BChw.max(),h_BChw.min())
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)#h_BChw是啥意思？
            
            # 残差是来自vq token的，那么fhat和vq token有什么关系？他这个vq空间的token是不是本来就是残差的意思？
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)

                if (not self.rotary_pos_emb) or (self.absolute_lvl_emb):
                    next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                else:
                    next_token_map = self.word_embed(next_token_map)
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG (B,?,C)
      
        for b in self.blocks: b.attn.kv_caching(False)
        with torch.autocast('cuda', enabled=False, dtype=torch.float32, cache_enabled=True):    # using bfloat16 can be faster
            return self.vae_proxy[0].fhat_to_img(f_hat.float()).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]


    def drop_scale(self, x_BLC, attn_bias, freqs_cis, start_idx, num_tokens_to_drop=200):
        B, L, C = x_BLC.shape
        # Generate random indices to drop, ensuring they are within the valid range [start_idx, L)
        all_indices = torch.arange(start_idx, L)
        dropped_indices = torch.randperm(len(all_indices))[:num_tokens_to_drop]
        dropped_indices = all_indices[dropped_indices]
        
        # Generate the mask for keeping tokens
        keep_indices = torch.ones(L, dtype=torch.bool)
        keep_indices[dropped_indices] = False
        
        # Apply mask to x_BLC, attn_bias, and freqs_cis
        x_BLC_dropped = x_BLC[:, keep_indices, :]
        attn_bias_dropped = attn_bias[:, :, keep_indices, :][:, :, :, keep_indices]
        freqs_cis_dropped = freqs_cis[keep_indices, :]

        # Adjust L dimension to length_dropped
        length_dropped = x_BLC_dropped.shape[1]
        
        return x_BLC_dropped, attn_bias_dropped, freqs_cis_dropped, keep_indices
    


    def select_square_region(self,x_BCHW, x_BLC, k, patch_nums, ranges, edge=3):
        B, C, H, W = x_BCHW.shape
        assert k <= H and k <= W, "k should be less than or equal to H and W"
        
        # Randomly choose the top-left corner for the k x k square
        H_1=round(ranges[0]);H_2=round(ranges[1])
        W_1=round(ranges[2]);W_2=round(ranges[3])
        # print(H_1,H_2,W_1,W_2)
        top_left_x = torch.randint(H_1, H_2 - k + 1, (1,)).item()
        top_left_y = torch.randint(W_1, W_2 - k + 1, (1,)).item()
        
        # Extract the k x k region
        selected_region = x_BCHW[:, :, top_left_x:top_left_x + k, top_left_y:top_left_y + k].reshape(B,C,-1).transpose(1, 2)
        
        # Generate a grid of indices for the k x k square
        row_indices = torch.arange(top_left_x, top_left_x + k).unsqueeze(1) * patch_nums
        col_indices = torch.arange(top_left_y, top_left_y + k)
        grid_indices = (row_indices + col_indices).flatten()
        # print([top_left_x,top_left_x+k,top_left_y,top_left_y+k],patch_nums)
        # Expand indices to match batch size
        selected_indices = grid_indices.view(-1)  # Shape (B, k*k)
        
        return selected_region, selected_indices, [top_left_x+edge,top_left_x+k-edge,top_left_y+edge,top_left_y+k-edge]


    def drop_scale_v2(self, x_BLC, attn_bias, freqs_cis, start_idx, k_list=[5,5,5,5,5]):
        # .view(B, C, -1).transpose(1, 2)
        # Generate the mask for keeping tokens
        B, L, C = x_BLC.shape
        keep_indices = torch.ones(L, dtype=torch.bool)

        ranges=[0,1,0,1]
        for idx in range(min(len(k_list),len(self.patch_nums)-self.drop_start_idx)):
            bg,ed=self.begin_ends[idx+self.drop_start_idx]
            pn=self.patch_nums[idx+self.drop_start_idx]
            x_BCHW=x_BLC[:,bg:ed].transpose(1, 2).reshape(B,C,pn,pn)
            x_BLC_dropped,dropped_indices,ranges=self.select_square_region(x_BCHW,x_BLC, k_list[idx], pn, [i*pn for i in ranges])
            ranges=[i/pn for i in ranges]
            dropped_indices=dropped_indices+bg
            keep_indices[bg:ed] = False
            keep_indices[dropped_indices] = True

        # Apply mask to x_BLC, attn_bias, and freqs_cis
        x_BLC_dropped = x_BLC[:, keep_indices, :]
        attn_bias_dropped = attn_bias[:, :, keep_indices, :][:, :, :, keep_indices]
        freqs_cis_dropped = freqs_cis[keep_indices, :]

        # Adjust L dimension to length_dropped
        length_dropped = x_BLC_dropped.shape[1]
        
        return x_BLC_dropped, attn_bias_dropped, freqs_cis_dropped, keep_indices

    
    def forward(self,
                x_BLCv_wo_first_l: torch.Tensor, 
                encoder_hidden_states=None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                encoder_pool_feat=None,
                embed_Cvae=None,) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """

        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            #label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.encoder_proj(encoder_pool_feat)#(B,1024)#self.class_emb(label_B)

            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: x_BLC = sos#self.prog_si始终是-1
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)

            if not self.rotary_pos_emb:
                x_BLC += self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
            if self.absolute_lvl_emb:
                x_BLC += self.lvl_embed(self.lvl_1L[:, :ed])
                cond_lvl_emb=None
            if self.shared_aln:
                cond_lvl_emb=self.lvl_embed_proj(
                        torch.cat([self.lvl_embed(self.lvl_1L),
                                self.lvl_embed(torch.full(self.lvl_1L.shape,self.lvl_1L[0,-1],device=self.lvl_1L.device))],dim=-1))#1,L,C
                cond_lvl_emb = self.lvl_embed_adaln(cond_lvl_emb)#cond_BD是加各种pos emb之前的class_emb
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        freqs_cis=self.freqs_cis
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)



        encoder_attention_mask=prepare_attn_mask(
            encoder_attention_mask=encoder_attention_mask)
        
        if not self.drop_scale_length==None:
            x_BLC,attn_bias,freqs_cis,drop_idxs=self.drop_scale(x_BLC,attn_bias,freqs_cis,
                                                                start_idx=self.drop_start,
                                                                num_tokens_to_drop=self.num_tokens_to_drop)
            # x_BLC,attn_bias,freqs_cis,drop_idxs=self.drop_scale_v2(x_BLC,attn_bias,freqs_cis,
            #                                                     start_idx=self.drop_start,
            #                                                     k_list=[31,26])
        else:
            drop_idxs=None
        

        AttnBlock.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, 
                      encoder_hidden_states=encoder_hidden_states,
                      encoder_attention_mask=encoder_attention_mask,
                      cond_BD=cond_lvl_emb, 
                      attn_bias=attn_bias,
                      freqs_cis=freqs_cis,
                      layer_id=i)
            
        logits_BLV = self.get_logits(x_BLC.float())#(B,pn*pn,4096)
        return logits_BLV,x_BLC,drop_idxs


    def forward_sampler(
            self,
            x_BLCv_wo_first_l: torch.Tensor, 
            encoder_hidden_states=None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            encoder_pool_feat=None, 
            embed_Cvae=None):
        
        with torch.no_grad():
            logits_BLV,feat_BlC,_=self.forward(x_BLCv_wo_first_l=x_BLCv_wo_first_l,
                                    encoder_hidden_states=encoder_hidden_states,
                                    encoder_attention_mask=encoder_attention_mask,
                                    encoder_pool_feat=encoder_pool_feat)
        B=embed_Cvae.shape[0]
        bg,ed=self.bg_last,self.ed_last#只计算最后一个scale
        logits_last=logits_BLV[:,bg:ed]
        feat_last=feat_BlC[:,bg:ed]
        pn=self.patch_nums[-1]#后续可以改成一个rnn🤔

        # prev_emb=None
        total_logits=[]
        mask=self.mask_scheduler.add_mask_for_training(embed_Cvae[...,0])
        embed_Cvae=embed_Cvae*mask[...,None]
        # embed_Cvae_=torch.cat([self.pos_start_last.repeat(B,1,1)+self.encoder_proj2(encoder_pool_feat.unsqueeze(1)),
        #                        self.word_embed_head(embed_Cvae[:,:-1])],dim=1)+self.pos_last

        text_pool_feat=self.encoder_proj2(encoder_pool_feat.unsqueeze(1))+self.pos_start_last

        freqs_cis=torch.cat([self.freqs_cis[0,None],self.freqs_cis[bg:ed]],dim=0)
        logits_BLV_=self.sample(feature=torch.cat([text_pool_feat,self.word_embed_head(embed_Cvae)+\
                                                   self.lvl_embed_2(self.lvl_1L[:, bg:ed])],dim=1),
                                prev_emb=torch.cat([text_pool_feat,feat_last],dim=1),
                                attn_bias=self.attn_mask,
                                freqs_cis=freqs_cis)#2B,1,32
                    # cur_logits=self.sample(feature=x[:,step,None],#x:2B,1,embed；当前token的feature，来自var
                    #                        prev_emb=prev_emb)#2B,1,32；上一个token的emb，来自自回归模型
        return logits_BLV_[:,1:],mask    # logits BLV, V is vocab_size,V最后一个dim是masked token对应的embed
    # 但是这里不用处理，因为后面直接加在loss上
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head_logits, nn.Linear):
                self.head_logits.weight.data.mul_(init_head)
                self.head_logits.bias.data.zero_()
            elif isinstance(self.head_logits, nn.Sequential):
                self.head_logits[-1].weight.data.mul_(init_head)
                self.head_logits[-1].bias.data.zero_()
        
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AttnBlock
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class VARHF(VAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )


if __name__ == '__main__':
    vae=VQVAE()
    var=VAR(vae_local=vae)
    B=3
    var.forward(label_B=None,
                x_BLCv_wo_first_l=torch.zeros([B,679,32]),
                encoder_hidden_states=torch.zeros([B,77,1024]),
                encoder_attention_mask=torch.zeros([B,77]),
                encoder_pool_feat=torch.zeros([B,1024])
                )
    
    var.autoregressive_infer_cfg(B=B,
            label_B=None,
            encoder_hidden_states=torch.zeros([2*B,77,1024]),
            encoder_attention_mask=torch.zeros([2*B,77]),
            encoder_pool_feat=torch.zeros([2*B,1024])
            )

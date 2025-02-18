from typing import Tuple
import torch.nn as nn

from .quant import VectorQuantizer2
from .var_drop import VAR
from .vqvae import VQVAE
# from .vqvae_chameleon import VQVAE
import yaml


def load_vqvae_chameleon(v_patch_nums,device):
    cfg_path='/nfs-26/maxiaoxiao/ckpts/chameleon/tokenizer/vqgan.yaml'
    ckpt_path='/nfs-140/maxiaoxiao/workspace/chameleon_vq_256/checkpoints/0175000.pt'

    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    params = config["model"]["params"]
    if "lossconfig" in params:
        del params["lossconfig"]
    params["ckpt_path"] = ckpt_path
    vq_model = VQVAE(**params,ignore_keys=['ema_vocab_hit_SV']).to(device).eval()
    vq_model.Cvae=params["embed_dim"]
    vq_model.vocab_size=params["n_embed"]

    return vq_model


def build_vae_var(
    # Shared args
    device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    # VQVAE args
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    # VAR args
    depth=16, shared_aln=False, attn_l2_norm=True,
    enable_cross=True,in_dim_cross=1024,
    flash_if_available=True, fused_if_available=True,
    init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
    rope_emb=True,lvl_emb=True,    # init_std < 0: automated
    rope_norm=32,
    drop_scale_length=None,
    enable_logit_norm=True,
    enable_adaptive_norm=True,
    train_mode='head_only',
    rope_theta=100.0,
    vae_ada=False,
    sample_from_idx=9,
) -> Tuple[VQVAE, VAR]:
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth/24
    
    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # # build models
    # vae_local=load_vqvae_chameleon(
    #     v_patch_nums=patch_nums,
    #     device=device
    # )
    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, 
                      share_quant_resi=share_quant_resi, v_patch_nums=patch_nums,
                      vae_ada=vae_ada).to(device)
    var_wo_ddp = VAR(
        vae_local=vae_local,
        depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        enable_cross=enable_cross,
        in_dim_cross=in_dim_cross,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        rotary_pos_emb=rope_emb,
        rope_norm=rope_norm,
        absolute_lvl_emb=lvl_emb,
        drop_scale_length=drop_scale_length,
        enable_logit_norm=enable_logit_norm,
        enable_adaptive_norm=enable_adaptive_norm,
        train_mode=train_mode,
        rope_theta=rope_theta,
        sample_from_idx=sample_from_idx
    ).to(device)
    if train_mode!='head_only':
        var_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)
    
    return vae_local, var_wo_ddp



def build_var(
    # Shared args
    vae_local,
    device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    # VAR args
    depth=16, shared_aln=False, attn_l2_norm=True,
    enable_cross=True,in_dim_cross=1024,
    flash_if_available=True, fused_if_available=True,
    init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
    rope_emb=True,lvl_emb=True,    # init_std < 0: automated
    drop_scale_length=None,
) -> Tuple[VQVAE, VAR]:
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth/24
    
    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    var_wo_ddp = VAR(
        vae_local=vae_local,
        depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        enable_cross=enable_cross,
        in_dim_cross=in_dim_cross,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        rotary_pos_emb=rope_emb,
        absolute_lvl_emb=lvl_emb,
        drop_scale_length=drop_scale_length,
    ).to(device)
    var_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)
    
    return var_wo_ddp

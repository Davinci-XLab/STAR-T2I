"""
References:
- VectorQuantizer2: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L110
- GumbelQuantize: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
- VQVAE (VQModel): https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/autoencoder.py#L14
"""
import sys
sys.path.insert(0,'../')
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F
import numpy as np
import pdb

from models.basic_vae import Decoder, Encoder
from models.quant import VectorQuantizer2,MultiscaleVectorQuantizer2


class VQVAE(nn.Module):
    def __init__(
        self, vocab_size=4096, z_channels=32, ch=128, dropout=0.0,
        beta=0.25,              # commitment loss weight
        using_znorm=False,      # whether to normalize when computing the nearest neighbors
        quant_conv_ks=3,        # quant conv kernel size
        quant_resi=0.5,         # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
        share_quant_resi=4,     # use 4 \phi layers for K scales: partially-shared \phi
        default_qresi_counts=0, # if is 0: automatically set to len(v_patch_nums)
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
        test_mode=True,
        enable_movq=False,
        vae_ada=False,
    ):
        super().__init__()
        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        # ddconfig is copied from https://github.com/CompVis/latent-diffusion/blob/e66308c7f2e64cb581c6d27ab6fbeb846828253b/models/first_stage_models/vq-f16/config.yaml
        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                       # from vq-f16/config.yaml above
            # resamp_with_conv=True,   # always True, removed.
        )
        ddconfig.pop('double_z', None)  # only KL-VAE should use double_z=True
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig,enable_movq=enable_movq)
        
        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult'])-1)
        # self.quantize: VectorQuantizer2 = VectorQuantizer2(
        #     vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
        #     default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        # )
        if vae_ada:
            self.quantize: VectorQuantizer2 = MultiscaleVectorQuantizer2(
                vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
                default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi,
            )
        else:
            self.quantize: VectorQuantizer2 = VectorQuantizer2(
                vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
                default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi,
            )
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        
        if self.test_mode==True:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]
        else:
            [p.requires_grad_(False) for p in self.parameters()]
            # [p.requires_grad_(True) for p in self.quantize.quant_resi.parameters()]
            [p.requires_grad_(True) for p in self.quantize.parameters()]

    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, inp, ret_usages=True):   # -> rec_B3HW, idx_N, loss
        VectorQuantizer2.forward
        h_BChw, usages, vq_loss, mean_entropy_loss = self.quantize(self.quant_conv(self.encoder(inp)), ret_usages=ret_usages)
        return self.decoder(self.post_quant_conv(h_BChw)), usages, vq_loss, mean_entropy_loss
    # ===================== `forward` is only used in VAE training =====================
    
    def fhat_to_img(self, f_hat: torch.Tensor):
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
    
    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:    # return List[Bl]
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat='idx', v_patch_nums=v_patch_nums)
    

    def img_to_quant_embed(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:    # return List[Bl]
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        # embed=self.quantize.f_to_idxBl_or_fhat(f, to_fhat='embed', v_patch_nums=v_patch_nums)
        idxBl=self.quantize.f_to_idxBl_or_fhat(f, to_fhat='idx', v_patch_nums=v_patch_nums)
        return idxBl#,embed

    
    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            ms_h_BChw.append(self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn))
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True))).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat='fhat', v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]

    def load_state_dict(self, state_dict: Dict[str, Any], strict=True):
        if 'quantize.ema_vocab_hit_SV' in state_dict and state_dict['quantize.ema_vocab_hit_SV'].shape[0] != self.quantize.ema_vocab_hit_SV.shape[0]:
            state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict)
    

if __name__ == '__main__':
    import os
    # v_patch_nums=[i for i in [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]]#[int(1024/(2**i)) for i in range(3,10)][::-1]
    v_patch_nums=[i for i in [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 24, 32, 48, 64]]#[int(1024/(2**i)) for i in range(3,10)][::-1]
    vae_ckpt='/home/disk2/nfs/maxiaoxiao/ckpts/FoundationVision-var/vae_ch160v4096z32.pth'
    vae_local = VQVAE(vocab_size=4096, z_channels=32, ch=160, 
                      test_mode=False, share_quant_resi=4,
                      v_patch_nums=v_patch_nums,enable_movq=False).cuda()



    inp=torch.rand((1,3,1024,1024)).cuda()
    recon,usages, mean_vq_loss, rec_loss=vae_local.forward(inp,ret_usages=False)
import sys
sys.path.insert(0,'/home/disk2/nfs/maxiaoxiao/VAR_new')
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
from transformers import CLIPTokenizer, CLIPTextModel
from models.t5_embedder import T5Embedder
import pdb
import torch.nn as nn


def build_text(pretrained_path,device,text_encoder='clip'):
    return Text_Model(pretrained_path=pretrained_path,device=device,text_encoder=text_encoder),1024 if text_encoder=='clip' else 2048


class Text_Model(nn.Module):
    def __init__(self, pretrained_path='',device='cpu',test_mode=True,text_encoder='clip'):
        super().__init__()
        self.device=device
        self.text_encoder_name=text_encoder
        if text_encoder=='t5':
            self.text_encoder_t5 = T5Embedder(
                device=device, 
                local_cache=True, 
                cache_dir='/nfs-141/maxiaoxiao/ckpts/flan_t5_xl', 
                dir_or_name='flan-t5-xl',
                torch_dtype=torch.float32,
                model_max_length=120,
            )
            self.avg_pool = nn.AvgPool1d(kernel_size=120)  # kernel_size 设置为 L 的长度
        # else:
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_path,subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_path,subfolder='text_encoder').to(device)
        print('using %s'%self.text_encoder_name)
        if test_mode:
            self.text_encoder.eval()
            [p.requires_grad_(False) for p in self.text_encoder.parameters()]
        for name, param in self.named_parameters():
            param.requires_grad_(False)

    # 2. 提取文本特征
    def extract_text_features(self,prompt):
        # pdb.set_trace()
        if self.text_encoder_name=='t5':
            outputs_t5, attn_mask = self.text_encoder_t5.get_text_embeddings(prompt)

            # 2. 使用 AvgPool1d 对 L 维度进行平均池化
            pooled_output = self.avg_pool(outputs_t5.permute(0, 2, 1)).squeeze(-1)  # 结果形状为 (B, C, 1)
            return outputs_t5,attn_mask,pooled_output  # 使用最后的隐藏状态作为文本特征
        else:
            inputs= self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, 
                                    truncation=True, return_tensors="pt").to(self.device)
            attn_mask=inputs.attention_mask
            # inputs{'input_ids': tensor; 'attention_mask': tensor; }
            # (Pdb) inputs.attention_mask
            # tensor([[1, 1, 1,  ..., 0, 0, 0],
            #         [1, 1, 1,  ..., 0, 0, 0],
            #         [1, 1, 1,  ..., 0, 0, 0],
            #         ...,
            #         [1, 1, 0,  ..., 0, 0, 0],
            #         [1, 1, 0,  ..., 0, 0, 0],
            #         [1, 1, 0,  ..., 0, 0, 0]], device='cuda:0')
            # attn_mask=torch.Size([B, 77])
            # outputs{last_hidden_state=tensor,pooler_output=tensor,hidden_states=None, attentions=None}
            outputs = self.text_encoder(**inputs)
            return outputs.last_hidden_state,attn_mask,outputs.pooler_output  # 使用最后的隐藏状态作为文本特征
        

if __name__ == '__main__':
    text_model,in_dim_cross=build_text(pretrained_path='/home/nfs/nfs-26/maxiaoxiao/ckpts/stable-diffusion-2-1', 
               device='cuda', text_encoder='t5')
    prompt=['a photo of an astronaut riding a horse on mars']
    text_features,text_mask,text_pooler=text_model.extract_text_features(prompt)
    pdb.set_trace()
    print(text_features.shape,text_mask.shape,text_pooler.shape,in_dim_cross)

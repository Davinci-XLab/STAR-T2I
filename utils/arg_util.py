import json
import os
import random
import re
import subprocess
import sys
import time
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import torch
import dist


class Args:
    def __init__(self,config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        for key, value in config.items():
            setattr(self, key, value)

        # 额外的依赖项
        self.vae_ckpt=os.path.join(self.ckpt_path,'vae_ch160v4096z32.pth')
        self.var_ckpt=os.path.join(self.ckpt_path,f'var_d{self.depth}.pth')

        # 如果 patch_nums 和 resos 依赖于其他值，需要在此设置
        if self.patch_nums is None:
            self.patch_nums = tuple(map(int, self.pn.replace('-', '_').split('_')))
        if self.resos is None:
            self.resos = tuple(p * self.patch_size for p in self.patch_nums)
        if self.data_load_reso is None:
            self.data_load_reso = max(self.patch_nums) * self.patch_size

        self.cmd = ' '.join(sys.argv[1:])

        # self.local_out_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_output')
        self.tb_log_dir_path = '...'  # Set according to your environment
        self.log_txt_path = '...'  # Set according to your environment
        self.last_ckpt_path = '...'  # Set according to your environment
        self.acc_mean: float = None      # [automatically set; don't specify this]
        self.acc_tail: float = None      # [automatically set; don't specify this]
        self.L_mean: float = None        # [automatically set; don't specify this]
        self.L_tail: float = None        # [automatically set; don't specify this]
        self.vacc_mean: float = None     # [automatically set; don't specify this]
        self.vacc_tail: float = None     # [automatically set; don't specify this]
        self.vL_mean: float = None       # [automatically set; don't specify this]
        self.vL_tail: float = None       # [automatically set; don't specify this]
        self.grad_norm: float = None     # [automatically set; don't specify this]
        self.cur_lr: float = None        # [automatically set; don't specify this]
        self.cur_wd: float = None        # [automatically set; don't specify this]
        self.cur_it: str = ''            # [automatically set; don't specify this]
        self.cur_ep: str = ''            # [automatically set; don't specify this]
        self.remain_time: str = ''       # [automatically set; don't specify this]
        self.finish_time: str = ''       # [automatically set; don't specify this]

        self.read_instance_prompts()

    def __str__(self):
        attrs = vars(self)
        return '\n'.join(f"{k}: {v}" for k, v in attrs.items())


    def read_instance_prompts(self):
        if type(self.instance_prompt)==str:
            prompts = []
            with open(self.instance_prompt) as prompt_file:
                for line in prompt_file.readlines():
                    prompts.append(line.strip())
            self.instance_prompt = prompts
        elif type(self.instance_prompt)==list:
            prompts = []
            for prompts_part in self.instance_prompt:
                assert os.path.exists(prompts_part)
                with open(prompts_part) as prompt_file:
                    for line in prompt_file.readlines():
                        prompts.append(line.strip())
            self.instance_prompt = prompts
        else:
            raise Exception("UnKnow instance_prompt")


    def seed_everything(self, benchmark: bool):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = benchmark
        if self.seed is None:
            torch.backends.cudnn.deterministic = False
        else:
            torch.backends.cudnn.deterministic = True
            seed = self.seed * dist.get_world_size() + dist.get_rank()
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
    same_seed_for_all_ranks: int = 0     # this is only for distributed sampler
    def get_different_generator_for_each_rank(self) -> Optional[torch.Generator]:   # for random augmentation
        if self.seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(self.seed * dist.get_world_size() + dist.get_rank())
        return g
    
    local_debug: bool = 'KEVIN_LOCAL' in os.environ
    dbg_nan: bool = False   # 'KEVIN_LOCAL' in os.environ
    
    def compile_model(self, m, fast):
        if fast == 0 or self.local_debug:
            return m
        return torch.compile(m, mode={
            1: 'reduce-overhead',
            2: 'max-autotune',
            3: 'default',
        }[fast]) if hasattr(torch, 'compile') else m
    
    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        # self.as_dict() would contain methods, but we only need variables
        for k in self.class_variables.keys():
            if k not in {'device'}:     # these are not serializable
                d[k] = getattr(self, k)
        return d
    
    def load_state_dict(self, d: Union[OrderedDict, dict, str]):
        if isinstance(d, str):  # for compatibility with old version
            d: dict = eval('\n'.join([l for l in d.splitlines() if '<bound' not in l and 'device(' not in l]))
        for k in d.keys():
            try:
                setattr(self, k, d[k])
            except Exception as e:
                print(f'k={k}, v={d[k]}')
                raise e
    
    @staticmethod
    def set_tf32(tf32: bool):
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
                print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')
    
    def dump_log(self):
        if not dist.is_local_master():
            return
        if '1/' in self.cur_ep: # first time to dump log
            with open(self.log_txt_path, 'w') as fp:
                json.dump({'is_master': dist.is_master(), 'name': self.exp_name, 'cmd': self.cmd, 'tb_log_dir_path': self.tb_log_dir_path}, fp, indent=0)
                fp.write('\n')
        
        log_dict = {}
        for k, v in {
            'it': self.cur_it, 'ep': self.cur_ep,
            'lr': self.cur_lr, 'wd': self.cur_wd, 'grad_norm': self.grad_norm,
            'L_mean': self.L_mean, 'L_tail': self.L_tail, 'acc_mean': self.acc_mean, 'acc_tail': self.acc_tail,
            'vL_mean': self.vL_mean, 'vL_tail': self.vL_tail, 'vacc_mean': self.vacc_mean, 'vacc_tail': self.vacc_tail,
            'remain_time': self.remain_time, 'finish_time': self.finish_time,
        }.items():
            if hasattr(v, 'item'): v = v.item()
            log_dict[k] = v
        with open(self.log_txt_path, 'a') as fp:
            fp.write(f'{log_dict}\n')
    
    # def __str__(self):
    #     s = []
    #     for k in self.class_variables.keys():
    #         if k not in {'device', 'dbg_ks_fp'}:     # these are not serializable
    #             s.append(f'  {k:20s}: {getattr(self, k)}')
    #     s = '\n'.join(s)
    #     return f'{{\n{s}\n}}\n'


def init_dist_and_get_args():
    print(sys.argv)
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('--config='):
            config_path=sys.argv[i].split('=')[-1]
        # elif sys.argv[i].startswith('--local-rank=') or sys.argv[i].startswith('--local_rank='):
        #     del sys.argv[i]

    args = Args(config_path)
    if args.local_debug:
        args.seed = 1
        args.aln = 1e-2
        args.alng = 1e-5
        # args.saln = False
        args.afuse = False
        args.pg = 0.8
        args.pg0 = 1

    # init torch distributed
    from utils import misc
    os.makedirs(args.local_out_dir_path, exist_ok=True)
    misc.init_distributed_mode(local_out_path=args.local_out_dir_path, timeout=30)
    
    # set env
    args.set_tf32(args.tf32)
    args.seed_everything(benchmark=args.pg == 0)
    
    # update args: data loading
    args.device = dist.get_device()
    if args.pn == '256':
        args.pn = '1_2_3_4_5_6_8_10_13_16'
    elif args.pn == '512':
        args.pn = '1_2_3_4_6_9_13_18_24_32'
    elif args.pn == '1024':
        args.pn = '1_2_3_4_5_7_9_12_16_21_27_36_48_64'
    args.patch_nums = tuple(map(int, args.pn.replace('-', '_').split('_')))
    args.resos = tuple(pn * args.patch_size for pn in args.patch_nums)
    args.data_load_reso = max(args.resos)
    
    # update args: bs and lr
    bs_per_gpu = round(args.bs / args.ac / dist.get_world_size())
    args.batch_size = bs_per_gpu
    args.bs = args.glb_batch_size = args.batch_size * dist.get_world_size()
    
    args.workers = min(max(0, args.workers), args.batch_size)
    
    # args.tlr = args.ac * args.tblr * args.glb_batch_size / 256
    if args.tlr==None:
        args.tlr = args.ac * args.tblr * args.glb_batch_size / 5120 #微调，学习率低一些
    args.twde = args.twde or args.twd
    
    if args.wp == 0:
        args.wp = args.ep * 1/50
    
    # update args: progressive training
    if args.pgwp == 0:
        args.pgwp = args.ep * 1/300
    if args.pg > 0:
        args.sche = f'lin{args.pg:g}'
    
    # update args: paths
    args.log_txt_path = os.path.join(args.local_out_dir_path, 'log.txt')
    args.last_ckpt_path = os.path.join(args.local_out_dir_path, f'ar-ckpt-last.pth')
    _reg_valid_name = re.compile(r'[^\w\-+,.]')
    tb_name = _reg_valid_name.sub(
        '_',
        f'tb-VARd{args.depth}'
        f'__pn{args.pn}'
        f'__b{args.bs}ep{args.ep}{args.opt[:4]}lr{args.tblr:g}wd{args.twd:g}'
    )
    args.tb_log_dir_path = os.path.join(args.local_out_dir_path, tb_name)
    
    return args

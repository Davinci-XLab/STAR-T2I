import gc
import os
import shutil
import sys
import time
import warnings
import pdb
from functools import partial
import math

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# build models
from torch.nn.parallel import DistributedDataParallel as DDP
from models import VAR, VQVAE, build_vae_var,build_var
from trainer import VARTrainer
from utils.amp_sc import AmpOptimizer
from utils.lr_control import filter_params

import dist
from utils import arg_util, misc
from dataset.data import build_dataset,build_dataset_webtar,gather_file_keys
from models.text_encoder import build_text
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume
from utils.weights import zero_out_cross_attention_weights,apply_lvl_emb_and_pos_1LC
import torchvision

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # Handle empty batch case
    return torch.utils.data.dataloader.default_collate(batch)


def build_everything(args: arg_util.Args):
    # resume
    auto_resume_info, start_ep, start_it, trainer_state = auto_resume(args, 'ar-ckpt*.pth')
    start_ep=0 if args.from_0 else start_ep
    # create tensorboard logger
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'), verbose=True)
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()
    
    # log args
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    
    # build data
    if not args.local_debug:
        print(f'[build PT data] ...\n')
        if args.using_webtar:
            print('[using webtar] ...\n')
            dataset_train,dataset_val=build_dataset_webtar(args)
            types = str((type(dataset_train).__name__, type(dataset_val).__name__))
            ld_train = DataLoader(
                dataset=dataset_train, num_workers=args.workers,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=custom_collate_fn
            )
            ld_val = DataLoader(
                dataset_val, num_workers=0,
                batch_size=round(args.batch_size*1.5),
                shuffle=False, drop_last=False,
                collate_fn=custom_collate_fn
            )
            # for idx,item in enumerate(ld_train):
            #     print(f"Rank {dist.get_local_rank()}: Prompt={item['prompt']}, File Key={item['file_key']}, url={item['url']}")
        else:
            print('[using lmdb] ...\n')
            dataset_train, dataset_val = build_dataset(args)
            types = str((type(dataset_train).__name__, type(dataset_val).__name__))

            ld_val = DataLoader(
                dataset_val, num_workers=0, pin_memory=True,
                batch_size=round(args.batch_size*1.5), sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
                shuffle=False, drop_last=False,
            )
            del dataset_val
            
            ld_train = DataLoader(
                dataset=dataset_train, num_workers=args.workers, pin_memory=True,
                generator=args.get_different_generator_for_each_rank(), # worker_init_fn=worker_init_fn,
                batch_sampler=DistInfiniteBatchSampler(
                    dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size, same_seed_for_all_ranks=args.same_seed_for_all_ranks,
                    shuffle=True, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(), start_ep=start_ep, start_it=start_it,
                ),
            )
            del dataset_train

        num_classes = 1000
        
        [print(line) for line in auto_resume_info]
        print(f'[dataloader multi processing] ...', end='', flush=True)
        try:
            stt = time.time()
            if args.using_webtar:
                iters_train = args.web_tar_len//args.glb_batch_size
            else:
                iters_train = len(ld_train)
                ld_train = iter(ld_train)
            # noinspection PyArgumentList
            print(f'     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True, clean=True)
            print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}')
        except:
            print(sys.exc_info())
    
    else:
        num_classes = 1000
        ld_val = ld_train = None
        iters_train = 10
    

    text_encoder,in_dim_cross=build_text(pretrained_path=args.text_enc_path,device=dist.get_device(),text_encoder=args.text_enc)

    vae_local, var_wo_ddp = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        device=dist.get_device(), patch_nums=args.patch_nums,
        depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        enable_cross=args.enable_cross,in_dim_cross=in_dim_cross,#TODO:换成从text enc得到的参数
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
        rope_emb=args.rope_emb,lvl_emb=args.lvl_emb,
        rope_norm=args.rope_norm,
        drop_scale_length=args.drop_scale_length,
        enable_logit_norm=args.logit_norm,
        enable_adaptive_norm=False,
        train_mode='all',
        rope_theta=args.rope_theta,
        vae_ada=False,
    )
    
    dist.barrier()
    vae_local.load_state_dict(torch.load(args.vae_ckpt, map_location='cpu'), strict=True)
    if trainer_state is not None and len(trainer_state):
        print('unsing strict=False in loading...')
        new_state_dict=apply_lvl_emb_and_pos_1LC(args,state_dict=trainer_state["var_wo_ddp"],patch_nums=args.patch_nums)
        missing, unexpected=var_wo_ddp.load_state_dict(new_state_dict, strict=False)
        print('checkpoints incompatible: ',missing,unexpected)

    if not args.from_scratch:
        zero_out_cross_attention_weights(var_wo_ddp)

    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    var_wo_ddp: VAR = args.compile_model(var_wo_ddp, args.tfast)
    var: DDP = (DDP if dist.initialized() else NullDDP)(var_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=True, broadcast_buffers=False)
    # var: FSDP = (FSDP if dist.initialized() else NullDDP)(var_wo_ddp, device_id=dist.get_local_rank(),
    #                                                       sharding_strategy=ShardingStrategy.FULL_SHARD)

    print(f'[INIT] VAR model = {var_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.2f}'
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAE', vae_local), ('VAE.enc', vae_local.encoder), ('VAE.dec', vae_local.decoder), ('VAE.quant', vae_local.quantize))]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAR', var_wo_ddp),)]) + '\n\n')
    
    # build optimizer
    # fsdp我暂时不会写混合adam和adamw的
    names, paras, para_groups = filter_params(var_wo_ddp, nowd_keys={#nowd_keys:没有权重衰减的参数名
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })
    opt_clz = {
        'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')
    
    var_optim = AmpOptimizer(
        mixed_precision=args.fp16, optimizer=opt_clz(params=para_groups, **opt_kw), names=names, paras=paras,
        grad_clip=args.tclip, n_gradient_accumulation=args.ac
    )
    del names, paras, para_groups
    
    # build trainer
    trainer = VARTrainer(
        device=args.device, patch_nums=args.patch_nums, resos=args.resos,
        vae_local=vae_local, var_wo_ddp=var_wo_ddp, var=var,
        var_opt=var_optim, label_smooth=args.ls
    )
    # if trainer_state is not None and len(trainer_state):
    #     print('unsing strict=False in loading...')
    #     missing,unexpected=trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
    #     print('checkpoints incompatible: ',missing,unexpected)


    del vae_local, var_wo_ddp, var, var_optim
    
    if args.local_debug:
        rng = torch.Generator('cpu')
        rng.manual_seed(0)
        B = 4
        inp = torch.rand(B, 3, args.data_load_reso, args.data_load_reso).to(args.device)
        label = torch.ones(B, dtype=torch.long)
        
        me = misc.MetricLogger(delimiter='  ')
        prompt_embeds_=[torch.zeros([B,77,1024],device=inp.device),
                        torch.zeros([B,77],device=inp.device),
                        torch.zeros([B,1024],device=inp.device)]
        trainer.train_step(
            it=0, g_it=0, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prompt_embeds=prompt_embeds_,
            prog_si=-1, prog_wp_it=20,
        )
        trainer.train_step(
            it=99, g_it=599, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prompt_embeds=prompt_embeds_,
            prog_si=-1, prog_wp_it=20,
        )
        print({k: meter.global_avg for k, meter in me.meters.items()})
        
        args.dump_log(); tb_lg.flush(); tb_lg.close()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
        exit(0)
    
    for name, param in text_encoder.named_parameters():
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
    for name, param in trainer.vae_local.named_parameters():
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
    for name, param in trainer.var_wo_ddp.named_parameters():
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    dist.barrier()
    return (
        tb_lg, trainer, start_ep, start_it,
        iters_train, ld_train, ld_val, text_encoder
    )


def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)
    
    (
        tb_lg, trainer,
        start_ep, start_it,
        iters_train, ld_train, ld_val,
        text_encoder
    ) = build_everything(args)
    
    # train
    start_time = time.time()
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999., 999., -1., -1.
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = 999, 999, -1, -1
    
    L_mean, L_tail = -1, -1
    print(f"===========> main training start")
    for ep in range(start_ep, args.ep):
        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            print(f"has attr sampler")
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                print(f"ep < 3")
                # noinspection PyArgumentList
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True, force=True)
        print(f"===========> in main training")
        tb_lg.set_step(ep * iters_train)
        print(f'===========> epoch:{ep}, before eval')
        if ep > 0:
            val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = trainer.eval_ep(args,ld_val,text_encoder)
        print(f'===========> epoch:{ep}, after eval')
# def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, tb_lg: misc.TensorboardLogger, ld_or_itrt, iters_train: int, trainer):
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep=ep, is_first_ep=(ep == start_ep), start_it=start_it if ep == start_ep else 0, 
            args=args, tb_lg=tb_lg, ld_or_itrt=ld_train, text_enc=text_encoder, iters_train=iters_train, trainer=trainer
        )
        print(f'===========> epoch:{ep}, after eval')
        
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(best_acc_mean, acc_mean)
        if L_tail != -1: best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(best_acc_tail, acc_tail)
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = L_mean, L_tail, acc_mean, acc_tail, grad_norm
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        
        AR_ep_loss = dict(L_mean=L_mean, L_tail=L_tail, acc_mean=acc_mean, acc_tail=acc_tail)
        if (ep + 1) % 5 == 0 or (ep + 1) == args.ep:#每10 epoch验证一次
        # if is_val_and_also_saving:
            if dist.is_local_master():
                local_out_ckpt = os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth')
                local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'ar-ckpt-best.pth')
                print(f'[saving ckpt] ...', end='', flush=True)
                torch.save({
                    'epoch':    ep+1,
                    'iter':     0,
                    'trainer':  trainer.state_dict(),
                    # 'text_enc': text_encoder.state_dict()
                    # 'args':     args.state_dict(),
                }, local_out_ckpt)
                # if best_updated:
                #     shutil.copy(local_out_ckpt, local_out_ckpt_best)
                trainer.inference_pic(args,text_encoder,cur_ep=ep,cur_iter=-1,top_k=600,top_p=0.8,w_mask=False)
                torch.cuda.empty_cache()
                print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True, clean=True)

            val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = trainer.eval_ep(args,ld_val,text_encoder)
            best_updated = best_val_loss_tail > val_loss_tail
            best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, val_loss_mean), min(best_val_loss_tail, val_loss_tail)
            best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, val_acc_mean), max(best_val_acc_tail, val_acc_tail)
            AR_ep_loss.update(vL_mean=val_loss_mean, vL_tail=val_loss_tail, vacc_mean=val_acc_mean, vacc_tail=val_acc_tail)
            args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail
            print(f' [*] [ep{ep}]  (val {tot})  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  Val cost: {cost:.2f}s')
            
            dist.barrier()
        
        print(    f'     [ep{ep}]  (training )  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        tb_lg.update(head='AR_ep_loss', step=ep+1, **AR_ep_loss)
        tb_lg.update(head='AR_z_burnout', step=ep+1, rest_hours=round(sec / 60 / 60, 2))
        args.dump_log(); tb_lg.flush()
    
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    del stats
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log(); tb_lg.flush(); tb_lg.close()
    dist.barrier()


def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, tb_lg: misc.TensorboardLogger, ld_or_itrt, text_enc, iters_train: int, trainer):
    # import heavy packages after Dataloader object creation
    # ld_or_itrt:dataloader
    print(f'===========> epoch:{ep}, train_one_ep')
    from trainer import VARTrainer
    from utils.lr_control import lr_wd_annealing
    trainer: VARTrainer
    
    step_cnt = 0
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
    me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # if args.using_webtar:
    #     me.add_meter('data_cnt', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    g_it, max_it = ep * iters_train, args.ep * iters_train
    
    if args.using_webtar:
        all_file_keys=[]
    for it, obj in me.log_every(start_it=start_it, 
                                         max_iters=iters_train, 
                                         itrt=ld_or_itrt,
                                         print_freq=math.floor(iters_train/args.print_every),
                                         header=header):
        # print(f'===========> epoch:{ep}, train_one_ep {args.val_it} {it}')
        if it % args.val_it == 0 and it!=0:
            if dist.is_local_master():
                local_out_ckpt = os.path.join(args.local_out_dir_path, 'ar-ckpt-ep%d-iter%d.pth'%(ep,it))
                print(f'[saving ckpt] ...', end='', flush=True)
                torch.save({
                    'epoch':    ep+1,
                    'iter':     0,
                    'trainer':  trainer.state_dict(),
                    # 'text_enc': text_enc.state_dict()
                    # 'args':     args.state_dict(),
                }, local_out_ckpt)
                trainer.inference_pic(args,text_enc,cur_ep=ep,cur_iter=it,top_k=600,top_p=0.8,w_mask=False)
                torch.cuda.empty_cache()
                print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True, clean=True)

        g_it = ep * iters_train + it
        if it < start_it: continue
        if is_first_ep and it == start_it: warnings.resetwarnings()
        
        # (inp, label)
        # if args.using_webtar:
        #     file_keys=obj['file_key']
        #     all_file_keys.extend(gather_file_keys(file_keys))
        #     # print('rank=',dist.get_rank(),file_keys)
        #     if dist.get_rank()==0:
        #         # print('total=', len(set(all_file_keys)),' file_keys')
        #         me.update(data_cnt=len(set(all_file_keys)))
        inp = obj['image'].to(args.device, non_blocking=True)
        obj['prompt_embeds']=text_enc.extract_text_features(obj['prompt'])
        B=inp.shape[0]
        label = torch.tensor([args.default_label]*B).to(args.device, non_blocking=True)
        prompt_embeds=obj['prompt_embeds']
        
        args.cur_it = f'{it+1}/{iters_train}'
        
        wp_it = args.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.var_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
        # warmup:迭代次数<wp_it时，学习率从wp0线性增加到1 (乘peak_lr=1e-5)
        args.cur_lr, args.cur_wd = max_tlr, max_twd
        
        if args.pg: # default: 0.0, no progressive training, won't get into this
            if g_it <= wp_it: prog_si = args.pg0#warmup阶段prog_si是默认值pg0=4
            elif g_it >= max_it*args.pg: prog_si = len(args.patch_nums) - 1#iter大于args.pg指定的iter时，prog_si是scale个数
            else:
                delta = len(args.patch_nums) - 1 - args.pg0
                progress = min(max((g_it - wp_it) / (max_it*args.pg - wp_it), 0), 1) # from 0 to 1
                prog_si = args.pg0 + round(progress * delta)    # from args.pg0 to len(args.patch_nums)-1
        else:
            prog_si = -1#prog_si似乎是指定不同训练stage focus在不同scale上
        
        stepping = (g_it + 1) % args.ac == 0
        step_cnt += int(stepping)
        
        grad_norm, scale_log2 = trainer.train_step(
            it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prompt_embeds=prompt_embeds,
            prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
        )
        
        me.update(tlr=max_tlr)
        tb_lg.set_step(step=g_it)
        tb_lg.update(head='AR_opt_lr/lr_min', sche_tlr=min_tlr)
        tb_lg.update(head='AR_opt_lr/lr_max', sche_tlr=max_tlr)
        tb_lg.update(head='AR_opt_wd/wd_max', sche_twd=max_twd)
        tb_lg.update(head='AR_opt_wd/wd_min', sche_twd=min_twd)
        tb_lg.update(head='AR_opt_grad/fp16', scale_log2=scale_log2)
        
        if args.tclip > 0:
            tb_lg.update(head='AR_opt_grad/grad', grad_norm=grad_norm)
            tb_lg.update(head='AR_opt_grad/grad', grad_clip=args.tclip)
            # t_ratio = 1 if grad_norm is None else min(1.0, args.tclip / (grad_norm + 1e-7))
            # tb_lg.update(head='AR_opt_lr/lr_max', actu_tlr=t_ratio*max_tlr)
            # tb_lg.update(head='AR_opt_lr/lr_min', actu_tlr=t_ratio*min_tlr)
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == '__main__':
    try: main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()

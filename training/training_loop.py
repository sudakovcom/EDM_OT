# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
import torch.nn.functional as F
import wandb
import gc
from src.stats_counter import save_model_samples
from fid import calculate_inception_stats, calculate_fid_from_inception_stats
from dnnlib.util import open_url


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)

def calc_fid(image_path, ref_path, num_expected, batch):
    with open_url(ref_path) as f:
        ref = dict(np.load(f))

    mu, sigma = calculate_inception_stats(image_path=image_path, num_expected=num_expected, max_batch_size=batch)
    fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
    return fid

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset1train_kwargs      = {}, # Options for training set.
    data_loader1train_kwargs  = {}, # Options for torch.utils.data.DataLoader.
    dataset2train_kwargs      = {}, # Options for training set.
    data_loader2train_kwargs  = {}, # Options for torch.utils.data.DataLoader.
    dataset1test_kwargs      = {},  # Options for training set.
    data_loader1test_kwargs  = {},  # Options for torch.utils.data.DataLoader.
    dataset2test_kwargs      = {},  # Options for training set.
    data_loader2test_kwargs  = {},  # Options for torch.utils.data.DataLoader.
    dataset1stats_path       = '',
    dataset2stats_path       = '',
    network1_kwargs      = {},      # Options for model and preconditioning.
    loss1_kwargs         = {},      # Options for loss function.
    optimizer1_kwargs    = {},      # Options for optimizer.
    network2_kwargs      = {},      # Options for model and preconditioning.
    loss2_kwargs         = {},      # Options for loss function.
    optimizer2_kwargs    = {},      # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    t_iters             = 10,
    f_iters             = 1,
    nz                  = 100,
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset1train_obj = dnnlib.util.construct_class_by_name(**dataset1train_kwargs) # subclass of training.dataset.Dataset
    dataset1train_sampler = misc.InfiniteSampler(dataset=dataset1train_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset1train_iterator = iter(torch.utils.data.DataLoader(dataset=dataset1train_obj, sampler=dataset1train_sampler, batch_size=batch_gpu, **data_loader1train_kwargs))

    dataset2train_obj = dnnlib.util.construct_class_by_name(**dataset2train_kwargs) # subclass of training.dataset.Dataset
    dataset2train_sampler = misc.InfiniteSampler(dataset=dataset2train_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset2train_iterator = iter(torch.utils.data.DataLoader(dataset=dataset2train_obj, sampler=dataset2train_sampler, batch_size=batch_gpu, **data_loader2train_kwargs))

    dataset1test_obj = dnnlib.util.construct_class_by_name(**dataset1test_kwargs) # subclass of training.dataset.Dataset
    dataset1test_sampler = misc.InfiniteSampler(dataset=dataset1test_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset1test_iterator = iter(torch.utils.data.DataLoader(dataset=dataset1test_obj, sampler=dataset1test_sampler, batch_size=batch_gpu, **data_loader1test_kwargs))

    dataset2test_obj = dnnlib.util.construct_class_by_name(**dataset2test_kwargs) # subclass of training.dataset.Dataset
    dataset2test_sampler = misc.InfiniteSampler(dataset=dataset2test_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset2test_iterator = iter(torch.utils.data.DataLoader(dataset=dataset2test_obj, sampler=dataset2test_sampler, batch_size=batch_gpu, **data_loader2test_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs1 = dict(img_resolution=dataset1train_obj.resolution, img_channels=dataset1train_obj.num_channels, label_dim=dataset1train_obj.label_dim)
    net1 = dnnlib.util.construct_class_by_name(**network1_kwargs, **interface_kwargs1) # subclass of torch.nn.Module
    net1.train().requires_grad_(True).to(device)

    interface_kwargs2 = dict(img_resolution=dataset2train_obj.resolution, img_channels=dataset2train_obj.num_channels, label_dim=dataset2train_obj.label_dim)
    net2 = dnnlib.util.construct_class_by_name(**network2_kwargs, **interface_kwargs2) # subclass of torch.nn.Module
    net2.train().requires_grad_(True).to(device)



    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    optimizer1 = dnnlib.util.construct_class_by_name(params=net1.parameters(), **optimizer1_kwargs) # subclass of torch.optim.Optimizer
    optimizer2 = dnnlib.util.construct_class_by_name(params=net2.parameters(), **optimizer2_kwargs) # subclass of torch.optim.Optimizer
    # augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe


    dist.print0(resume_state_dump)
    if resume_state_dump:
        resume_state_dump
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net1'], dst_module=net1, require_all=True)
        misc.copy_params_and_buffers(src_module=data['net2'], dst_module=net2, require_all=True)
        optimizer1.load_state_dict(data['optimizer1_state'])
        optimizer2.load_state_dict(data['optimizer2_state'])
        del data # conserve memory
    
    net1.train().requires_grad_(True).to(device)
    net2.train().requires_grad_(True).to(device)
    ddp1 = torch.nn.parallel.DistributedDataParallel(net1, device_ids=[device])
    ddp2 = torch.nn.parallel.DistributedDataParallel(net2, device_ids=[device])
    ema1 = copy.deepcopy(net1).eval().requires_grad_(False)
    ema2 = copy.deepcopy(net2).eval().requires_grad_(False)

    if dist.get_rank() == 0:
        # wandb.init(name='tester', project='EDM-NOT', id="g7icxun0", resume="must")
        wandb.init(name=f'4gpu_strongD_D_{f_iters}it_G_{t_iters}it', project='EDM-NOT')

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    
    
    
    step = 0
    optimizer1.zero_grad(set_to_none=True)
    optimizer2.zero_grad(set_to_none=True)
    while True:
        unfreeze(ddp1)
        freeze(ddp2)
        for t_iter in range(t_iters):
            # Accumulate gradients.
            for round_idx in range(num_accumulation_rounds):
                with misc.ddp_sync(ddp1, (round_idx == num_accumulation_rounds - 1)):
                    images, _ = next(dataset1train_iterator)
                    X = images.to(device).to(torch.float32) / 127.5 - 1
                    with torch.no_grad():
                        latent_z = torch.randn(X.shape[0], nz, device=X.device)*0.1
                    T_X = ddp1(X, latent_z)
                    loss = F.mse_loss(X, T_X).mean() - ddp2(T_X).mean()
                    training_stats.report('Loss/loss', loss)
                    loss.sum().mul(loss_scaling / batch_gpu_total).backward()

            # Update weights.
            for g in optimizer1.param_groups:
                g['lr'] = optimizer1_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
            for param in net1.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            optimizer1.step()
            optimizer1.zero_grad(set_to_none=True)
            
        if dist.get_rank() == 0:
            wandb.log({f'T_loss' : loss.item()}, step=step)


        # Accumulate gradients.
        unfreeze(ddp2)
        freeze(ddp1)
        for f_iter in range(f_iters):
            for round_idx in range(num_accumulation_rounds):
                with misc.ddp_sync(ddp2, (round_idx == num_accumulation_rounds - 1)):
                    images, _ = next(dataset1train_iterator)
                    X = images.to(device).to(torch.float32) / 127.5 - 1
                    with torch.no_grad():
                        latent_z = torch.randn(X.shape[0], nz, device=X.device)*0.1
                        T_X = ddp1(X, latent_z)
                    images, _ = next(dataset2train_iterator)
                    Y = images.to(device).to(torch.float32) / 127.5 - 1
                    loss = ddp2(T_X).mean() - ddp2(Y).mean()
                    training_stats.report('Loss/loss', loss)
                    loss.sum().mul(loss_scaling / batch_gpu_total).backward()

            # Update weights.
            for g in optimizer2.param_groups:
                g['lr'] = optimizer2_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
            for param in net2.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            optimizer2.step()
            optimizer2.zero_grad(set_to_none=True)
        
        if dist.get_rank() == 0:
            wandb.log({f'f_loss' : loss.item()}, step=step)
            
            

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema1.parameters(), net1.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
        for p_ema, p_net in zip(ema2.parameters(), net2.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)


        if dist.get_rank() == 0:
            print(step)
            if step % 100 == 0:
                l2, lpips, X_array, T_X_array, Y_array = save_model_samples('samples', ema1, dataset1test_iterator, batch_gpu, len(dataset1test_obj), device, dataset2test_iterator)
                fid = calc_fid('samples', dataset2stats_path, len(dataset1test_obj), batch_gpu)
                print(fid, l2, lpips)
                wandb.log({f'FID' : fid}, step=step)
                wandb.log({f'l2' : l2}, step=step)
                wandb.log({f'lpips' : lpips}, step=step)
                wandb.log({"examples_X": [wandb.Image(image) for image in X_array]}, step=step)
                wandb.log({"examples_Y": [wandb.Image(image) for image in Y_array]}, step=step)
                wandb.log({"examples_T_X": [wandb.Image(image) for image in T_X_array]}, step=step)
                
        step+=1
        
        # if dist.get_rank() == 0:
        #     if step % 1000 == 0:
        #         torch.save(dict(net1=net1, net2=net2, ema1=ema1, ema2=ema2, optimizer1_state=optimizer1.state_dict(), optimizer2_state=optimizer2.state_dict()), os.path.join(run_dir, f'training-state-{step}.pt'))

            
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')
            
        gc.collect(); torch.cuda.empty_cache()

        # # Save network snapshot.
        # if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
        #     data = dict(ema1=ema1, ema2=ema2, dataset_kwargs=dict(dataset_kwargs))
        #     for key, value in data.items():
        #         if isinstance(value, torch.nn.Module):
        #             value = copy.deepcopy(value).eval().requires_grad_(False)
        #             misc.check_ddp_consistency(value)
        #             data[key] = value.cpu()
        #         del value # conserve memory
        #     if dist.get_rank() == 0:
        #         with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
        #             pickle.dump(data, f)
        #     del data # conserve memory
        
        
        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break
    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------

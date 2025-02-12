# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop
import lpips
import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data1train',    help='Path to the dataset1', metavar='ZIP|DIR',                    type=str, required=True)
@click.option('--data2train',    help='Path to the dataset2', metavar='ZIP|DIR',                    type=str, required=True)
@click.option('--data1test',     help='Path to the dataset1', metavar='ZIP|DIR',                    type=str, required=True)
@click.option('--data2test',     help='Path to the dataset2', metavar='ZIP|DIR',                    type=str, required=True)
@click.option('--data1stats',    help='Path to the dataset1', metavar='ZIP|DIR',                    type=str, required=True)
@click.option('--data2stats',    help='Path to the dataset2', metavar='ZIP|DIR',                    type=str, required=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm',          type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ddpmpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm',       type=click.Choice(['no', 'vp', 've', 'edm']), default='edm', show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--t_iters',       help='T iters', metavar='INT',                                     type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--f_iters',       help='f iters', metavar='INT',                                     type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=1e-4, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)

def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """    
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    
    c.t_iters = opts.t_iters
    c.f_iters = opts.f_iters
    
    
    c.dataset1train_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data1train, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    c.dataset2train_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data2train, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    c.dataset1test_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data1test, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    c.dataset2test_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data2test, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    c.data_loader1train_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.data_loader2train_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.data_loader1test_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.data_loader2test_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)

    c.dataset1stats_path=opts.data1stats
    c.dataset2stats_path=opts.data2stats

    c.network1_kwargs = dnnlib.EasyDict()
    c.loss1_kwargs = dnnlib.EasyDict()
    c.optimizer1_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)
    
    c.network2_kwargs = dnnlib.EasyDict()
    c.loss2_kwargs = dnnlib.EasyDict()
    c.optimizer2_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)

    # Validate datasets options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset1train_kwargs)
        dataset1train_name = dataset_obj.name
        c.dataset1train_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset1train_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
    
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset2train_kwargs)
        dataset2train_name = dataset_obj.name
        c.dataset2train_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset2train_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
    
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset1test_kwargs)
        dataset1test_name = dataset_obj.name
        c.dataset1test_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset1test_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
    
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset2test_kwargs)
        dataset2test_name = dataset_obj.name
        c.dataset2test_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset2test_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')


    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network1_kwargs.update(model_type='SongUNet', encoder_type='standard', decoder_type='standard')
        c.network1_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])

        c.network2_kwargs.update(model_type='SongUNetD', encoder_type='standard', decoder_type='standard')
        c.network2_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=192, channel_mult=[2,2,2])
    else:
        assert opts.arch == 'adm'
        c.network1_kwargs.update(model_type='SongUNet', encoder_type='residual', decoder_type='standard')
        c.network1_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])

        c.network2_kwargs.update(model_type='SongUNetD', encoder_type='residual', decoder_type='standard')
        c.network2_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])


    # Preconditioning & loss function.
    assert opts.precond == 'no'
    if opts.precond == 'no':
        c.network1_kwargs.class_name = 'src.SongUnet.NOPrecond'
        c.network2_kwargs.class_name = 'src.SongUnet.NOPrecond'
        # c.loss_kwargs.class_name = 'training.loss.VPLoss'

    # Network options.
    # if opts.cbase is not None:
    #     c.network_kwargs.model_channels = opts.cbase
    # if opts.cres is not None:
    #     c.network_kwargs.channel_mult = opts.cres
    # if opts.augment:
    #     c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
    #     c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
    #     c.network_kwargs.augment_dim = 9
    c.network1_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)
    c.network2_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    if opts.resume is not None:
        c.resume_state_dump = opts.resume

    # Description string.
    cond_str = 'cond' if c.dataset1train_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network1_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset1train_name:s}-{dataset2train_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset1 path:           {c.dataset1train_kwargs.path}')
    dist.print0(f'Dataset2 path:           {c.dataset2train_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset1train_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network1_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------

# torchrun --standalone --nproc_per_node=4 train.py --outdir=outdir --data1train=datasets/male64train.zip --data2train=datasets/female64train.zip --data1test=datasets/male64test.zip --data2test=datasets/female64test.zip --data1stats=datasets/male64train.npz --data2stats=datasets/female64train.npz --batch=256 --precond=no --batch-gpu=32 --resume=/home/iasudakov/project/edm/outdir/00030-male64train-female64train-uncond-ddpmpp-no-gpus2-batch256-fp32/training-state-2000.pt 
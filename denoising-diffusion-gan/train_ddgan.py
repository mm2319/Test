# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import argparse
import torch
import numpy as np
import rawpy
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from datasets_prep.lsun import LSUN
from datasets_prep.stackmnist_data import StackedMNIST, _data_transforms_stacked_mnist
from datasets_prep.lmdb_datasets import LMDBDataset

from network import Network

from torch.multiprocessing import Process
import torch.distributed as dist
import shutil

def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6020'
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()  

def cleanup():
    dist.destroy_process_group()    
#%%
if __name__ == '__main__':
    opt = argparse.ArgumentParser('ddgan parameters')
    opt.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    
    opt.add_argument('--resume', action='store_true',default=False)
    
    opt.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    opt.add_argument('--num_channels', type=int, default=4,
                            help='channel of image')
    opt.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    opt.add_argument('--use_geometric', action='store_true',default=False)
    opt.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    opt.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')
    
    
    opt.add_argument('--num_channels_dae', type=int, default=128,
                            help='number of initial channels in denosing model')
    opt.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    opt.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')
    opt.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    opt.add_argument('--attn_resolutions', default=(16,),
                            help='resolution of applying attention')
    opt.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    opt.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    opt.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    opt.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    opt.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    opt.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    opt.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    opt.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    opt.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    opt.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')
    
    opt.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    opt.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    opt.add_argument('--not_use_tanh', action='store_true',default=False)
    
    #geenrator and training
    opt.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    opt.add_argument('--dataset', default='cifar10', help='name of dataset')
    opt.add_argument('--nz', type=int, default=100)
    opt.add_argument('--num_timesteps', type=int, default=4)

    opt.add_argument('--z_emb_dim', type=int, default=256)
    opt.add_argument('--t_emb_dim', type=int, default=256)
    opt.add_argument('--batch_size', type=int, default=128, help='input batch size')
    opt.add_argument('--num_epoch', type=int, default=1200)
    opt.add_argument('--ngf', type=int, default=64)

    opt.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    opt.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    opt.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    opt.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    opt.add_argument('--no_lr_decay',action='store_true', default=False)
    
    opt.add_argument('--use_ema', action='store_true', default=False,
                            help='use EMA or not')
    opt.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    
    opt.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    opt.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    opt.add_argument('--save_content', action='store_true',default=False)
    opt.add_argument('--save_content_every', type=int, default=50, help='save content for resuming every x epochs')
    opt.add_argument('--save_ckpt_every', type=int, default=25, help='save ckpt every x epochs')
   
    ###ddp
    opt.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    opt.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    opt.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    opt.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    opt.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    opt.add_argument('--adaptive_loss', action="store_true", help='whether to use a learned weight of loss for different stages')
    opt.add_argument('--name', type=str, default=None, help='name of the experiment. It decides where to store samples and models')
    opt.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    opt.add_argument('--model', type=str, default='eld_model', help='chooses which model to use.', choices=model_names)
    opt.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    opt.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    opt.add_argument('--resume_epoch', '-re', type=int, default=None, help='checkpoint to use. (default: latest')
    opt.add_argument('--seed', type=int, default=2018, help='random seed to use. Default=2018')

    # for setting input
    opt.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    opt.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
    opt.add_argument('--chop', action='store_true', help='enable forward_chop')

    # for display
    opt.add_argument('--no-log', action='store_true', help='disable tf logger?')
    opt.add_argument('--no-verbose', action='store_true', help='disable verbose info?')
    opt.add_argument('--debug', action='store_true', help='debugging mode')

    opt.add_argument('--iter_num', type=int, default=2)
    opt.add_argument('--netG', type=str, default='unet', help='chooses which architecture to use for netG.')
    opt.add_argument('--adaptive_res_and_x0', action="store_true", help='adaptively combine the clean image and the image removed noise')
    opt.add_argument('--with_photon', action="store_true")
    opt.add_argument('--concat_origin', action="store_true")
    
    opt.add_argument('--resid', action="store_true", help='predict the noise instead of the clean image')
    opt.add_argument('--channels', '-c', type=int, default=4, help='in/out channels (4: bayer; 9: xtrans')
    opt.add_argument('--stage_in', type=str, default='raw', help='input stage [raw|srgb]')
    opt.add_argument('--stage_out', type=str, default='raw', help='output stage [raw|srgb]')
    opt.add_argument('--stage_eval', type=str, default='raw', help='output stage [raw|srgb]')
    opt.add_argument('--model_path', type=str, default=None, help='model checkpoint to use.')
    opt.add_argument('--include', type=int, default=None, help='select camera in ELD dataset')
    opt.add_argument('--gt_wb', action='store_true', help='use white balance of ground truth')
    opt.add_argument('--crf', action='store_true', help='use CRF to render sRGB images')
    opt.add_argument('--epoch', type=int, default=200)
   
    opt = opt.parse_args()
    opt.world_size = opt.num_proc_node * opt.num_process_per_node
    size = opt.num_process_per_node
    opt.display_freq = 20
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 100
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True
    if size > 1:
        processes = []
        for rank in range(size):
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.num_process_per_node
            global_size = opt.num_proc_node * opt.num_process_per_node
            opt.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (opt.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, opt))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:
        print('starting in debug mode')
        
        init_processes(0, size, train, opt)
   
                
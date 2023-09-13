import torch
import util.util as util
import ELD_iter_model
import time
import os
import torch.nn.functional as F
import sys
from os.path import join
from torchvision.utils import save_image
import dataset
from dataset import lmdb_dataset
from util import process
import noise
import torch.nn as nn
from os.path import join
import torch.optim as optim
import torch
import torch.backends.cudnn as cudnn
import dataset.sid_dataset as datasets
import dataset.lmdb_dataset as lmdb_dataset
import dataset
import numpy as np
from dataset.sid_dataset import worker_init_fn
from score_sde.models.discriminator import Discriminator_small, Discriminator_large
from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from EMA import EMA
import shutil

def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

class Network(object):
    def __init__(self, rank, opt):
        self.opt = opt
        self.writer = None
        self.model = None
        self.best_val_loss = 1e6
        self.__setup(rank)

    def __setup(self, rank):
        torch.manual_seed(self.opt.seed + rank)
        torch.cuda.manual_seed(self.opt.seed + rank)
        torch.cuda.manual_seed_all(self.opt.seed + rank)
        device = torch.device('cuda:{}'.format(rank))
        self.basedir = join('checkpoints', self.opt.name)
        if not os.path.exists(self.basedir):
            os.mkdir(self.basedir)
        
        opt = self.opt
        
        """Model"""
        netG = NCSNpp(self.opt).to(device)
        self.netG = netG
        self.model = ELD_iter_model.ELDModelIter(self.netG)
        self.model.initialize(opt)
        if not opt.no_log:
            self.writer = util.get_summary_writer(os.path.join(self.basedir, 'logs'))

    def train(self, gpu, rank):
        device = torch.device('cuda:{}'.format(rank))
        print('\nEpoch: %d' % self.epoch)
        avg_meters = util.AverageMeters()
        opt = self.opt
        model = self.model
        epoch = self.epoch
        epoch_start_time = time.time()
        batch_size = opt.batch_size
        nz = opt.nz #latent dimension
        # model.print_optimizer_param()

        cudnn.benchmark = True

        evaldir = './datasets/SID/Sony'
        traindir = './datasets/train'

        expo_ratio = [100, 300] # [100, 250, 300]
        read_expo_ratio = lambda x: float(x.split('_')[-1][:-5])
        # evaluate 15 indoor scenes (but you can also evaluate the performance on the whole dataset)
        indoor_ids = dataset.read_paired_fns('./SID_Sony_15_paired.txt')
        eval_fns_list = [[(fn[0], fn[1]) for fn in indoor_ids if int(fn[2]) == ratio] for ratio in expo_ratio]

        cameras = ['CanonEOS5D4', 'CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']
        noise_model = noise.NoiseModel(model="P+G+r+u", include=4)

        repeat = 1 if opt.max_dataset_size is None else 1288 / opt.max_dataset_size
        print('[i] repeat:', repeat)

        CRF = None
        if opt.crf:
            print('[i] enable CRF')
            CRF = process.load_CRF()

        if opt.stage_out == 'srgb':
            target_data = lmdb_dataset.LMDBDataset(join(traindir, 'SID_Sony_SRGB_CRF.db'))
        else:
            target_data = lmdb_dataset.LMDBDataset(
                join(traindir, 'SID_Sony_Raw.db'),
                size=opt.max_dataset_size, repeat=repeat)
        if opt.stage_in == 'srgb':
            input_data = datasets.ISPDataset(
                lmdb_dataset.LMDBDataset(join(traindir, 'SID_Sony_Raw.db')),
                noise_maker=noise_model, CRF=CRF)
        else:
            ## Synthesizing noise on-the-fly by noise model    
            input_data = datasets.SynDataset(
                lmdb_dataset.LMDBDataset(join(traindir, 'SID_Sony_Raw.db')),
                noise_maker=noise_model, num_burst=1,
                size=opt.max_dataset_size, repeat=repeat, continuous_noise=opt.continuous_noise)

            ## Noise generated offline    
            # camera = cameras[opt.include]
            # input_data = lmdb_dataset.LMDBDataset(
            #     join(traindir, f'SID_Sony_syn_Raw_{camera}.db'),
            #     size=opt.max_dataset_size, repeat=repeat)

        train_dataset =  datasets.ELDTrainDataset(target_dataset=target_data, input_datasets=[input_data])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas= self.opt.world_size,
                                                                    rank=self.opt.rank)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=4,
                                                pin_memory=True,
                                                sampler=train_sampler,
                                                drop_last = True)
        
        netD = Discriminator_large(nc = 2*self.opt.num_channels, ngf = self.opt.ngf, 
                                   t_emb_dim = opt.t_emb_dim,
                                   act=nn.LeakyReLU(0.2)).to(device)
        
        broadcast_params(self.netG.parameters())
        broadcast_params(netD.parameters())
        optimizerD = optim.Adam(netD.parameters(), lr=self.opt.lr_d, betas = (self.opt.beta1, self.opt.beta2))
    
        optimizerG = optim.Adam(netG.parameters(), lr=self.opt.lr_g, betas = (self.opt.beta1, self.opt.beta2))
    
        if self.opt.use_ema:
            optimizerG = EMA(optimizerG, ema_decay=opt.ema_decay)
        
        schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, self.opt.num_epoch, eta_min=1e-5)
        schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, self.opt.num_epoch, eta_min=1e-5)

        netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu])
        netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])

        exp = opt.exp
        parent_dir = "./saved_info/dd_gan/{}".format(opt.dataset)

        exp_path = os.path.join(parent_dir,exp)
        if rank == 0:
            if not os.path.exists(exp_path):
                os.makedirs(exp_path)
                copy_source(__file__, exp_path)
                shutil.copytree('score_sde/models', os.path.join(exp_path, 'score_sde/models'))
        # 是否接着训练
        if self.opt.resume:
            checkpoint_file = os.path.join(exp_path, 'content.pth')
            checkpoint = torch.load(checkpoint_file, map_location=device)
            init_epoch = checkpoint['epoch']
            epoch = init_epoch
            netG.load_state_dict(checkpoint['netG_dict'])
            # load G
            
            optimizerG.load_state_dict(checkpoint['optimizerG'])
            schedulerG.load_state_dict(checkpoint['schedulerG'])
            # load D
            netD.load_state_dict(checkpoint['netD_dict'])
            optimizerD.load_state_dict(checkpoint['optimizerD'])
            schedulerD.load_state_dict(checkpoint['schedulerD'])
            global_step = checkpoint['global_step']
            print("=> loaded checkpoint (epoch {})"
                    .format(checkpoint['epoch']))
        else:
            global_step, epoch, init_epoch = 0, 0, 0
        
        for epoch in range(init_epoch, opt.num_epoch+1):
            train_sampler.set_epoch(epoch)
            for i, data in enumerate(train_loader):
                for p in netD.parameters():  
                    p.requires_grad = True 

                model.set_input(data, mode='train')
                latent_z = torch.randn(batch_size, nz, device=device)
                output_list, target_list, ratio_list = model.forward(train_loader, opt.iter_num, latent_z)
                iter_num=len(ratio_list)-1 if not isinstance(ratio_list, torch.Tensor) else ratio_list.shape[1]-1
                netD.zero_grad()
                errD_real_total = 0
                for i in range(iter_num):
                    target_list[i].requires_grad = True
                    D_real = netD(target_list[i], i/iter_num, target_list[i+1].detach()).view(-1)
                    errD_real = F.softplus(-D_real)
                    errD_real = errD_real.mean()
                    errD_real_total += errD_real
                errD_real_total.backward(retain_graph=True)

                if opt.lazy_reg is None:
                    grad_real = torch.autograd.grad(
                                outputs=D_real.sum(), inputs=target_list[i], create_graph=True
                                )[0]
                    grad_penalty = (
                                    grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                                    ).mean()
                    
                    
                    grad_penalty = opt.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()
                else:
                    if global_step % opt.lazy_reg == 0:
                        grad_real = torch.autograd.grad(
                                outputs=D_real.sum(), inputs=target_list[i], create_graph=True
                                )[0]
                        grad_penalty = (
                                    grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                                    ).mean()

                
                
                    grad_penalty = opt.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()

               
                errD_fake_total = 0
                for i in range(iter_num):
                    target_list[i].requires_grad = True
                    D_real = netD(output_list[i], i/iter_num, output_list[i+1].detach()).view(-1)
                    errD_fake = F.softplus(-D_real)
                    errD_fake = errD_fake.mean()
                    errD_fake_total += errD_fake
                errD_fake_total.backward(retain_graph=True)
        
                
                errD = errD_real_total + errD_fake_total
                # Update D
                optimizerD.step()

                #update G
                for p in netD.parameters():
                    p.requires_grad = False
                netG.zero_grad()
                errG_total = 0
                for i in range(iter_num):
                    output = netD(output_list[i], ratio_list[i]/np.max(ratio_list), target_list[i+1].detach()).view(-1)
                    errG = F.softplus(-output)
                    errG = errG.mean()
                    errG_total += errG
                errG_total.backward()   
                optimizerG.step()        

                global_step += 1
                if global_step % 100 == 0:
                    if rank == 0:
                        print('epoch {} iteration{}, G Loss: {}, D Loss: {}'.format(epoch,global_step, errG.item(), errD.item()))

                if not opt.no_lr_decay:
                    
                    schedulerG.step()
                    schedulerD.step()

                if rank == 0:
                    if epoch % 10 == 0:
                        torchvision.utils.save_image(output_list[-1], os.path.join(exp_path, 'xpos_epoch_{}.png'.format(epoch)), normalize=True)
                    
        
                    
                    if opt.save_content:
                        if epoch % opt.save_content_every == 0:
                            print('Saving content.')
                            content = {'epoch': epoch + 1, 'global_step': global_step, 'args': opt,
                                    'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                                    'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                                    'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
                            
                            torch.save(content, os.path.join(exp_path, 'content.pth'))
                        
                    if epoch % opt.save_ckpt_every == 0:
                        if opt.use_ema:
                            optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                            
                        torch.save(netG.state_dict(), os.path.join(exp_path, 'netG_{}.pth'.format(epoch)))
                        if opt.use_ema:
                            optimizerG.swap_parameters_with_ema(store_params_in_ema=True)


        # train_loader.reset()

    def eval(self, val_loader, dataset_name, savedir=None, loss_key=None, **kwargs):
        iter_num = kwargs.get("iter_num", None)
        if iter_num:
            print("[i] Evaluation using iterartion number of %d"%iter_num)
        avg_meters = util.AverageMeters()
        model = self.model
        opt = self.opt
        with torch.no_grad():
            for i, data in enumerate(val_loader):                
                index = model.eval(data, savedir=savedir, **kwargs)
                # print(data['fn'], index)
                avg_meters.update(index)
                
                util.progress_bar(i, len(val_loader), str(avg_meters))
                
        if not opt.no_log:
            util.write_loss(self.writer, join('eval', dataset_name), avg_meters, self.epoch)
        
        if loss_key is not None:
            val_loss = avg_meters[loss_key]
            if val_loss < self.best_val_loss: # larger value indicates better
                self.best_val_loss = val_loss
                print('saving the best model at the end of epoch %d, iters %d' % 
                    (self.epoch, self.iterations))
                model.save(label='best_{}_{}'.format(loss_key, dataset_name))

        return avg_meters

    def test(self, test_loader, savedir=None, **kwargs):
        model = self.model
        opt = self.opt
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                model.test(data, savedir=savedir, **kwargs)
                util.progress_bar(i, len(test_loader))

    def set_learning_rate(self, lr):
        for optimizer in self.model.optimizers:
            print('[i] set learning rate to {}'.format(lr))
            util.set_opt_param(optimizer, 'lr', lr)

    @property
    def iterations(self):
        return self.model.iterations

    @iterations.setter
    def iterations(self, i):
        self.model.iterations = i

    @property
    def epoch(self):
        return self.model.epoch

    @epoch.setter
    def epoch(self, e):
        self.model.epoch = e
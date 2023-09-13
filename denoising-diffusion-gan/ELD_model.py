import torch
from torch import nn
import torch.nn.functional as F

import os
import numpy as np
import fnmatch
from collections import OrderedDict

import util.util as util
import util.index as index
import models.networks as networks
from models import arch, losses

from .base_model import BaseModel
from PIL import Image
from os.path import join

import rawpy
import util.process as process
from torchvision.utils import save_image

def tensor2im(image_tensor, visualize=False, video=False):    
    image_tensor = image_tensor.detach()

    if visualize:                
        image_tensor = image_tensor[:, 0:3, ...]

    if not video: 
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    else:
        image_numpy = image_tensor.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1))) * 255.0

    image_numpy = np.clip(image_numpy, 0, 255)

    return image_numpy


def postprocess_bayer(rawpath, img4c):
    img4c = img4c.detach()
    img4c = img4c[0].cpu().float().numpy()
    img4c = np.clip(img4c, 0, 1)

    #unpack 4 channels to Bayer image
    raw = rawpy.imread(rawpath)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    G2 = np.where(raw_pattern==3)
    B = np.where(raw_pattern==2)
    
    black_level = np.array(raw.black_level_per_channel)[:,None,None]

    white_point = 16383

    img4c = img4c * (white_point - black_level) + black_level
    
    img_shape = raw.raw_image_visible.shape
    H = img_shape[0]
    W = img_shape[1]

    raw.raw_image_visible[R[0][0]:H:2, R[1][0]:W:2] = img4c[0, :,:]
    raw.raw_image_visible[G1[0][0]:H:2,G1[1][0]:W:2] = img4c[1, :,:]
    raw.raw_image_visible[B[0][0]:H:2,B[1][0]:W:2] = img4c[2, :,:]
    raw.raw_image_visible[G2[0][0]:H:2,G2[1][0]:W:2] = img4c[3, :,:]
    
    # out = raw.postprocess(use_camera_wb=False, user_wb=[1,1,1,1], half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    # out = raw.postprocess(use_camera_wb=False, user_wb=[1.96875, 1, 1.444, 1], half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)    
    out = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    return out


def postprocess_bayer_v2(rawpath, img4c):    
    with rawpy.imread(rawpath) as raw:
        out_srgb = process.raw2rgb_postprocess(img4c.detach(), raw)        
    
    return out_srgb


def postprocess_xtrans(rawpath, img9c):
    img9c = img9c.detach()
    img9c = img9c[0].cpu().float().numpy()
    img9c = np.clip(img9c, 0, 1)

    #unpack 9 channels to xtrans image
    raw = rawpy.imread(rawpath)
    img_shape = raw.raw_image_visible.shape
    H = (img_shape[0] // 6) * 6
    W = (img_shape[1] // 6) * 6
    
    black_level = 1024
    white_point = 16383

    img9c = img9c * (white_point - black_level) + black_level

    # 0 R
    raw.raw_image_visible[0:H:6, 0:W:6] = img9c[0, 0::2, 0::2]
    raw.raw_image_visible[0:H:6, 4:W:6] = img9c[0, 0::2, 1::2]
    raw.raw_image_visible[3:H:6, 1:W:6] = img9c[0, 1::2, 0::2]
    raw.raw_image_visible[3:H:6, 3:W:6] = img9c[0, 1::2, 1::2]

    # 1 G
    raw.raw_image_visible[0:H:6, 2:W:6] = img9c[1, 0::2, 0::2]
    raw.raw_image_visible[0:H:6, 5:W:6] = img9c[1, 0::2, 1::2]
    raw.raw_image_visible[3:H:6, 2:W:6] = img9c[1, 1::2, 0::2]
    raw.raw_image_visible[3:H:6, 5:W:6] = img9c[1, 1::2, 1::2]

    # 1 B
    raw.raw_image_visible[0:H:6, 1:W:6] = img9c[2, 0::2, 0::2]
    raw.raw_image_visible[0:H:6, 3:W:6] = img9c[2, 0::2, 1::2]
    raw.raw_image_visible[3:H:6, 0:W:6] = img9c[2, 1::2, 0::2]
    raw.raw_image_visible[3:H:6, 4:W:6] = img9c[2, 1::2, 1::2]

    # 4 R
    raw.raw_image_visible[1:H:6, 2:W:6] = img9c[3, 0::2, 0::2]
    raw.raw_image_visible[2:H:6, 5:W:6] = img9c[3, 0::2, 1::2] 
    raw.raw_image_visible[5:H:6, 2:W:6] = img9c[3, 1::2, 0::2] 
    raw.raw_image_visible[4:H:6, 5:W:6] = img9c[3, 1::2, 1::2] 

    # 5 B
    raw.raw_image_visible[2:H:6, 2:W:6] = img9c[4, 0::2, 0::2]
    raw.raw_image_visible[1:H:6, 5:W:6] = img9c[4, 0::2, 1::2]
    raw.raw_image_visible[4:H:6, 2:W:6] = img9c[4, 1::2, 0::2]
    raw.raw_image_visible[5:H:6, 5:W:6] = img9c[4, 1::2, 1::2]

    raw.raw_image_visible[1:H:3, 0:W:3] = img9c[5, :, :]
    raw.raw_image_visible[1:H:3, 1:W:3] = img9c[6, :, :]
    raw.raw_image_visible[2:H:3, 0:W:3] = img9c[7, :, :]
    raw.raw_image_visible[2:H:3, 1:W:3] = img9c[8, :, :]
    
    out = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    # out = raw.postprocess(use_camera_wb=True, demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    return out


class IlluminanceCorrect(nn.Module):
    def __init__(self):
        super(IlluminanceCorrect, self).__init__()
    
    # Illuminance Correction
    def forward(self, predict, source):
        if predict.shape[0] != 1:
            output = torch.zeros_like(predict)
            if source.shape[0] != 1:
                for i in range(predict.shape[0]):
                    output[i:i+1, ...] = self.correct(predict[i:i+1, ...], source[i:i+1, ...])               
            else:                                     
                for i in range(predict.shape[0]):
                    output[i:i+1, ...] = self.correct(predict[i:i+1, ...], source)                    
        else:
            output = self.correct(predict, source)
        return output

    def correct(self, predict, source):
        N, C, H, W = predict.shape        
        predict = torch.clamp(predict, 0, 1)
        assert N == 1
        output = torch.zeros_like(predict, device=predict.device)
        pred_c = predict[source != 1]
        source_c = source[source != 1]
        
        num = torch.dot(pred_c, source_c)
        den = torch.dot(pred_c, pred_c)        
        output = num / den * predict
        # print(num / den)

        return output


class ELDModelBase(BaseModel):
    def set_input(self, data, mode='train'):
        target = None
        data_name = None

        mode = mode.lower()
        if mode == 'train':
            input, target, self.ratio, self.K = data['input'], data['target'], data["ratio"], data["K"]
        elif mode == 'eval':
            input, target, data_name, self.ratio, self.K = data['input'], data['target'], data['fn'], data["ratio"], data["K"].item()
        elif mode == 'test':
            input, data_name = data['input'], data['fn']
        else:
            raise NotImplementedError('Mode [%s] is not implemented' % mode)
        
        if len(self.gpu_ids) > 0:  # transfer data into gpu
            input = input.to(device=self.gpu_ids[0])
            if target is not None:
                target = target.to(device=self.gpu_ids[0]) 

        self.input = input
        self.target = target
        self.data_name = data_name

        self.rawpath = data['rawpath'][0] if 'rawpath' in data else None
        self.cfa = data['cfa'][0] if 'cfa' in data else 'bayer'

        # self.issyn = False if 'real' in data else True
        self.aligned = False if 'unaligned' in data else True

            
    def eval(self, data, savedir=None, suffix=None, correct=False, crop=True, frame_id=None, iter_num=0, old_diffusion=False):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'eval')

        # if self.data_name is not None and savedir is not None:
        #     name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
        #     if not os.path.exists(join(savedir, name)):
        #         os.makedirs(join(savedir, name))
            
        #     for fn in os.listdir(join(savedir, name)):                
        #         if fnmatch.fnmatch(fn, '*{}_*'.format(self.opt.name)):
        #             return {}

        with torch.no_grad():
            ### evaluate center region to avoid fixed pattern noise
            cropx = 512; cropy = 512
            if crop:
                self.target = util.crop_center(self.target, cropx, cropy)
                self.input = util.crop_center(self.input, cropx, cropy)
            if not old_diffusion:
                if not self.opt.chop:
                    self.target, para = util.auto_padding(self.target, scale=16)
                    self.input, para = util.auto_padding(self.input, scale=16)
                output_list = self.forward(iter_num=iter_num)
                self.output = output_list[0]
            else:
                ABLATION_NUM = 0
                num_steps = 3
                if ABLATION_NUM == 0:
                    def to_photon(x, ratio, K):
                        # x: 0-1 image
                        return x/ratio/K*15583
                    def to_image(x, ratio_next, target_ratio, K):
                        return x/ratio_next*target_ratio*K/15583
                    ratio = int(self.ratio)
                    ratio_list = [1,50, 100]
                    # ratio_list = [1,2, 3]
                    # ratio_list =sorted(set(np.logspace(np.log10(1),np.log10(300),num_steps).astype(int)))
                    # ratio_list =[1,3]
                    # ratio_list =sorted(set(np.logspace(np.log10(1),np.log10(3),10).astype(float)))
                    for i in range(len(ratio_list)-1):
                        output_list = self.forward()
                        self.output = output_list[0]
                        ratio_current, ratio_next = ratio_list[i], ratio_list[i+1]
                        self.input = to_image(to_photon(self.input.clamp(0,1), ratio/ratio_current, self.K) + torch.poisson(to_photon(output.clamp(0,1), ratio/(ratio_next-ratio_current), self.K)), ratio_next, ratio, self.K)
                        
                        # self.input = to_image(to_photon(self.input, ratio, self.K)*ratio_current + to_photon(output, ratio, self.K)*(ratio_next-ratio_current), ratio_next, ratio, self.K)
                    output_list = self.forward()
                    self.output = output[0]
            #     elif ABLATION_NUM == 1:
            #         for i in range(num_steps):
            #             self.input = self.forward()
            #         self.output=self.input
                    
            #     elif ABLATION_NUM == 2:
            #         original_input = self.input
            #         for i in range(1,num_steps+1):
            #             self.output = self.forward()
            #             self.input = self.output * i/num_steps + (num_steps-i)/num_steps*original_input

            if not self.opt.chop:
                self.input = F.pad(self.input, para)
                self.target = F.pad(self.target, para)
                self.output = F.pad(self.output, para)
            if correct:
                self.output = self.corrector(self.output, self.target)
            
            if self.opt.stage_out == 'raw' and self.opt.stage_eval == 'srgb':
                target = postprocess_bayer_v2(self.rawpath, self.target)
                output = postprocess_bayer_v2(self.rawpath, self.output)
                input = postprocess_bayer_v2(self.rawpath, self.input)
            else:
                output = self.output
                target = self.target
                input = self.input

            output = tensor2im(output)
            target = tensor2im(target)   
            input = tensor2im(input)

            if target.shape[0] != output.shape[0]:
                target = np.repeat(target, output.shape[0], axis=0)

            res = index.quality_assess(output, target, data_range=255)
            res_in = index.quality_assess(input, target, data_range=255)  

            if savedir is not None and not crop:
                ## raw postprocessing
                if self.rawpath:
                    if self.cfa == 'bayer':
                        output = postprocess_bayer(self.rawpath, self.output)
                        target = postprocess_bayer(self.rawpath, self.target)
                        input = postprocess_bayer(self.rawpath, self.input)

                        # target = tensor2im(postprocess_bayer_v2(self.rawpath, self.target))
                        # output = tensor2im(postprocess_bayer_v2(self.rawpath, self.output))
                        # input = tensor2im(postprocess_bayer_v2(self.rawpath, self.input))

                    elif self.cfa == 'xtrans':
                        output = postprocess_xtrans(self.rawpath, self.output)
                        target = postprocess_xtrans(self.rawpath, self.target)
                        input = postprocess_xtrans(self.rawpath, self.input)
                    else:
                        raise NotImplementedError

                if self.data_name is not None:
                    if "ELD" in self.data_name[0]:
                        name = "_".join(self.data_name[0].split("/")[-2:]).split(".")[0]
                    else:
                        name = os.path.splitext(os.path.basename(self.data_name[0]))[0]

                    if not os.path.exists(join(savedir, name)):
                        os.makedirs(join(savedir, name))

                    if frame_id is not None:
                        if not os.path.exists(join(savedir, name, self.opt.name)):
                            os.makedirs(join(savedir, name, self.opt.name))

                        if not os.path.exists(join(savedir, name, 'input')):
                            os.makedirs(join(savedir, name, 'input'))                            

                        Image.fromarray(output.astype(np.uint8)).save(join(savedir, name, self.opt.name, '{}_{:.2f}.png'.format(frame_id, res['PSNR'])))

                        if not os.path.exists(join(savedir, name, 'input', '{}_{:.2f}.png'.format(frame_id, res_in['PSNR']))):
                            Image.fromarray(input.astype(np.uint8)).save(join(savedir, name, 'input', '{}_{:.2f}.png'.format(frame_id, res_in['PSNR'])))

                        if not os.path.exists(join(savedir, name, 'label')):
                            os.makedirs(join(savedir, name, 'label'))   

                        if not os.path.exists(join(savedir, name, 'label', '{}.png'.format(frame_id))):
                            Image.fromarray(target.astype(np.uint8)).save(join(savedir, name, 'label', '{}.png'.format(frame_id)))
                    else:
                        if suffix is not None:
                            Image.fromarray(output.astype(np.uint8)).save(join(savedir, name,'{}_{:.1f}_{}.png'.format(self.opt.name, res['PSNR'], suffix)))
                            # Image.fromarray(output.astype(np.uint8)).save(join(savedir, name,'{}_{:.1f}_{}.jpg'.format(self.opt.name, res['PSNR'], suffix)), optimize=True, quality=90)
                            Image.fromarray(input.astype(np.uint8)).save(join(savedir, name, 'm_input_{}.png'.format(suffix)))
                            # Image.fromarray(input.astype(np.uint8)).save(join(savedir, name, 'm_input_{}.jpg'.format(suffix)), optimize=True, quality=90)
                        else:
                            # Image.fromarray(output.astype(np.uint8)).save(join(savedir, name, '{}_{:.1f}.jpg'.format(self.opt.name, res['PSNR'])), optimize=True, quality=90)
                            Image.fromarray(output.astype(np.uint8)).save(join(savedir, name, '{}_{:.2f}_{:.2f}.png'.format(self.opt.model_path.split("/")[-2], res['PSNR'], res['SSIM'])))
                            # Image.fromarray(output.astype(np.uint8)).save(join(savedir, name, '{}.png'.format(self.opt.name)))
                            Image.fromarray(input.astype(np.uint8)).save(join(savedir, name, 'm_input_{:.2f}_{:.2f}.png'.format(res_in['PSNR'], res_in['SSIM'])))
                            # Image.fromarray(input.astype(np.uint8)).save(join(savedir, name, 'm_input.jpg'), optimize=True, quality=90)
                        
                        Image.fromarray(target.astype(np.uint8)).save(join(savedir, name, 't_label.png'))
                        # Image.fromarray(target.astype(np.uint8)).save(join(savedir, name, 't_label.jpg'), optimize=True, quality=90)

            return res

    def test(self, data, savedir=None, video_mode=False):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'test')

        if self.data_name is not None and savedir is not None:
            name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
            if not video_mode:
                if not os.path.exists(join(savedir, name)):
                    os.makedirs(join(savedir, name))

                # if os.path.exists(join(savedir, name, '{}.png'.format(self.opt.name))):
                for fn in os.listdir(join(savedir, name)):
                    if fnmatch.fnmatch(fn, '*{}_*'.format(self.opt.name)):
                        return
            else:
                if not os.path.exists(join(savedir, self.opt.name)):
                    os.makedirs(join(savedir, self.opt.name))
                
        with torch.no_grad():
            output = self.forward()
            
            # if self.opt.netG == 'fastdvd': # video network  
            #     self.input = self.input[:, 8:12, ...]

            ## raw postprocessing
            if self.rawpath:
                if self.opt.stage_in == 'srgb':
                    output = tensor2im(self.output)
                    input = tensor2im(self.input)
                else:
                    output = postprocess_bayer(self.rawpath, self.output)
                    input = postprocess_bayer(self.rawpath, self.input)                  

                if not video_mode:
                    Image.fromarray(output.astype(np.uint8)).save(join(savedir, name,'{}.jpg'.format(self.opt.name)), optimize=True, quality=90)
                    Image.fromarray(input.astype(np.uint8)).save(join(savedir, name, 'm_input.jpg'), optimize=True, quality=90)
                else:
                    Image.fromarray(output.astype(np.uint8)).save(join(savedir, self.opt.name,'{}.jpg'.format(name)), optimize=True, quality=90)

        return output
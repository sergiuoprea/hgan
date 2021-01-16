import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import transforms
from torch.nn import init
import functools
import random
import numpy as np
import os
from PIL import Image
from torch.nn import functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, padding=0,
                 norm_layer=None, act_layer=None):
        super(ConvBlock, self).__init__()

        layer_list = []

        layer_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=ksize,
                                    stride=stride, padding=padding))

        if norm_layer:
            layer_list.append(norm_layer(out_channels))

        if act_layer:
            layer_list.append(act_layer)

        self.module = nn.Sequential(*layer_list)

    def forward(self, inp):
        """ Simple forward function """
        return self.module(inp)

class TransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, padding=0,
                 output_padding=0, norm_layer=None, act_layer=None):
        super(TransConvBlock, self).__init__()

        layer_list = []

        layer_list.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=ksize,
                                             stride=stride, padding=padding,
                                             output_padding=output_padding))
        if norm_layer:
            layer_list.append(norm_layer(out_channels))

        if act_layer:
            layer_list.append(act_layer)

        self.module = nn.Sequential(*layer_list)

    def forward(self, inp):
        """ Simple forward function """
        return self.module(inp)

class ResidualBlock(nn.Module):
    def __init__(self, channels, norm_layer=None, use_dropout=False):
        super(ResidualBlock, self).__init__()
        layer_list = []
        layer_list += [nn.ReflectionPad2d(1),
                       ConvBlock(channels, channels, ksize=3, norm_layer=norm_layer,
                                 act_layer=nn.ReLU(True))]

        if use_dropout:
            layer_list += [nn.Dropout(0.5)]

        layer_list += [nn.ReflectionPad2d(1),
                       ConvBlock(channels, channels, ksize=3, norm_layer=norm_layer)]

        self.module = nn.Sequential(*layer_list)

    def forward(self, inp):
        """ Forward function with skip connections """
        return inp + self.module(inp)
    
class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError(f'Normalization layer {norm_layer} is not found!')

    return norm_layer

def init_weights(net, init_type='xavier', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier
                           | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    Function extracted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or
                                     classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] \
                                           is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def set_mode(nets, mode='train'):
    for _, net in nets.items():
        if mode == 'train':
            net.train()
        elif mode == 'eval':
            net.eval()
        else:
            print(f'Unable to set network in mode {mode}!')

    print(f'Models {nets.keys()} are now in {mode} mode.!')

def set_requires_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad

def save_model(nets, optimizers, tag, dir_path):
    _out_path = os.path.join(dir_path, "chk_" + tag + '.pkl')
    _save_format = {}

    for name, net in nets.items():
        _save_format[name+"_state_dict"] = net.state_dict()
    for name, opt in optimizers.items():
        _save_format[name+"_state_dict"] = opt.state_dict()

    torch.save(_save_format, _out_path)

def load_model(nets, optimizers, start_epoch, start_iter, dir_path, device):
    _chk_file = os.path.join(dir_path, f'chk_epoch:{start_epoch}_iteration:{start_iter}.pkl')

    if os.path.isfile(_chk_file):
        _state_dict = torch.load(_chk_file, map_location=device)

        if hasattr(_state_dict, '_metadata'):
            del _state_dict._metadata

        for name in nets.keys():
            nets[name].load_state_dict(_state_dict[name+"_state_dict"])
        for opt in optimizers.keys():
            optimizers[opt].load_state_dict(_state_dict[opt+"_state_dict"])
    else:
        print(f'Loading checkpoint file {_chk_file} failed!')

    print(f'Models {nets.keys()} and optimizers {optimizers.keys()} were initilized successfully with weights {_chk_file}.')

def get_mse_loss(output, label):
    if label.lower() == 'real':
        target = torch.ones_like(output)
    else:
        target = torch.zeros_like(output)

    return F.mse_loss(output, target)

"""def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def save_imgs(images, out_dir):
    to_pil = transforms.ToPILImage()
    for i, image in enumerate(images):
        _filename = os.path.join(out_dir, f'output_{i}.png')
        _pil_img = to_pil(denormalize(image.detach().cpu()))#Image.fromarray(np.uint8(denormalize(image)))
        _pil_img.save(_filename, "JPEG")
        
    print(f'{len(images)} properly saved to {out_dir}!')"""

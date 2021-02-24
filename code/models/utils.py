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

        layer_list = [nn.Conv2d(in_channels, out_channels, kernel_size=ksize,
                                stride=stride, padding=padding)]

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

        layer_list = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=ksize,
                                         stride=stride, padding=padding,
                                         output_padding=output_padding)]
        if norm_layer:
            layer_list.append(norm_layer(out_channels))

        if act_layer:
            layer_list.append(act_layer)

        self.module = nn.Sequential(*layer_list)

    def forward(self, inp):
        """ Simple forward function """
        return self.module(inp)

class ResidualBlock(nn.Module):
    """
                        Defines a ResnetBlock
    X ------------------------identity------------------------
    |-- Convolution -- Norm -- ReLU -- Convolution -- Norm --|

    Parameters:
        in_ch: Number of input channels
        norm_layer: Batch or Instance normalization
        use_dropout: If set to True, activations will be 0'ed with a probability of 0.5 by default
    """

    def __init__(self, in_ch, norm_layer=None, use_dropout=False):
        super(ResidualBlock, self).__init__()
        layer_list = [nn.ReflectionPad2d(1),
                      ConvBlock(in_ch, in_ch, ksize=3, stride=1, norm_layer=norm_layer,
                                act_layer=nn.ReLU(True))]

        if use_dropout:
            layer_list += [nn.Dropout(0.5)]

        layer_list += [nn.ReflectionPad2d(1),
                       ConvBlock(in_ch, in_ch, ksize=3, stride=1, norm_layer=norm_layer)]

        self.module = nn.Sequential(*layer_list)

    def forward(self, inp):
        """ Forward function with skip connections """
        return inp + self.module(inp)
    
class ImagePool:
    """
    This class implements an image buffer that stores previously generated images!
    This enables to update the discriminators using a buffer of generated images rather
    than the latest one produced by the generator.
    """

    def __init__(self, pool_size: int = 50):
        """
        Parameters:
            pool_size: size of the buffer, i.e. max number of images stored in the buffer.
                       If pool_size = 0 no buffer is created.
        """
        self.pool_size = pool_size

        # Creating and empty pool
        if self.pool_size > 0:
            self.images = []
            self.num_imgs = 0

    def query(self, images):
        """Return a batch of images from the pool after adding the current
        images to the buffer.

        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0: # No buffer, so return the input batch
            return images

        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)

            # If the buffer is not full; keep inserting current images to the buffer
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                # 50% chance to return an image from the buffer, 
                # previously swapping it with the current image
                if random.uniform(0, 1) > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else: # by another 50% chance, the buffer will return the current image
                    return_images.append(image)

        return torch.cat(return_images, 0)

class Initializer:
    """
    To initialize network weights.
    """

    def __init__(self, init_type: str = 'normal', init_gain: float = 0.02):
        """
        Initialize the weights of the network.

        Parameters:
            init_type: Initialization method: normal | xavier | kaiming | orthogonal
            init_gain: Scaling factor
        """

        self.init_type = init_type
        self.init_gain = init_gain

    def init_module(self, m):
        cls_name = m.__class__.__name__
        if hasattr(m, 'weight') and (cls_name.find('Conv') != -1 or cls_name.find('Linear') != -1):

            if   self.init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif self.init_type == 'xavier' :
                nn.init.xavier_normal_ (m.weight.data,  gain = self.init_gain)
            elif self.init_type == 'normal' :
                nn.init.normal_(m.weight.data, mean = 0, std = self.init_gain)
            else: raise ValueError('Initialization not found!!')

            if m.bias is not None:
                nn.init.constant_(m.bias.data, val = 0)

        if hasattr(m, 'weight') and cls_name.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, mean = 1.0, std = self.init_gain)
            nn.init.constant_(m.bias.data, val = 0)

    def __call__(self, net):
        net.apply(self.init_module)
        return net

def set_requires_grad(nets, requires_grad: bool = False):
    """
    Set requires_grad to False for all the networks in the provided list to
    avoid unnecessary computations.

    Parameters:
        nets (list): a list of networks
        requires_grad (bool): whether the networks require gradients or not
    """
    if not isinstance(nets, list): nets = [nets]
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad

def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError(f'Normalization layer {norm_layer} is not found!')

    return norm_layer

def set_mode(nets, mode='train'):
    for _, net in nets.items():
        if mode == 'train':
            net.train()
        elif mode == 'eval':
            net.eval()
        else:
            print(f'Unable to set network in mode {mode}!')

    print(f'Models {nets.keys()} are now in {mode} mode.!')

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

def save_to_disk(batch, batch_idx, path, denorm=None, name='test'):
    to_pil = transforms.ToPILImage()

    if denorm:
        batch = denorm(batch)

    for i, image in enumerate(batch):
        _filename = os.path.join(path, f'{name}_{batch_idx}_{i}.jpg')
        _pil_img = to_pil(image.detach().cpu())
        _pil_img.save(_filename, "JPEG")

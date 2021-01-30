import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.utils import ConvBlock, ResidualBlock, TransConvBlock, get_norm_layer

class Resnet(nn.Module):
    """Resnet-based generator consisting of a downsampling step (7x7 conv to encode color features
    + two 3x3 2-strided convs for the downsampling), Resnet blocks in between, and the upsampling
    step (two 3x3 2-strided transposed convs + one 7x7 final conv)"""
    def __init__(self, hparams, deconvolution=True, n_blocks=6):
        """Construct a Resnet-based generator
        Parameters:
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
        """
        super(Resnet, self).__init__()

        norm_layer = get_norm_layer(hparams.norm)

        # First 7x7 convolution to encode color information + 2 downsampling steps
        model = [nn.ReflectionPad2d(3),
                 ConvBlock(hparams.inp_ch, hparams.ngf * 1, ksize=7, stride=1, padding=0,
                           norm_layer=norm_layer, act_layer=nn.ReLU(True)),
                 ConvBlock(hparams.ngf * 1, hparams.ngf * 2, ksize=3, stride=2, padding=1,
                           norm_layer=norm_layer, act_layer=nn.ReLU(True)),
                 ConvBlock(hparams.ngf * 2, hparams.ngf * 4, ksize=3, stride=2, padding=1,
                           norm_layer=norm_layer, act_layer=nn.ReLU(True))]

        # ResNet blocks
        for _ in range(n_blocks):
            model += [ResidualBlock(hparams.ngf * 4, norm_layer=norm_layer,
                                    use_dropout=hparams.use_dropout)]

        # Upsampling phase
        if deconvolution:
            model += [TransConvBlock(hparams.ngf * 4, hparams.ngf * 2, ksize=3, stride=2, padding=1,
                                     output_padding=1, norm_layer=norm_layer,
                                     act_layer=nn.ReLU(True)),
                      TransConvBlock(hparams.ngf * 2, hparams.ngf * 1, ksize=3, stride=2, padding=1,
                                     output_padding=1, norm_layer=norm_layer,
                                     act_layer=nn.ReLU(True))]
        else:
            model += [nn.Upsample(scale_factor=2, mode='bilinear'),
                      nn.ReflectionPad2d(1),
                      ConvBlock(hparams.ngf * 4, hparams.ngf * 2, ksize=3, norm_layer=norm_layer,
                                act_layer=nn.ReLU(True))]
            model += [nn.Upsample(scale_factor=2, mode='bilinear'),
                      nn.ReflectionPad2d(1),
                      ConvBlock(hparams.ngf * 2, hparams.ngf * 1, ksize=3, norm_layer=norm_layer,
                                act_layer=nn.ReLU(True))]

        model += [nn.ReflectionPad2d(3),
                  ConvBlock(hparams.ngf, hparams.out_ch, ksize=7, padding=0, act_layer=nn.Tanh())]

        self.model = nn.Sequential(*model)

    def forward(self, inp):
        """ Standard forward """
        return self.model(inp)


def generator(hparams):
    model = None

    if hparams.netG == 'resnet_9blocks':
        model = Resnet(hparams, n_blocks=9)
    elif hparams.netG == 'resnet_6blocks':
        model = Resnet(hparams, n_blocks=6)
    elif hparams.netG == 'resnet_4blocks':
        model = Resnet(hparams, n_blocks=4)
    else:
        raise NotImplementedError(f'Generator model name {hparams.netG} is not recognized.')

    return model

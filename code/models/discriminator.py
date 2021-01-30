import torch
from math import log
import torch.nn as nn
import pytorch_lightning as pl
from models.utils import ConvBlock, get_norm_layer

class PatchGAN_Discriminator(pl.LightningModule):
    """PatchGAN discriminator. It outputs an one channel prediction map
    """
    def __init__(self, in_ch, ndf=64, ndl=3, norm_layer=None):
        super(PatchGAN_Discriminator, self).__init__()

        layers = [ConvBlock(in_ch, ndf, ksize=4, stride=2, padding=1,
                            act_layer=nn.LeakyReLU(0.2, True))]

        nf_mult = 1
        nf_mult_prev = 1

        for i in range(1, ndl):
            nf_mult_prev = nf_mult
            nf_mult = min(2**i, 8)
            layers += [ConvBlock(ndf * nf_mult_prev, ndf * nf_mult, ksize=4, stride=2,
                                 padding=1, norm_layer=norm_layer,
                                 act_layer=nn.LeakyReLU(0.2, True))]

        nf_mult_prev = nf_mult
        nf_mult = min(2**ndl, 8)

        layers += [ConvBlock(ndf * nf_mult_prev, ndf * nf_mult, ksize=4, stride=1,
                             padding=1, norm_layer=norm_layer,
                             act_layer=nn.LeakyReLU(0.2, True)),
                   ConvBlock(ndf * nf_mult, 1, ksize=4, stride=1, padding=1)]

        self.model = nn.Sequential(*layers)

    def forward(self, inp):
        return [self.model(inp)]

class Pixel_Discriminator(pl.LightningModule):
    """ 1x1 PatchGAN discriminator from the pixelGAN """
    def __init__(self, in_ch, ndf=64, ndl=3, norm_layer=None):
        super(Pixel_Discriminator, self).__init__()

        layers = [ConvBlock(in_ch, ndf, ksize=1, stride=1, padding=0,
                            act_layer=nn.LeakyReLU(0.2, True)),
                  ConvBlock(ndf, ndf * 2, ksize=1, stride=1, padding=0,
                            norm_layer=norm_layer, act_layer=nn.LeakyReLU(0.2, True)),
                  ConvBlock(ndf * 2, 1, ksize=1, stride=1, padding=0)]

        self.model = nn.Sequential(*layers)

    def forward(self, inp):
        """ Standard forward implementation. """
        return [self.model(inp)]

class Feature_Discriminator(pl.LightningModule):
    """ Feature discriminator"""
    def __init__(self, in_ch, in_size, ndf=64, output_scales='64, 32, 8, 4', max_channels=512):
        super(Feature_Discriminator, self).__init__()

        self.output_scales = [int(s) for s in output_scales.split(',')]

        in_num_channels = []
        num_layers = int(log(in_size // max(self.output_scales[-1], 4), 2))

        power = log(ndf, 2) + 1
        for i in range(0, num_layers):
            val = min(int(2**(power + i)), max_channels)
            in_num_channels.append(val)

        self.bottleneck_layers = nn.ModuleList()
        self.input_layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()

        in_channels = in_ch
        out_channels = ndf
        spatial_size = in_size

        for _ in range(num_layers):
            self.bottleneck_layers += [ConvBlock(in_channels, out_channels, ksize=4, stride=2, padding=1, act_layer=nn.LeakyReLU(0.2, True))]

            in_channels = in_num_channels[len(self.input_layers)]
            self.input_layers += [ConvBlock(in_channels, out_channels, ksize=3, stride=1, padding=1, act_layer=nn.LeakyReLU(0.2, True))]
            spatial_size //= 2

            if spatial_size in self.output_scales:
                self.output_layers += [ConvBlock(in_channels, 1, ksize=3, stride=1, padding=1)]

            out_channels = min(out_channels * 2, max_channels)
            in_channels = out_channels

    def forward(self, inp):
        #Encode inputs after denormalizing
        ms_preds = [] # tensor to store multiscale predictions
        out_idx = 0
        out = inp[0]

        for i, layer in enumerate(self.bottleneck_layers):
            out = layer(out)

            if i+1 < len(inp):
                next_inp = self.input_layers[i](inp[i+1])
                out = torch.cat([out, next_inp], 1) # stack on channel dimension

            if out.shape[2] in self.output_scales:
                ms_preds += [self.output_layers[out_idx](out)]
                out_idx += 1

        return ms_preds

def discriminator(hparams):
    model = None
    norm_layer = get_norm_layer(norm_type=hparams.norm)

    if hparams.netD == 'patch':
        model = PatchGAN_Discriminator(in_ch=hparams.inp_ch, ndf=hparams.ndf,
                                       ndl=hparams.ndl, norm_layer=norm_layer)
    elif hparams.netD == 'pixel':
        model = Pixel_Discriminator(in_ch=hparams.inp_ch, ndf=hparams.ndf, norm_layer=norm_layer)
    elif hparams.netD == 'perceptual':
        model = Feature_Discriminator(in_ch=64, in_size=hparams.train_inp_size, ndf=hparams.ndf,
                                      output_scales=hparams.output_scales)
    else:
        raise NotImplementedError(f'Discriminator model name {hparams.netD} is not recognized.')

    return model

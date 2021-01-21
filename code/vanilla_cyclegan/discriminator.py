import torch
import torch.nn as nn
from torchsummary import summary
import pytorch_lightning as pl
from vanilla_cyclegan.utils import ConvBlock, get_norm_layer

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
        """Standard forward implementation"""
        return self.model(inp)

class Pixel_Discriminator(pl.LightningModule):
    """ 1x1 PatchGAN discriminator from the pixelGAN """
    def __init__(self, in_ch, ndf=64, norm_layer=None):
        super(Pixel_Discriminator, self).__init__()

        layers = [ConvBlock(in_ch, ndf, ksize=1, stride=1, padding=0,
                            act_layer=nn.LeakyReLU(0.2, True)),
                  ConvBlock(ndf, ndf * 2, ksize=1, stride=1, padding=0,
                            norm_layer=norm_layer, act_layer=nn.LeakyReLU(0.2, True)),
                  ConvBlock(ndf * 2, 1, ksize=1, stride=1, padding=0)]

        self.model = nn.Sequential(*layers)

    def forward(self, inp):
        """ Standard forward implementation. """
        return self.model(inp)

def define_discriminator(in_ch, ndf, net, ndl=3, norm='batch'):
    model = None
    norm_layer = get_norm_layer(norm_type=norm)

    if net == 'patch':
        model = PatchGAN_Discriminator(in_ch, ndf, ndl, norm_layer)
    elif net == 'pixel':
        model = Pixel_Discriminator(in_ch, ndf, norm_layer)
    else:
        raise NotImplementedError(f'Discriminator model name {net} is not recognized.')

    return model

#def _test_net(net='patch'):
#    inp = torch.randn(1, 3, 256, 256, device='cuda')
#    network = define_discriminator(3, 64, net).to('cuda')
#    out = network(inp)
#    print(out)
#    summary(network, [(3, 128, 128)])

#if __name__ == '__main__':
#    _test_net('patch')

import torch
import torch.nn as nn
import pytorch_lightning as pl
#from torchsummary import summary
from vanilla_cyclegan.utils import ConvBlock, ResidualBlock, TransConvBlock, get_norm_layer

class Resnet(nn.Module):
    """Resnet-based generator consisting of a downsampling step (7x7 conv to encode color features
    + two 3x3 2-strided convs for the downsampling), Resnet blocks in between, and the upsampling
    step (two 3x3 2-strided transposed convs + one 7x7 final conv)"""
    def __init__(self, in_ch=3, out_ch=3, ngf=64,
                 norm_layer=None, use_dropout=False, deconvolution=True, n_blocks=6):
        """Construct a Resnet-based generator
        Parameters:
            in_ch (int)      -- the number of channels in input images
            out_ch (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
        """
        super(Resnet, self).__init__()

        # First 7x7 convolution to encode color information + 2 downsampling steps
        model = [nn.ReflectionPad2d(3),
                 ConvBlock(in_ch, ngf* 1, ksize=7, stride=1, padding=0,
                           norm_layer=norm_layer, act_layer=nn.ReLU(True)),
                 ConvBlock(ngf * 1, ngf * 2, ksize=3, stride=2, padding=1,
                           norm_layer=norm_layer, act_layer=nn.ReLU(True)),
                 ConvBlock(ngf * 2, ngf * 4, ksize=3, stride=2, padding=1,
                           norm_layer=norm_layer, act_layer=nn.ReLU(True))]

        # ResNet blocks
        for _ in range(n_blocks):
            model += [ResidualBlock(ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout)]

        # Upsampling phase
        if deconvolution:
            model += [TransConvBlock(ngf * 4, ngf * 2, ksize=3, stride=2, padding=1,
                                     output_padding=1, norm_layer=norm_layer,
                                     act_layer=nn.ReLU(True)),
                      TransConvBlock(ngf * 2, ngf * 1, ksize=3, stride=2, padding=1,
                                     output_padding=1, norm_layer=norm_layer,
                                     act_layer=nn.ReLU(True))]
        else:
            model += [nn.Upsample(scale_factor=2, mode='bilinear'),
                      nn.ReflectionPad2d(1),
                      ConvBlock(ngf * 4, ngf * 2, ksize=3, norm_layer=norm_layer,
                                act_layer=nn.ReLU(True))]
            model += [nn.Upsample(scale_factor=2, mode='bilinear'),
                      nn.ReflectionPad2d(1),
                      ConvBlock(ngf * 2, ngf * 1, ksize=3, norm_layer=norm_layer,
                                act_layer=nn.ReLU(True))]

        model += [nn.ReflectionPad2d(3),
                  ConvBlock(ngf, out_ch, ksize=7, padding=0, act_layer=nn.Tanh())]

        self.model = nn.Sequential(*model)

    def forward(self, inp):
        """ Standard forward """
        return self.model(inp)


def generator(in_ch, out_ch, ngf, net, norm='batch', use_dropout=False):
    model = None
    norm_layer = get_norm_layer(norm)

    if net == 'resnet_9blocks':
        model = Resnet(in_ch=in_ch, out_ch=out_ch, ngf=ngf, norm_layer=norm_layer,
                       use_dropout=use_dropout, n_blocks=9)
    elif net == 'resnet_6blocks':
        model = Resnet(in_ch=in_ch, out_ch=out_ch, ngf=ngf, norm_layer=norm_layer,
                       use_dropout=use_dropout, n_blocks=6)
    elif net == 'resnet_4blocks':
        model = Resnet(in_ch=in_ch, out_ch=out_ch, ngf=ngf, norm_layer=norm_layer,
                       use_dropout=use_dropout, n_blocks=4)
    else:
        raise NotImplementedError(f'Generator model name {net} is not recognized.')

    return model

#def _test_net():
#    inp = torch.randn(1, 3, 256, 256, device='cuda')
#    network = define_generator(3, 3, 64, 'resnet_9blocks').to('cuda')
#    out = network(inp)
#    print(out)
#    summary(network, [(3, 256, 256)])

#if __name__ == '__main__':
#    _test_net()

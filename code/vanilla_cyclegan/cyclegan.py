import torch
import os
from torch.optim import Adam
import pytorch_lightning as pl
from torch.nn import functional as F
from torchvision.utils import make_grid
from pytorch_lightning.callbacks import Callback

from vanilla_cyclegan.discriminator import discriminator
from vanilla_cyclegan.generator import generator
from vanilla_cyclegan.utils import Initializer, ImagePool, set_requires_grad, get_mse_loss, save_to_disk

import datasets.synthhands as sh
import datasets.realhands as rh

from itertools import chain
from argparse import ArgumentParser

from metrics.inception import InceptionV3
from metrics.fid import fid

from loss.vgg_perceptual import VGGPerceptualLoss

class CycleGAN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        print("Initializing Vanilla CycleGAN!")
        self.hparams = hparams
        #self.models = {} #network modules (generators and discriminators)

        init = Initializer(init_type=hparams.init_type, init_gain=0.02)

        # Network architecture
        # Define the two generators, one for each domain:
        #  - gAtoB: A to B domain translation
        #  - gBtoA: B to A domain translation
        self.g_ab = init(generator(in_ch=hparams.inp_ch, out_ch=hparams.out_ch,
                            ngf=hparams.ngf, net=hparams.netG,
                            norm=hparams.norm, use_dropout=hparams.use_dropout))
        self.g_ba = init(generator(in_ch=hparams.inp_ch, out_ch=hparams.out_ch,
                            ngf=hparams.ngf, net=hparams.netG,
                            norm=hparams.norm, use_dropout=hparams.use_dropout))

        # Define the discriminators:
        #  - dA: discriminator for domain A
        #  - dB: discriminator for domain B
        self.d_a = init(discriminator(in_ch=hparams.inp_ch, ndf=hparams.ndf,
                                 net=hparams.netD, ndl=hparams.ndl,
                                 norm=hparams.norm))
        self.d_b = init(discriminator(in_ch=hparams.inp_ch, ndf=hparams.ndf,
                                 net=hparams.netD, ndl=hparams.ndl,
                                 norm=hparams.norm))

        if hparams.perceptual > 0:
            self.vgg = VGGPerceptualLoss().eval()

        # ImagePool from where we randomly get generated images in both domains
        self.pool_fakeA = ImagePool(50)
        self.pool_fakeB = ImagePool(50)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Model specific arguments
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--inp_ch', type=int, default=3, help='# of input image channels, e.g. 3 for RGB')
        parser.add_argument('--out_ch', type=int, default=3, help='# of output image channels, e.g. 3 for RGB')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--ndl', type=int, default=3, help='only used if netD==patch')
        parser.add_argument('--netD', type=str, default='patch', help='specify discriminator architecture [patch | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | resnet_4blocks]')
        parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--use_dropout', action='store_false', help='no dropout for the generator')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--glr', type=float, default=0.0002, help='initial learning rate for generator (adam)')
        parser.add_argument('--dlr', type=float, default=0.0002, help='initial learning rate for discriminator (adam)')
        parser.add_argument('--lambda_idt', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        parser.add_argument('--lambda_a', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambda_b', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        parser.add_argument('--valid_out_pth', type=str, default="/src/github_repos/a-gan-in-your-hands/valid_outputs/")
        parser.add_argument('--masked', type=bool, default=True)
        parser.add_argument('--fid_dims', type=int, default=2048)
        parser.add_argument('--perceptual', type=float, default=0.25)
        parser.add_argument('--log_freq', type=int, default=25)

        return parser

    def configure_optimizers(self):
        # optimizers
        g_opt = Adam(chain(self.g_ab.parameters(), self.g_ba.parameters()),
                           lr=self.hparams.glr, betas=(0.5, 0.999))
        da_opt = Adam(self.d_a.parameters(), lr=self.hparams.dlr, betas=(0.5, 0.999))
        db_opt = Adam(self.d_b.parameters(), lr=self.hparams.dlr, betas=(0.5, 0.999))

        return [g_opt, da_opt, db_opt]

    def generator_pass(self, realA, realB, maskA, maskB):
        _losses = {}

        self.fakeA = self.g_ba(realB) # G_BtoA(B)
        self.fakeB = self.g_ab(realA) # G_AtoB(A)

        if self.hparams.masked:
            self.fakeA = self.fakeA * maskB + realB * (1 - maskB)
            self.fakeB = self.fakeB * maskA + realA * (1 - maskA)

        recA = self.g_ba(self.fakeB)  # G_BtoA(G_AtoB(A))
        recB = self.g_ab(self.fakeA)  # G_AtoB(G_BtoA(B))

        idtA = self.g_ab(realB)  # G_AtoB(B)
        idtB = self.g_ba(realA)  # G_BtoA(A)

        # Generators must fool the discriminators, so the label is real
        _losses['gBtoA'] = get_mse_loss(self.d_b(self.fakeA), 'real') # D_A(G_BtoA(B))
        _losses['gAtoB'] = get_mse_loss(self.d_a(self.fakeB), 'real') # D_B(G_AtoB(A))

        #Identity losses
        if self.hparams.lambda_idt > 0:
            _losses['idtA'] = F.l1_loss(idtA, realB) * self.hparams.lambda_b * self.hparams.lambda_idt
            _losses['idtB'] = F.l1_loss(idtB, realA) * self.hparams.lambda_a * self.hparams.lambda_idt

        if self.hparams.perceptual > 0:
            _perceptual_loss = 0
            with torch.no_grad():
                _vgg_realA = self.vgg(realA)
                _vgg_fakeA = self.vgg(self.fakeA)
                _vgg_recA = self.vgg(recA)
                _vgg_realB = self.vgg(realB)
                _vgg_fakeB = self.vgg(self.fakeB)
                _vgg_recB = self.vgg(recB)

            _perceptual_loss += torch.nn.functional.l1_loss(_vgg_realA[1], _vgg_recA[1]) * self.hparams.perceptual
            _perceptual_loss += torch.nn.functional.l1_loss(_vgg_realB[1], _vgg_recB[1]) * self.hparams.perceptual
            _perceptual_loss += torch.nn.functional.l1_loss(_vgg_fakeA[3], _vgg_realA[3]) * self.hparams.perceptual
            _perceptual_loss += torch.nn.functional.l1_loss(_vgg_fakeB[3], _vgg_realB[3]) * self.hparams.perceptual

            _losses['per'] = _perceptual_loss

        # Cycle-consistency losses
        _losses['cycleAB'] = F.l1_loss(recA, realA) * self.hparams.lambda_a
        _losses['cycleBA'] = F.l1_loss(recB, realB) * self.hparams.lambda_b

        # Total generator loss
        _losses['genTotal'] = sum(_losses.values())

        # Logging
        if self.trainer.global_step % self.hparams.log_freq == 0: # Log images every log_freq steps
            self.log_images(realA, realB, maskA, maskB)
        
        self.log_losses(_losses) # Log losses each training step

        return _losses['genTotal']

    def discriminator_pass(self, net, real, fake, name):
        _losses = {}

        _losses['real_'+ name] = get_mse_loss(net(real), 'real')
        _losses['fake_'+ name] = get_mse_loss(net(fake.detach()), 'fake')
        _losses['tot_'+ name] = (_losses['real_'+ name] + _losses['fake_'+ name]) * 0.5

        # Logging
        self.log_losses(_losses) # Log losses each trainings step

        return _losses['tot_'+ name]

    def training_step(self, batch, batch_idx, optimizer_idx):
        domainA, domainB = batch
        realA, realB, maskA, maskB = domainA['rgb'], domainB['rgb'], domainA['mask'], domainB['mask']

        # Generators
        if optimizer_idx == 0:
            set_requires_grad([self.d_a, self.d_b], requires_grad= False)
            return self.generator_pass(realA, realB, maskA, maskB)

        # Discriminator A
        if optimizer_idx == 1:
            set_requires_grad(self.d_a, requires_grad=True)
            fakeB = self.pool_fakeB.query(self.fakeB)
            return self.discriminator_pass(self.d_a, realB, fakeB, 'dA')

        # Discriminator B
        if optimizer_idx == 2:
            set_requires_grad(self.d_b, requires_grad=True)
            fakeA = self.pool_fakeA.query(self.fakeA)
            return self.discriminator_pass(self.d_b, realA, fakeA, 'dB')

    def validation_step(self, batch, batch_idx):
        domainA, domainB = batch
        real_imgs = domainA['rgb']
        synth_imgs = domainB['rgb']

        real_imgs = self.trainer.datamodule.denormalizers[0](real_imgs)

        test = self.inception_model(real_imgs)[0]
        # Calculate FID metric
        # Inceltion activations for real images of hands
        self.inception_real = torch.cat((self.inception_real,
                                         self.inception_model(real_imgs)[0].squeeze()))

        # Generator forward step to translate synthetic into real
        _outputs = self.g_ba(synth_imgs)
        _outputs = self.trainer.datamodule.denormalizers[1](_outputs)
        # Inception activations for generated images
        self.inception_gen = torch.cat((self.inception_gen,
                                        self.inception_model(_outputs)[0].squeeze()))

        #outputs = self.trainer.datamodule.denormalizers[0](self.g_ba(imgs))
        #save_to_disk(outputs, batch_idx, self.hparams.valid_out_pth)

    def log_images(self, inp_A, inp_B, maskA, maskB):
        merged_out = torch.cat((inp_A, self.fakeB, inp_B, self.fakeA), 0)
        merged_out = self.trainer.datamodule.denormalizers[0](merged_out)
        masks = torch.cat((maskA, maskB), 0)

        self.trainer.logger.experiment.log_image(log_name='From top to bottom: inpA, fakeB, inpB, fakeA',
                                                 x = make_grid(merged_out, nrow=self.hparams.batch_size).permute(1, 2, 0).cpu())

        self.trainer.logger.experiment.log_image(log_name='Masks',
                                                 x = make_grid(masks, nrow=self.hparams.batch_size).permute(1, 2, 0).cpu())

    def log_losses(self, losses):
        for key, value in losses.items():
            self.trainer.logger.experiment.log_metric(log_name='Loss '+key, x = value)

class ValidationCallback(Callback):
    def on_validation_start(self, trainer, pl_module):
        pl_module.inception_real = torch.empty(0, device='cuda')
        pl_module.inception_gen = torch.empty(0, device='cuda')

        _block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[pl_module.hparams.fid_dims]
        pl_module.inception_model = InceptionV3([_block_idx])
        pl_module.inception_model.cuda()
        pl_module.inception_model.eval()

    def on_validation_end(self, trainer, pl_module):
        fid_score = fid(pl_module.inception_real.cpu().data.numpy(),
                        pl_module.inception_gen.cpu().data.numpy())
        trainer.logger.experiment.log_metric(log_name = 'FID', x = fid_score)

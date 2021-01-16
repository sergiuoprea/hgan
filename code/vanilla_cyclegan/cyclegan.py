import torch
from torch.optim import Adam
import pytorch_lightning as pl
from torch.nn import functional as F
from torchvision.utils import make_grid

from vanilla_cyclegan.discriminator import define_discriminator
from vanilla_cyclegan.generator import define_generator
from vanilla_cyclegan.utils import init_weights, ImagePool, set_requires_grad, get_mse_loss

import datasets.synthhands as sh
import datasets.realhands as rh

from itertools import chain
from argparse import ArgumentParser

class CycleGAN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        print("Initializing Vanilla CycleGAN!")
        self.hparams = hparams
        self.models = {} #network modules (generators and discriminators)

        # Network architecture
        # Define the two generators, one for each domain:
        #  - gAtoB: A to B domain translation
        #  - gBtoA: B to A domain translation
        self.models['gAtoB'] = define_generator(in_ch=hparams.inp_ch, out_ch=hparams.out_ch,
                                                ngf=hparams.ngf, net=hparams.netG,
                                                norm=hparams.norm, use_dropout=hparams.use_dropout).cuda()
        self.models['gBtoA'] = define_generator(in_ch=hparams.inp_ch, out_ch=hparams.out_ch,
                                                ngf=hparams.ngf, net=hparams.netG,
                                                norm=hparams.norm, use_dropout=hparams.use_dropout).cuda()

        # Define the discriminators:
        #  - dA: discriminator for domain A
        #  - dB: discriminator for domain B
        self.models['dA'] = define_discriminator(in_ch=hparams.inp_ch, ndf=hparams.ndf,
                                                 net=hparams.netD, ndl=hparams.ndl,
                                                 norm=hparams.norm).cuda()
        self.models['dB'] = define_discriminator(in_ch=hparams.inp_ch, ndf=hparams.ndf,
                                                 net=hparams.netD, ndl=hparams.ndl,
                                                 norm=hparams.norm).cuda()

        # Initialize models' weights
        for _model in self.models:
            init_weights(self.models[_model], init_type=hparams.init_type)

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

        return parser

    def configure_optimizers(self):
        generator_optimizer = Adam(chain(self.models['gAtoB'].parameters(),
                                         self.models['gBtoA'].parameters()),
                                   lr=self.hparams.glr, betas=(0.5, 0.999))
        discriminator_optimizer = Adam(chain(self.models['dA'].parameters(),
                                             self.models['dB'].parameters()),
                                       lr=self.hparams.dlr, betas=(0.5, 0.999))

        return [generator_optimizer, discriminator_optimizer]

    def generator_pass(self, realA, realB):
        gen_losses = {}

        self.fakeA = self.models['gBtoA'](realB) # G_BtoA(B)
        self.fakeB = self.models['gAtoB'](realA) # G_AtoB(A)

        recA = self.models['gBtoA'](self.fakeB)  # G_BtoA(G_AtoB(A))
        recB = self.models['gAtoB'](self.fakeA)  # G_AtoB(G_BtoA(B))

        idtA = self.models['gAtoB'](realB)  # G_AtoB(B)
        idtB = self.models['gBtoA'](realA)  # G_BtoA(A)

        # Generators must fool the discriminators, so the label is real
        gen_losses['gBtoA'] = get_mse_loss(self.models['dA'](self.fakeA), 'real') # D_A(G_BtoA(B))
        gen_losses['gAtoB'] = get_mse_loss(self.models['dB'](self.fakeB), 'real') # D_B(G_AtoB(A))

        #Identity losses
        if self.hparams.lambda_idt > 0:
            gen_losses['idtA'] = F.l1_loss(idtA, realB) * self.hparams.lambda_a * self.hparams.lambda_idt
            gen_losses['idtB'] = F.l1_loss(idtB, realA) * self.hparams.lambda_b * self.hparams.lambda_idt
        else:
            gen_losses['idtA'] = 0
            gen_losses['idtB'] = 0

        # Cycle-consistency losses
        gen_losses['cycleAB'] = F.l1_loss(recA, realA) * self.hparams.lambda_a
        gen_losses['cycleBA'] = F.l1_loss(recB, realB) * self.hparams.lambda_b

        # Total generator loss
        gen_losses['genTotal'] = sum(gen_losses.values())

        # Tensorboard Logging
        self.log_images(realA, realB)
        self.log_losses(gen_losses)

        return gen_losses['genTotal'] 

    def discriminator_pass(self, realA, realB):
        dis_losses = {}

        fakeA = self.pool_fakeA.query(self.fakeA)
        fakeB = self.pool_fakeB.query(self.fakeB)

        dis_losses['realAd'] = get_mse_loss(self.models['dA'](realA), 'real')
        dis_losses['realBd'] = get_mse_loss(self.models['dB'](realB), 'real')

        dis_losses['fakeAd'] = get_mse_loss(self.models['dA'](fakeA.detach()), 'fake')
        dis_losses['fakeBd'] = get_mse_loss(self.models['dB'](fakeB.detach()), 'fake')

        # Total discriminators loss
        dis_losses['disTotal'] = sum(dis_losses.values()) * 0.5

        #Tensorboard logging
        self.log_losses(dis_losses)

        return dis_losses['disTotal']

    def training_step(self, batch, batch_idx, optimizer_idx):
        domainA, domainB = batch
        realA = domainA['rgb']
        realB = domainB['rgb']

        discriminator_requires_grad = (optimizer_idx == 1)
        set_requires_grad([self.models['dA'], self.models['dB']], discriminator_requires_grad)        

        if optimizer_idx == 0:
            return self.generator_pass(realA, realB)
        else:
            return self.discriminator_pass(realA, realB)

    def log_images(self, inp_A, inp_B):
        merged_out = torch.cat((inp_A, self.fakeB, inp_B, self.fakeA), 0)
        merged_out = self.trainer.datamodule.denormalizers[0](merged_out)

        self.trainer.logger.experiment.add_image('From top to bottom: inpA, fakeB, inpB, fakeA',
                                                  make_grid(merged_out,
                                                  nrow=self.hparams.batch_size),
                                                  self.trainer.global_step)

    def log_losses(self, losses):
        for key, value in losses.items():
            self.trainer.logger.experiment.add_scalar('Loss ' + key, value,
                                                      self.trainer.global_step)

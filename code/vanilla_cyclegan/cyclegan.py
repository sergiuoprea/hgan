import torch
import os
import torch.optim as optim
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

        init = Initializer(init_type=hparams.init_type, init_gain=hparams.init_gain)

        # Network architecture
        # Two generators, one for each domain:
        #  - g_ab: translation from domain A to domain B
        #  - g_ba: translation from domain B to domain A
        self.g_ab = init(generator(in_ch=hparams.inp_ch, out_ch=hparams.out_ch, ngf=hparams.ngf, net=hparams.netG,
                                   norm=hparams.norm, use_dropout=hparams.use_dropout))
        self.g_ba = init(generator(in_ch=hparams.inp_ch, out_ch=hparams.out_ch, ngf=hparams.ngf, net=hparams.netG,
                                   norm=hparams.norm, use_dropout=hparams.use_dropout))

        # Discriminators:
        #  - d_a: domain A discriminator
        #  - d_b: domain B discriminator
        self.d_a = init(discriminator(in_ch=hparams.inp_ch, ndf=hparams.ndf, net=hparams.netD, ndl=hparams.ndl,
                                      norm=hparams.norm))
        self.d_b = init(discriminator(in_ch=hparams.inp_ch, ndf=hparams.ndf, net=hparams.netD, ndl=hparams.ndl,
                                      norm=hparams.norm))

        # If we are using perceptual loss. We will need VGG network as a feature extractor
        if hparams.perceptual > 0: 
            self.vgg = VGGPerceptualLoss().eval()

        # ImagePool from where we randomly get generated images in both domains
        self.pool_fake_a = ImagePool(50)
        self.pool_fake_b = ImagePool(50)

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
        parser.add_argument('--perceptual', type=float, default=0.0)
        parser.add_argument('--train_log_freq', type=int, default=25)
        parser.add_argument('--valid_log_freq', type=int, default=10)
        parser.add_argument('--epoch_decay', type=int, default=3)
        parser.add_argument('--init_gain', type=float, default=0.02)
        parser.add_argument('--deterministic', type=bool, default=False)
        parser.add_argument('--precision', type=int, default=32)

        return parser

    def lr_lambda(self, epoch):
        fraction = (epoch - self.hparams.epoch_decay) / self.hparams.epoch_decay
        return 1 if epoch < self.hparams.epoch_decay else 1 - fraction

    def configure_optimizers(self):
        # optimizers
        g_opt = optim.Adam(chain(self.g_ab.parameters(), self.g_ba.parameters()),
                           lr=self.hparams.glr, betas=(0.5, 0.999))
        da_opt = optim.Adam(self.d_a.parameters(), lr=self.hparams.dlr, betas=(0.5, 0.999))
        db_opt = optim.Adam(self.d_b.parameters(), lr=self.hparams.dlr, betas=(0.5, 0.999))

        # schedulers
        g_sch = optim.lr_scheduler.LambdaLR(g_opt, lr_lambda=self.lr_lambda)
        da_sch = optim.lr_scheduler.LambdaLR(da_opt, lr_lambda=self.lr_lambda)
        db_sch = optim.lr_scheduler.LambdaLR(db_opt, lr_lambda=self.lr_lambda)

        return [g_opt, da_opt, db_opt], [g_sch, da_sch, db_sch]

    def generator_pass(self, real_a, real_b, mask_a, mask_b):
        losses = {}

        self.fake_a = self.g_ba(real_b) # g_ba(real_b)
        self.fake_b = self.g_ab(real_a) # g_ab(real_a)

        if self.hparams.masked: # make the generators to only focus on the hands
            self.fake_a = self.fake_a * mask_b + real_b * (1 - mask_b)
            self.fake_b = self.fake_b * mask_a + real_a * (1 - mask_a)

        rec_a = self.g_ba(self.fake_b)  # g_ba(g_ab(real_a))
        rec_b = self.g_ab(self.fake_a)  # g_ab(g_ba(real_b))

        idt_a = self.g_ba(real_a)  # g_ba(real_a)
        idt_b = self.g_ab(real_b)  # g_ab(real_b)

        # Generators must fool the discriminators
        losses['g_ba'] = get_mse_loss(self.d_a(self.fake_a), 'real') # D_A(G_BtoA(B))
        losses['g_ab'] = get_mse_loss(self.d_b(self.fake_b), 'real') # D_B(G_AtoB(A))

        #Identity losses
        if self.hparams.lambda_idt > 0:
            losses['idt_a'] = F.l1_loss(idt_a, real_a) * self.hparams.lambda_a * self.hparams.lambda_idt
            losses['idt_b'] = F.l1_loss(idt_b, real_b) * self.hparams.lambda_b * self.hparams.lambda_idt

        if self.hparams.perceptual > 0:
            perceptual_loss = 0
            with torch.no_grad():
                vgg_real_a = self.vgg(real_a)
                vgg_fake_a = self.vgg(self.fake_a)
                #_vgg_rec_a = self.vgg(rec_a)
                vgg_real_b = self.vgg(real_b)
                vgg_fake_b = self.vgg(self.fake_b)
                #_vgg_rec_b = self.vgg(rec_b)

            #perceptual_loss += torch.nn.functional.l1_loss(vgg_real_a[1], _vgg_rec_a[1]) * self.hparams.perceptual
            #perceptual_loss += torch.nn.functional.l1_loss(vgg_real_b[1], _vgg_rec_b[1]) * self.hparams.perceptual
            perceptual_loss += torch.nn.functional.l1_loss(vgg_fake_a[3], vgg_real_a[3]) * self.hparams.perceptual
            perceptual_loss += torch.nn.functional.l1_loss(vgg_fake_b[3], vgg_real_b[3]) * self.hparams.perceptual

            losses['per'] = perceptual_loss

        # Cycle-consistency losses
        losses['cycle_ab'] = F.l1_loss(rec_a, real_a) * self.hparams.lambda_a
        losses['cycle_ba'] = F.l1_loss(rec_b, real_b) * self.hparams.lambda_b

        # Total generator loss
        losses['gen_total'] = sum(losses.values())

        # Logging
        if self.trainer.global_step % self.hparams.train_log_freq == 0: # Log images every train_log_freq steps
            self.log_visuals(images=(real_a, self.fake_b, real_b, self.fake_a),
                             denorm=self.trainer.datamodule.denormalizers[0],
                             log_name='Training visual outputs')
        
        self.log_losses(losses) # Log losses each training step

        return losses['gen_total']

    def discriminator_pass(self, net, real, fake, name):
        losses = {}

        losses['real_'+name] = get_mse_loss(net(real), 'real')
        losses['fake_'+name] = get_mse_loss(net(fake.detach()), 'fake')
        losses[name+'_total'] = (losses['real_'+ name] + losses['fake_'+ name]) * 0.5

        # Logging
        self.log_losses(losses) # Log losses each trainings step

        return losses[name+'_total']

    def training_step(self, batch, batch_idx, optimizer_idx):
        domain_a, domain_b = batch
        real_a, real_b, mask_a, mask_b = domain_a['rgb'], domain_b['rgb'], domain_a['mask'], domain_b['mask']

        # Generators
        if optimizer_idx == 0:
            set_requires_grad([self.d_a, self.d_b], requires_grad= False)
            return self.generator_pass(real_a, real_b, mask_a, mask_b)

        # Discriminator A
        if optimizer_idx == 1:
            set_requires_grad(self.d_a, requires_grad=True)
            fake_a = self.pool_fake_a.query(self.fake_a)
            return self.discriminator_pass(self.d_a, real_b, fake_a, 'd_a')

        # Discriminator B
        if optimizer_idx == 2:
            set_requires_grad(self.d_b, requires_grad=True)
            fake_b = self.pool_fake_b.query(self.fake_b)
            return self.discriminator_pass(self.d_b, real_a, fake_b, 'd_b')

    def validation_step(self, batch, batch_idx):
        domain_a, domain_b = batch
        imgs_a = domain_a['rgb']
        imgs_b = domain_b['rgb']

        # Calculate FID metric
        # Get features of images from domain a (in our case: real images of hands)
        imgs_a_denorm = self.trainer.datamodule.denormalizers[0](imgs_a)
        self.inception_a = torch.cat((self.inception_a,
                                      self.inception_model(imgs_a_denorm)[0].squeeze()))

        # Generator forward step to translate images from domain b to domain a (synthetic hands to real hands)
        imgs_a_fake = self.g_ba(imgs_b)
        imgs_a_fake_denorm = self.trainer.datamodule.denormalizers[1](imgs_a_fake)
        # Inception activations for generated images
        self.inception_a_fake = torch.cat((self.inception_a_fake,
                                           self.inception_model(imgs_a_fake_denorm)[0].squeeze()))

        # Logging
        if batch_idx % self.hparams.valid_log_freq == 0:
            self.log_visuals(images=(imgs_b, imgs_a_fake),
                             denorm=self.trainer.datamodule.denormalizers[1],
                             log_name='Validation visual outputs')

    def log_visuals(self, images, denorm=None, log_name="Visuals"):
        out = torch.cat(images, 0)
        nrow = out.shape[0] // len(images)

        if denorm:
            out = denorm(out)

        out = make_grid(out, nrow=nrow).permute(1, 2, 0).cpu()
        self.trainer.logger.experiment.log_image(log_name=log_name, x=out)

    def log_losses(self, losses):
        for key, value in losses.items():
            self.trainer.logger.experiment.log_metric(log_name='Loss '+key, x = value)

class ValidationCallback(Callback):
    def on_validation_start(self, trainer, pl_module):
        pl_module.inception_a = torch.empty(0, device='cuda')
        pl_module.inception_a_fake = torch.empty(0, device='cuda')

        _block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[pl_module.hparams.fid_dims]
        pl_module.inception_model = InceptionV3([_block_idx])
        pl_module.inception_model.cuda()
        pl_module.inception_model.eval()

    def on_validation_end(self, trainer, pl_module):
        fid_score = fid(pl_module.inception_a.cpu().data.numpy(),
                        pl_module.inception_a_fake.cpu().data.numpy())
        trainer.logger.experiment.log_metric(log_name = 'FID', x = fid_score)

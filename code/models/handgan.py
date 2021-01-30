## Pytorch imports
import torch
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn import functional as F
from torchvision.utils import make_grid
from pytorch_lightning.callbacks import Callback

## Basic Python imports
import os
from itertools import chain
from argparse import ArgumentParser

## Model-specific imports
from models.discriminator import discriminator
from models.generator import generator
from models.inception import InceptionV3
from models.vgg import Vgg16
import models.utils as mutils

## Metrics
from metrics.fid import fid

## Criterions
from loss.losses import DiscriminatorLoss

class HandGAN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        print("Initializing our HandGAN model...")

        self.hparams = hparams

        # Used to initialize the networks
        init = mutils.Initializer(init_type=hparams.init_type,
                                  init_gain=hparams.init_gain)

        # Network architecture
        # Two generators, one for each domain:
        self.g_ab = init(generator(hparams)) #  - g_ab: translation from domain A to domain B
        self.g_ba = init(generator(hparams)) #  - g_ba: translation from domain B to domain A

        # Discriminators:
        self.d_a = init(discriminator(hparams)) #  - d_a: domain A discriminator
        self.d_b = init(discriminator(hparams)) #  - d_b: domain B discriminator

        # For the perceptual discriminator we will need a feature extractor
        if hparams.netD == 'perceptual':
            self.vgg_net = Vgg16().eval()

        # For validation we will need Inception network to compute FID metric
        if hparams.valid_interval > 0.0:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.inception_net = InceptionV3([block_idx]).eval()

        # ImagePool from where we randomly get generated images in both domains
        self.fake_a_pool = mutils.ImagePool(hparams.pool_size)
        self.fake_b_pool = mutils.ImagePool(hparams.pool_size)

        # Criterions
        self.crit_cycle = torch.nn.L1Loss()
        self.crit_discr = DiscriminatorLoss('lsgan')

        if hparams.lambda_idt > 0.0:
            self.crit_idt = torch.nn.L1Loss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Model specific arguments
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--inp_ch', type=int, default=3, help='# of input image channels, e.g. 3 for RGB')
        parser.add_argument('--out_ch', type=int, default=3, help='# of output image channels, e.g. 3 for RGB')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer. Only for [patch | pixel] discriminators')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--ndl', type=int, default=3, help='# of donwsampling layers in the discriminator. Only for [patch | pixel] discriminators')
        parser.add_argument('--netD', type=str, default='perceptual', help='specify discriminator architecture [patch | pixel | perceptual]')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | resnet_4blocks]')
        parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--use_dropout', type=bool, default=True, help='use dropout for the generator')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02)
        #parser.add_argument('--lr_epoch_decay', type=int, default=3, help='learning rate decay for the LR scheduler')
        parser.add_argument('--glr', type=float, default=0.0002, help='initial learning rate for generator (adam optimizer)')
        parser.add_argument('--dlr', type=float, default=0.0002, help='initial learning rate for discriminator (adam optimizer)')
        parser.add_argument('--lambda_idt', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        parser.add_argument('--lambda_a', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambda_b', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        parser.add_argument('--use_masks', type=bool, default=True, help='generator output is masked to only focus the loss on the hands')
        parser.add_argument('--log_freq_train', type=int, default=25, help='frequency to log training visuals (each n batches)')
        parser.add_argument('--log_freq_valid', type=int, default=10, help='frequency to log validation visuales (each n batches)')
        parser.add_argument('--be_deterministic', type=bool, default=True, help='make the model deterministic.')
        parser.add_argument('--fp_precision', type=int, default=32, help='floating point precision. use 16 to handle larger batch sizes')
        parser.add_argument('--valid_interval', type=float, default=0.2, help='validation epoch interval. This values is a percentage')
        parser.add_argument('--pool_size', type=int, default=50, help='# of generated images stored in the pool')
        parser.add_argument('--output_scales', type=str, default='16,8,4', help='output sizes for the multi-scale perceptual discriminator')
        parser.add_argument('--exp_name', type=str, default='alpha', help='prefix name to indetify the current experiment when saving the checkpoints')

        return parser

    #def lr_lambda(self, epoch):
    #    fraction = (epoch - self.hparams.lr_epoch_decay) / self.hparams.lr_epoch_decay
    #    return 1 if epoch < self.hparams.lr_epoch_decay else 1 - fraction

    def configure_optimizers(self):
        # optimizers
        g_opt = optim.Adam(chain(self.g_ab.parameters(), self.g_ba.parameters()),
                           lr=self.hparams.glr, betas=(0.5, 0.999))
        da_opt = optim.Adam(self.d_a.parameters(), lr=self.hparams.dlr, betas=(0.5, 0.999))
        db_opt = optim.Adam(self.d_b.parameters(), lr=self.hparams.dlr, betas=(0.5, 0.999))

        # schedulers
    #    g_sch = optim.lr_scheduler.LambdaLR(g_opt, lr_lambda=self.lr_lambda)
    #    da_sch = optim.lr_scheduler.LambdaLR(da_opt, lr_lambda=self.lr_lambda)
    #    db_sch = optim.lr_scheduler.LambdaLR(db_opt, lr_lambda=self.lr_lambda)

        return [g_opt, da_opt, db_opt]#, [g_sch, da_sch, db_sch]

    def generator_pass(self, real_a, real_b, mask_a, mask_b):
        losses = {}

        self.fake_a = self.g_ba(real_b) # g_ba(real_b)
        self.fake_b = self.g_ab(real_a) # g_ab(real_a)

        if self.hparams.use_masks: # mask generations to focus on the hands
            self.fake_a = self.fake_a * mask_b + real_b * (1 - mask_b)
            self.fake_b = self.fake_b * mask_a + real_a * (1 - mask_a)

        # Generators must fool the discriminators
        if self.hparams.netD == 'perceptual':
            fake_a_feat = self.vgg_net(self.trainer.datamodule.denormalize(self.fake_a))
            fake_b_feat = self.vgg_net(self.trainer.datamodule.denormalize(self.fake_b))
            losses['g_ba'] = self.crit_discr(real=self.d_a(fake_a_feat))
            losses['g_ab'] = self.crit_discr(real=self.d_b(fake_b_feat))
        else:
            losses['g_ba'] = self.crit_discr(real=self.d_a(self.fake_a))
            losses['g_ab'] = self.crit_discr(real=self.d_b(self.fake_b))

        #Identity losses
        if self.hparams.lambda_idt > 0.0:
            idt_a = self.g_ba(real_a)  # g_ba(real_a)
            idt_b = self.g_ab(real_b)  # g_ab(real_b)

            losses['idt_a'] = self.crit_idt(idt_a, real_a) * self.hparams.lambda_a * self.hparams.lambda_idt
            losses['idt_b'] = self.crit_idt(idt_b, real_b) * self.hparams.lambda_b * self.hparams.lambda_idt

        rec_a = self.g_ba(self.fake_b)  # g_ba(g_ab(real_a))
        rec_b = self.g_ab(self.fake_a)  # g_ab(g_ba(real_b))

        # Cycle-consistency losses
        losses['cycle_ab'] = self.crit_cycle(rec_a, real_a) * self.hparams.lambda_a
        losses['cycle_ba'] = self.crit_cycle(rec_b, real_b) * self.hparams.lambda_b

        # Total generator loss
        losses['gen_total'] = sum(losses.values())

        ## Logging step
        self.log_losses(losses) # Log losses each training step

        # Log Images every log_freq_train steps
        if self.trainer.global_step % self.hparams.log_freq_train == 0:
            self.log_visuals(images=(real_a, self.fake_b, real_b, self.fake_a),
                             denorm=self.trainer.datamodule.denormalize,
                             log_name='Training visual outputs')

        return losses['gen_total']

    def discriminator_pass(self, net, real, fake, name):
        if self.hparams.netD == 'perceptual':
            real = self.vgg_net(self.trainer.datamodule.denormalize(real))
            fake = self.vgg_net(self.trainer.datamodule.denormalize(fake))

        loss = self.crit_discr(real=net(real), fake=net(fake))

        # Logging
        self.log_losses({name: loss}) # Log losses each trainings step

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        domain_a, domain_b = batch
        real_a, real_b, mask_a, mask_b = domain_a['rgb'], domain_b['rgb'], domain_a['mask'], domain_b['mask']

        # Generators
        if optimizer_idx == 0:
            mutils.set_requires_grad([self.d_a, self.d_b], requires_grad= False)
            return self.generator_pass(real_a, real_b, mask_a, mask_b)

        # Discriminator A
        if optimizer_idx == 1:
            mutils.set_requires_grad(self.d_a, requires_grad=True)
            fake_a = self.fake_a_pool.query(self.fake_a)
            return self.discriminator_pass(self.d_a, real_a, fake_a.detach(), 'dA')

        # Discriminator B
        if optimizer_idx == 2:
            mutils.set_requires_grad(self.d_b, requires_grad=True)
            fake_b = self.fake_b_pool.query(self.fake_b)
            return self.discriminator_pass(self.d_b, real_b, fake_b.detach(), 'dB')

    def validation_step(self, batch, batch_idx):
        domain_a, domain_b = batch
        imgs_a = domain_a['rgb']
        imgs_b = domain_b['rgb']

        # Calculate FID metric
        # Get features of images from domain a (in our case: real images of hands)
        _imgs_a_real = self.trainer.datamodule.denormalize(imgs_a)
        self.inception_a_real = torch.cat((self.inception_a_real, self.inception_net(_imgs_a_real)[0].squeeze()))

        # Generator forward step to translate images from domain b to domain a (synthetic hands to real hands)
        imgs_a_fake = self.g_ba(imgs_b)
        _imgs_a_fake = self.trainer.datamodule.denormalize(imgs_a_fake)
        self.inception_a_fake = torch.cat((self.inception_a_fake, self.inception_net(_imgs_a_fake)[0].squeeze()))

        # Logging
        if batch_idx % self.hparams.log_freq_valid == 0:
            self.log_visuals(images=(imgs_b, imgs_a_fake),
                             denorm=self.trainer.datamodule.denormalize,
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
    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.inception_a_real = torch.empty(0, device='cuda')
        pl_module.inception_a_fake = torch.empty(0, device='cuda')

    def on_validation_epoch_end(self, trainer, pl_module):
        fid_score = fid(pl_module.inception_a_real.cpu().data.numpy(),
                        pl_module.inception_a_fake.cpu().data.numpy())
        pl_module.log('FID', fid_score, prog_bar=True, logger=True)

class PrintModels(Callback):
    def setup(self, trainer, pl_module, stage):
        # Log generators' and discriminators' model layers
        for chunk in str(pl_module.g_ab).split('\n'):
            trainer.logger.experiment.log_text('Generators_arch', str(chunk))

        for chunk in str(pl_module.d_a).split('\n'):
            trainer.logger.experiment.log_text('Discriminators_arch', str(chunk))

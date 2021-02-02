# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid

# Pytorch lightning imports
import pytorch_lightning as pl

# Model imports
from models.unet import UNet
import models.utils as mutils

# Other basic imports
from argparse import ArgumentParser, Namespace
import random

class SilNet(pl.LightningModule):
    def __init__(self, hparams, requires_grad=False):
        super().__init__()
        print("Initializing SilNet model...")

        # Workaround from https://github.com/PyTorchLightning/pytorch-lightning/issues/3998
        # Happens when loading model from checkpoints. save_hyperparameters() not working
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)

        init = mutils.Initializer(init_type=hparams.init_type,
                                  init_gain=hparams.init_gain)

        self.hparams = hparams
        self.unet = init(UNet(num_classes=1, input_channels=3, num_layers=3, features_start=64))
        mutils.set_requires_grad(self.unet, requires_grad=requires_grad)

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        SilNet specific arguments
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--sn_lr', type=float, default=0.001, help='initial learning rate for SilNet (adam optimizer)')
        parser.add_argument('--sn_log_img_freq', type=int, default=25, help='frequency to log visuals')

        return parser

    def configure_optimizers(self):
        return optim.RMSprop(self.unet.parameters(), lr=self.hparams.sn_lr, weight_decay=1e-8, momentum=0.9)

    def training_step(self, batch, batch_idx):
        domain_a, domain_b = batch

        # 50% chance to choose a sample from one domain
        if random.uniform(0, 1) > 0.5:
            x, y = domain_a['rgb'], domain_a['mask']
        else:
            x, y = domain_b['rgb'], domain_b['mask']

        y_hat = self.unet(x)
        loss = self.criterion(y_hat, y)

        self.trainer.logger.experiment.log_metric(log_name='SilNet_train_loss', x=loss)

        if batch_idx % self.hparams.sn_log_img_freq == 0:
            in_imgs = self.trainer.datamodule.denormalize(x)
            y_hats = torch.cat(([torch.sigmoid(y_hat) > 0.5] * 3), dim=1)
            ys = torch.cat(([y] * 3), dim=1)
            self.log_visuals(images=(in_imgs, y_hats, ys), log_name='SilNet train visuals')

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        domain_a, domain_b = batch

        # 50% chance to choose a sample from one domain
        if random.uniform(0, 1) > 0.5:
            x, y = domain_a['rgb'], domain_a['mask']
        else:
            x, y = domain_b['rgb'], domain_b['mask']

        y_hat = self.unet(x)
        loss = self.criterion(y_hat, y)

        if batch_idx % self.hparams.sn_log_img_freq == 0:
            in_imgs = self.trainer.datamodule.denormalize(x)
            y_hats = torch.cat(([torch.sigmoid(y_hat) > 0.5] * 3), dim=1)
            ys = torch.cat(([y] * 3), dim=1)
            self.log_visuals(images=(in_imgs, y_hats, ys), log_name='SilNet valid visuals')

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.trainer.logger.experiment.log_metric(log_name='SilNet_valid_loss', x=avg_loss)
        self.log('avg_loss', avg_loss)

    def log_visuals(self, images, denorm=None, log_name="Visuals"):
        out = torch.cat(images, 0)
        nrow = out.shape[0] // len(images)

        if denorm:
            out = denorm(out)

        out = make_grid(out, nrow=nrow).permute(1, 2, 0).cpu()
        self.trainer.logger.experiment.log_image(log_name=log_name, x=out)

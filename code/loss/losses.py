import torch
from torch import nn

class DiscriminatorLoss:
    """
    Criterion used for the discriminators. It handles the multi-scale predictions obtained from
    the perceptual discriminator.
    """
    def __init__(self, loss_type='lsgan'):
        self.labels_real = []
        self.labels_fake = []

        if loss_type == 'lsgan':
            self.criterion = nn.MSELoss()

    def __call__(self, real, fake=None):
        losses = []
        loss = 0

        for i in range(len(real)):
            losses.append(self.criterion(real[i], torch.ones_like(real[i], device='cuda')))

            if fake:
                losses[-1] += self.criterion(fake[i], torch.zeros_like(fake[i], device='cuda'))
                losses[-1] *= 0.5

            loss += losses[-1]

        return loss

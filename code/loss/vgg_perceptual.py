import torch
import torch.nn as nn
import torchvision

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def __call__(self, inp):
        features = []

        if inp.shape[1] != 3:
            inp = inp.repeat(1, 3, 1, 1)

        inp = (inp - self.mean) / self.std

        if self.resize:
            inp = self.transform(inp, mode='bilinear', size=(224, 224), align_corners=False)

        x = inp
        for block in self.blocks:
            x = block(x)
            features.append(x)

        return features

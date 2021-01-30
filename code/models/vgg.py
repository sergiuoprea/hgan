import torch
import torch.nn as nn
import torchvision

class Vgg16(nn.Module):
    def __init__(self, requires_grad=False, resize=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.mean = torch.nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize
        self.transform = torch.nn.functional.interpolate

    def forward(self, inp):
        if inp.shape[1] != 3:
            inp = inp.repeat(1, 3, 1, 1)

        inp = (inp-self.mean) / self.std

        if self.resize:
            inp = self.transform(inp, mode='bilinear', size=(224, 224), align_corners=False)

        h = self.slice1(inp)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h

        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]

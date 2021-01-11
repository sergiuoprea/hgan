import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch.utils.data
from datasets.utils import calculate_mean_and_std

from argparse import ArgumentParser
import json
import os
import random
import numpy as np

class SynthHandsDataset(torch.utils.data.Dataset):
    def __init__(self, data, ops):
        self.data = data
        self.ops = ops
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        _out = {}
        _path = self.data[idx]

        _rgb = Image.open(_path)
        _rgb = np.array(_rgb)

        _rgb, _mask = self.preprocess_rgb_and_get_mask(_rgb)
        _mask = np.uint8(_mask * 255)

        _rgb = Image.fromarray(_rgb[:, :, :3])
        _mask = Image.fromarray(_mask)

        if self.ops:
            _rgb = self.ops['rgb'](_rgb)
            _mask = self.ops['mask'](_mask)

        _out['rgb'] = _rgb
        _out['mask'] = _mask
        _out['paths'] = _path

        return _out

    def __len__(self):
        return len(self.data)

    def preprocess_rgb_and_get_mask(self, image):
        red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        mask = (red == 14) & (green == 255) & (blue == 14)
        image[mask == 1] = 255

        return image, ~mask

class SynthHandsDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.data = [] # absolute paths to images
        self.datasets = {}
        self.ops = {}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--sh_path', type=str, default='/src/datasets/synthhands', help='path to the SynthHands dataset root folder.')
        parser.add_argument('--sh_json', type=bool, default=False, help='if true: gather data paths of SynthHands dataset to a json file, pairing the rgb images with their respective masks.')
        parser.add_argument('--sh_mean_std', type=bool, default=True, help='if true: calculate the mean and standard deviation for the SynthHands dataset.')

        return parser

    def prepare_data(self, shuffle=True):
        if os.path.exists(os.path.join(self.hparams.sh_path, 'synthhands.json')) and not self.hparams.sh_json:
            with open(os.path.join(self.hparams.sh_path, 'synthhands.json')) as file:
                self.data = json.load(file)

            print("SynthHands dataset paths loaded from JSON file!")

        else:
            print("Creating a JSON file with dataset paths...")

            subfolders = [os.path.join(self.hparams.sh_path, _folder)
                          for _folder in os.listdir(self.hparams.sh_path)
                          if os.path.isdir(os.path.join(self.hparams.sh_path, _folder))]

            for _folder in subfolders:
                for _sequence in os.listdir(_folder):
                    _path = os.path.join(_folder, _sequence)
                    self.data.extend(self.get_data_by_extension(_path))

            if shuffle:
                random.shuffle(self.data)

            with open(os.path.join(self.hparams.sh_path, 'synthhands.json'), 'w') as _file:
                json.dump(self.data, _file)

        if self.hparams.sh_mean_std:
            print("Computing the mean and std for the SynthHands dataset...")
            dataset = SynthHandsDataset(random.sample(self.data, 20000), self.ops)
            dataloader = DataLoader(dataset=dataset, batch_size=64)
            mean, std = calculate_mean_and_std(dataloader)

    def setup(self):
        self.ops['rgb'] = transforms.Compose([transforms.CenterCrop(300),
                                      transforms.Normalize(mean=[0.8733, 0.8294, 0.8045],
                                                           std =[0.2139, 0.2529, 0.2788]),
                                      transforms.ToTensor()])
        self.ops['mask'] = transforms.Compose([transforms.CenterCrop(300),
                                       transforms.ToTensor()])

    def get_data_by_extension(self, folder, suffix="_color.png"):
        buffer = []
        for root, _, files in os.walk(folder):
            for name in files:
                if name.endswith(suffix):
                    buffer.append(os.path.join(root, name))

        return buffer

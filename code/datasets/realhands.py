import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch.utils.data
from datasets.utils import calculate_mean_and_std, Denormalize

from argparse import ArgumentParser
import json
import os
import random
import numpy as np

class RealHandsDataset(torch.utils.data.Dataset):
    def __init__(self, data, ops):
        self.data = data
        self.ops = ops

    def __getitem__(self, idx):
        _out = {}
        _path = self.data[idx]

        _rgb = Image.open(_path[0])
        _mask = Image.open(_path[1])

        # Setting white background
        _rgb = np.array(_rgb)
        _mask = np.array(_mask)
        _rgb[_mask < 10] = 255

        _rgb = Image.fromarray(_rgb)
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

class RealHandsDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.data = [] # absolute paths to images
        self.datasets = {}

        # Calculated from the data
        #self.mean = [-0.7522, 0.1588, -0.1367]
        #self.std = [0.4952, 0.1567, 0.2317]

        # To normalize in range [-1, 1]
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--rh_path', type=str, default='/src/datasets/realhands', help='path to the RealHands dataset root folder.')
        parser.add_argument('--rh_json', type=bool, default=False, help='if true: gather data paths of RealHands dataset to a json file, pairing the rgb images with their respective masks.')
        parser.add_argument('--rh_mean_std', type=bool, default=False, help='if true: calculate the mean and standard deviation for the RealHands dataset.')
        parser.add_argument('--rh_shuffle', type=bool, default=True)
        parser.add_argument('--rh_crop_size', type=int, default=256)

        return parser

    def prepare_data(self):
        """
        If get_jason = True -> The paths will be stored in a JSON file for speedup.
        Paths will be stored in a JSON file for performance issues. I
        Image, mask pairs will be stored in a json file 
        """
        if os.path.exists(os.path.join(self.hparams.rh_path, 'realhands.json')) and not self.hparams.rh_json:
            with open(os.path.join(self.hparams.rh_path, 'realhands.json')) as file:
                self.data = json.load(file)

            print("RealHands dataset paths loaded from JSON file!")

        else:
            print("Creating a JSON file with dataset paths...")

            subfolders = [os.path.join(self.hparams.rh_path, _folder)
                          for _folder in os.listdir(self.hparams.rh_path)
                          if os.path.isdir(os.path.join(self.hparams.rh_path, _folder))]

            _buffer = []
            for _folder in subfolders:
                _buffer.clear()
                _path = os.path.join(_folder, "mask")

                for _img in os.listdir(_path):
                    if _img.endswith("_mask.jpg"):
                        _rgb_name = os.path.basename(_img).replace('_mask.jpg', '_color.png')
                        _buffer.append([os.path.join(_folder, "color", _rgb_name),
                                        os.path.join(_path, _img)])

                self.data.extend(_buffer)

            if self.hparams.rh_shuffle:
                random.shuffle(self.data)

            with open(os.path.join(self.hparams.rh_path, 'realhands.json'), 'w') as _file:
                json.dump(self.data, _file)

            print("JSON file for the RealHands datset ready to go!")

        if self.hparams.rh_mean_std:
            print("Computing the mean and std for the RealHands dataset...")
            dataloader = DataLoader(dataset=self.datasets['mean_std'], batch_size=64)
            mean, std = calculate_mean_and_std(dataloader)

    def setup(self):
        ops = {}
        ops['rgb'] = transforms.Compose([transforms.CenterCrop(self.hparams.rh_crop_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean= self.mean,
                                                           std = self.std)])

        ops['mask'] = transforms.Compose([transforms.CenterCrop(self.hparams.rh_crop_size),
                                       transforms.ToTensor()])

        self.datasets['train'] = RealHandsDataset(self.data, ops)
        self.datasets['mean_std'] = RealHandsDataset(self.data, transforms.ToTensor())
        #ToDo make the splits into training, validation, test sets if necessary

    def get_dataset(self, mode='train'):
        return self.datasets[mode]

    def get_denormalizer(self):
        return Denormalize(self.mean, self.std)

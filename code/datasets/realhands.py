import pytorch_lightning as pl
from datasets.transforms import CenterCrop, MaskedRandomCrop, Normalize, ToTensor, Compose, CentroidCrop, Denormalize
from torch.utils.data import DataLoader
from PIL import Image
import torch.utils.data
from torch.utils.data import Subset
from datasets.utils import calculate_mean_and_std

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
        _path = self.data[idx]

        _rgb = np.array(Image.open(_path[0]))
        _mask = np.array(Image.open(_path[1]))

        # Setting white background
        _rgb[_mask < 10] = 255

        if self.ops:
            _sample = self.ops({'rgb': _rgb, 'mask': _mask})

        _sample['paths'] = _path

        return _sample

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
        parser.add_argument('--rh_path', type=str, default='/src/datasets/real_hands', help='path to the RealHands dataset root folder.')
        parser.add_argument('--rh_json', type=bool, default=False, help='if true: gather data paths of RealHands dataset to a json file, pairing the rgb images with their respective masks.')
        parser.add_argument('--rh_mean_std', type=bool, default=False, help='if true: calculate the mean and standard deviation for the RealHands dataset.')

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

            if self.hparams.shuffle:
                random.shuffle(self.data)

            with open(os.path.join(self.hparams.rh_path, 'realhands.json'), 'w') as _file:
                json.dump(self.data, _file)

            print("JSON file for the RealHands datset ready to go!")

        if self.hparams.rh_mean_std:
            print("Computing the mean and std for the RealHands dataset...")
            dataloader = DataLoader(dataset=self.datasets['mean_std'], batch_size=64)
            mean, std = calculate_mean_and_std(dataloader)

    def setup(self):
        train_ops = []
        valid_ops = []

        # Image cropping for the train set
        if self.hparams.random_crop:
            train_ops.append(MaskedRandomCrop(self.hparams.train_inp_size))
        else:
            train_ops.append(CenterCrop(self.hparams.train_inp_size))

        # Image cropping for the valid set
        valid_ops.append(CentroidCrop(self.hparams.valid_inp_size))

        # ToTensor
        train_ops.append(ToTensor())
        valid_ops.append(ToTensor())

        # Normalize
        train_ops.append(Normalize(mean= self.mean, std= self.std))
        valid_ops.append(Normalize(mean= self.mean, std= self.std))

        # Data split into train, valid and test sets.
        indices = list(range(len(self.data)))
        test_dataset = Subset(self.data, indices[:self.hparams.test_size])
        valid_dataset = Subset(self.data, indices[self.hparams.test_size:self.hparams.test_size + self.hparams.valid_size])
        train_dataset = Subset(self.data, indices[self.hparams.test_size + self.hparams.valid_size:])

        self.datasets['train'] = RealHandsDataset(train_dataset, Compose(train_ops))
        self.datasets['valid'] = RealHandsDataset(valid_dataset, Compose(valid_ops))
        self.datasets['test'] = RealHandsDataset(test_dataset, Compose(valid_ops))
        self.datasets['mean_std'] = RealHandsDataset(self.data, ToTensor())

    def get_dataset(self, mode='train'):
        return self.datasets[mode]

    def get_denormalizer(self):
        return Denormalize(self.mean, self.std)

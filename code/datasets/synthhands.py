import pytorch_lightning as pl
from datasets.transforms import CenterCrop, MaskedRandomCrop, Normalize, ToTensor, Compose, Resize, CentroidCrop
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

class SynthHandsDataset(torch.utils.data.Dataset):
    def __init__(self, data, ops):
        self.data = data
        self.ops = ops

    def __getitem__(self, idx):
        _out = {}
        _path = self.data[idx]

        _rgb = np.array(Image.open(_path))[:, :, :3]
        _rgb, _mask = self.preprocess_rgb_and_get_mask(_rgb)

        if self.ops:
            _sample = self.ops({'rgb': _rgb, 'mask': _mask})

        _sample['paths'] = _path

        return _sample

    def __len__(self):
        return len(self.data)

    def preprocess_rgb_and_get_mask(self, image):
        red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        mask = (red == 14) & (green == 255) & (blue == 14)
        image[mask == 1] = 255

        return image, np.uint8(~mask * 255)

class SynthHandsDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.data = [] # absolute paths to images
        self.datasets = {}

        # Calculated from the data
        #self.mean = [0.8733, 0.8294, 0.8045]
        #self.std = [0.2139, 0.2529, 0.2788]

        # To normalize in range [-1, 1]
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--sh_path', type=str, default='/src/datasets/synth_hands', help='path to the SynthHands dataset root folder.')
        parser.add_argument('--sh_json', type=bool, default=False, help='if true: gather data paths of SynthHands dataset to a json file, pairing the rgb images with their respective masks.')
        parser.add_argument('--sh_mean_std', type=bool, default=False, help='if true: calculate the mean and standard deviation for the SynthHands dataset.')
        parser.add_argument('--sh_with_objects', type=bool, default=False)

        return parser

    def prepare_data(self):
        if os.path.exists(os.path.join(self.hparams.sh_path, 'synthhands.json')) and not self.hparams.sh_json:
            with open(os.path.join(self.hparams.sh_path, 'synthhands.json')) as file:
                self.data = json.load(file)

            print("SynthHands dataset paths loaded from JSON file!")

        else:
            print("Creating a JSON file with dataset paths...")

            subfolders = [os.path.join(self.hparams.sh_path, _folder)
                          for _folder in os.listdir(self.hparams.sh_path)
                          if os.path.isdir(os.path.join(self.hparams.sh_path, _folder))]

            if not self.hparams.sh_with_objects:
                aux = []
                for _folder in subfolders:
                    if _folder.endswith("_noobject"):
                        aux.append(_folder)
                subfolders = aux

            for _folder in subfolders:
                for _sequence in os.listdir(_folder):
                    _path = os.path.join(_folder, _sequence)
                    self.data.extend(self.get_data_by_extension(_path))

            if self.hparams.shuffle:
                random.shuffle(self.data)

            with open(os.path.join(self.hparams.sh_path, 'synthhands.json'), 'w') as _file:
                json.dump(self.data, _file)

        if self.hparams.sh_mean_std:
            print("Computing the mean and std on a image subset of the SynthHands dataset...")
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

        self.datasets['train'] = SynthHandsDataset(train_dataset, Compose(train_ops))
        self.datasets['valid'] = SynthHandsDataset(valid_dataset, Compose(valid_ops))
        self.datasets['test'] = SynthHandsDataset(test_dataset, Compose(valid_ops))
        # self.datasets['mean_std'] = SynthHandsDataset(random.sample(self.data, 20000), ToTensor())

    def get_dataset(self, mode='train'):
        return self.datasets[mode]

    def get_data_by_extension(self, folder, suffix="_color.png"):
        buffer = []
        for root, _, files in os.walk(folder):
            for name in files:
                if name.endswith(suffix):
                    buffer.append(os.path.join(root, name))

        return buffer

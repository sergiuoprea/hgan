import os
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image
import random
import numpy as np
import json
from argparse import ArgumentParser
import torch.utils.data
import imagesize

class MitIndoorDataset(torch.utils.data.Dataset):
    def __init__(self, data, ops):
        self.data = data
        self.ops = ops

    def __getitem__(self, idx):
        path = self.data[idx]

        rgb = Image.open(path).convert("RGB")

        if self.ops:
            rgb = self.ops(rgb)

        return {'rgb': rgb}

    def __len__(self):
        return len(self.data)

class MitIndoorDataModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.data = []
        self.dataset = None

        # To normalize in range [-1, 1]
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--mi_path', type=str, default='/src/datasets/mit_indoor')
    
        return parser

    def prepare_data(self):
        """
        TODO
        """
        if os.path.exists(os.path.join(self.hparams.mi_path, 'mit_indoor.json')):
            with open(os.path.join(self.hparams.mi_path, 'mit_indoor.json')) as file:
                self.data = json.load(file)

            print("Mit Indoor data paths loaded from JSON file!")

        else:
            print("Creating a JSON file with dataset paths...")

            buffer = []
            for root, _version, f_names in os.walk(self.hparams.mi_path):
                buffer.clear()
    
                for f in f_names:
                    width, height = imagesize.get(os.path.join(root, f))
                    if width >= 256 and height >= 256:
                        buffer.append(os.path.join(root, f))

                self.data.extend(buffer)

            # We randomly shuffle the data
            random.shuffle(self.data)

            # Dump the paths to data into the json file
            with open(os.path.join(self.hparams.mi_path, 'mit_indoor.json'), 'w') as file:
                json.dump(self.data, file)

            print("JSON file for the Mit Indoor Dataset ready to go!")

    def setup(self):
        ops = []
        ops.append(transforms.RandomCrop(self.hparams.valid_inp_size))
        ops.append(transforms.ToTensor())
        ops.append(transforms.Normalize(self.mean, self.std))
        self.dataset = MitIndoorDataset(self.data, transforms.Compose(ops))

    def get_dataset(self, mode='None'):
        return self.dataset

import pytorch_lightning as pl
from datasets.utils import ConcatDataset
from torch.utils.data import DataLoader

from datasets.realhands import RealHandsDataModule
from datasets.synthhands import SynthHandsDataModule

from argparse import ArgumentParser

DATAMODULES = {'real_hands'    : RealHandsDataModule,
               'synth_hands'   : SynthHandsDataModule}

def update_dataset_specific_args(datasets, parser):
    for dataset in datasets:
        parser = DATAMODULES[dataset].add_model_specific_args(parser)

    return parser

class PairedDataModule(pl.LightningDataModule):
    def __init__(self, datasets, hparams):
        self.hparams = hparams
        self.datasets = datasets
        self.denormalizers = []
        self.concat_datasets = []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--num_workers', type=int, default=8)

        return parser

    def prepare_data(self):
        """no need for this"""

    def setup(self, mode='train'):
        datasets = []

        for dataset in self.datasets:
            _datamodule = DATAMODULES[dataset](self.hparams)
            _datamodule.prepare_data()
            _datamodule.setup()
            datasets.append(_datamodule.get_dataset(mode))
            self.denormalizers.append(_datamodule.get_denormalizer())

        self.dataset = ConcatDataset(datasets)

    def train_dataloader(self):
        return DataLoader(dataset=self.dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

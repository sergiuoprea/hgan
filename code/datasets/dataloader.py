import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.utils import ConcatDataset
from datasets.transforms import Denormalize

from datasets.realhands import RealHandsDataModule
from datasets.synthhands import SynthHandsDataModule

from argparse import ArgumentParser

DATAMODULES = {'real_hands'    : RealHandsDataModule,
               'synth_hands'   : SynthHandsDataModule}

class MultipleDataModule(pl.LightningDataModule):
    """
    Master datamodule with concatenates several datasets provided in the input list
    """
    def __init__(self, datasets, hparams):
        self.hparams = hparams
        self.datasets = datasets
        self.data = {}
        self.denormalize = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    @staticmethod
    def add_model_specific_args(parent_parser, datasets):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--train_bs', type=int, default=8, help='training batch size')
        parser.add_argument('--valid_bs', type=int, default=4, help='validation batch size')
        parser.add_argument('--test_bs', type=int, default=4, help='test batch size')
        parser.add_argument('--num_workers', type=int, default=8, help='# of workers for the dataloader')
        parser.add_argument('--valid_size', type=int, default=1000, help='validation set size')
        parser.add_argument('--test_size', type=int, default=2000, help='test set size')
        parser.add_argument('--train_inp_size', type=int, default=128, help='size of input images for training')
        parser.add_argument('--valid_inp_size', type=int, default=256, help='size of input images for validation')
        parser.add_argument('--shuffle', type=bool, default=True, help='if True, data will be shuffled')
        parser.add_argument('--random_crop', type=bool, default=True, help='if True, random crop the images (for training)')

        # Take into account the arguments of each dataset
        for dataset in datasets:
            parser = DATAMODULES[dataset].add_model_specific_args(parser)

        return parser

    def prepare_data(self):
        """no need for this"""

    def setup(self):
        _train = []
        _valid = []
        _test = []

        for dataset in self.datasets:
            _datamodule = DATAMODULES[dataset](self.hparams)
            _datamodule.prepare_data()
            _datamodule.setup()
            _train.append(_datamodule.get_dataset('train'))
            _valid.append(_datamodule.get_dataset('valid'))
            _test.append(_datamodule.get_dataset('test'))

        self.data['train'] = ConcatDataset(_train)
        self.data['valid'] = ConcatDataset(_valid)
        self.data['test'] = ConcatDataset(_test)

    def train_dataloader(self):
        return DataLoader(dataset=self.data['train'],
                          batch_size=self.hparams.train_bs,
                          num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.data['valid'],
                          batch_size=self.hparams.valid_bs,
                          num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.data['test'],
                          batch_size=self.hparams.test_bs,
                          num_workers=self.hparams.num_workers)

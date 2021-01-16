#from vanilla_cyclegan.cyclegan import CycleGAN
from vanilla_cyclegan.cyclegan import CycleGAN
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from datasets.dataloader import PairedDataModule, update_dataset_specific_args
from argparse import ArgumentParser
import torch

DATASETS = ["real_hands", "synth_hands"]
parser = ArgumentParser()

if __name__ == '__main__':
    parser = update_dataset_specific_args(DATASETS, parser)
    parser = PairedDataModule.add_model_specific_args(parser)
    parser = CycleGAN.add_model_specific_args(parser)
    args = parser.parse_args()

    logger = TensorBoardLogger('tb_logs', name='cycle_gan')

    dm = PairedDataModule(DATASETS, args)
    dm.prepare_data()
    dm.setup(mode='train')

    vanilla_cyclegan = CycleGAN(hparams=args)

    trainer = pl.Trainer(gpus=1, precision=32, logger=logger, log_every_n_steps=5)
    trainer.fit(vanilla_cyclegan, dm)

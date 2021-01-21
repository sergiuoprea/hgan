import pytorch_lightning as pl
from vanilla_cyclegan.cyclegan import CycleGAN
from vanilla_cyclegan.cyclegan import ValidationCallback
from pytorch_lightning.loggers.neptune import NeptuneLogger
import neptune_cfg
from datasets.dataloader import PairedDataModule, update_dataset_specific_args
from datasets.synthhands import SynthHandsDataModule

from argparse import ArgumentParser
import torch

DATASETS = ["real_hands", "synth_hands"]
parser = ArgumentParser()

if __name__ == '__main__':
    parser = update_dataset_specific_args(DATASETS, parser)
    parser = PairedDataModule.add_model_specific_args(parser)
    parser = CycleGAN.add_model_specific_args(parser)
    args = parser.parse_args()

    # Loggers
    logger = NeptuneLogger(api_key=neptune_cfg.key, project_name=neptune_cfg.project, params=vars(args))

    # Datamodule
    dm = PairedDataModule(DATASETS, args)
    dm.prepare_data()
    dm.setup()

    # CycleGAN instance
    net = CycleGAN(hparams=args)

    # Set seed and deterministic flag in the Trainer to True if a deterministic behavior is expected
    pl.seed_everything(23)

    # Trainer instance
    trainer = pl.Trainer(gpus=1, precision=32, logger=logger, val_check_interval=0.10,
                         deterministic=True, callbacks=[ValidationCallback()],
                         limit_val_batches=100, limit_train_batches=4000)

    # Train the model
    trainer.fit(net, dm)

    # Stop Neptune logging
    logger.experiment.stop()

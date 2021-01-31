import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models.handgan import HandGAN
from models.handgan import ValidationCallback, PrintModels
import neptune_cfg
from datasets.dataloader import MultipleDataModule
from datasets.synthhands import SynthHandsDataModule

from argparse import ArgumentParser
import torch

DATASETS = ["real_hands", "synth_hands"]
parser = ArgumentParser()

if __name__ == '__main__':
    parser = MultipleDataModule.add_model_specific_args(parser, DATASETS)
    parser = HandGAN.add_model_specific_args(parser)
    args = parser.parse_args()

    # Loggers
    logger = NeptuneLogger(api_key=neptune_cfg.key, project_name=neptune_cfg.project, params=vars(args),
                           experiment_name=args.exp_name, experiment_id=args.exp_id)

    # Datamodule
    dm = MultipleDataModule(DATASETS, args)
    dm.prepare_data()
    dm.setup()

    # Model Checkpoint callback
    checkpoint_callback = ModelCheckpoint(monitor='FID', dirpath='./checkpoints/' + args.exp_name,
                                          filename=args.exp_name + '_handgan-{epoch:02d}-{FID:.2f}',
                                          save_top_k=3,
                                          mode='min')

    # CycleGAN instance
    net = HandGAN(hparams=args)

    # Set seed and deterministic flag in the Trainer to True if a deterministic behavior is expected
    if args.deterministic:
        pl.seed_everything(23)

    # Trainer instance
    trainer = pl.Trainer(gpus=args.gpus, precision=args.fp_precision, logger=logger, val_check_interval=args.valid_interval,
                         deterministic=args.deterministic, callbacks=[ValidationCallback(), PrintModels(), checkpoint_callback],
                         limit_val_batches=150, limit_train_batches=5000)

    # Train the mode
    trainer.fit(net, dm)

    # Stop Neptune logging
    logger.experiment.stop()

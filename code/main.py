# Pytorch lightning imports
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Model
from models.handgan import HandGAN
from models.silnet import SilNet

# Callbacks
from models.handgan import ValidationCallback, PrintModels, TestCallback

# Dataloader
from datasets.dataloader import MultipleDataModule

# Other imports:
# Neptune credentials
import neptune_cfg

from argparse import ArgumentParser

SILNET = False
DATASETS = ["real_hands", "synth_hands"]
parser = ArgumentParser()

if __name__ == '__main__':
    parser = MultipleDataModule.add_model_specific_args(parser, DATASETS)
    parser = HandGAN.add_model_specific_args(parser)

    if SILNET:
        parser = SilNet.add_model_specific_args(parser)

    args = parser.parse_args()

    # Loggers
    logger = NeptuneLogger(api_key=neptune_cfg.key, project_name=neptune_cfg.project,
                           params=vars(args), experiment_name=args.exp_name)

    # Datamodule
    dm = MultipleDataModule(DATASETS, args)
    dm.prepare_data()
    dm.setup()

    # Set seed and deterministic flag in the Trainer to True if a deterministic behavior is expected
    if args.deterministic:
        pl.seed_everything(23)

    if SILNET:
        silnet = SilNet(hparams=args)

        stop_callback = EarlyStopping(monitor='avg_loss', mode='auto', patience=5, verbose=True)

        checkpoint_callback = ModelCheckpoint(monitor='avg_loss', dirpath=args.chk_dir + args.exp_name,
                                              filename=args.exp_name + '_silent-{epoch:02d}-{avg_loss:.2f}',
                                              save_top_k=1,
                                              mode='min')
        trainer = pl.Trainer(gpus=args.gpus, logger=logger, val_check_interval=args.valid_interval,
                             callbacks=[checkpoint_callback, stop_callback], 
                             limit_val_batches=200, limit_train_batches=5000,
                             max_epochs=args.max_epochs, gradient_clip_val=0.5)

        trainer.fit(silnet, dm)

    else:
        # Model instance
        net = HandGAN(hparams=args)

        # Model Checkpoint callback
        checkpoint_callback = ModelCheckpoint(monitor='FID', dirpath=args.chk_dir + args.exp_name,
                                              filename=args.exp_name + '_handgan-{epoch:02d}-{FID:.2f}',
                                              save_top_k=2,
                                              mode='min')

        # Trainer instance
        trainer = pl.Trainer(gpus=args.gpus, precision=args.fp_precision, logger=logger, val_check_interval=args.valid_interval,
                            deterministic=args.deterministic, callbacks=[ValidationCallback(), PrintModels(), TestCallback(), checkpoint_callback],
                            limit_val_batches=1.0, limit_train_batches=1.0, max_epochs=args.max_epochs, accumulate_grad_batches=1)

        # Training step
        trainer.fit(net, dm)

        # Test step
        # Load the best model
        net = HandGAN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        trainer.test(net, dm.test_dataloader())
    
    # Stop Neptune logging
    logger.experiment.stop()

import os
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import seed_everything

import utils
import argparse
from dataset.lit_data_module import TemporalShapeDataModule
from models.model_utils import count_parameters
from lit_convlstm import ConvLSTMModule
from lit_3dconv import ThreeDCNNModule
from lit_timesformer import TimeSformerModule


def main(parser, hidden_units=None, config_path=None, seed=None):

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if config_path:
        config = utils.load_json_config(config_path)
    else:
        config = utils.load_json_config(args.config)

    if hidden_units:
        config['dim_head'] = hidden_units
        config['hidden_per_layer'] = [hidden_units, hidden_units, hidden_units]

    if not seed:
        seed = 42

    wandb_logger = WandbLogger(project='temporal-shape', config=config)

    seed_everything(seed, workers=True)

    if config['model_name'] == 'lit_convlstm':
        model = ConvLSTMModule(input_size=(config['batch_size'], config['clip_size'], 1,
                               config['input_spatial_size'], config['input_spatial_size']),
                               optimizer=config['optimizer'],
                               nb_labels=config['nb_labels'],
                               hidden_per_layer=config['hidden_per_layer'],
                               kernel_size_per_layer=config['kernel_size_per_layer'],
                               conv_stride=config['conv_stride'],
                               lr=config['lr'], reduce_lr=config['reduce_lr'],
                               momentum=config['momentum'], weight_decay=config['weight_decay'],
                               dropout_classifier=config['dropout_classifier'],
                               return_sequence=config['return_sequence'],
                               if_not_sequence=config['if_not_sequence'])

    if config['model_name'] == 'lit_3dconv':
        model = ThreeDCNNModule(input_size=(config['batch_size'], config['clip_size'], 1,
                                config['input_spatial_size'], config['input_spatial_size']),
                                optimizer=config['optimizer'],
                                hidden_per_layer=config['hidden_per_layer'],
                                kernel_size_per_layer=config['kernel_size_per_layer'],
                                conv_stride=config['conv_stride'],
                                dropout_encoder=config['dropout_encoder'],
                                pooling=config['pooling'],
                                nb_labels=config['nb_labels'],
                                lr=config['lr'], reduce_lr=config['reduce_lr'],
                                momentum=config['momentum'], weight_decay=config['weight_decay'],
                                dropout_classifier=config['dropout_classifier'])

    if config['model_name'] == 'lit_timesformer':
        model = TimeSformerModule(input_size=(config['batch_size'], config['clip_size'], 1,
                                config['input_spatial_size'], config['input_spatial_size']),
                                dim_head=config['dim_head'], patch_size=config['patch_size'],
                                num_heads=config['num_heads'],
                                num_layers=config['num_layers'],
                                optimizer=config['optimizer'],
                                nb_labels=config['nb_labels'],
                                lr=config['lr'], reduce_lr=config['reduce_lr'],
                                attn_dropout=config['attn_dropout'], ff_dropout=config['ff_dropout'],
                                momentum=config['momentum'], weight_decay=config['weight_decay'],
                                dropout_classifier=config['dropout_classifier'])

    config['nb_encoder_params'], config['nb_trainable_params'] = count_parameters(model)
    print('\n Nb encoder params: ', config['nb_encoder_params'], 'Nb params total: ', config['nb_trainable_params'])

    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max',
                                          verbose=True,
                                          filename='{epoch}-{val_loss:.2f}-{val_acc:.4f}')

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=config['early_stopping_patience'],
        verbose=False,
        mode='max'
    )

    callbacks = [checkpoint_callback, early_stop_callback]
    
    if config['reduce_lr']:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    trainer = pl.Trainer.from_argparse_args(
        args, max_epochs=config['num_epochs'],
        progress_bar_refresh_rate=1,
        callbacks=callbacks,
        weights_save_path=os.path.join(config['output_dir'], args.job_identifier),
        logger=wandb_logger,
        plugins=DDPPlugin(find_unused_parameters=False))

    if trainer.gpus is not None:
        config['num_workers'] = int(trainer.gpus/8 * 128)
    else:
        config['num_workers'] = 2

    test_dm = TemporalShapeDataModule(data_dir=config['test_data_folder'], config=config, seq_first=model.seq_first)
    test_dm_2 = TemporalShapeDataModule(data_dir=config['test_data_folder_2'], config=config, seq_first=model.seq_first)
    test_dm_3 = TemporalShapeDataModule(data_dir=config['test_data_folder_3'], config=config, seq_first=model.seq_first)

    test_accuracies = []

    if config['inference_from_checkpoint_only']:
        if config['model_name'] == 'lit_convlstm':
            model_from_checkpoint = ConvLSTMModule.load_from_checkpoint(config['checkpoint_path'])
        if config['model_name'] == 'lit_3dconv':
            model_from_checkpoint = ThreeDCNNModule.load_from_checkpoint(config['checkpoint_path'])
        trainer.test(datamodule=test_dm, model=model_from_checkpoint)
        trainer.test(datamodule=test_dm_2, model=model_from_checkpoint)
        trainer.test(datamodule=test_dm_3, model=model_from_checkpoint)

    else:
        train_dm = TemporalShapeDataModule(data_dir=config['data_folder'], config=config, seq_first=model.seq_first)
        trainer.fit(model, train_dm)

        best_val_acc = trainer.checkpoint_callback.best_model_score
        wandb_logger.log_metrics({'best_val_acc': best_val_acc})
        test_accuracies.append(best_val_acc)

        trainer.test(datamodule=test_dm)
        test_accuracies.append(trainer.callback_metrics['test_acc'])

        trainer.test(datamodule=test_dm_2)
        test_accuracies.append(trainer.callback_metrics['test_acc'])

        trainer.test(datamodule=test_dm_3)
        test_accuracies.append(trainer.callback_metrics['test_acc'])

    if hasattr(args, 'results_persist'):
        return test_accuracies


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # load configurations
    parser.add_argument('--config', '-c', help='json config file path')
    parser.add_argument('--eval_only', '-e', action='store_true',
                        help="evaluate trained model on validation data.")
    parser.add_argument('--resume', '-r', action='store_true',
                        help="resume training from a given checkpoint.")
    parser.add_argument('--test_run', action='store_true',
                        help="quick test run")
    parser.add_argument('--job_identifier', '-j', help='Unique identifier for run,'
                                                       'avoids overwriting model.')

    main(parser)

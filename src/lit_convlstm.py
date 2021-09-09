from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.convlstm import StackedConvLSTMModel
import pytorch_lightning as pl
from pl_bolts.metrics.object_detection import iou as iou_metric
from pl_bolts.losses.object_detection import iou_loss
from torch import nn
import torch


class ConvLSTMModule(pl.LightningModule):
    def __init__(self, input_size, optimizer, hidden_per_layer,
                 kernel_size_per_layer, conv_stride, lr, reduce_lr,
                 momentum, weight_decay, dropout):
        super(ConvLSTMModule, self).__init__()

        self.b, self.t, self.c, self.h, self.w = input_size
        self.seq_first = True
        self.num_layers = len(hidden_per_layer)
        self.optimizer = optimizer
        self.lr = lr
        self.reduce_lr = reduce_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.convlstm_encoder = StackedConvLSTMModel(
            self.c, hidden_per_layer, kernel_size_per_layer, conv_stride)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.dropout = nn.Dropout(p=dropout)
        self.encoder_out_dim = self.t * hidden_per_layer[-1] * int(self.h /(2**self.num_layers*conv_stride)) * int(self.w/(2**self.num_layers*conv_stride))
        self.linear = nn.Linear(
            in_features=self.encoder_out_dim,
            out_features=12)
        self.iou = iou_metric
        self.sigmoid = nn.Sigmoid()
        self.save_hyperparameters()

    def forward(self, x) -> torch.Tensor:
        x = self.convlstm_encoder(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.sigmoid(x) * self.h
        return x

    @staticmethod
    def loss_function(y_hat, y):
        criterion = nn.SmoothL1Loss()
        # criterion = nn.MSELoss()
        # criterion = iou_loss
        loss = criterion(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss, iou = self.get_loss_iou(y_hat, y)
        self.log('train_iou', iou, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss, iou = self.get_loss_iou(y_hat, y)

        # By default, on_step=False, on_epoch=True for log calls in val and test
        self.log('val_iou', iou, prog_bar=True, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True)
        return {'val_loss': loss, 'val_iou': iou}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss, iou = self.get_loss_iou(y_hat, y)
        self.log('test_iou', iou, prog_bar=True, sync_dist=True)
        self.log('test_loss', loss, sync_dist=True)
        return {'test_loss': loss}

    def get_loss_iou(self, y_hat, y):
        loss = self.loss_function(y_hat, y)
        # losses = self.loss_function(y_hat, y).diag()
        # loss = sum(losses)/self.b
        ious = self.iou(y_hat, y).diag()
        iou = sum(ious)/self.b
        return loss, iou

    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                self.parameters(), self.lr, momentum=self.momentum,
                weight_decay=self.weight_decay)

        if self.optimizer == 'Adadelta':
            optimizer = torch.optim.Adadelta(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.reduce_lr:
            scheduler = ReduceLROnPlateau(
                optimizer, 'max', factor=0.5, patience=2, verbose=True)
            return {'optimizer': optimizer,
                    'lr_scheduler': scheduler,
                    'monitor': 'val_iou'}
        else:
            return optimizer


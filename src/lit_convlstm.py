from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report
from models.convlstm import StackedConvLSTMModel
import pytorch_lightning as pl
from torch import nn
import torchmetrics
import torch


class ConvLSTMModule(pl.LightningModule):
    def __init__(self, input_size, optimizer, hidden_per_layer, nb_labels,
                 kernel_size_per_layer, conv_stride, lr, reduce_lr,
                 momentum, weight_decay, dropout, return_sequence):
        super(ConvLSTMModule, self).__init__()

        self.b, self.t, self.c, self.h, self.w = input_size
        self.seq_first = True
        self.num_layers = len(hidden_per_layer)
        self.out_features = nb_labels
        self.optimizer = optimizer
        self.lr = lr
        self.reduce_lr = reduce_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.return_sequence = return_sequence
        self.convlstm_encoder = StackedConvLSTMModel(
            self.c, hidden_per_layer, kernel_size_per_layer, conv_stride, return_sequence=return_sequence)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.dropout = nn.Dropout(p=dropout)

        self.spatial_out_dim = hidden_per_layer[-1] * int(self.h /(2**self.num_layers*conv_stride)) * int(self.w/(2**self.num_layers*conv_stride))

        if self.return_sequence:
            self.encoder_out_dim = self.t * self.spatial_out_dim
        else:
            self.encoder_out_dim = self.spatial_out_dim

        self.linear = nn.Linear(
            in_features=self.encoder_out_dim,
            out_features=self.out_features)

        self.softmax = nn.Softmax(dim=1)

        self.accuracy = torchmetrics.Accuracy()
        self.top5_accuracy = torchmetrics.Accuracy(top_k=2)
        self.confmat = torchmetrics.ConfusionMatrix(num_classes=nb_labels)
        self.save_hyperparameters()

    def forward(self, x) -> torch.Tensor:
        x = self.convlstm_encoder(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    @staticmethod
    def loss_function(y_hat, y):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss, acc, _ = self.get_loss_acc(y_hat, y)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss, acc, top5_acc = self.get_loss_acc(y_hat, y)
        # By default, on_step=False, on_epoch=True for log calls in val and test
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        self.log('val_top5_acc', top5_acc, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True)
        return {'val_loss': loss, 'val_acc': acc, 'y_hat': y_hat, 'y': y}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss, acc, top5_acc = self.get_loss_acc(y_hat, y)
        self.log('test_acc', acc, prog_bar=True, sync_dist=True)
        self.log('test_top5_acc', top5_acc, sync_dist=True)
        self.log('test_loss', loss, sync_dist=True)
        return {'test_loss': loss, 'y_hat': y_hat, 'y': y}

    def validation_epoch_end(self, validation_step_outputs):
        outs_y = []
        outs_y_hat = []
        for out in validation_step_outputs:
            outs_y.append(out['y'])
            outs_y_hat.append(out['y_hat'])
        y_pred = torch.argmax(torch.cat(outs_y_hat), dim=1)
        y = torch.cat(outs_y)
        cm = self.confmat(y_pred, y)
        print('\n', cm)
        cr = classification_report(y.cpu().numpy(), y_pred.cpu().numpy(), digits=4)
        print(cr)

    def test_epoch_end(self, test_step_outputs):
        outs_y = []
        outs_y_hat = []
        for out in test_step_outputs:
            outs_y.append(out['y'])
            outs_y_hat.append(out['y_hat'])
        y_pred = torch.argmax(torch.cat(outs_y_hat), dim=1)
        y = torch.cat(outs_y)
        cm = self.confmat(y_pred, y)
        print('\n', cm)
        cr = classification_report(y.cpu().numpy(), y_pred.cpu().numpy(), digits=4)
        print(cr)

    def get_loss_acc(self, y_hat, y):
        loss = self.loss_function(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds, y)
        top5_acc = self.top5_accuracy(self.softmax(y_hat), y)
        return loss, acc, top5_acc


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
                    'monitor': 'val_acc'}
        else:
            return optimizer


from timesformer_pytorch.timesformer_pytorch import attn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from timesformer_pytorch import TimeSformer
from sklearn.metrics import classification_report
import pytorch_lightning as pl

from torch import nn
import torchmetrics
import torch

class TransformerModule(pl.LightningModule):
    def __init__(self, input_size, optimizer, hidden_per_layer, nb_labels, 
                 lr, reduce_lr, dim, patch_size, num_frames, attn_dropout, ff_dropout,
                 depth, heads, dim_head, momentum, weight_decay, dropout_classifier):
                 
        super(TransformerModule, self).__init__()

        self.b, self.t, self.c, self.h, self.w = input_size
        self.seq_first = True
        self.num_layers = len(hidden_per_layer)
        self.out_features = nb_labels
        self.optimizer = optimizer
        self.lr = lr
        self.reduce_lr = reduce_lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.dim = dim
        self.patch_size = patch_size
        self.num_frames = self.t
        self.depth = depth

        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout

        self.transformer_encoder = TimeSformer(
                dim = self.dim,  #?
                image_size = self.w if self.w <= self.h else self.h,  # NOTE: TimeSformer accepts square images
                patch_size = self.patch_size,
                num_frames = self.num_frames,
                num_classes = self.out_features,
                depth = self.depth,  #?
                heads = self.heads,  #?
                dim_head =  self.dim_head,  #?
                attn_dropout = self.attn_dropout,
                ff_dropout = self.ff_dropout,
                channels=self.c
            )

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.dropout = nn.Dropout(p=dropout_classifier)

        # sample_input = torch.autograd.Variable(torch.rand(1, 20, 1, 64, 64)) 
        #TimeSformer(dim = 64,image_size = 64, patch_size = 16, num_frames = 20, channels=1, num_classes = 5, depth = 3, heads = 8, dim_head =  64, attn_dropout = 0.1, ff_dropout = 0.1 )(sample_input)

        sample_input = torch.autograd.Variable(torch.rand(1, self.t, self.c, self.h, self.w)) 
        sample_output = self.transformer_encoder(sample_input)
        self.encoder_out_dim = torch.prod(torch.tensor(sample_output.shape[1:]))

        self.linear = nn.Linear(
            in_features=self.encoder_out_dim,
            out_features=self.out_features)

        self.softmax = nn.Softmax(dim=1)

        self.accuracy = torchmetrics.Accuracy()
        self.top5_accuracy = torchmetrics.Accuracy(top_k=2)
        self.confmat = torchmetrics.ConfusionMatrix(num_classes=nb_labels)
        # self.save_hyperparameters()

    def forward(self, x) -> torch.Tensor:
        x = self.transformer_encoder(x)
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

        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                self.parameters(), self.lr, momentum=self.momentum,
                weight_decay=self.weight_decay)

        if self.reduce_lr:
            scheduler = ReduceLROnPlateau(
                optimizer, 'max', factor=0.5, patience=2, verbose=True)
            return {'optimizer': optimizer,
                    'lr_scheduler': scheduler,
                    'monitor': 'val_acc'}
        else:
            return optimizer

from timesformer_pytorch.timesformer_pytorch import attn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from timesformer_pytorch import TimeSformer
from lit_convlstm import ConvLSTMModule
from sklearn.metrics import classification_report
import pytorch_lightning as pl

from torch import nn
import torchmetrics
import torch

class TimeSformerModule(ConvLSTMModule):
    def __init__(self, input_size, dim_head, num_heads, patch_size, num_layers,  
                 nb_labels, lr, reduce_lr,  attn_dropout, ff_dropout,
                 optimizer, momentum, weight_decay):
                 
        super(ConvLSTMModule, self).__init__()

        self.b, self.t, self.c, self.h, self.w = input_size
        self.seq_first = True
        self.out_features = nb_labels
        self.optimizer = optimizer
        self.lr = lr
        self.reduce_lr = reduce_lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.patch_size = patch_size
        self.num_frames = self.t
        self.depth = num_layers

        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim = self.dim_head * self.num_heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout

        self.timesformer_encoder = TimeSformer(
                dim = self.dim,
                image_size = self.w if self.w <= self.h else self.h,  # NOTE: TimeSformer accepts square images
                patch_size = self.patch_size,
                num_frames = self.num_frames,
                num_classes = self.out_features,
                depth = self.depth,
                heads = self.num_heads,
                dim_head =  self.dim_head,
                attn_dropout = self.attn_dropout,
                ff_dropout = self.ff_dropout,
                channels=self.c
            )

        self.softmax = nn.Softmax(dim=1)

        self.accuracy = torchmetrics.Accuracy()
        self.top5_accuracy = torchmetrics.Accuracy(top_k=2)
        self.confmat = torchmetrics.ConfusionMatrix(num_classes=nb_labels)
        self.save_hyperparameters()

    def forward(self, x) -> torch.Tensor:
        x = self.timesformer_encoder(x)
        return x


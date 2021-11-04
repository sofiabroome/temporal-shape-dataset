from timesformer_pytorch.timesformer_pytorch import attn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from timesformer_pytorch import TimeSformer
from lit_convlstm import ConvLSTMModule
from sklearn.metrics import classification_report
import pytorch_lightning as pl

from torch import nn
import torchmetrics
import torch

class TransformerModule(ConvLSTMModule):
    def __init__(self, input_size, optimizer, hidden_per_layer, nb_labels, 
                 lr, reduce_lr, dim, patch_size, num_frames, attn_dropout, ff_dropout,
                 depth, heads, dim_head, momentum, weight_decay, dropout_classifier):
                 
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
        self.save_hyperparameters()

    def forward(self, x) -> torch.Tensor:
        x = self.transformer_encoder(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


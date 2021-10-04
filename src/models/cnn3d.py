import torch
import torch.nn as nn


class VGGStyle3DCNN(nn.Module):
    """
    - A VGG-style 3D CNN with 11 layers.
    - Kernel size is kept 3 for all three dimensions - (time, H, W)
      except the first layer has kernel size of (3, 5, 5)
    - Time dimension is preserved with `padding=1` and `stride=1` (except a
      stride of 2 after third conv layer), and is averaged at the end

    Arguments:
    - Input: a (batch_size, 3, sequence_length, W, H) tensor
    - Returns: a (batch_size, 512) sized tensor
    """

    def __init__(self, input_channels, hidden_per_layer, kernel_size_per_layer, conv_stride, pooling, dropout):
        super(VGGStyle3DCNN, self).__init__()
        self.hidden_per_layer = hidden_per_layer
        self.input_channels = input_channels
        self.conv_stride = conv_stride
        self.pooling = pooling
        self.dropout = dropout
        self.num_layers = len(hidden_per_layer)
        self.blocks = []

        assert(len(hidden_per_layer) == len(kernel_size_per_layer))

        if self.num_layers > 4:
            self.pooling_padding = 1
        else:
            self.pooling_padding = 0

        for i, nb_channels in enumerate(self.hidden_per_layer):
            cur_input_dim = self.input_channels if i == 0 else self.hidden_per_layer[i - 1]
            self.blocks.append(Conv3dBlock(cur_input_dim, hidden_dim=hidden_per_layer[i],
                                           kernel_size=kernel_size_per_layer[i],
                                           stride=self.conv_stride, pooling=self.pooling[i],
                                           pooling_padding=self.pooling_padding, dropout=self.dropout))
        self.conv3d_blocks = nn.ModuleList(self.blocks)
        # self.block1 = nn.Sequential(
        #     nn.Conv3d(1, 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 2, 2)),
        #     nn.BatchNorm3d(2),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout3d(p=0),
        # )

        # self.block2 = nn.Sequential(
        #     nn.Conv3d(2, 2, kernel_size=(3, 3, 3), stride=2, dilation=(1, 1, 1), padding=(1, 1, 1)),
        #     nn.BatchNorm3d(2),
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
        #     nn.Dropout3d(p=0),
        # )
        # self.block3 = nn.Sequential(
        #     nn.Conv3d(2, 2, kernel_size=(3, 3, 3), stride=2, dilation=(1, 1, 1), padding=(1, 1, 1)),
        #     nn.BatchNorm3d(2),
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
        #     nn.Dropout3d(p=0),
        # )

    def forward(self, x):
        
        # print(x.size())
        for layer_idx in range(self.num_layers):
            x = self.conv3d_blocks[layer_idx](x)
            # print(x.size())

        return x


class Conv3dBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, stride,
                 pooling, pooling_padding, dropout):
        super().__init__()
        # Pad according to kernel size to keep size in convolutions
        self.padding = kernel_size // 2, kernel_size // 2, kernel_size // 2 
        self.conv3d = nn.Conv3d(input_dim, hidden_dim,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride, dilation=(1, 1, 1), padding=self.padding)
        self.bn = nn.BatchNorm3d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        if pooling == 'max':
            self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2),
                padding=pooling_padding)
        if pooling == 'avg':
            self.pool = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2),
                padding=pooling_padding)
        self.dropout3d = nn.Dropout3d(p=dropout)

    def forward(self, x):
        # print('begin block')
        # print(x.size())
        x = self.conv3d(x)
        # print(x.size())
        x = self.relu(x)
        x = self.pool(x)
        # print(x.size())
        # print('end block \n')
        x = self.bn(x)
        x = self.dropout3d(x)
        return x


if __name__ == "__main__":
    input_tensor = torch.autograd.Variable(torch.rand(64, 1, 20, 64, 64))
    model = VGGStyle3DCNN(input_channels=1, hidden_per_layer=[ 5, 5, 5, 5],
        kernel_size_per_layer=[3, 3, 3, 3],
        conv_stride=1,
        pooling=["max", "max", "max", "max"],
        dropout=0
        )
    output = model(input_tensor)

    print('\n Output size:')
    print(output.size(), '\n')

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('pytorch_total_params', pytorch_total_params)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('pytorch_total_params, only trainable', pytorch_total_params)


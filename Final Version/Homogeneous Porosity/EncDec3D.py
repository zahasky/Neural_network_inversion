import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

from torch.autograd import Variable


#-------------------------------------------------------------------------------------------------------------------------
#References:
# 1. https://github.com/zahasky/Neural_network_inversion/blob/master/Examples/dense_network_Mo2020/CAAE3D/CAAE3D_models.py
#    (Integration of adversarial autoencoders with residual dense convolutional networks for estimation of
#    non-Gaussian hydraulic conductivities, Water Resources Research)
# 2. https://github.com/yulunzhang/RDN
#    (Residual Dense Network for Image Super-Resolution, CVPR 18)
#-------------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Residual Dnense Encoder-Decoder Convolutional Network')
# Residual-in-residual dense network
parser.add_argument('--dropout-rate', type=float, default=0, help='Dropout rate for the dense blocks')
parser.add_argument('--res-scale', type=float, default=0.2, help='Scaling factor for each residual')
parser.add_argument("--num-blocks", type=int, default=5, help='Number of dense blocks in a residual dense block')
parser.add_argument("--num-RDB", type=int, default=3, help='Number of residual dense blocks in a residual in residual dense block')
# Encoding-decoding process
parser.add_argument('--in-channels', type=int, default=1, help='Number of channels for the input field')
parser.add_argument('--out-channels', type=int, default=1, help='Number of channels for the output field')
parser.add_argument('--lat-channels', type=int, default=2, help='Number of channels for the latent space')
parser.add_argument('--num-filters', type=int, default=48, help='Number of feature maps during the encoding-decoding process')
parser.add_argument('--num-res-blocks', type=int, default=1, help='Number of residual in residual dense block in the network')

args = parser.parse_args()


class DenseBlock(nn.Module):
    def __init__(self, in_features, growth_rate, dropout_rate=args.dropout_rate, kernel_size=3, stride=1, padding=1, non_linearity=True):
        super(DenseBlock, self).__init__()

        def Block(in_channels, non_linearity):
            layers = [nn.BatchNorm3d(in_features)]
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout3d(p=dropout_rate))
            layers.append(nn.Conv3d(in_features, growth_rate, kernel_size, stride, padding, bias=True))
            return nn.Sequential(*layers)

        self.block = Block(in_features, non_linearity)

    def forward(self,x):
        return self.block(x)

class ResiduleDenseBlock(nn.Module):
    def __init__(self, in_features, num_blocks, growth_rate, res_scale):
        super(ResiduleDenseBlock, self).__init__()
        self.res_scale = res_scale
        DBlocks = [DenseBlock(in_features, growth_rate)]

        for i in range(num_blocks-1):
            in_features += growth_rate
            if i == num_blocks-2:
                DBlocks.append(DenseBlock(in_features, growth_rate, non_linearity=False))
                break
            DBlocks.append(DenseBlock(in_features, growth_rate))

        self.dblocks = nn.Sequential(*DBlocks)

    def forward(self, x):
        inp = x
        for dblock in self.dblocks:
            out = dblock(inp)
            inp = torch.cat((inp, out), 1)
        RDB = out.mul(self.res_scale)
        return torch.add(RDB,x)

class ResiduleDenseNetwork(nn.Module):
    def __init__(self, in_features, num_blocks=args.num_blocks, num_RDB = args.num_RDB, res_scale=args.res_scale):
        super(ResiduleDenseNetwork, self).__init__()
        self.res_scale = res_scale
        growth_rate = in_features
        RDBs = [ResiduleDenseBlock(in_features, num_blocks, growth_rate, res_scale)]

        for i in range(num_RDB-1):
            RDBs.append(ResiduleDenseBlock(in_features, num_blocks, growth_rate, res_scale))

        self.residule_dense_blocks = nn.Sequential(*RDBs)

    def forward(self, x):
        RDN = self.residule_dense_blocks(x).mul(self.res_scale)
        return torch.add(RDN,x)

class Encoder(nn.Module):
    def __init__(self, inchannels=args.in_channels, outchannels=args.lat_channels, filters=args.num_filters, num_res_blocks=args.num_res_blocks):
        super(Encoder, self).__init__()
        # input size, inchannels x 20 x 20 x 40
        self.conv1 = nn.Conv3d(inchannels, filters, kernel_size=3, stride=1, padding=1)
        # state size. filters x 20 x 20 x 40
        self.trans1 = nn.Sequential(
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(4,2,1),
        )
        # state size. filters x 10 x 10 x 20
        # Residual-in-residual dense blocks
        self.res_blocks1 = nn.Sequential(*[ResiduleDenseNetwork(filters) for _ in range(num_res_blocks)])
        # state size. filters x 10 x 10 x 20
        self.trans2 = nn.Sequential(
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(4,2,1),
            nn.Conv3d(filters, outchannels, kernel_size=3, stride=1, padding=1),
        )
        # output size, outchannels x 5 x 5 x 10

    def forward(self, inp):
        # inp: inchannels x 20 x 20 x 40
        out1 = self.conv1(inp)         # filterd x 20 x 20 x 40
        out2 = self.trans1(out1)       # filterd x 10 x 10 x 20
        out3 = self.res_blocks1(out2)  # filterd x 10 x 10 x 20
        out4 = self.trans2(out3)       # outchannels x 5 x 5 x 10

        return out4

    def _n_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params

class Decoder(nn.Module):
    def __init__(self, inchannels=args.lat_channels, outchannels=args.out_channels, filters=args.num_filters, num_res_blocks=args.num_res_blocks):
        super(Decoder, self).__init__()

        # input size, inchannels x 5 x 5 x 10
        self.conv1 = nn.Conv3d(inchannels, filters, kernel_size=3, stride=1, padding=1)

        # state size. filters x 5 x 5 x 10
        self.res_block2 = nn.Sequential(*[ResiduleDenseNetwork(filters) for _ in range(num_res_blocks)])
        # state size. filters x 5 x 5 x 10
        self.transup1 = nn.Sequential(
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(filters,filters,
                                kernel_size=4, stride=2, padding=1, bias=False),
           )
        # state size. filters x 10 x 10 x 20
        self.res_block3 = nn.Sequential(*[ResiduleDenseNetwork(filters) for _ in range(num_res_blocks)])
        # state size. filters x 10 x 10 x 20
        self.transup2 = nn.Sequential(
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(filters, outchannels,
                                kernel_size=4, stride=2, padding=1, bias=False),
        )
        # output size, outchannels x 20 x 20 x 40

    def forward(self, z):
        # z: in_channels x 5 x 5 x 10
        out1 = self.conv1(z)           # filters x 5 x 5 x 10
        out2 = self.res_block2(out1)   # filters x 5 x 5 x 10
        out = torch.add(out1, out2)    # filters x 5 x 5 x 10
        out3 = self.transup1(out)      # filters x 10 x 10 x 20
        out4 = self.res_block3(out3)   # filters x 10 x 10 x 20
        out5 = self.transup2(out4)     # outchannels x 20 x 20 x 40

        return out5

    def _n_parameters(self):
        n_params= 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params

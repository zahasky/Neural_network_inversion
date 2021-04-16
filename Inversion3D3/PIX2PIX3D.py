import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.BatchNorm3d(in_features)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Conv3d(in_features, filters, 3, 1, 1, bias=True)]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters),
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class Encoder(nn.Module):
    def __init__(self, inchannels=1, outchannels=2, filters=48, num_res_blocks=1):
        super(Encoder, self).__init__()
        # input size, inchannels x 20 x 20 x 41
        self.conv1 = nn.Conv3d(inchannels, filters, kernel_size=3, stride=1, padding=1)
        # state size. filters x 20 x 20 x 41
        self.trans1 = nn.Sequential(
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(4,2,1),
        )
        # state size. filters x 10 x 10 x 20
        self.res_blocks2 = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # state size. filters x 10 x 10 x 20
        self.trans2 = nn.Sequential(
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(4,2,1),
            nn.Conv3d(filters, outchannels, kernel_size=3, stride=1, padding=1),
        )
        # output size, outchannels x 5 x 5 x 10

    def forward(self, img):
        # img: inchannels x 20 x 20 x 41
        out1 = self.conv1(img)
        out2 = self.trans1(out1)
        out3 = self.res_blocks2(out2)
        out4 = self.trans2(out3)

        return out4

    def _n_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params

class Decoder(nn.Module):
    def __init__(self, inchannels=2, outchannels=1, filters=48, num_res_blocks=1, num_upsample=2):
        super(Decoder, self).__init__()

        # First layer. input size, inchannels x 5 x 5 x 10
        self.conv1 = nn.Conv3d(inchannels, filters, kernel_size=3, stride=1, padding=1)

        # state size. filters x 5 x 5 x 10
        # Residual blocks
        self.res_block1 = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        self.transup1 = nn.Sequential(
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(filters,filters,
                                kernel_size=4, stride=2, padding=1, bias=False),
           )
        self.res_block2 = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        self.transup2 = nn.Sequential(
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(filters, outchannels,
                                kernel_size=4, stride=2, padding=1, bias=False),
        )

    def forward(self, z):
        # z: in_channels x 5 x 5 x 10
        out1 = self.conv1(z)           # filters x 5 x 5 x 10
        out2 = self.res_block1(out1)   # filters x 5 x 5 x 10
        out = torch.add(out1, out2)    # filters x 5 x 5 x 10
        out3 = self.transup1(out)      # filters x 10 x 10 x 20
        out4 = self.res_block2(out3)   # filters x 10 x 10 x 20
        out5 =  self.transup2(out4)    # outchannels x 20 x 20 x 40

        return out5

    def _n_parameters(self):
        n_params= 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params

class Discriminator(nn.Module):
    def __init__(self, inchannels=1, outchannels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(inchannels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )

        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = nn.Sequential(
            nn.Linear(512*2,128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final2 = nn.Sequential(
            nn.Linear(128, outchannels),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.model(input)
        # state size. 512 x 1 x 1 x 2
        x = x.view(x.size(0), -1)
        # state size. 1 x 1024
        x = self.final1(x)
        # state size. 1 x 128
        x = self.final2(x)
        # state size. 1 x 1
        return x

    def _n_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params

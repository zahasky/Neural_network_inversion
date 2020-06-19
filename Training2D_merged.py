# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:50:41 2020

@author: zahas
"""

import torch
import torchvision
import numpy as np
import math
import os
import CAAE_models
import itertools

from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from CAAE_models import Encoder, Decoder, Discriminator
from torch.utils.data import Dataset, DataLoader


# ----------------
#  Dataset Object
# ----------------
class TrainDataset(Dataset):

    def __init__(self, transform=None):
         #initial conditions
         # training data iteration
         td = [401, 402, 403, 404, 405]
         label_list = []
         input_list = []
         # number of layers
         nlay = 1

         perm_field_dir = os.path.join('.', 'matlab_perm_fields/k_training_data')
         workdir = os.path.join('.', 'Tdata_2D')

         #use existing permeability maps as labels
         for i in td:

             tdata_km2 = np.loadtxt(perm_field_dir + '/td_km2_' + str(i) +'.csv', delimiter=',', dtype=np.float32)

             nrow = int(tdata_km2[-2]) # number of rows / grid cells
             ncol = int(tdata_km2[-1]) # number of columns (parallel to axis of core)
             tdata_km2 = torch.from_numpy(tdata_km2[0:-2])
             #input should be four-dimension (Nf*Df*H*W)
             tdata_km2 = tdata_km2.view(1,1,20,40)
             label_list.append(tdata_km2)


             #use time difference maps as inputs

             model_inp_filename_sp = workdir + '/td' + str(i) + '_' + str(nlay) + '_' + str(nrow) + '_' + str(ncol)\
                                 +'.csv'
             tdata_ex = np.loadtxt(model_inp_filename_sp, delimiter=',', dtype=np.float32)
             tdata_ex = torch.from_numpy(tdata_ex[0:-2])
             tdata_ex = tdata_ex.view(1,1,20,40)
             input_list.append(tdata_ex)


         self.input = input_list
         self.label = label_list
         self.nsamples = nrow*ncol*len(td)
         self.transform = transform

    def __getitem__(self, index):
         sample = self.input[index], self.label[index]

         if self.transform:
             sample = self.transform(sample)

         return sample


    def __len__(self):
         return self.nsamples


# --------------------------------
#  Initializing Training Datasets
# --------------------------------
#initialize dataset object
dataset = TrainDataset()

#get the first image
dataset_input = dataset[0][0]
dataset_label = dataset[0][1]
dataloader_input = DataLoader(dataset=dataset_input, batch_size=40, shuffle=False, num_workers=2)
dataloader_label = DataLoader(dataset=dataset_label, batch_size=40, shuffle=False, num_workers=2)


# --------------------------------------------------------
#  Initializing Parameters and Key Components of the Model
# --------------------------------------------------------
# loss functions
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

# latent dimension = nf*h*w
nf, h, w = 1, 10, 20

# Initialize generator and discriminator
encoder = Encoder(outchannels=nf)
decoder = Decoder(inchannels=nf)
discriminator = Discriminator()

print("number of parameters: {}".format(encoder._n_parameters()+decoder._n_parameters()+discriminator._n_parameters()))

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=0.0002, betas=(0.5, 0.999))

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

Tensor = torch.FloatTensor
nepochs = 50


# ----------
#  Training
# ----------
dataiter = iter(dataloader_label)

for epoch in range(1,nepochs+1):
    encoder.train()
    decoder.train()
    discriminator.train()

    for i, (imgs_inp) in enumerate(dataloader_input):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs_inp.shape[0],1).fill_(1.0), requires_grad=False)
        fake  = Variable(Tensor(imgs_inp.shape[0],1).fill_(0.0), requires_grad=False)


        imgs_lab = dataiter.next()
        # Configure input and label
        input_imgs = Variable(imgs_inp.type(Tensor))
        label_imgs = Variable(imgs_lab.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs = encoder(input_imgs)
        decoded_imgs = decoder(encoded_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss_a = adversarial_loss(discriminator(encoded_imgs), valid)
        g_loss_c = pixelwise_loss(decoded_imgs, label_imgs)
        g_loss = 0.01 * g_loss_a + (1 - 0.01) * g_loss_c

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (imgs_inp.shape[0], 1, h, w))))

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)

        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

    print(
        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f /G_A loss: %f/ G_C loss: %f]"
        % (epoch, 50, i, len(dataloader_input), d_loss.item(), g_loss.item(), g_loss_a.item(), g_loss_c.item())
    )
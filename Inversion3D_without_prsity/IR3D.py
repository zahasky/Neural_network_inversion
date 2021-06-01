import torch
import torchvision
import torch.nn as nn
import numpy as np
import os
import itertools
import argparse

from numpy import savetxt
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from EncDec3D import Encoder, Decoder
from DataLoader import InputDataset
from Accuracy3D import SSIM, R2, RMSE

parser = argparse.ArgumentParser(description='Training and Validation Loop for the Residual Dnense Encoder-Decoder Convolutional Network')
# Traning/Validation Process
parser.add_argument('--idx-train', type=int, default=19980, help='Index of the last training data')
parser.add_argument('--idx-val', type=int, default=24960, help='Index of the last validation data')
parser.add_argument('--train-batch-size', type=int, default=90, help='Number of the batches for training')
parser.add_argument('--val-batch-size', type=int, default=60, help='Number of the batches for validation')
parser.add_argument('--num-epochs', type=int, default=210, help='Number of epochs for both the training and validation')
# Optimizer
parser.add_argument('--lr-ini', type=float, default=0.005, help='Learnign rate for the Adam optimizer')
parser.add_argument("--beta-1", type=float, default=0.5, help='Adam: decay of first order momentum of gradient')
parser.add_argument("--beta-2", type=float, default=0.999, help='Adam: decay of second order momentum of gradient')
# Learning rate decay
parser.add_argument('--factor', type=float, default=0.5, help='Factor of the learning rate reduction')
parser.add_argument("--patience", type=float, default=10, help='Number of epochs before the learning rate is reduced due to no improvement')
# Loss function
parser.add_argument('--w-KL', type=float, default=0.15, help='Weight of the KL-Divergence loss in the overall loss function')
parser.add_argument('--w-L1', type=float, default=1, help='Weight of the L1 loss in the overall loss function')
# Early stopping
parser.add_argument('--loss-stop', type=float, default=1.5, help='Trigger for early stopping based on the training loss')
parser.add_argument('--loss-stopv', type=float, default=1, help='Trigger for early stopping based on the validation loss')

args = parser.parse_args()

#--------------------------------
# Initializing Training Datasets
#--------------------------------
# Initialize dataset object
dataset = InputDataset()

# Loading the processed input (arrival time) with boundary condition (mean permeability) and labeling (permeability) data
train_dataset = dataset.input[0:args.idx_train]
validation_dataset = dataset.input[args.idx_train:args.idx_val]
train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)
validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=args.val_batch_size, shuffle=True, num_workers=2)


#--------------------------------------------------------
# Initializing Parameters and Key Components of the Model
#--------------------------------------------------------
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# List storing the training loss
g_l = []
# List storing the validation loss
g_lv = []
# List storing the validation accuracy
ssim_v = []
r2_v = []
rmse_v = []

# Loss functions
pixelwise_loss = torch.nn.L1Loss()
mse_loss = torch.nn.MSELoss()
kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

# Accuracy calculations
ssim_accu = SSIM()
r2_accu = R2()
rmse_accu = RMSE()

# Initialize generator and discriminator
encoder = Encoder()
decoder = Decoder()

if cuda:
    encoder.cuda()
    decoder.cuda()
    pixelwise_loss.cuda()
    mse_loss.cuda()
    kl_loss.cuda()

# Optimizer
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=args.lr_ini, betas=(args.beta_1, args.beta_2))

# Scheduler
scheduler = ReduceLROnPlateau(
                    optimizer_G, mode='min', factor=args.factor, patience=args.patience,
                    verbose=True, threshold=0.0001, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-08)

print("number of parameters: {}".format(encoder._n_parameters()+decoder._n_parameters()))

for epoch in range(1,args.num_epochs+1):
    encoder.train()
    decoder.train()

#----------
# Training
#----------
    for i, (imgs_inp) in enumerate(train_dataloader):
        for j, (image) in enumerate(imgs_inp):
            # Get inputs and targets
            if j == 0:
                # The real input image with the mean permeability as boundary condition
                true_inp_P = Variable(image.type(Tensor))
            else:
                # The real labeling permeability field
                true_target = Variable(image.type(Tensor))

        # Encoding-Decoding model
        lat_target = encoder(true_inp_P)
        fake_target = decoder(lat_target)

        #-----------------
        # Train Generator
        #-----------------
        optimizer_G.zero_grad()

        # Using softmax on each layer of both the input and decoded image to generate their distribution
        # for KL divergence loss calculation
        dec_soft_lay = F.log_softmax(fake_target, dim=-3)
        inp_soft_lay = F.softmax(true_target, dim=-3)

        # Using softmax on each row of both the input and decoded image to generate their distribution
        # for KL divergence loss calculation
        dec_soft_row = F.log_softmax(fake_target, dim=-2)
        inp_soft_row = F.softmax(true_target, dim=-2)

        # Using softmax on each column of both the input and decoded image to generate their distribution
        # for KL divergence loss calculation
        dec_soft_col = F.log_softmax(fake_target, dim=-1)
        inp_soft_col = F.softmax(true_target, dim=-1)

        # Traing loss propogation
        g_loss_k_row = kl_loss(dec_soft_row, inp_soft_row)
        g_loss_k_col = kl_loss(dec_soft_col, inp_soft_col)
        g_loss_k_lay = kl_loss(dec_soft_lay, inp_soft_lay)
        g_loss_p = pixelwise_loss(fake_target, true_target)

        KL_loss = args.w_KL*(g_loss_k_row + g_loss_k_col + g_loss_k_lay)
        g_loss = args.w_L1*g_loss_p + KL_loss

        g_loss.backward()
        optimizer_G.step()

    print(
        "[Epoch %d/%d] [Batch %d/%d] [G loss: %f/ G_pix loss: %f/ G_KL loss: %f]"
        % (epoch, args.num_epochs, i+1, len(train_dataloader), g_loss.item(), g_loss_p.item(), KL_loss.item())
    )

    g_l.append(g_loss.item())

    encoder.eval()
    decoder.eval()

    # Accuracy between the labeling and prediction
    ssim = 0
    r2 = 0
    rmse = 0

#------------------------------------
# Validation and Accuracy Calculation
#------------------------------------
    for i, (imgs_v) in enumerate(validation_dataloader):
        for j, (imagev) in enumerate(imgs_v):
            # Get inputs and targets
            if j == 0:
                # The real input image with the mean permeability as boundary condition
                true_inp_Pv = Variable(imagev.type(Tensor))
            else:
                # The real labeling permeability field
                true_targetv = Variable(imagev.type(Tensor))
           
        with torch.no_grad():
            # Encoding-Decoding model
            lat_targetv = encoder(true_inp_Pv)
            fake_targetv = decoder(lat_targetv)

            # ---------------------
            #  Validate Generator
            # ---------------------
            # Using softmax on each layer of both the input and decoded image to generate their distribution
            # for KL divergence loss calculation
            dec_soft_layv = F.log_softmax(fake_targetv, dim=-3)
            inp_soft_layv = F.softmax(true_targetv, dim=-3)

            # Using softmax on each row of both the input and decoded image to generate their distribution
            # for KL divergence loss calculation
            dec_soft_rowv = F.log_softmax(fake_targetv, dim=-2)
            inp_soft_rowv = F.softmax(true_targetv, dim=-2)

            # Using softmax on each column of both the input and decoded image to generate their distribution
            # for KL divergence loss calculation
            dec_soft_colv = F.log_softmax(fake_targetv, dim=-1)
            inp_soft_colv = F.softmax(true_targetv, dim=-1)

            # Validation loss calculation
            g_loss_k_rowv = kl_loss(dec_soft_rowv, inp_soft_rowv)
            g_loss_k_colv = kl_loss(dec_soft_colv, inp_soft_colv)
            g_loss_k_layv = kl_loss(dec_soft_layv, inp_soft_layv)
            g_loss_pv = pixelwise_loss(fake_targetv, true_targetv)

            KL_lossv = args.w_KL*(g_loss_k_rowv + g_loss_k_colv + g_loss_k_layv)
            g_lossv = args.w_L1*g_loss_pv + KL_lossv

            # Accuracy calculation
            ssim += ssim_accu(fake_targetv, true_targetv)
            r2 += r2_accu(fake_targetv, true_targetv)
            rmse += rmse_accu(fake_targetv, true_targetv)

    ssim = ssim/len(validation_dataloader)
    r2 = r2/len(validation_dataloader)
    rmse = rmse/len(validation_dataloader)

    scheduler.step(g_lossv.item())

    print(
        "[Epoch %d/%d] [Batch %d/%d] [G lossv: %f/ G_pix lossv: %f/ G_KL lossv: %f]"
        % (epoch, args.num_epochs, i+1, len(validation_dataloader), g_lossv.item(), g_loss_pv.item(), KL_lossv.item())
    )

    g_lv.append(g_lossv.item())
    ssim_v.append(ssim.item())
    r2_v.append(r2.item())
    rmse_v.append(rmse.item())

    # Stop the train
    if g_lossv.item() < args.loss_stopv and g_loss.item() < args.loss_stop:
        break

print(g_l)
print(g_lv)
print(ssim_v)
print(r2_v)
print(rmse_v)

#-------------------------
# Storing Training Results
#-------------------------
result_dir = os.path.join('.', 'results')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
enc_dir = result_dir + '/encoder_epoch{}.pth'.format(args.num_epochs)
dec_dir = result_dir + '/decoder_epoch{}.pth'.format(args.num_epochs)

torch.save(encoder.state_dict(), enc_dir)
torch.save(decoder.state_dict(), dec_dir)

target = true_target[0][0]
inp_img = true_inp_P[0][0]
dec_img = fake_target[0][0]

target = target.flatten()
inp_img = inp_img.flatten()
dec_img = dec_img.flatten()

target = target.cpu().detach().numpy()
inp_img = inp_img.cpu().detach().numpy()
dec_img = dec_img.cpu().detach().numpy()

# Save sample images
savetxt('./results/target.csv', target, delimiter=',')
savetxt('./results/inp.csv', inp_img, delimiter=',')
savetxt('./results/dec.csv', dec_img, delimiter=',')

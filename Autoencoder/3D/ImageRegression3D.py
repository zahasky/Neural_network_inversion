import torch
import torchvision
import torch.nn as nn
import numpy as np
import math
import os
import CAAE3D
import itertools

from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.utils.data import Dataset, DataLoader
from CAAE3D import Encoder, Decoder
from numpy import savetxt


# ----------------
#  Dataset Object
# ----------------
class InputDataset(Dataset):

    def __init__(self, transform=None):
        #initial conditions
        td = [x for x in range(1,9000)]
        label_list = []
        input_list = []

        #perm_field_dir = os.path.join('.', 'perm3D')
        workdir = os.path.join('.', 'bt3D')
        # training data iteration
        for i in td:
            #use existing permeability maps as labels
            #tdata_km2 = np.loadtxt(perm_field_dir + '/td_3dk_m2_' + str(i) +'.csv', delimiter=',', dtype=np.float32)

            nlay = 20 # number of layers
            nrow = 20 # number of rows / grid cells
            ncol = 40 # number of columns (parallel to axis of core)

            #use time difference maps as inputs
            if (i <= 500):
                model_inp_filename_sp = workdir + '500' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'
            elif (i <= 1000):
                model_inp_filename_sp = workdir + '1000' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'
            elif (i <= 1500):
                model_inp_filename_sp = workdir + '1500' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'
       	    elif (i <= 2000):
                model_inp_filename_sp = workdir + '2000' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'
       	    elif (i <= 2500):
                model_inp_filename_sp = workdir + '2500' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'
       	    elif (i <= 3000):
                model_inp_filename_sp = workdir + '3000' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'
       	    elif (i <= 3500):
                model_inp_filename_sp = workdir + '3500' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'
       	    elif (i <= 4000):
                model_inp_filename_sp = workdir + '4000' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'
       	    elif (i <= 4500):
                model_inp_filename_sp = workdir + '4500' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'
       	    elif (i <= 5000):
                model_inp_filename_sp = workdir + '5000' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'
            elif (i <= 5500):
                model_inp_filename_sp = workdir + '5500' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'
            elif (i <= 6000):
                model_inp_filename_sp = workdir + '6000' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'
            elif (i <= 6500):
                model_inp_filename_sp = workdir + '6500' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'
            elif (i <= 7000):
                model_inp_filename_sp = workdir + '7000' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'
            elif (i <= 7500):
                model_inp_filename_sp = workdir + '7500' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'
            elif (i <= 8000):
                model_inp_filename_sp = workdir + '8000' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'
            elif (i <= 8500):
                model_inp_filename_sp = workdir + '8500' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'
            elif (i <= 9000):
                model_inp_filename_sp = workdir + '9000' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'
            # elif (i <= 9500):
            #     model_inp_filename_sp = workdir + '9500' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
            #                             + str(ncol) +'.csv'
            # elif (i <= 10000):
            #     model_inp_filename_sp = workdir + '10000' + '/norm_td' + str(i) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
            #                             + str(ncol) +'.csv'

            tdata_ex = np.loadtxt(model_inp_filename_sp, delimiter=',', dtype=np.float32)
            tdata_ex = torch.from_numpy(tdata_ex[0:-2])
            tdata_ex = tdata_ex.view(1,20,20,40)
            input_list.append(tdata_ex)

        self.input = input_list
        # self.label = label_list
        self.td = td
        self.nrow = nrow
        self.ncol = ncol
        self.nlay = nlay
        self.transform = transform
        self.len = nlay*nrow*ncol

    def __getitem__(self, index):
        sample = self.input[index]

        return sample

    def __len__(self):
        return self.len


# --------------------------------
#  Initializing Training Datasets
# --------------------------------
# #initialize dataset object
dataset = InputDataset()
train_dataset = dataset[0:8000]
validation_dataset = dataset[8000:9000]

train_dataloader = DataLoader(dataset=train_dataset, batch_size=125, shuffle=True, num_workers=2)
validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=12, shuffle=True, num_workers=2)


# --------------------------------------------------------
#  Initializing Parameters and Key Components of the Model
# --------------------------------------------------------
cuda = True if torch.cuda.is_available() else False

# latent dimension = nf*h*w
nf, h, w = 1, 10, 20
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
nepochs = 100

# lists storing training loss
g_l = []

# lists storing validation loss
g_lv = []

# loss functions
pixelwise_loss = torch.nn.L1Loss()

# initialize generator and discriminator
encoder = Encoder(outchannels=nf)
decoder = Decoder(inchannels=nf)

if cuda:
    encoder.cuda()
    decoder.cuda()
    pixelwise_loss.cuda()

# optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=0.001, betas=(0.9, 0.999))

# learning rate decay
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma = 0.95)

print("number of parameters: {}".format(encoder._n_parameters()+decoder._n_parameters()))


for epoch in range(1,nepochs+1):
    encoder.train()
    decoder.train()

# ----------
#  Training
# ----------
    for i, (imgs_inp) in enumerate(train_dataloader):

        # Configure input
        input_imgs = Variable(imgs_inp.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs = encoder(input_imgs)
        decoded_imgs = decoder(encoded_imgs)
        decoded_imgs = np.float32(0.7)*decoded_imgs

        g_loss_p = pixelwise_loss(decoded_imgs, input_imgs)
        g_loss = g_loss_p
   
        g_loss.backward()
        optimizer_G.step()

    print(
        "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
        % (epoch, nepochs, i+1, len(train_dataloader), g_loss.item())
    )

    g_l.append(g_loss.item())

    scheduler.step()

    encoder.eval()
    decoder.eval()
# -----------
#  Validation
# -----------
    for i, (imgs_v) in enumerate(validation_dataloader):

        # Configure input
        img_v = Variable(imgs_v.type(Tensor))

        encoded_imgs_v = encoder(img_v)
        decoded_imgs_v = decoder(encoded_imgs_v)
       	decoded_imgs_v = np.float32(0.7)*decoded_imgs_v
        
        g_loss_p_v = pixelwise_loss(decoded_imgs_v, img_v)
        g_loss_v = g_loss_p_v
    
    print(
        "[Epoch %d/%d] [Batch %d/%d] [G loss validation: %f]"
        % (epoch, nepochs, i+1, len(validation_dataloader), g_loss_v.item())
    )


    g_lv.append(g_loss_v.item())

print(g_l)
print(g_lv)

# -------------------------
#  Storing Training Results
# -------------------------
result_dir = os.path.join('.', 'results')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
encoder_dir = result_dir + '/encoder_epoch{}.pth'.format(nepochs)
decoder_dir = result_dir + '/decoder_epoch{}.pth'.format(nepochs)

torch.save(decoder.state_dict(), decoder_dir)
torch.save(encoder.state_dict(), encoder_dir)
inp_img = input_imgs[0][0]
dec_img = decoded_imgs[0][0]
latent_img = encoded_imgs[0][0]

inp_img = input_imgs.flatten()
dec_img = decoded_imgs.flatten()
latent_img = encoded_imgs.flatten()

inp_img = inp_img.cpu().detach().numpy()
latent_img = latent_img.cpu().detach().numpy()
dec_img = dec_img.cpu().detach().numpy()

#save sample images
savetxt('./results/inp.csv', inp_img, delimiter=',')
savetxt('./results/dec.csv', dec_img, delimiter=',')
savetxt('./results/lat.csv', latent_img, delimiter=',')



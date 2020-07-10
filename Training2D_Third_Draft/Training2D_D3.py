import torch
import torchvision
import numpy as np
import math
import os
import CAAE_models
import itertools
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from CAAE import Encoder, Decoder
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ----------------
#  Dataset Object
# ----------------
class TrainDataset(Dataset):

    def __init__(self, transform=None):
        #initial conditions
        td = [x for x in range(401,491)]
        label_list = []
        input_list = []
        # number of layers
        nlay = 1

        perm_field_dir = os.path.join('.', 'matlab_perm_fields/k_training_data')
        workdir = os.path.join('.', 'Tdata_2D')
        # training data iteration
        for i in td:
            #use existing permeability maps as labels
            tdata_km2 = np.loadtxt(perm_field_dir + '/td_km2_' + str(i) +'.csv', delimiter=',', dtype=np.float32)

            nrow = int(tdata_km2[-2]) # number of rows / grid cells
            ncol = int(tdata_km2[-1]) # number of columns (parallel to axis of core)

            #use time difference maps as inputs
            model_inp_filename_sp = workdir + '/td' + str(i) + '_' + str(nlay) + '_' + str(nrow) + '_' + str(ncol) \
                                +'.csv'
            tdata_ex = np.loadtxt(model_inp_filename_sp, delimiter=',', dtype=np.float32)
            tdata_ex = torch.from_numpy(tdata_ex[0:-2])
            tdata_ex = tdata_ex.view(1,20,40)
            input_list.append(tdata_ex)

        self.input = input_list
        # self.label = label_list
        self.td = td
        self.nrow = nrow
        self.ncol = ncol
        self.transform = transform
        self.len = nrow*ncol

    def __getitem__(self, index):
        sample = self.input[index]

        return sample

    def __len__(self):
        return self.len


# --------------------------------
#  Initializing Training Datasets
# --------------------------------
# #initialize dataset object
dataset = TrainDataset()
dataset_input = dataset[0:90]
dataloader_input = DataLoader(dataset=dataset_input, batch_size=10, shuffle=True, num_workers=2)


# --------------------------------------------------------
#  Initializing Parameters and Key Components of the Model
# --------------------------------------------------------
# latent dimension = nf*h*w
nf, h, w = 1, 10, 20
Tensor = torch.FloatTensor
nepochs = 60

#list storing generator's loss and pixel-wise loss
g_l = []
g_lc = []

# loss functions
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
encoder = Encoder(outchannels=nf)
decoder = Decoder(inchannels=nf)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=0.0002, betas=(0.5, 0.999))

print("number of parameters: {}".format(encoder._n_parameters()+decoder._n_parameters()))

# ----------
#  Training
# ----------
for epoch in range(1,nepochs+1):
    encoder.train()
    decoder.train()
#    discriminator.train()

    for i, (imgs_inp) in enumerate(dataloader_input):

        # Configure input
        input_imgs = Variable(imgs_inp.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs = encoder(input_imgs)
        decoded_imgs = decoder(encoded_imgs)

        g_loss_c = pixelwise_loss(decoded_imgs, input_imgs)
        g_loss = g_loss_c

        g_loss.backward()
        optimizer_G.step()


    print(
        "[Epoch %d/%d] [Batch %d/%d] [G loss: %f / G_C loss: %f]"
        % (epoch, nepochs, i+1, len(dataloader_input), g_loss.item(), g_loss_c.item())
    )

    g_l.append(g_loss.item())
    g_lc.append(g_loss_c.item())


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


# -------------
#  Plot Losses
# -------------
fig1 = plt.figure(figsize=(18, 10))
fig1.add_subplot(224)
x_data = list(range(nepochs))
plt.plot(x_data, g_l, 'g')
plt.plot(x_data, g_lc, 'c')
plt.xlabel('number of epochs')
plt.ylabel('pixel-wise loss of generator')


# ---------------------------------
#  Plot the Input and Decoded Image
# ---------------------------------
# layer to plot
ilayer = 0
# fontsize
fs = 18
hfont = {'fontname':'Arial'}

# number of layers
nlay = 1
# Grid cell size
grid_size = [0.25, 0.25, 0.25] # selected units [cm]

# Define grid
# Describe grid for results
Lx = (dataset.ncol) * grid_size[1]   # length of model in selected units
Ly = (dataset.nrow) * grid_size[0]   # length of model in selected units
y, x = np.mgrid[slice(0, Ly + grid_size[0], grid_size[0]),
                slice(0, Lx + grid_size[1], grid_size[1])]

# The latent vector's size is 5 by 10
latent_x = (10) * grid_size[1]
latent_y = (5) * grid_size[0]
l_y, l_x = np.mgrid[slice(0, latent_y + grid_size[0], grid_size[0]),
                    slice(0, latent_x + grid_size[1], grid_size[1])]

inp_img = imgs_inp[0][0]
latent_img = encoded_imgs[0][0]
dec_img = decoded_imgs[0][0]
print(inp_img)
print(latent_img)
print(dec_img)
inp_img = inp_img.detach().numpy()
latent_img = latent_img.detach().numpy()
dec_img = dec_img.detach().numpy()

ax0 = fig1.add_subplot(2, 2, 1, aspect='equal')
imp = plt.pcolor(x, y, inp_img, cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Pore Volumes', fontsize=fs-5, **hfont)
cbar.ax.tick_params(labelsize= (fs-2))
ax0.tick_params(axis='both', which='major', labelsize=fs)
ax0.set_xlabel('Distance from inlet [cm]', fontsize=fs-5, **hfont)
plt.ylabel('Distance [cm]', fontsize=fs-5, **hfont)
plt.title('Input Breakthrough Time Difference Map', fontsize=fs, **hfont)

ax2 = fig1.add_subplot(2, 2, 2, aspect='equal')
imp = plt.pcolor(l_x, l_y, latent_img, cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize= (fs-2))
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.title('Latent Map', fontsize=fs, **hfont)

ax2 = fig1.add_subplot(2, 2, 3, aspect='equal')
imp = plt.pcolor(x, y, dec_img, cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Pore Volumes', fontsize=fs-5, **hfont)
cbar.ax.tick_params(labelsize= (fs-2))
ax2.tick_params(axis='both', which='major', labelsize=fs)
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs-5, **hfont)
plt.ylabel('Distance [cm]', fontsize=fs-5, **hfont)
plt.title('Decoded Breakthrough Time Difference Map', fontsize=fs, **hfont)

plt.show()

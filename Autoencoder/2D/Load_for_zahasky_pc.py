#This file is used to load the trained .pth files.
#The results of the encoder and decoder will be plotted.
import torch
import torchvision
import numpy as np
import math
import os
import itertools
import matplotlib.pyplot as plt
import ED

from ED import Encoder, Decoder
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ----------------
#  Dataset Object
# ----------------
class TrainDataset(Dataset):

    def __init__(self, transform=None):
        #initial conditions
        td = [x for x in range(1,10)]
        label_list = []
        input_list = []
        # number of layers
        nlay = 1
        
        # Zitong paths
        # perm_field_dir = os.path.join('.', 'matlab_perm_fields/k_training_data/')
        # workdir = os.path.join('.', 'Tdata_2D/')
        # Zahasky paths
        perm_field_dir = os.path.join('C:\\Users\\zahas\\Dropbox\\Matlab\\Deep_learning\\Neural_network_inversion\\training_data_generation\\2D_fields\\matlab_perm_fields\k_training_data\\')
        workdir = os.path.join('C:\\Users\\zahas\\Dropbox\\Matlab\\Deep_learning\\Neural_network_inversion\\training_data_generation\\2D_fields\\Tdata_2D\\')
        
        # training data iteration
        for i in td:
            #use existing permeability maps as labels
            tdata_km2 = np.loadtxt(perm_field_dir + 'td_km2_' + str(i) +'.csv', delimiter=',', dtype=np.float32)

            nrow = int(tdata_km2[-2]) # number of rows / grid cells
            ncol = int(tdata_km2[-1]) # number of columns (parallel to axis of core)

            #use time difference maps as inputs
            model_inp_filename_sp = workdir + 'td' + str(i) + '_' + str(nlay) + '_' + str(nrow) + '_' + str(ncol) \
                                +'.csv'
            tdata_ex = np.loadtxt(model_inp_filename_sp, delimiter=',', dtype=np.float32)
            tdata_ex = torch.from_numpy(tdata_ex[0:-2])
            tdata_ex = tdata_ex.view(nlay, nrow, ncol)
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
dataset_input = dataset[3:5]
dataloader_input = DataLoader(dataset=dataset_input, batch_size=1, shuffle=True, num_workers=0)


nf, h, w = 1, 10, 20
Tensor = torch.FloatTensor

encoder = Encoder(outchannels=nf)
decoder = Decoder(inchannels=nf)

encoder.load_state_dict(torch.load('./encoder_epoch50.pth', map_location='cpu'))
encoder.eval()

decoder.load_state_dict(torch.load('./decoder_epoch50.pth', map_location='cpu'))
decoder.eval()

for i, (imgs_inp) in enumerate(dataloader_input):
    imgs_inp = Variable(imgs_inp.type(Tensor))
    encoded_imgs = encoder(imgs_inp)
    decoded_imgs = decoder(encoded_imgs)


fig1 = plt.figure(figsize=(18, 10))

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
Lx = (40) * grid_size[1]   # length of model in selected units
Ly = (20) * grid_size[0]   # length of model in selected units
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
# print(inp_img)
# print(latent_img)
# print(dec_img)
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

import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
import os
import itertools
import matplotlib.pyplot as plt
import CAAE3D

from CAAE3D import Encoder, Decoder
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import savetxt

# ----------------
#  Dataset Object
# ----------------
class InputDataset(Dataset):

    def __init__(self, transform=None):
        #initial conditions
        label_list = []
        input_list = []

        perm_field_dir = os.path.join('.', 'perm3D')
        workdir = os.path.join('.', 'bt3D')

        nlay = 20 # number of layers
        nrow = 20 # number of rows / grid cells
        ncol = 40 # number of columns (parallel to axis of core)

        index_1 = [x for x in range(30000,30500,500)]
        invalid = False

        numb = index_1.__getitem__(-1)
        input_list = [[] for p in range(numb-10000)]

        q = 0   # index of the elements in the input list
        for i in index_1:
            index_2 = [y for y in range(228,229)] #indeces for every .csv file

            for j in index_2:
                model_lab_filename_sp = perm_field_dir + str(i) + '/core_k_3d_m2_' + str(i-j) + '.csv'
                model_inp_filename_sp = workdir + str(i) +'/norm_td' + str(i-j) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'

                if i > 10000:
                    model_lab_filename_sp = perm_field_dir + str(i) + '/core_k_3d_m2_' + str(i-10000-j) + '.csv'
                    model_inp_filename_sp = workdir + str(i) + '/norm_td' + str(i-10000-j) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                            + str(ncol) +'.csv'

                    if i-10000-j == 4476:
                        continue
                    tdata_ex = np.loadtxt(model_inp_filename_sp, delimiter=',', dtype= np.float32)
                    pdata_ex = np.loadtxt(model_lab_filename_sp, delimiter=',', dtype= np.float32)

                    pressure = tdata_ex[-2]
                    tdata_ex = tdata_ex[0:-2]
                    for p in range(len(tdata_ex[0:-2])):
                        if tdata_ex[p] == 0:
                            continue
                        elif tdata_ex[p] < 0:
                            tdata_ex[p] = -tdata_ex[p]
                            tdata_ex[p] = np.log(tdata_ex[p])
                            tdata_ex[p] = -(tdata_ex[p]**2)
                        else:
                            tdata_ex[p] = np.log(tdata_ex[p])
                            tdata_ex[p] = (tdata_ex[p]**2)

                    pdata_ex = pdata_ex[0:-3]/np.float64(9.8692e-16)
                    for g in range(len(pdata_ex[0:-3])):
                        if pdata_ex[g] == 0:
                            continue
                        elif pdata_ex[g] < 0:
                            pdata_ex[g] = -pdata_ex[g]
                            pdata_ex[g] = np.log(pdata_ex[g])
                            pdata_ex[g] = -(pdata_ex[g]**2)
                        else:
                            pdata_ex[g] = np.log(pdata_ex[g])
                            if pdata_ex[g] < 0:
                                #pdata_ex[g] = -(pdata_ex[g]**2)
                            #else:
                            pdata_ex[g] = (pdata_ex[g]**2)

                    tdata_ex = torch.from_numpy(tdata_ex)
                    pdata_ex = torch.from_numpy(pdata_ex)

                    for w in range(0,16000):
                        if torch.isnan(tdata_ex[w]) or not torch.is_floating_point(tdata_ex[w]) or torch.isnan(pdata_ex[w]) or not torch.is_floating_point(pdata_ex[w]):
                            print(str(i-10000-j))
                            invalid = True
                            break
                    if invalid == True:
                        invalid = False
                        continue

                    tdata_ex = tdata_ex.reshape(1,20,20,40)
                    pdata_ex = pdata_ex.reshape(1,20,20,40)
                    Pad = nn.ConstantPad3d((1,0,0,0,0,0), pressure)
                    tdata_ex = Pad(tdata_ex)
                    input_list[q].append(tdata_ex)
                    input_list[q].append(pdata_ex)
                    q=q+1


        self.input = input_list
        self.nrow = nrow
        self.ncol = ncol
        self.nlay = nlay
        self.transform = transform

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
dataset_input = dataset.input[0:1]
dataloader_input = DataLoader(dataset=dataset_input, batch_size=1, shuffle=True, num_workers=2)

nf, h, w = 1, 10, 20
Tensor = torch.FloatTensor

encoder = Encoder(outchannels=nf)
decoder = Decoder(inchannels=nf)

encoder.load_state_dict(torch.load('./encoder_epoch100.pth', map_location='cpu'))
encoder.eval()

decoder.load_state_dict(torch.load('./decoder_epoch100.pth', map_location='cpu'))
decoder.eval()

for i, (imgs_inp) in enumerate(dataloader_input):
    for j, (image) in enumerate(imgs_inp):
        # Get inputs and targets
        if j == 0:
            imgs_inp_N = Variable(image.type(Tensor))
        else:
            target = Variable(image.type(Tensor))

    encoded_imgs = encoder(imgs_inp_N)
    decoded_imgs = decoder(encoded_imgs)

inp_img = target[0][0]
dec_img = decoded_imgs[0][0]
latent_img = encoded_imgs[0][0]

inp_img = inp_img.cpu().detach().numpy()
latent_img = latent_img.cpu().detach().numpy()
dec_img = dec_img.cpu().detach().numpy()

inp_img = inp_img.flatten()
dec_img = dec_img.flatten()
latent_img = latent_img.flatten()



for p in range(len(inp_img)):
    if inp_img[p] == 0:
        continue
    elif inp_img[p] < 0:
        inp_img[p] = -inp_img[p]
        inp_img[p] = np.sqrt(inp_img[p])
        inp_img[p] = -inp_img[p]
        inp_img[p] = np.exp(inp_img[p])
    else:
        inp_img[p] = np.sqrt(inp_img[p])
        inp_img[p] = np.exp(inp_img[p])

for g in range(len(dec_img)):
    if abs(dec_img[g]) == 0:
        dec_img[g] = 0
    elif dec_img[g] < 0:
        dec_img[g] = -dec_img[g]
        dec_img[g] = np.sqrt(dec_img[g])
        dec_img[g] = -dec_img[g]
        dec_img[g] = np.exp(dec_img[g])
    else:
        dec_img[g] = np.sqrt(dec_img[g])
        dec_img[g] = np.exp(dec_img[g])

dec_img = dec_img*np.float64(9.8692e-16);
inp_img = inp_img*np.float64(9.8692e-16);
inp_img = inp_img.reshape(20,20,40)
dec_img = dec_img.reshape(20,20,40)

# =============================================================================
# PLOT DATA
# =============================================================================
# layer to plot
ilayer = 0
# fontsize
fs = 13
hfont = {'fontname':'Arial'}


# Grid cell size
grid_size = [0.25, 0.25, 0.25] # selected units [cm]

# Define grid
# Describe grid for results
Lx = dataset.ncol * grid_size[2]   # length of model in selected units
Ly = dataset.nrow * grid_size[1]   # length of model in selected units
y, x = np.mgrid[slice(0, Ly + grid_size[1], grid_size[1]),
                 slice(0, Lx + grid_size[2], grid_size[2])]


# First figure with head and breakthrough time difference maps
fig1 = plt.figure(figsize=(18, 10))
ax2 = fig1.add_subplot(2, 2, 1, aspect='equal')
imp = plt.pcolor(x, y, inp_img[round(dataset.nlay/2),:,:], cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Pore Volumes', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2))
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Permeability Layer 10', fontsize=fs+2, **hfont)
plt.clim(np.min(inp_img[round(dataset.nlay/2),:,:]), np.max(inp_img[round(dataset.nlay/2),:,:]))

ax2 = fig1.add_subplot(2, 2, 2, aspect='equal')
imp = plt.pcolor(x, y, dec_img[round(dataset.nlay/2),:,:], cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Pore Volumes', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2))
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Decoded Permeability Layer 10', fontsize=fs+2, **hfont)
plt.clim(np.min(dec_img[round(dataset.nlay/2),:,:]), np.max(dec_img[round(dataset.nlay/2),:,:]))

ax2 = fig1.add_subplot(2, 2, 3, aspect='equal')
imp = plt.pcolor(x, y, inp_img[:,round(dataset.nrow/2),:], cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Pore Volumes', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2))
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Permeability Row 10', fontsize=fs+2, **hfont)
plt.clim(np.min(inp_img[:,round(dataset.nrow/2),:]), np.max(inp_img[:,round(dataset.nrow/2),:]))

ax2 = fig1.add_subplot(2, 2, 4, aspect='equal')
imp = plt.pcolor(x, y, dec_img[:,round(dataset.nrow/2),:], cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Pore Volumes', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2))
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Decoded Permeability Row 10', fontsize=fs+2, **hfont)
plt.clim(np.min(dec_img[:,round(dataset.nrow/2),:]), np.max(dec_img[:,round(dataset.nrow/2),:]))

plt.show()

inp_img = inp_img.flatten()
dec_img = dec_img.flatten()
latent_img = latent_img.flatten()

savetxt('./results/inp.csv', inp_img, delimiter=',')
savetxt('./results/dec.csv', dec_img, delimiter=',')
savetxt('./results/lat.csv', latent_img, delimiter=',')

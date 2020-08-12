import torch
import torchvision
import numpy as np
import math
import os
import CAAE
import itertools
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.nn import functional as F
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
        td = [x for x in range(1,10000)]
        label_list = []
        input_list = []
        # number of layers
        nlay = 1

        perm_field_dir = os.path.join('.', 'matlab_perm_fields/k_training_data')
        workdir = os.path.join('.', 'Tdata_2D')
        # training data iteration
        for i in td:
            #use existing permeability maps as labels
            tdata_km2 = np.loadtxt(perm_field_dir + '/tdr_km2_' + str(i) +'.csv', delimiter=',', dtype=np.float32)

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
dataset_input = dataset[0:10000]
dataloader_input = DataLoader(dataset=dataset_input, batch_size=50, shuffle=True, num_workers=2)


# --------------------------------------------------------
#  Initializing Parameters and Key Components of the Model
# --------------------------------------------------------
cuda = True if torch.cuda.is_available() else False

# latent dimension = nf*h*w
nf, h, w = 1, 10, 20
Tensor = torch.FloatTensor
nepochs = 50

# list storing generator's loss and pixel-wise loss
g_l = []
g_lc = []
g_lk_row = []
g_lk_col = []

# loss functions
pixelwise_loss = torch.nn.L1Loss()
kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

# initialize generator and discriminator
encoder = Encoder(outchannels=nf)
decoder = Decoder(inchannels=nf)

# optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=0.001, betas=(0.5, 0.999))

# learning rate decay
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma = 0.95)

print("number of parameters: {}".format(encoder._n_parameters()+decoder._n_parameters()))

if cuda:
    encoder.cuda()
    decoder.cuda()
    kl_loss.cuda()
    pixelwise_loss.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------
for epoch in range(1,nepochs+1):
    encoder.train()
    decoder.train()

    for i, (imgs_inp) in enumerate(dataloader_input):

        # Configure input
        input_imgs = Variable(imgs_inp.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs = encoder(input_imgs)
        decoded_imgs = decoder(encoded_imgs)

        #using softmax on each column of both the input and decoded image to generate their distribution
        #for KL divergence loss calculation
        dec_soft_col = F.log_softmax(decoded_imgs, dim=-2)
        inp_soft_col = F.softmax(input_imgs, dim=-2)

        #using softmax on each row of both the input and decoded image to generate their distribution
        #for KL divergence loss calculation
        dec_soft_row = F.log_softmax(decoded_imgs, dim=-1)
        inp_soft_row = F.softmax(input_imgs, dim=-1)

        g_loss_c = pixelwise_loss(decoded_imgs, input_imgs)
        g_loss_k_row = kl_loss(dec_soft_row, inp_soft_row)
        g_loss_k_col = kl_loss(dec_soft_col, inp_soft_col)
        g_loss = 0.99*g_loss_c + 0.005*g_loss_k_row + 0.005*g_loss_k_col

        g_loss.backward()
        optimizer_G.step()

    print(
        "[Epoch %d/%d] [Batch %d/%d] [G loss: %f / G_C loss: %f / G_k loss row: %f / G_k loss col: %f]"
        % (epoch, nepochs, i+1, len(dataloader_input), g_loss.item(), g_loss_c.item(), g_loss_k_row.item(),
           g_loss_k_col.item())
    )

    g_l.append(g_loss.item())
    g_lc.append(g_loss_c.item())
    g_lk_row.append(g_loss_k_row.item())
    g_lk_col.append(g_loss_k_col.item())

    scheduler.step()


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

print(g_l)
print(g_lc)
print(g_lk_row)
print(g_lk_col)

inp_img = imgs_inp[0][0]
latent_img = encoded_imgs[0][0]
dec_img = decoded_imgs[0][0]
print(inp_img)
print(latent_img)
print(dec_img)

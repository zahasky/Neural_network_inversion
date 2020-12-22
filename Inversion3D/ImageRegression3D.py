import torch
import torchvision
import torch.nn as nn
import numpy as np
import math
import os
import CAAE3D2
import itertools
import random

from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.utils.data import Dataset, DataLoader
from CAAE3D2 import Encoder, Decoder
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
   
        index_1 = [x for x in range(10500,30000,500)]
        invalid = False

        numb = index_1.__getitem__(-1)
        input_list = [[] for p in range(numb-10000)]
        
        q = 0   # index of the elements in the input list 
        for i in index_1:
            index_2 = [y for y in range(500)] #indeces for every .csv file

            for j in index_2:
                model_lab_filename_sp = perm_field_dir + str(i) + '/core_k_3d_m2_' + str(i-j) + '.csv'
                model_inp_filename_sp = workdir + str(i) +'/norm_td' + str(i-j) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'

                if i > 10000:
                    model_lab_filename_sp = perm_field_dir + str(i) + '/core_k_3d_m2_' + str(i-10000-j) + '.csv'
                    model_inp_filename_sp = workdir + str(i) + '/norm_td' + str(i-10000-j) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' \
                                            + str(ncol) +'.csv' 

                try:
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
                                pdata_ex[g] = -(pdata_ex[g]**2)
                            else:
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
                except:
                    print(str(i-10000-j))
                    continue

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
train_dataset = dataset.input[0:16000]
validation_dataset = dataset.input[16000:18500]

train_dataloader = DataLoader(dataset=train_dataset, batch_size=75, shuffle=True, num_workers=2)
validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=25, shuffle=True, num_workers=2)


# --------------------------------------------------------
#  Initializing Parameters and Key Components of the Model
# --------------------------------------------------------
cuda = True if torch.cuda.is_available() else False

# latent dimension = nf*h*w
nf, h, w = 1, 10, 20
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
nepochs = 105

# lists storing training loss
g_l = []

# lists storing validation loss
g_lv = []

# loss functions
pixelwise_loss = torch.nn.L1Loss()
mse_loss = torch.nn.MSELoss()
kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

# initialize generator and discriminator
encoder = Encoder(outchannels=nf)
decoder = Decoder(inchannels=nf)

if cuda:
    encoder.cuda()
    decoder.cuda()
    pixelwise_loss.cuda()
    mse_loss.cuda()
    kl_loss.cuda()

# optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=0.0005, betas=(0.5, 0.999))

# scheduler
#scheduler = ReduceLROnPlateau(
#                    optimizer_G, mode='min', factor=0.1, patience=10,
#                    verbose=True, threshold=0.0001, threshold_mode='rel',
#                    cooldown=0, min_lr=0, eps=1e-08)

print("number of parameters: {}".format(encoder._n_parameters()+decoder._n_parameters()))
 
#mse = 0
#noise = []

for epoch in range(1,nepochs+1):
    encoder.train()
    decoder.train()

# ----------
#  Training
# ----------
    for i, (imgs_inp) in enumerate(train_dataloader):
        for j, (image) in enumerate(imgs_inp):
            # Get inputs and targets
            if j == 0:
               imgs_inp_N = Variable(image.type(Tensor))
            else:
                target = Variable(image.type(Tensor))
       
        optimizer_G.zero_grad()

        encoded_imgs = encoder(imgs_inp_N)
        decoded_imgs = decoder(encoded_imgs)
 
        dec_soft_lay = F.log_softmax(decoded_imgs, dim=-3)
        inp_soft_lay = F.softmax(target, dim=-3)

        #using softmax on each column of both the input and decoded image to generate their distribution
        #for KL divergence loss calculation
        dec_soft_col = F.log_softmax(decoded_imgs, dim=-2)
        inp_soft_col = F.softmax(target, dim=-2)

        #using softmax on each row of both the input and decoded image to generate their distribution
        #for KL divergence loss calculation
        dec_soft_row = F.log_softmax(decoded_imgs, dim=-1)
        inp_soft_row = F.softmax(target, dim=-1)

        #g_loss_p = pixelwise_loss(decoded_imgs, target)
        g_loss_k_row = kl_loss(dec_soft_row, inp_soft_row)
        g_loss_k_col = kl_loss(dec_soft_col, inp_soft_col)
        g_loss_k_lay = kl_loss(dec_soft_lay, inp_soft_lay)
        g_loss_mse = mse_loss(decoded_imgs, target)
        #g_loss_rmse = torch.sqrt(g_loss_mse)
        g_loss = 0.7*g_loss_mse + 0.1*g_loss_k_row + 0.1*g_loss_k_col + 0.1*g_loss_k_lay


        g_loss.backward()
        optimizer_G.step()
        
        #mse += g_loss_mse.item()

    #rmse = np.sqrt(mse / 16000)
    #scheduler.step(rmse)

    print(
        "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
        % (epoch, nepochs, i+1, len(train_dataloader), g_loss.item())
    )

    g_l.append(g_loss.item())

    encoder.eval()
    decoder.eval()
# -----------
#  Validation
# -----------
    for i, (imgs_v) in enumerate(validation_dataloader):
        for j, (imagev) in enumerate(imgs_v):
            # Get inputs and targets
            if j == 0:
               imgs_inp_Nv = Variable(imagev.type(Tensor))
            else:
                targetv = Variable(imagev.type(Tensor))

        with torch.no_grad():

            encoded_imgs_v = encoder(imgs_inp_Nv)
            decoded_imgs_v = decoder(encoded_imgs_v)

            dec_soft_lay_v = F.log_softmax(decoded_imgs_v, dim=-3)
            inp_soft_lay_v = F.softmax(targetv, dim=-3)

            #using softmax on each column of both the input and decoded image to generate their distribution
            #for KL divergence loss calculation
            dec_soft_col_v = F.log_softmax(decoded_imgs_v, dim=-2)
            inp_soft_col_v = F.softmax(targetv, dim=-2)

            #using softmax on each row of both the input and decoded image to generate their distribution
            #for KL divergence loss calculation
            dec_soft_row_v = F.log_softmax(decoded_imgs_v, dim=-1)
            inp_soft_row_v = F.softmax(targetv, dim=-1)

            g_loss_p_v = pixelwise_loss(decoded_imgs_v, targetv)
            g_loss_k_row_v = kl_loss(dec_soft_row_v, inp_soft_row_v)
            g_loss_k_col_v = kl_loss(dec_soft_col_v, inp_soft_col_v)
            g_loss_k_lay_v = kl_loss(dec_soft_lay_v, inp_soft_lay_v)
            g_loss_mse_v = mse_loss(decoded_imgs_v, targetv)
            #g_loss_rmse_v = torch.sqrt(g_loss_mse_v)
            g_loss_v = 0.7*g_loss_mse_v + 0.1*g_loss_k_row_v + 0.1*g_loss_k_col_v + 0.1*g_loss_k_lay_v

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
target = target[0][0]
inp_img = imgs_inp_N[0][0]
dec_img = decoded_imgs[0][0]
latent_img = encoded_imgs[0][0]

target = target.flatten()
inp_img = inp_img.flatten()
dec_img = dec_img.flatten()
latent_img = latent_img.flatten()

target = target.cpu().detach().numpy()
inp_img = inp_img.cpu().detach().numpy()
latent_img = latent_img.cpu().detach().numpy()
dec_img = dec_img.cpu().detach().numpy()

#save sample images
savetxt('./results/target.csv', target, delimiter=',')
savetxt('./results/inp.csv', inp_img, delimiter=',')
savetxt('./results/dec.csv', dec_img, delimiter=',')
savetxt('./results/lat.csv', latent_img, delimiter=',')


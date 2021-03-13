import torch
import torchvision
import torch.nn as nn
import numpy as np
import math
import os
import PIX2PIX3D
import itertools
import random

from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.utils.data import Dataset, DataLoader
from PIX2PIX3D import Encoder, Decoder, Discriminator
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

                    gaussian_arr = np.random.normal(0,0.08,len(tdata_ex))
                    for w in range(len(tdata_ex)):
                        num = np.random.randint(15999,size=len(tdata_ex))
                        random_arr = tdata_ex[num]

                    noise = np.multiply(random_arr,gaussian_arr)

                    for p in range(len(tdata_ex)):
                        if tdata_ex[p] == 0:
                            continue

                        tdata_ex[p] = tdata_ex[p] + noise[p]
                        tdata_ex[p] = np.sign(tdata_ex[p])*np.log(np.abs(tdata_ex[p]))


                    pdata_ex = pdata_ex[0:-3]/np.float64(9.8692e-16)
                    for g in range(len(pdata_ex)):
                        if pdata_ex[g] == 0:
                            continue
                        elif pdata_ex[g] < 0:
                            print("Warning: Negative Permeability at " + str(i-10000-j))

                        pdata_ex[g] = np.sign(pdata_ex[g])*np.log(np.abs(pdata_ex[g]))

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

train_dataloader = DataLoader(dataset=train_dataset, batch_size=85, shuffle=True, num_workers=2)
validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=85, shuffle=True, num_workers=2)


# --------------------------------------------------------
#  Initializing Parameters and Key Components of the Model
# --------------------------------------------------------
torch.set_default_dtype(torch.float32)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
nepochs = 199

# lists storing training loss
g_l = []
d_l = []

# lists storing validation loss
g_lv = []
d_lv = []

# loss functions
pixelwise_loss = torch.nn.L1Loss()
mse_loss = torch.nn.MSELoss()
kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
adversarial_loss = torch.nn.BCELoss()

# initialize generator and discriminator
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    pixelwise_loss.cuda()
    mse_loss.cuda()
    kl_loss.cuda()

# optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

print("number of parameters: {}".format(encoder._n_parameters()+decoder._n_parameters()+discriminator._n_parameters()))

for epoch in range(1,nepochs+1):
    encoder.train()
    decoder.train()
    discriminator.train()

# ----------
#  Training
# ----------
    for i, (imgs_inp) in enumerate(train_dataloader):
        for j, (image) in enumerate(imgs_inp):
            # Get inputs and targets
            if j == 0:
                # The real input image with pressure
                true_inp_P = Variable(image.type(Tensor))
                img = np.delete(image,0,axis=4)
                # The real input image without pressure
                true_inp = Variable(img.type(Tensor))
            else:
                true_target = Variable(image.type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(true_target.shape[0],1).fill_(1.0), requires_grad=False)
        fake  = Variable(Tensor(true_target.shape[0],1).fill_(0.0), requires_grad=False)

        lat_target = encoder(true_inp_P)
        fake_target = decoder(lat_target)

        fake_result = discriminator(fake_target)
        real_result = discriminator(true_target)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        #using softmax on each layer of both the input and decoded image to generate their distribution
        #for KL divergence loss calculation
        dec_soft_lay = F.log_softmax(fake_target, dim=-3)
        inp_soft_lay = F.softmax(true_target, dim=-3)

        #using softmax on each column of both the input and decoded image to generate their distribution
        #for KL divergence loss calculation
        dec_soft_col = F.log_softmax(fake_target, dim=-2)
        inp_soft_col = F.softmax(true_target, dim=-2)

        #using softmax on each row of both the input and decoded image to generate their distribution
        #for KL divergence loss calculation
        dec_soft_row = F.log_softmax(fake_target, dim=-1)
        inp_soft_row = F.softmax(true_target, dim=-1)

        g_loss_k_row = kl_loss(dec_soft_row, inp_soft_row)
        g_loss_k_col = kl_loss(dec_soft_col, inp_soft_col)
        g_loss_k_lay = kl_loss(dec_soft_lay, inp_soft_lay)
        g_loss_adv = adversarial_loss(fake_result, valid)
        g_loss_mse = mse_loss(fake_target, true_target)
        # g_loss_k = kl_loss(fake_target, true_target)
        # print(g_loss_k.item())

        KL_loss = 0.15*g_loss_k_row + 0.15*g_loss_k_col + 0.15*g_loss_k_lay
        g_loss = 0.53*g_loss_mse + KL_loss + 0.02*g_loss_adv

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Loss for real images
        d_real_loss = adversarial_loss(real_result, valid)
        # Loss for fake images
        d_fake_loss = adversarial_loss(fake_result.detach(), fake)
        # Total adversarial loss
        d_loss = 0.5*d_real_loss + 0.5*d_fake_loss

        d_loss.backward()
        optimizer_D.step()

    print(
        "[Epoch %d/%d] [Batch %d/%d] [G loss: %f/ G_mse loss: %f/ G_KL loss: %f/ G_Adv loss: %f]"
        % (epoch, nepochs, i+1, len(train_dataloader), g_loss.item(), g_loss_mse.item(), KL_loss.item(), g_loss_adv.item())
    )

    g_l.append(g_loss.item())
    d_l.append(d_loss.item())

    encoder.eval()
    decoder.eval()
    discriminator.eval()

# -----------
#  Validation
# -----------
    for i, (imgs_v) in enumerate(validation_dataloader):
        for j, (imagev) in enumerate(imgs_v):
            # Get inputs and targets
            if j == 0:
                # The real input image with pressure
                true_inp_Pv = Variable(imagev.type(Tensor))
                imgv = np.delete(imagev,0,axis=4)
                # The real input image without pressure
                true_inpv = Variable(imgv.type(Tensor))
            else:
                true_targetv = Variable(imagev.type(Tensor))

        # Adversarial ground truths
        validv = Variable(Tensor(true_targetv.shape[0],1).fill_(1.0), requires_grad=False)
        fakev  = Variable(Tensor(true_targetv.shape[0],1).fill_(0.0), requires_grad=False)

        with torch.no_grad():
            lat_targetv = encoder(true_inp_Pv)
            fake_targetv = decoder(lat_targetv)

            fake_resultv = discriminator(fake_targetv)
            real_resultv = discriminator(true_targetv)

            # ---------------------
            #  Validate Generator
            # ---------------------
            #using softmax on each layer of both the input and decoded image to generate their distribution
            #for KL divergence loss calculation
            dec_soft_layv = F.log_softmax(fake_targetv, dim=-3)
            inp_soft_layv = F.softmax(true_targetv, dim=-3)

            #using softmax on each column of both the input and decoded image to generate their distribution
            #for KL divergence loss calculation
            dec_soft_colv = F.log_softmax(fake_targetv, dim=-2)
            inp_soft_colv = F.softmax(true_targetv, dim=-2)

            #using softmax on each row of both the input and decoded image to generate their distribution
            #for KL divergence loss calculation
            dec_soft_rowv = F.log_softmax(fake_targetv, dim=-1)
            inp_soft_rowv = F.softmax(true_targetv, dim=-1)

            g_loss_k_rowv = kl_loss(dec_soft_rowv, inp_soft_rowv)
            g_loss_k_colv = kl_loss(dec_soft_colv, inp_soft_colv)
            g_loss_k_layv = kl_loss(dec_soft_layv, inp_soft_layv)
            g_loss_advv = adversarial_loss(fake_resultv, validv)
            g_loss_msev = mse_loss(fake_targetv, true_targetv)

            KL_lossv = 0.15*g_loss_k_rowv + 0.15*g_loss_k_colv + 0.15*g_loss_k_layv
            g_lossv = 0.53*g_loss_msev +  + 0.02*g_loss_advv

            # -----------------------
            #  Validate Discriminator
            # -----------------------

            # Loss for real images
            d_real_lossv = adversarial_loss(real_resultv, validv)
            # Loss for fake images
            d_fake_lossv = adversarial_loss(fake_resultv.detach(), fakev)
            # Total adversarial loss
            d_lossv = 0.5*d_real_lossv + 0.5*d_fake_lossv

    print(
        "[Epoch %d/%d] [Batch %d/%d] [G lossv: %f/ G_mse lossv: %f/ G_KL lossv: %f/ G_Advv loss: %f]"
        % (epoch, nepochs, i+1, len(train_dataloader), g_lossv.item(), g_loss_msev.item(), KL_lossv.item(), g_loss_advv.item())
    )

    g_lv.append(g_lossv.item())
    d_lv.append(d_lossv.item())

    # Stop the train
    if g_lossv.item() < 1 and epoch > 50:
        break

print(g_l)
print(d_l)
print(g_lv)
print(d_lv)

# -------------------------
#  Storing Training Results
# -------------------------
result_dir = os.path.join('.', 'results')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
enc_dir = result_dir + '/encoder_epoch{}.pth'.format(nepochs)
dec_dir = result_dir + '/decoder_epoch{}.pth'.format(nepochs)

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

#save sample images
savetxt('./results/target.csv', target, delimiter=',')
savetxt('./results/inp.csv', inp_img, delimiter=',')
savetxt('./results/dec.csv', dec_img, delimiter=',')

# prerequisites
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from torchvision.utils import save_image
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import imageio
import os
from torch.utils.data import DataLoader, Subset

from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
from joblib import dump, load
import copy

#torch.set_default_dtype(torch.float64)

# load a saved vae checkpoint
def load_checkpoint(filepath, vae_type, d=0):
    vae, z = vae_builder(vae_type=vae_type)
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{d}')
        torch.cuda.set_device(d)
    else:
        device = 'cpu'
    
    torch_version = torch.__version__
    if torch_version == '2.4.0':
        checkpoint = torch.load(filepath, device, weights_only = True)
    else:
        checkpoint = torch.load(filepath, device)
    vae.load_state_dict(checkpoint['state_dict'])
    vae.to(device)
    return vae

global colorlabels
numcolors = 0

colornames = ["red", "blue", "green", "purple", "yellow", "cyan", "orange", "brown", "pink", "white"]
colorlabels = np.random.randint(0, 10, 1000000)
colorrange = .1
colorvals = [
    [1 - colorrange, colorrange * 1, colorrange * 1],
    [colorrange * 1, 1 - colorrange, colorrange * 1],
    [colorrange * 2, colorrange * 2, 1 - colorrange],
    [1 - colorrange * 2, colorrange * 2, 1 - colorrange * 2],
    [1 - colorrange, 1 - colorrange, colorrange * 2],
    [colorrange, 1 - colorrange, 1 - colorrange],
    [1 - colorrange, .5, colorrange * 2],
    [.6, .4, .2],
    [1 - colorrange, 1 - colorrange * 3, 1 - colorrange * 3],
    [1-colorrange,1-colorrange,1-colorrange]
]


#comment this
def Colorize_func(img):
    global numcolors,colorlabels

    thiscolor = colorlabels[numcolors]  # what base color is this?

    rgb = colorvals[thiscolor];  # grab the rgb for this base color
    numcolors += 1  # increment the index

    r_color = rgb[0] + np.random.uniform() * colorrange * 2 - colorrange  # generate a color randomly in the neighborhood of the base color
    g_color = rgb[1] + np.random.uniform() * colorrange * 2 - colorrange
    b_color = rgb[2] + np.random.uniform() * colorrange * 2 - colorrange
    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img * r_color, np_img * g_color, np_img * b_color])
    backup = np_img
    np_img = np_img.astype(np.uint8)
    img = Image.fromarray(np_img, 'RGB')
    return img

#comment
def Colorize_func_specific(col,img):
    # col: an int index for which base color is being used
    rgb = colorvals[col]  # grab the rgb for this base color
    r_color = rgb[0] + np.random.uniform() * colorrange * 2 - colorrange  # generate a color randomly in the neighborhood of the base color
    g_color = rgb[1] + np.random.uniform() * colorrange * 2 - colorrange
    b_color = rgb[2] + np.random.uniform() * colorrange * 2 - colorrange
    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img * r_color, np_img * g_color, np_img * b_color])
    backup = np_img
    np_img = np_img.astype(np.uint8)
    img = Image.fromarray(np_img, 'RGB')

    return img

# model training data set and dimensions
data_set_flag = 'red_green_mnist'
imgsize = 28
vae_type_flag = 'CNN' # must be CNN or FC
h_dim1 = 256
h_dim2 = 128
z_dim = 8
x_dim = 3*28*28


# FC VAE

class VAE_FC(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, l_dim, sc_dim):
        super(VAE_FC, self).__init__()
        # encoder part
        self.l_dim = l_dim
        self.z_dim = z_dim

        #encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)  # shape
        self.fc32 = nn.Linear(h_dim2, z_dim)
        self.fc33 = nn.Linear(h_dim2, z_dim)  # color
        self.fc34 = nn.Linear(h_dim2, z_dim)
        
        #decoder part
        self.fc4s = nn.Linear(z_dim, h_dim2)  # shape
        self.fc4c = nn.Linear(z_dim, h_dim2)  # color
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

        # map scalars
        self.shape_scale = 1 #1.9
        self.color_scale = 1 #2

        self.relu = nn.ReLU()

    def encoder(self, x):
        x=x.view(x.size(0),-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        return self.fc31(h), self.fc32(h), self.fc33(h), self.fc34(h) # mu, log_var

    def activations(self, x):
        x = x.cuda()
        mu_shape, log_var_shape, mu_color, log_var_color = self.encoder(x)
        
        z_shape = self.sampling(mu_shape, log_var_shape)
        z_color = self.sampling(mu_color, log_var_color)

        return z_shape, z_color

    def sampling(self, mu, log_var, noise=1):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) * noise
        return mu + eps * std

    def decoder_color(self, z_color):
        h = F.relu(self.fc4c(z_color))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h)).view(-1, 3, imgsize, imgsize)
    
    #decodes from the shape map
    def decoder_shape(self, z_shape):
        h = F.relu(self.fc4s(z_shape))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h)).view(-1, 3, imgsize, imgsize)
    
    #decodes from shape and color maps
    def decoder_cropped(self, z_shape, z_color):
        h = F.relu(self.fc4c(z_color)) + F.relu(self.fc4s(z_shape))
        h = (F.relu(self.fc5(h)))
        return torch.sigmoid(self.fc6(h)).view(-1, 3, imgsize, imgsize)

    def forward(self, x, whichdecode='cropped', keepgrad=[]):
        x = x.cuda()
        mu_shape, log_var_shape, mu_color, log_var_color = self.encoder(x)

        #what maps are used in the training process.. the others are detached to zero out those gradients
        if ('shape' in keepgrad):
            z_shape = self.sampling(mu_shape, log_var_shape)
        else:
            z_shape = self.sampling(mu_shape, log_var_shape).detach()

        if ('color' in keepgrad):
            z_color = self.sampling(mu_color, log_var_color)
        else:
            z_color = self.sampling(mu_color, log_var_color).detach()

        if(whichdecode == 'cropped'):
            output = self.decoder_cropped(z_shape, z_color)
        elif (whichdecode == 'color'):
            output = self.decoder_color(z_color)
        elif (whichdecode == 'shape'):
            output = self.decoder_shape(z_shape)

        return output, mu_color, log_var_color, mu_shape, log_var_shape

# CNN VAE

class VAE_CNN(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, l_dim, sc_dim):
        super(VAE_CNN, self).__init__()
        # encoder part
        self.l_dim = l_dim
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc2 = nn.Linear(int(imgsize / 4) * int(imgsize / 4)*16, h_dim2) #
        self.fc_bn2 = nn.BatchNorm1d(h_dim2) # remove
        # bottle neck part  # Latent vectors mu and sigma
        self.fc31 = nn.Linear(h_dim2, z_dim)  # shape
        self.fc32 = nn.Linear(h_dim2, z_dim)
        self.fc33 = nn.Linear(h_dim2, z_dim)  # color
        self.fc34 = nn.Linear(h_dim2, z_dim)

        # decoder part
        self.fc4s = nn.Linear(z_dim, h_dim2)  # shape
        self.fc4c = nn.Linear(z_dim, h_dim2)  # color

        self.fc5 = nn.Linear(h_dim2, int(imgsize/4) * int(imgsize/4) * 16)

        self.conv5 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(3)

        # map scalars
        self.shape_scale = 1 #1.9
        self.color_scale = 1 #2

        self.relu = nn.ReLU()

    def encoder(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.relu(self.bn2(self.conv2(h)))        
        h = self.relu(self.bn3(self.conv3(h)))
        h = self.relu(self.bn4(self.conv4(h)))
        h = h.view(-1,int(imgsize / 4) * int(imgsize / 4)*16)
        h = self.relu(self.fc_bn2(self.fc2(h)))

        return self.fc31(h), self.fc32(h), self.fc33(h), self.fc34(h) # mu, log_var

    def activations(self, x):
        x = x.cuda()
        mu_shape, log_var_shape, mu_color, log_var_color = self.encoder(x)
        
        z_shape = self.sampling(mu_shape, log_var_shape)
        z_color = self.sampling(mu_color, log_var_color)

        return z_shape, z_color


    def sampling(self, mu, log_var, noise = 1):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        #if self.training:
        #    eps = eps * 5
        return mu + eps * std

    def decoder_color(self, z_color):
        h = F.relu(self.fc4c(z_color)) * self.color_scale
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def decoder_shape(self, z_shape):
        h = F.relu(self.fc4s(z_shape)) * self.shape_scale
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def decoder_cropped(self, z_shape, z_color):
        #print('crop',z_shape.size())
        h = (F.relu(self.fc4c(z_color)) * self.color_scale) + (F.relu(self.fc4s(z_shape)) * self.shape_scale)
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def forward(self, x, whichdecode='cropped', keepgrad=[]):
        x = x.cuda()
        mu_shape, log_var_shape, mu_color, log_var_color = self.encoder(x)

        #what maps are used in the training process.. the others are detached to zero out those gradients
        if ('shape' in keepgrad):
            z_shape = self.sampling(mu_shape, log_var_shape)
        else:
            z_shape = self.sampling(mu_shape, log_var_shape).detach()

        if ('color' in keepgrad):
            z_color = self.sampling(mu_color, log_var_color)
        else:
            z_color = self.sampling(mu_color, log_var_color).detach()


        if(whichdecode == 'cropped'):
            output = self.decoder_cropped(z_shape, z_color)
        elif (whichdecode == 'color'):
            output = self.decoder_color(z_color)
        elif (whichdecode == 'shape'):
            output = self.decoder_shape(z_shape)

        return output, mu_color, log_var_color, mu_shape, log_var_shape

# function to build an  actual model instance
# function to build a model instance
def vae_builder(vae_type = vae_type_flag, x_dim = x_dim, h_dim1 = h_dim1, h_dim2 = h_dim2, z_dim = z_dim, l_dim = 0, sc_dim = 0):
    if vae_type == 'CNN':
        vae = VAE_CNN(x_dim, h_dim1, h_dim2, z_dim, l_dim, sc_dim)
    else:
        vae = VAE_FC(x_dim, h_dim1, h_dim2, z_dim, l_dim, sc_dim) 

    return vae, z_dim

######the loss functions
#pixelwise loss for just the cropped image
def loss_function_crop(recon_x, x, mu, log_var, mu_c, log_var_c):
    beta = 0.7
    x = x.clone().cuda()
    BCE = F.binary_cross_entropy(recon_x.view(-1, imgsize * imgsize * 3), x.view(-1, imgsize * imgsize * 3), reduction='sum')
    return BCE

# loss for shape in a cropped image
def loss_function_shape(recon_x, x, mu, log_var):
    beta = 1
    x = x.clone().cuda()
    # make grayscale reconstruction
    gray_x = x.view(-1, 3, imgsize, imgsize).mean(1)
    gray_x = torch.stack([gray_x, gray_x, gray_x], dim=1)
    # here's a loss BCE based only on the grayscale reconstruction.  Use this in the return statement to kill color learning
    BCEGray = F.binary_cross_entropy(recon_x.view(-1, imgsize * imgsize * 3), gray_x.view(-1,imgsize * imgsize * 3), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCEGray + (beta * KLD)

#loss for just color in a cropped image
def loss_function_color(recon_x, x, mu, log_var):
    beta = 1
    x = x.clone().cuda()
    # make color-only (no shape) reconstruction and use that as the loss function
    recon = recon_x.clone().view(-1, 3 * imgsize * imgsize)
    # compute the maximum color for the r,g and b channels for each digit separately
    maxr, maxi = torch.max(x[:, 0, :], -1, keepdim=True)
    maxg, maxi = torch.max(x[:, 1, :], -1, keepdim=True)
    maxb, maxi = torch.max(x[:, 2, :], -1, keepdim=True)
    newx = x.clone()
    newx[:, 0, :] = maxr
    newx[:, 1, :] = maxg
    newx[:, 2, :] = maxb
    newx = newx.view(-1, imgsize * imgsize * 3)
    BCE = F.binary_cross_entropy(recon, newx, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + (beta * KLD)

#loss function for color map
def loss_function_color_old(recon_x, x, mu, log_var):
    # make color-only (no shape) reconstruction and use that as the loss function
    x = x.clone().cuda()
    recon = recon_x.clone().view(-1, 3, imgsize * imgsize)
    # compute the maximum color for the r,g and b channels for each digit separately
    maxr, maxi = torch.max(recon[:, 0, :], -1, keepdim=True)
    maxg, maxi = torch.max(recon[:, 1, :], -1, keepdim=True)
    maxb, maxi = torch.max(recon[:, 2, :], -1, keepdim=True)
    
    #now build a new reconsutrction that has only the max color, and no shape information at all
    recon[:, 0, :] = maxr
    recon[:, 1, :] = maxg
    recon[:, 2, :] = maxb
    recon = recon.view(-1, 784 * 3)
    maxr, maxi = torch.max(x[:, 0, :], -1, keepdim=True)
    maxg, maxi = torch.max(x[:, 1, :], -1, keepdim=True)
    maxb, maxi = torch.max(x[:, 2, :], -1, keepdim=True)
    newx = x.clone()
    newx[:, 0, :] = maxr
    newx[:, 1, :] = maxg
    newx[:, 2, :] = maxb
    newx = newx.view(-1, 784 * 3)
    BCE = F.binary_cross_entropy(recon, newx, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD 

def test_loss(vae, test_data, whichdecode = []):
    loss_dict = {}

    for decoder in whichdecode:
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(test_data, decoder)
        
        if decoder == 'cropped':
            loss = loss_function_crop(recon_batch, test_data, mu_shape, log_var_shape, mu_color, log_var_color)
        
        loss_dict[decoder] = loss.item()

    return loss_dict

def update_seen_labels(batch_labels, current_labels):
    new_label_lst = []
    for i in range(len(batch_labels)):
        s = batch_labels[0][i].item() # shape label
        c = batch_labels[1][i].item() # color label
        r = batch_labels[2][i].item() # retina location label
        new_label_lst += [(s, c, r)]
    seen_labels = set(new_label_lst) | set(current_labels) # creates a new set 
    return seen_labels

def component_to_grad(comp): # determine gradient for component training
    if comp == 'shape':
        return ['shape']
    elif comp == 'color':
        return ['color']
    elif comp == 'cropped':
        return ['shape', 'color']
    else:
        raise Exception(f'Invalid component: {comp}')

def train(vae, optimizer, epoch, dataloaders, return_loss = False, seen_labels = {}, components = {}):
    vae.train()
    train_loader_noSkip, test_loader = dataloaders[0], dataloaders[1]
    train_loss = 0
    dataiter_noSkip = iter(train_loader_noSkip) # the latent space is trained on EMNIST, MNIST, and f-MNIST
    test_iter = iter(test_loader)
    #sample_iter = iter(sample_loader)
    count = 0
    max_iter = 200
    loader=tqdm(train_loader_noSkip, total = max_iter)

    cropped_loss_train = 0 # loss metrics returned to Training.py

    for i,j in enumerate(loader):
        count += 1
        data, batch_labels = next(dataiter_noSkip)
        
        optimizer.zero_grad()
        
        # determine which component is being trained
        comp_ind = count % len(components)
        whichdecode_use = components[comp_ind]
        keepgrad = component_to_grad(whichdecode_use)
        
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use, keepgrad)
            
        if whichdecode_use == 'shape':  # shape
            loss = loss_function_shape(recon_batch, data, mu_shape, log_var_shape)

        elif whichdecode_use == 'color': # color
            loss = loss_function_color_old(recon_batch, data, mu_color, log_var_color)

        elif whichdecode_use == 'cropped': # cropped
            loss = loss_function_crop(recon_batch, data, mu_shape, log_var_shape, mu_color, log_var_color)
            cropped_loss_train = loss.item()
        
        loss.backward()

        train_loss += loss.item()
        optimizer.step()
        loader.set_description((f'epoch: {epoch}; mse: {loss.item():.5f};'))
        seen_labels = None #update_seen_labels(batch_labels,seen_labels)
        if i == max_iter +1:
            break

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader_noSkip.dataset)))
    
    if return_loss is True:
        # get test losses for cropped and retinal
        test_data = next(test_iter)
        test_data = test_data[0]

        test_loss_dict = test_loss(vae, test_data, ['cropped'])
    
        return [cropped_loss_train, test_loss_dict['cropped']], seen_labels
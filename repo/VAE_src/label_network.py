import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torchvision.utils import save_image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
from joblib import dump, load
import copy

bs = 100
s_classes = 36
c_classes = 10

# shape label network
class VAEshapelabels(nn.Module):
    def __init__(self, xlabel_dim, hlabel_dim,  zlabel_dim):
        super(VAEshapelabels, self).__init__()

        # encoder part
        self.fc1label = nn.Linear(xlabel_dim, hlabel_dim)
        self.fc21label= nn.Linear(hlabel_dim,  zlabel_dim) #mu shape
        self.fc22label = nn.Linear(hlabel_dim, zlabel_dim) #log-var shape


    def sampling_labels (self, mu, log_var, n=1):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) * n
        return mu + eps * std

    def forward(self, x_labels, n):
        h = F.relu(self.fc1label(x_labels))
        mu_shape_label = self.fc21label(h)
        log_var_shape_label=self.fc22label(h)
        z_shape_label = self.sampling_labels(mu_shape_label, log_var_shape_label, n)
        return  z_shape_label

# color label network
class VAEcolorlabels(nn.Module):
    def __init__(self, xlabel_dim, hlabel_dim, zlabel_dim):
        super(VAEcolorlabels, self).__init__()

        # encoder part
        self.fc1label = nn.Linear(xlabel_dim, hlabel_dim)
        self.fc21label = nn.Linear(hlabel_dim, zlabel_dim)  # mu color
        self.fc22label = nn.Linear(hlabel_dim, zlabel_dim)  # log-var color

    def sampling_labels(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_labels):
        h = F.relu(self.fc1label(x_labels))
        mu_shape_label = self.fc21label(h)
        log_var_shape_label = self.fc22label(h)
        z_color_label = self.sampling_labels(mu_shape_label, log_var_shape_label)
        return  z_color_label

vae_shape_labels= VAEshapelabels(xlabel_dim=s_classes, hlabel_dim=20,  zlabel_dim=8)
vae_color_labels= VAEcolorlabels(xlabel_dim=10, hlabel_dim=7,  zlabel_dim=8)
if torch.cuda.is_available():
    vae_shape_labels.cuda()
    vae_color_labels.cuda()
    print('CUDA')

def image_recon(z_labels):
    with torch.no_grad():
        vae.eval()
        output=vae.decoder_noskip(z_labels)
    return output

def load_checkpoint_shapelabels(filepath):
    checkpoint = torch.load(filepath)
    vae_shape_labels.load_state_dict(checkpoint['state_dict_shape_labels'])
    for parameter in vae_shape_labels.parameters():
        parameter.requires_grad = False
    vae_shape_labels.eval()
    return vae_shape_labels

def load_checkpoint_colorlabels(filepath):
    checkpoint = torch.load(filepath)
    vae_color_labels.load_state_dict(checkpoint['state_dict_color_labels'])
    for parameter in vae_color_labels.parameters():
        parameter.requires_grad = False
    vae_color_labels.eval()
    return vae_color_labels


def loss_label(label_act,image_act):

    criterion=nn.MSELoss(reduction='sum')
    e=criterion(label_act,image_act)

    return e

def train_labels(vae, epoch, train_loader, optimizer_shapelabels, optimizer_colorlabels, folder_path):
    global colorlabels, numcolors    
    device = next(vae.parameters()).device
    numcolors = 0
    train_loss_shapelabel = 0
    train_loss_colorlabel = 0

    vae_shape_labels.train().to(device)
    vae_color_labels.train().to(device)

    dataiter = iter(train_loader)

    # labels_color=0
    max_iter = 100
    for i in tqdm(range(len(train_loader)), total= max_iter):
        optimizer_shapelabels.zero_grad()
        optimizer_colorlabels.zero_grad()

        image, labels = next(dataiter)
        labels_for_shape=labels[0].clone()
        labels_for_color=labels[1].clone()
              
        image = image[1].cuda()
        labels_shape = labels_for_shape.to(device)
        input_oneHot = F.one_hot(labels_shape, num_classes=s_classes) # 36 classes in emnist, 10 classes in f-mnist
        input_oneHot = input_oneHot.float()
        input_oneHot = input_oneHot.to(device)

        labels_color = labels_for_color  # get the color labels
        labels_color = labels_color.to(device)
        color_oneHot = F.one_hot(labels_color, num_classes=10)
        color_oneHot = color_oneHot.float()
        color_oneHot = color_oneHot.to(device)
        
        n = 1 # sampling noise
        z_shape_label = vae_shape_labels(input_oneHot,n)
        z_color_label = vae_color_labels(color_oneHot)

        z_shape, z_color, z_location = vae.activations(image)

        # train shape label net
        
        loss_of_shapelabels = loss_label(z_shape_label, z_shape)
        loss_of_shapelabels.backward(retain_graph = True)
        train_loss_shapelabel += loss_of_shapelabels.item()

        optimizer_shapelabels.step()

        # train color label net
        loss_of_colorlabels = loss_label(z_color_label, z_color)
        loss_of_colorlabels.backward(retain_graph = True)
        train_loss_colorlabel += loss_of_colorlabels.item()

        optimizer_colorlabels.step()

        if i % 1000 == 0:
            vae_shape_labels.eval()
            vae_color_labels.eval()
            vae.eval()

            with torch.no_grad():
                recon_imgs = vae.decoder_cropped(z_shape, z_color,0,0)
                recon_imgs_shape = vae.decoder_shape(z_shape, z_color,0)
                recon_imgs_color = vae.decoder_color(z_shape, z_color,0)

                recon_labels = vae.decoder_cropped(z_shape_label, z_color_label,0,0)
                recon_shapeOnly = vae.decoder_shape(z_shape_label, 0,0)
                recon_colorOnly = vae.decoder_color(0, z_color_label,0)

                sample_size = 20
                orig_imgs = image[:sample_size]
                recon_labels = recon_labels[:sample_size]
                recon_imgs = recon_imgs[:sample_size]
                recon_imgs_shape = recon_imgs_shape[:sample_size]
                recon_imgs_color = recon_imgs_color[:sample_size]
                recon_shapeOnly = recon_shapeOnly[:sample_size]
                recon_colorOnly = recon_colorOnly[:sample_size]

            utils.save_image(
                torch.cat(
                    [orig_imgs,
                     recon_imgs.view(sample_size, 3, 28, 28),
                     recon_imgs_shape.view(sample_size, 3, 28, 28),
                     recon_imgs_color.view(sample_size, 3, 28, 28),
                     recon_labels.view(sample_size, 3, 28, 28),
                     recon_shapeOnly.view(sample_size, 3, 28, 28),
                     recon_colorOnly.view(sample_size, 3, 28, 28)], 0),
                f'{folder_path}{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                nrow=sample_size,
                normalize=False,
            )
        if i > max_iter + 1:
            break
    print(
        '====> Epoch: {} Average loss shape: {:.4f}'.format(epoch, train_loss_shapelabel / (len(train_loader.dataset) / bs)))
    print('====> Epoch: {} Average loss color: {:.4f}'.format(epoch,train_loss_colorlabel / (len(train_loader.dataset) / bs)))


def test_outputs(test_loader, n = 0.5):
        vae_shape_labels.eval()
        vae_color_labels.eval()
        vae.eval()

        dataiter = iter(test_loader)
        image, labels = dataiter.next()
        labels_for_shape=labels[0].clone()
        labels_for_color=labels[1].clone()
              
        image = image.cuda()
        labels_shape = labels_for_shape.cuda()
        input_oneHot = F.one_hot(labels_shape, num_classes=s_classes) # 47 classes in emnist, 10 classes in f-mnist
        input_oneHot = input_oneHot.float()
        input_oneHot = input_oneHot.cuda()

        labels_color = labels_for_color  # get the color labels
        labels_color = labels_color.cuda()
        color_oneHot = F.one_hot(labels_color, num_classes=10)
        color_oneHot = color_oneHot.float()
        color_oneHot = color_oneHot.cuda()
        
        n=1
        z_shape_label = vae_shape_labels(input_oneHot,n)
        z_color_label = vae_color_labels(color_oneHot)

        z_shape, z_color, z_location = image_activations(image)

        with torch.no_grad():
                recon_imgs = vae.decoder_cropped(z_shape, z_color,0,0)
                recon_imgs_shape = vae.decoder_shape(z_shape, z_color,0)
                recon_imgs_color = vae.decoder_color(z_shape, z_color,0)

                recon_labels = vae.decoder_cropped(z_shape_label, z_color_label,0,0)
                recon_shapeOnly = vae.decoder_shape(z_shape_label, 0,0)
                recon_colorOnly = vae.decoder_color(0, z_color_label,0)

                sample_size = 20
                orig_imgs = image[:sample_size]
                recon_labels = recon_labels[:sample_size]
                recon_imgs = recon_imgs[:sample_size]
                recon_imgs_shape = recon_imgs_shape[:sample_size]
                recon_imgs_color = recon_imgs_color[:sample_size]
                recon_shapeOnly = recon_shapeOnly[:sample_size]
                recon_colorOnly = recon_colorOnly[:sample_size]

        utils.save_image(
                torch.cat(
                    [orig_imgs,
                     recon_imgs.view(sample_size, 3, 28, 28),
                     recon_imgs_shape.view(sample_size, 3, 28, 28),
                     recon_imgs_color.view(sample_size, 3, 28, 28),
                     recon_labels.view(sample_size, 3, 28, 28),
                     recon_shapeOnly.view(sample_size, 3, 28, 28),
                     recon_colorOnly.view(sample_size, 3, 28, 28)], 0),
                f'sample_training_labels/labeltest_with_{n}.png',
                nrow=sample_size,
                normalize=False,
                range=(-1, 1),
            )

def test_opposite_colors(test_loader, n = 0.5):
        vae_shape_labels.eval()
        vae_color_labels.eval()
        vae.eval()

        dataiter = iter(test_loader)
        image, labels = dataiter.next()
        labels_for_shape=labels[0].clone()
        labels_for_color=labels[1].clone()
              
        image = image.cuda()
        labels_shape = labels_for_shape.cuda()
        input_oneHot = F.one_hot(labels_shape, num_classes=s_classes) # 47 classes in emnist, 10 classes in f-mnist
        input_oneHot = input_oneHot.float()
        input_oneHot = input_oneHot.cuda()

        labels_color = labels_for_color  # get the color labels
        labels_color = labels_color.cuda()
        color_oneHot = F.one_hot(labels_color, num_classes=10)
        color_oneHot = color_oneHot.float()
        color_oneHot = color_oneHot.cuda()
        
        n=1
        z_shape_label = vae_shape_labels(input_oneHot,n)
        z_color_label = vae_color_labels(color_oneHot)

        z_shape, z_color, z_location = image_activations(image)

        with torch.no_grad():
                recon_imgs = vae.decoder_cropped(z_shape, z_color,0,0)
                recon_imgs_shape = vae.decoder_shape(z_shape, z_color,0)
                recon_imgs_color = vae.decoder_color(z_shape, z_color,0)

                recon_labels = vae.decoder_cropped(z_shape_label, z_color_label,0,0)
                recon_shapeOnly = vae.decoder_shape(z_shape_label, 0,0)
                recon_colorOnly = vae.decoder_color(0, z_color_label,0)

                sample_size = 20
                orig_imgs = image[:sample_size]
                recon_labels = recon_labels[:sample_size]
                recon_imgs = recon_imgs[:sample_size]
                recon_imgs_shape = recon_imgs_shape[:sample_size]
                recon_imgs_color = recon_imgs_color[:sample_size]
                recon_shapeOnly = recon_shapeOnly[:sample_size]
                recon_colorOnly = recon_colorOnly[:sample_size]

        utils.save_image(
                torch.cat(
                    [orig_imgs,
                     recon_imgs.view(sample_size, 3, 28, 28),
                     recon_imgs_shape.view(sample_size, 3, 28, 28),
                     recon_imgs_color.view(sample_size, 3, 28, 28),
                     recon_labels.view(sample_size, 3, 28, 28),
                     recon_shapeOnly.view(sample_size, 3, 28, 28),
                     recon_colorOnly.view(sample_size, 3, 28, 28)], 0),
                f'sample_training_labels_red_green/opposite_color_test.png',
                nrow=sample_size,
                normalize=False,
                range=(-1, 1),
            )
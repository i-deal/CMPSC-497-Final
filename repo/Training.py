# prerequisites
import torch
import os
from VAE_src.mVAE import load_checkpoint, vae_builder
from torch.utils.data import DataLoader, ConcatDataset
from VAE_src.dataset_builder import Dataset
from VAE_src.train_mVAE import train_mVAE
from VAE_src.train_labels import train_labelnet
from VAE_src.train_classifiers import train_classifiers
from torchvision import datasets, transforms, utils
import sys
import argparse

parser = argparse.ArgumentParser(description="Training of MLR-2.0")
parser.add_argument("--load_prev", type=bool, default=False, help="Begin training from previous checkpoints")
parser.add_argument("--cuda_device", type=int, default=1, help="Which cuda device to use")
parser.add_argument("--cuda", type=bool, default=True, help="Cuda availability")
parser.add_argument("--folder", type=str, default='red_green_betadf', help="Where to store checkpoints in checkpoints/")
parser.add_argument("--train_list", nargs='+', type=str, default=['mVAE', 'label_net', 'SVM'], help="Which components to train")
parser.add_argument("--wandb", type=bool, default=False, help="Track training with wandb")
parser.add_argument("--checkpoint_name", type=str, default='mVAE_checkpoint.pth', help="file name of checkpoint .pth")
parser.add_argument("--vae_type", type=str, default='CNN', help="type of vae, CNN, FC")
#parser.add_argument("--batch_size", nargs='+', type=int, default=['mVAE', 'label_net', 'SVM'], help="Which components to train")
args = parser.parse_args()


transforms_train = {'colorize':True, 'color_targets':{0:[0,1,2,3,4],1:[5,6,7,8,9]}, 'retina':False} # red:0-4, green:5-9
mnist_dataset= Dataset('mnist', transforms_train)

transforms_test = {'colorize':True, 'color_targets':{1:[0,1,2,3,4],0:[5,6,7,8,9]}, 'retina':False} # green:0-4, red:5-9
mnist_test_dataset= Dataset('mnist', transforms_test, train=False)

print(mnist_dataset.all_possible_labels())

#concat datasets and init dataloaders

bs = 200
train_loader = mnist_dataset.get_loader(bs)
test_loader = mnist_test_dataset.get_loader(bs)



dataloaders = [train_loader, test_loader]

print(f'Training: {args.train_list}')

folders = ['red_green', 'red_green_betadf', 'red_green_df', 'red_green_olddf']
vae_types = ['CNN', 'FC', 'FC', 'FC']
betas = [1, 1.7, 1.7]

for q in range(3,4):
    folder_name = folders[q]
    #torch.set_default_dtype(torch.float64)
    checkpoint_folder_path = f'checkpoints/{folder_name}/' # the output folder for the trained model versions

    if not os.path.exists('checkpoints/'):
        os.mkdir('checkpoints/')

    if not os.path.exists(checkpoint_folder_path):
        os.mkdir(checkpoint_folder_path)

    if args.cuda is True:
        d = args.cuda_device

    load = args.load_prev

    print(f'Device: {d}')
    print(f'Load: {load}')

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{d}')
        torch.cuda.set_device(d)
        print('CUDA')
    else:
        device = 'cpu'

    bs=100

    # to resume training an existing model checkpoint, uncomment the following line with the checkpoints filename
    if load is True:
        vae = load_checkpoint(f'{checkpoint_folder_path}/{args.checkpoint_name}', vae_types[q], d)
        print('checkpoint loaded')
    else:
        vae, z = vae_builder(vae_types[q], betas[q])

    vae.to(device)
    print('Training: mVAE')
    train_mVAE(dataloaders, vae, 120, folder_name, args.wandb)
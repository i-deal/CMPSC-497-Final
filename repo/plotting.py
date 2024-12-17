import torch
import os
from VAE_src.mVAE import load_checkpoint
from VAE_src.dataset_builder import Dataset
from torchvision import utils
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np

d = 1

device = torch.device(f'cuda:{d}')
torch.cuda.set_device(d)
print('checkpoint loaded')

transforms_train = {'colorize':True, 'color_targets':{0:[0,1,2,3,4],1:[5,6,7,8,9]}, 'retina':False} # red:0-4, green:5-9
mnist_dataset= Dataset('mnist', transforms_train)

transforms_test = {'colorize':True, 'color_targets':{1:[0,1,2,3,4],0:[5,6,7,8,9]}, 'retina':False} # green:0-4, red:5-9
mnist_test_dataset= Dataset('mnist', transforms_test, train=False)

print(mnist_dataset.all_possible_labels())

#concat datasets and init dataloaders
sample_size = 400
train_loader = mnist_dataset.get_loader(sample_size)
test_loader = mnist_test_dataset.get_loader(sample_size)

train_iter = iter(train_loader)
test_iter = iter(test_loader)

imgsize = 28
shape_color_dim = imgsize

from sklearn.model_selection import StratifiedShuffleSplit

class DisentanglementMetrics:
    def __init__(self, device):
        self.device = device
        
    def collect_batch_representations(self, z_shape, z_color, labels):
        shape_labels = labels[0].detach()
        color_labels = labels[1].detach()
        return z_shape.detach(), z_color.detach(), shape_labels, color_labels

    def compute_representation_classification(self, representations, labels):
        """Compute classification accuracy."""
        num_classes = labels.unique().size(0)  # Infer class count dynamically
        representations = representations.to(self.device)
        labels = labels.to(self.device)
        
        # Classifier
        classifier = nn.Sequential(
            nn.Linear(representations.size(1), 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        ).to(self.device)
        
        # Stratified train-test split
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(representations.cpu(), labels.cpu()))
        train_X, test_X = representations[train_idx], representations[test_idx]
        train_y, test_y = labels[train_idx], labels[test_idx]
        
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        best_acc = 0
        
        for epoch in range(20):  # Train for up to 20 epochs
            classifier.train()
            optimizer.zero_grad()
            output = classifier(train_X)
            loss = nn.CrossEntropyLoss()(output, train_y)
            loss.backward()
            optimizer.step()
            
            # Evaluate on the test set
            classifier.eval()
            with torch.no_grad():
                predictions = classifier(test_X).argmax(dim=1)
                accuracy = (predictions == test_y).float().mean().item()
                best_acc = max(best_acc, accuracy)
        
        # Adjust for chance
        chance = 1.0 / num_classes
        adjusted_accuracy = max(0, (best_acc - chance) / (1 - chance))
        return adjusted_accuracy

    def compute_nk(self, shape_rep, color_rep, labels, is_shape_task=True):
        """Compute NK metric."""
        # Combine representations
        full_rep = torch.cat([shape_rep, color_rep], dim=1)
        acc_all = self.compute_representation_classification(full_rep, labels)

        # Exclude relevant representation
        wrong_rep = color_rep if is_shape_task else shape_rep
        acc_without_relevant = self.compute_representation_classification(wrong_rep, labels)

        return acc_all - acc_without_relevant

def sort_by_shape(data, labels):
    """
    Sort data and labels by shape identity in increasing order.
    
    Args:
        data: tensor of shape [batch_size, channels, height, width]
        labels: list of [shape_labels, color_labels] tensors
        
    Returns:
        Sorted data and labels tensors
    """
    shape_labels = labels[0]
    color_labels = labels[1]
    
    # Get sorting indices from shape labels
    _, indices = torch.sort(shape_labels)
    indices = indices.cpu()
    
    # Sort all tensors using these indices
    sorted_data = data[indices]
    sorted_shape_labels = shape_labels[indices]
    sorted_color_labels = color_labels[indices]
    
    return sorted_data, [sorted_shape_labels, sorted_color_labels]

folders = ['red_green', 'red_green_betadf', 'red_green_df', 'red_green_olddf']
vae_types = ['CNN', 'FC', 'FC', 'FC']

# Get representations and labels from your batch
data, labels = next(train_iter)
data = data.to(device)

test_data, test_labels = next(test_iter)
test_data.to(device)

labels[0] = labels[0].to(device)
labels[1] = labels[1].to(device)

test_labels[0] = test_labels[0].to(device)
test_labels[1] = test_labels[1].to(device)

data, labels = sort_by_shape(data, labels)
test_data, junk = sort_by_shape(test_data, test_labels)

for q in range(0,4):
        folder_name = folders[q]
        print(q)

        checkpoint_folder_path = f'checkpoints/{folder_name}/' # the output folder for the trained model versions
        vae = load_checkpoint(f'{checkpoint_folder_path}/mVAE_checkpoint.pth', vae_types[q], d)

        # Get latent representations using your VAE
        _, mu_color, log_var_color, mu_shape, log_var_shape = vae(data)
        z_color = vae.sampling(mu_color, log_var_color)
        z_shape = vae.sampling(mu_shape, log_var_shape)
        # Usage with your existing code:
        metrics = DisentanglementMetrics(device)

        # Process batch
        # Usage
        metrics = DisentanglementMetrics(device)

        # Process batch
        shape_reps, color_reps, shape_labels, color_labels = metrics.collect_batch_representations(
        z_shape, z_color, labels
        )

        # Compute metrics
        shape_classification_acc = metrics.compute_representation_classification(shape_reps, shape_labels)
        color_classification_acc = metrics.compute_representation_classification(color_reps, color_labels)
        shape_nk = metrics.compute_nk(shape_reps, color_reps, shape_labels, is_shape_task=True)
        color_nk = metrics.compute_nk(shape_reps, color_reps, color_labels, is_shape_task=False)

        print(f"Shape SNC Score: {shape_classification_acc:.3f}")
        print(f"Color SNC Score: {color_classification_acc:.3f}")
        print(f"Shape NK Score: {shape_nk:.3f}")
        print(f"Color NK Score: {color_nk:.3f}")
        
        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data)
        recon_batch_test, mu_color1, log_var_color1, mu_shape1, log_var_shape1 = vae(test_data)

        MSE_train = torch.mean((recon_batch.cuda() - data.cuda()) ** 2)
        MSE_test = MSE = torch.mean((recon_batch_test.cuda() - test_data.cuda()) ** 2)
        print(f'MSE train: {MSE_train}')
        print(f'MSE test: {MSE_test}')

        utils.save_image(torch.cat([data.view(sample_size, 3, imgsize, shape_color_dim).cuda(), recon_batch.view(sample_size, 3, imgsize, shape_color_dim).cuda(), test_data.view(sample_size, 3, imgsize, shape_color_dim).cuda(),
                    recon_batch_test.view(sample_size, 3, imgsize, shape_color_dim).cuda()], 0),
            f'{q}red_green_sample{q}.png',
            nrow=sample_size, normalize=False,)
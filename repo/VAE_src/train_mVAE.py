# prerequisites
import torch
from VAE_src.mVAE import train
import torch.optim as optim

def train_mVAE(dataloaders, vae, epoch_count, checkpoint_folder, use_wandb, start_epoch = 1):
    if use_wandb is True:
        import wandb
        from VAE_src.wandb_setup import initialize_wandb, log_system_metrics
        initialize_wandb('2d-retina-train', {'version':'MLR_2.0_2D_RETINA'})

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    seen_labels = {}
    components = ['color'] + ['shape'] + ['cropped']

    for epoch in range(start_epoch, epoch_count):

        loss_lst, seen_labels = train(vae, optimizer, epoch, dataloaders, True, seen_labels, components)

        if use_wandb is True:
            wandb.log({
            'epoch': epoch,
            'training_loss': loss_lst[0],
            'test_loss': loss_lst[1]
            })

        torch.cuda.empty_cache()
        vae.eval()
        checkpoint =  {
            'state_dict': vae.state_dict(),
            #'labels': seen_labels
                    }
        torch.save(checkpoint, f'checkpoints/{checkpoint_folder}/mVAE_checkpoint.pth')
    
    if use_wandb is True:
        wandb.finish()
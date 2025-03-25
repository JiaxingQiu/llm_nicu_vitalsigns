import torch
import matplotlib.pyplot as plt
import numpy as np
def plot_reconstructions(model, dataloader, num_samples=20, start_idx=0, title="Data Reconstructions"):
    """
    Plot original and reconstructed time series from a VAE model.
    
    Args:
        model: The VAE model
        dataloader: DataLoader containing the dataset
        num_samples: Number of samples to plot (default=20)
        start_idx: Starting index for samples (default=0)
        title: Title for the plot (default="Data Reconstructions")
    """
    model.eval()
    rows = num_samples // 5
    fig, axes = plt.subplots(rows, 5, figsize=(20, 4*rows), facecolor='white')
    
    with torch.no_grad():
        for i in range(start_idx, start_idx + num_samples):
            # Get data
            x, text_features, _ = dataloader.dataset[i]
            x = x.unsqueeze(0).to(model.device)
            text_features = [text_feature.unsqueeze(0).to(model.device) 
                           for text_feature in text_features]
            
            # Get reconstruction
            _, x_hat, _, _ = model(x, text_features)
            
            # Plot
            row, col = i//5, i%5
            axes[row, col].plot(x[0].cpu().numpy(), label='Original')
            axes[row, col].plot(x_hat[0].cpu().detach().numpy(), label='Reconstructed')
            axes[row, col].legend()
            axes[row, col].set_title(f'Sample {i}')
    
    plt.tight_layout()
    fig.suptitle(title, y=1.02, fontsize=16)
    plt.show()


def plot_reconstruction_from_distances(model, dataloader, sample_idx=1, distances=[0, 5e-4, 7.5e-4, 1e-3, 2e-3]):
    """
    Visualize how different levels of latent space noise affect reconstruction quality.
    
    Args:
        model: The VAE model
        dataloader: DataLoader containing the dataset
        sample_idx: Index of the sample to analyze (default=1)
        distances: List of noise variances to add to latent space (default=[0, 5e-4, 7.5e-4, 1e-3, 2e-3])
    """
    model.eval()
    x, _, _ = dataloader.dataset[sample_idx]

    # Create figure and subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Original Signal vs Reconstructions with Different Noise Levels', fontsize=14)
    axs = axs.flatten()

    # Plot original signal in first subplot
    axs[0].plot(x.cpu().detach().numpy(), 'b-', label='Original')
    axs[0].set_title('Original Signal')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Value')
    axs[0].grid(True)
    axs[0].legend()

    with torch.no_grad():
        # Plot reconstructions with different distances
        for i, distance in enumerate(distances, 1):
            # Get embeddings and reconstruction
            _, z_mean, z_log_var = model.ts_encoder(x)
            z = model.ts_encoder.reparameterization(z_mean, z_log_var + distance)
            x_hat = model.ts_decoder(z)
            _, z_mean_hat, _ = model.ts_encoder(x_hat)

            # Calculate Euclidean distance between original and reconstructed latents
            z_mean = z_mean.cpu().detach().numpy()
            z_mean_hat = z_mean_hat.cpu().detach().numpy()
            euc_dist = np.sqrt(np.sum((z_mean - z_mean_hat) ** 2))

            # Plot reconstruction
            axs[i].plot(x_hat.cpu().detach().numpy(), 'r-', label='Reconstruction')
            axs[i].plot(x.cpu().detach().numpy(), 'b--', alpha=0.5, label='Original')
            axs[i].set_title(f'Noise var={distance:.1e}\nLatent space distance={euc_dist:.4f}')
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Value')
            axs[i].grid(True)
            axs[i].legend()

    plt.tight_layout()
    plt.show()

# Usage example:
# plot_reconstruction_noise_analysis(model, train_dataloader)
# 
# # With custom parameters:
# plot_reconstruction_noise_analysis(
#     model, 
#     train_dataloader,
#     sample_idx=5,
#     distances=[0, 1e-4, 1e-3, 1e-2]
# )
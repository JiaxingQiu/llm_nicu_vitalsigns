from config import *
import torch
from torch.utils.data import Dataset
import numpy as np
from encoder import *
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def augment_ts_df(ts_df, pretrained_model_path, K = 50, dist = 5e-4):

    # df to return
    df_aug = pd.DataFrame()

    # K number of augmentations for each sample
    model = VAE_Linear_Medium().to(device)
    model.load_state_dict(torch.load(pretrained_model_path))
    model.eval()
    ts_dataset = ScaledTSDataset(ts_df)

    for i in tqdm(range(len(ts_dataset))):
        x, ts_mean, ts_std = ts_dataset[i]
        ts_mean = ts_mean.cpu().detach().numpy()
        ts_std = ts_std.cpu().detach().numpy()
        ts_hat_ls = []
        euc_dist_ls = []
        for k in range(K): 
            # augment x k times
            distance = np.random.uniform(0, dist)
            z_mean, z_log_var = model.encode(x)
            z = model.reparameterization(z_mean, z_log_var + distance)
            x_hat = model.decode(z) # length of 300
            z_mean_hat, _ = model.encode(x_hat)

            # Calculate Euclidean distance
            z_mean = z_mean.cpu().detach().numpy()
            z_mean_hat = z_mean_hat.cpu().detach().numpy()
            euc_dist = np.sqrt(np.sum((z_mean - z_mean_hat) ** 2))

            x_hat = x_hat.cpu().detach().numpy()
            ts_hat = x_hat * ts_std + ts_mean
            ts_hat_ls.append(ts_hat)
            euc_dist_ls.append(euc_dist)

        # Convert to numpy array with shape (K, 300)
        ts_hat_ls = np.array(ts_hat_ls)
        euc_dist_ls = np.array(euc_dist_ls)

        #  the dataframe
        df_aug_i = pd.DataFrame(ts_hat_ls, columns=[str(i) for i in range(1, 301)])  # '1' to '300'
        # make all cells integer 
        df_aug_i = df_aug_i.round().astype(int)
        df_aug_i.insert(0, 'rowid', [i] * K)  # Add rowid at the beginning
        df_aug_i['euc_dist'] = euc_dist_ls     # Add euc_dist at the end

        df_aug = pd.concat([df_aug, df_aug_i], ignore_index=True)


    return df_aug






# Create class vitalsign timeseries which is a child of Dataset class from torch.util.data
class ScaledTSDataset(Dataset):

    def __init__(self, ts_df):
        self.ts_df = ts_df # Shape: [obs, time]
        self.ts_scaled, self.ts_means, self.ts_stds = self.scale_ts(ts_df)
 

    def scale_ts(self, ts_df):
        data = ts_df.values
        obs_mean = np.nanmean(data, axis=1)
        osb_std = np.nanstd(data, axis=1)
        osb_std = np.where(osb_std == 0, 1e-8, osb_std)
        data_t = (data.T - obs_mean.T) / osb_std.T
        data_scaled = data_t.T
        return data_scaled, obs_mean, osb_std
        
    def __len__(self):
        return len(self.ts_df)
    
    def __getitem__(self, idx):
        # Get single time series and convert to tensor
        ts = torch.tensor(self.ts_scaled[idx], dtype=torch.float32).to(device)
        ts_mean = torch.tensor(self.ts_means[idx], dtype=torch.float32).to(device)
        ts_std = torch.tensor(self.ts_stds[idx], dtype=torch.float32).to(device)
        return ts, ts_mean, ts_std



def plot_augmentor(x, pretrained_model_path):

    # x has to be a output of scaledtsdataset
    
    # K number of augmentations for each sample
    model = VAE_Linear_Medium().to(device)
    model.load_state_dict(torch.load(pretrained_model_path))
    model.eval()

    
    #  Create figure and subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Original Signal vs Reconstructions with Different Distances', fontsize=14)

    # Flatten axs for easier iteration
    axs = axs.flatten()

    # Plot original signal in first subplot
    axs[0].plot(x.detach().numpy(), 'b-', label='Original')
    axs[0].set_title('Original Signal')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Value')
    axs[0].grid(True)
    axs[0].legend()

    # Plot reconstructions with different distances
    distances = [0, 5e-4, 7.5e-4, 1e-3, 2e-3]
    for i, distance in enumerate(distances, 1):
        # Get embeddings and reconstruction
        z_mean, z_log_var = model.encode(x)
        z = model.reparameterization(z_mean, z_log_var + distance)
        x_hat = model.decode(z)
        z_mean_hat, z_log_var_hat = model.encode(x_hat)

        # Calculate Euclidean distance
        z_mean = z_mean.cpu().detach().numpy()
        z_mean_hat = z_mean_hat.cpu().detach().numpy()
        euc_dist = np.sqrt(np.sum((z_mean - z_mean_hat) ** 2))

        # Plot reconstruction
        axs[i].plot(x_hat.detach().numpy(), 'r-', label=f'Reconstruction')
        axs[i].plot(x.detach().numpy(), 'b--', alpha=0.5, label='Original')
        axs[i].set_title(f'Larger Gaussian var={distance:.1e}\nEuclidean dist between latents={euc_dist:.4f}')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Value')
        axs[i].grid(True)
        axs[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()
        


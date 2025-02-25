#%pip install pandas==2.2.0
#%pip install python-calamine
#%pip install torchinfo



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchinfo import summary as nn_summary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# global variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("using device: ", device)


def prepare_data(data_path, batch_size = 128):
    # data_path = './data/PAS Challenge HR Data.xlsx'    
    df = pd.read_excel(data_path, sheet_name='Sheet1', engine="calamine")
    df['id_time'] = 'id_' + df['VitalID'].astype(str) + '_' + df['VitalTime'].astype(str)
    df.set_index('id_time', inplace=True, drop=True)
    df.drop(columns=['VitalID', 'VitalTime'], inplace=True)  


    data = df.values
    obs_mean = np.nanmean(data, axis=1)
    osb_std = np.nanstd(data, axis=1)
    osb_std = np.where(osb_std == 0, 1e-8, osb_std)
    data_t = (data.T - obs_mean.T) / osb_std.T
    data_scaled = data_t.T
    
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.ravel()  # Flatten the 2D array of axes for easier indexing
    # Plot each pair of lines
    for i in range(12):
        ax = axes[i]
        ax.plot(data_scaled[i,:], label='data_scaled', alpha=0.7)
        ax.plot(data[i,:], label='data', alpha=0.7)
        #ax.legend()
        ax.set_title(f'Time Series {i+1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
    # Adjust the layout to prevent overlap
    plt.tight_layout()
    plt.show()

    # # sanity check
    # import matplotlib.pyplot as plt
    # data_scaled_row0 = data_scaled[0,:]
    # row0 = df.loc['id_1007_122170',]
    # scaled_row0 = (row0 - np.nanmean(row0))/np.nanstd(row0)
    # # plot scaled_row0 and row0
    # plt.plot(scaled_row0, label='scaled_row0')
    # plt.plot(data_scaled_row0, label='data_scaled_row0')
    # plt.legend()
    # plt.show()
    # # scaled_row0 and data_scaled_row0 are they identical?
    # # they are float numbers, so we need to check if they are very close
    # np.allclose(scaled_row0, data_scaled_row0)

    df_scaled = df.copy() # [obs, time]
    df_scaled = df_scaled.astype(float)  # Convert all columns to float
    df_scaled.loc[:,:] = data_scaled
    dataset = VSTSDataset(df_scaled)

    # Split dataset into train and test in the ratio of 80:20
    train_dataset, test_dataset = random_split(dataset, [0.8,0.2])

    # Use DataLoader for batching and shuffling
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader, df_scaled, df




# Create class vitalsign timeseries which is a child of Dataset class from torch.util.data
class VSTSDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.input = self.dataframe.values  # Shape: [obs, time]
        # No need to reshape here since we want to keep the time series structure
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Get single time series and convert to tensor
        ts = torch.tensor(self.input[idx], dtype=torch.float32).to(device)
        return ts





def loss_function(x, x_hat, mean, log_var):
    # Reproduction Loss
    reproduction_loss = nn.functional.mse_loss(x_hat, x)
    # KL Divergence Loss
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD



def train_epoch(model, train_dataloader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, x in enumerate(train_dataloader):
        x = x.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches



def test_epoch(model, test_dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():  # No gradient computation during testing
        for batch_idx, x in enumerate(test_dataloader):
            x = x.to(device)
            
            # Forward pass only
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches



def train(model, train_dataloader, test_dataloader, optimizer, num_epochs, device):
    train_losses = []
    test_losses = []
    
    # Learning rate scheduler
    # Reduce LR by factor of 0.5 if loss doesn't improve for k epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           # Reduce LR when metric stops decreasing
        factor=0.8,          # Multiply LR by this factor
        patience=30,          # Number of epochs to wait before reducing LR
        verbose=True,        # Print message when LR is reduced
        min_lr=1e-20         # Don't reduce LR below this value
    )
    
    # Keep track of best model
    best_loss = float('inf')
    best_model_state = None
    
    try:
        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss = train_epoch(model, train_dataloader, optimizer, device)
            
            # Test for one epoch
            test_loss = test_epoch(model, test_dataloader, device)
            
            # Store losses
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            # Update learning rate based on test loss
            scheduler.step(test_loss)
            
            # Save best model
            if test_loss < best_loss:
                best_loss = test_loss
                best_model_state = model.state_dict().copy()
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print progress
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'\tTraining Loss: {train_loss:.6f}')
            print(f'\tTesting Loss: {test_loss:.6f}')
            print(f'\tLearning Rate: {current_lr:.9f}')
            
            # Early stopping if learning rate becomes too small
            if current_lr <= scheduler.min_lrs[0]:
                print("Learning rate too small. Stopping training.")
                break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")
    
    finally:
        # Load best model if we found one
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return train_losses, test_losses
   


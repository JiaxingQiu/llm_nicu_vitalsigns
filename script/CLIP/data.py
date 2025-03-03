import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from config import *
from encoders import *
from models import get_similarity_targets

class CLIPDataset(Dataset):
    def __init__(self, ts_features, text_features, labels):
        """
        Args:
            ts_features: time series features tensor [N, ts_dim]
            text_features: text features tensor [N, text_dim]
            labels: class labels tensor [N]
        """
        # Verify inputs are tensors
        if not isinstance(ts_features, torch.Tensor):
            ts_features = torch.FloatTensor(ts_features)
        if not isinstance(text_features, torch.Tensor):
            text_features = torch.FloatTensor(text_features)
        if not isinstance(labels, torch.Tensor):
            labels = torch.LongTensor(labels)
            
        assert len(ts_features) == len(text_features) == len(labels), "All inputs must have the same length"
        self.ts_features = ts_features
        self.text_features = text_features
        self.labels = labels
        self.targets_org = get_similarity_targets(ts_features, text_features)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            idx,
            self.ts_features[idx],
            self.text_features[idx],
            self.labels[idx],
            self.targets_org[idx]
        )
    def dataloader(self, batch_size=32):
        return DataLoader(self, 
                          batch_size=batch_size, 
                          shuffle=False)
    


class TSFeature(Dataset):
    def __init__(self, norm_ts_df, encoder_model_name):
        self.ts_df = norm_ts_df # normalized time series dataframe. Shape: [obs, time]
        self.encoder_model_name = encoder_model_name
        self.encoder = TSEncoder(model_name = encoder_model_name)
        self.features = self.encode_batch(self.ts_df) # tensor shape: [obs, embed_dim]
        # Convert to tensor if not already
        if not isinstance(self.features, torch.Tensor):
            self.features = torch.tensor(self.features)

    def encode_batch(self, df, batch_size=32):
        encoded = []
        model = self.encoder.model
        data_tensor = torch.tensor(df.values, dtype=torch.float32).to(device)
        with torch.no_grad():  # Disable gradient computation for inference
            for i in range(0, len(df), batch_size):
                batch = data_tensor[i:i + batch_size]
                z_mean, z_log_var = model.encode(batch)
                z = model.reparameterization(z_mean, z_log_var)
                encoded.append(z)
        return torch.cat(encoded, dim=0)
        

    def __len__(self):
        return len(self.ts_df)
    
    def __getitem__(self, idx):
        ts_f = torch.tensor(self.features[idx], dtype=torch.float32).to(device)
        return ts_f


class TXTFeature(Dataset):
    def __init__(self, txt_ls, encoder_model_name):
        self.txt_ls = txt_ls # list of strings
        self.encoder_model_name = encoder_model_name
        self.encoder = TXTEncoder(model_name = encoder_model_name)
        self.features = self.encoder.encode_text_list(txt_ls) # tensor shape: [obs, embed_dim]
        # Convert to tensor if not already
        if not isinstance(self.features, torch.Tensor):
            self.features = torch.tensor(self.features)
        # Global min-max normalization (across all values)
        min_val = torch.min(self.features)
        max_val = torch.max(self.features)
        self.features = (self.features - min_val) / (max_val - min_val + 1e-8)  # Add small epsilon to avoid division by zero

    def __len__(self):
        return len(self.txt_ls)
    
    def __getitem__(self, idx):
        txt_f = torch.tensor(self.features[idx], dtype=torch.float32).to(device)
        return txt_f







def get_ts_txt_org(df):
    # df must have columns: 
    # VitalID, VitalTime, 
    # '1', '2', ..., '300' 
    # 'text'
    # 'label'
    
    df = df.assign(id_time='id_' + df['VitalID'].astype(str) + '_' + df['VitalTime'].astype(str))
    df = df.set_index('id_time')
    df = df.drop(columns=['VitalID', 'VitalTime'])
    
    # get normalized time series dataframe norm_ts_df
    ts_df = df.loc[:,'1':'300']
    data = ts_df.values
    obs_mean = np.nanmean(data, axis=1)
    osb_std = np.nanstd(data, axis=1)
    osb_std = np.where(osb_std == 0, 1e-8, osb_std)
    data_t = (data.T - obs_mean.T) / osb_std.T
    data_scaled = data_t.T
    norm_ts_df = ts_df.copy() # [obs, time]
    norm_ts_df = norm_ts_df.astype(float)
    norm_ts_df.loc[:,:] = data_scaled

    # get text list txt_ls
    txt_ls = df.loc[:,'text'].tolist()
    labels = df.loc[:,'label'].tolist()

    return norm_ts_df, txt_ls, labels


def get_features(df, 
                 ts_encoder_name='hr_vae_linear_medium', 
                 text_encoder_name='sentence-transformers/all-mpnet-base-v2'):
    norm_ts_df, txt_ls, labels = get_ts_txt_org(df)
    ts_f = TSFeature(norm_ts_df, encoder_model_name=ts_encoder_name).features
    tx_f = TXTFeature(txt_ls, encoder_model_name=text_encoder_name).features
    labels = torch.tensor(labels)
    
    return ts_f, tx_f, labels



# def get_dataloaders(ts_f, tx_f, labels, train_ratio=0.8, batch_size=32):
#     # get the unique values of labels
#     unique_labels = torch.unique(labels)
#     # first 80% of unique labels are train labels, the rest are test labels
#     train_labels = unique_labels[:int(train_ratio*len(unique_labels))]
#     test_labels = unique_labels[int(train_ratio*len(unique_labels)):]

#     # Use tensor operations instead of isin()
#     train_idx = torch.where(torch.isin(labels, train_labels))[0]
#     test_idx = torch.where(torch.isin(labels, test_labels))[0]

#     train_dataset = CLIPDataset(ts_f[train_idx], tx_f[train_idx], labels[train_idx])
#     test_dataset = CLIPDataset(ts_f[test_idx], tx_f[test_idx], labels[test_idx])

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import *

from data_utils.preprocessor import *
from data_utils.augmentor import *
from data_utils.masker import *
from data_utils.describer import *

class VITALDataset(Dataset):
    def __init__(self, ts_features, text_features, labels, targets = None):
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
        self.targets = targets
        self.ts_features = self.ts_features.to(device)
        self.text_features = self.text_features.to(device)
        self.labels = self.labels.to(device)
        if targets is None:
            labels_equal = (self.labels.unsqueeze(0) == self.labels.unsqueeze(1))
            self.targets = labels_equal.float()
        else:
            self.targets = targets.float()
        self.targets = self.targets.to(device)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            idx,
            self.ts_features[idx],
            self.text_features[idx],
            self.labels[idx],
            self.targets[idx]
        )
    def dataloader(self, batch_size=32):
        return DataLoader(self, 
                          batch_size=batch_size, 
                          shuffle=True,
                          num_workers=0,
                          pin_memory=False) # they will already be in GPU
    

class VITAL3DDataset(Dataset):
    def __init__(self, ts_features, text_features_list, labels, targets = None):
        """
        Dataset for VITAL3D model handling multiple text features per time series.
        
        Args:
            ts_features: time series features tensor [N, ts_dim]
            text_features_list: list of text features tensors, each [N, text_dim]
            labels: class labels tensor [N]
        """
        # Convert inputs to tensors if needed
        self.ts_features = torch.FloatTensor(ts_features) if not isinstance(ts_features, torch.Tensor) else ts_features
        self.text_features_list = [
            torch.FloatTensor(text_features) if not isinstance(text_features, torch.Tensor) else text_features
            for text_features in text_features_list
        ]
        self.labels = torch.LongTensor(labels) if not isinstance(labels, torch.Tensor) else labels
        
        # Verify dimensions
        n_samples = len(self.ts_features)
        assert all(len(text_features) == n_samples for text_features in self.text_features_list), \
            "All text features must have the same number of samples as ts_features"
        assert len(self.labels) == n_samples, \
            "Labels must have the same number of samples as features"
        
        self.ts_features = self.ts_features.to(device)
        self.text_features_list = [t.to(device) for t in self.text_features_list]
        self.labels = self.labels.to(device)
        if targets is None:
            labels_equal = (self.labels.unsqueeze(0) == self.labels.unsqueeze(1))
            self.targets = labels_equal.float()
        else:
            self.targets = targets.float()
        self.targets = self.targets.to(device)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (
                idx: sample index,
                ts_features: time series features [ts_dim],
                text_features: list of text features, each [text_dim],
                label: class label,
                target_org: similarity target
            )
        """
        return (
            idx,
            self.ts_features[idx],
            [text_features[idx] for text_features in self.text_features_list],
            self.labels[idx],
            self.targets[idx]
        )
    
    def dataloader(self, batch_size=32, shuffle=True):
        """
        Creates a DataLoader for this dataset.
        
        Args:
            batch_size: int, number of samples per batch
            shuffle: bool, whether to shuffle the data
            num_workers: int, number of worker processes for data loading
        
        Returns:
            DataLoader: PyTorch DataLoader object
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )
    

class TSFeature(Dataset):
    # mean, std can be computed from this object
    # get the normalized / encoded time series features of raw time series
    def __init__(self, ts_df, normalize_mean=None, normalize_std=None, global_norm = True, local_norm = False):
        """Initialize dataset with optional normalization
        
        Args:
            ts_df (pd.DataFrame): Time series dataframe with shape [obs, time]
            normalize_mean (float, optional): Pre-computed mean for normalization
            normalize_std (float, optional): Pre-computed std for normalization
        """
        # Copy the original dataframe
        self.ts_df = ts_df.copy().astype(float)
        data = self.ts_df.values

        # ---- normalize the time series data ----
        # default is raw scaled data
        data_scaled = data 
        # do a global normalization
        if global_norm: 
            if normalize_mean is None:
                self.normalize_mean = data.mean()
            else:
                self.normalize_mean = normalize_mean
            if normalize_std is None:
                self.normalize_std = data.std()
            else:
                self.normalize_std = normalize_std
            data_scaled = (data_scaled - self.normalize_mean) / (self.normalize_std + 1e-8)
        # continue with a local normalization
        if local_norm: 
            obs_mean = np.nanmean(data_scaled, axis=1)
            osb_std = np.nanstd(data_scaled, axis=1)
            osb_std = np.where(osb_std == 0, 1e-8, osb_std)
            data_scaled_T = (data_scaled.T - obs_mean.T) / osb_std.T
            data_scaled = data_scaled_T.T
        
            
        
        # Store normalized data
        self.norm_ts_df = pd.DataFrame(
            data_scaled,
            index=ts_df.index,
            columns=ts_df.columns
        )
        self.norm_ts_df = self.norm_ts_df.astype(float)
        self.features = self.norm_ts_df.values # numpy array shape: [obs, time]
        self.features = torch.tensor(self.features, dtype=torch.float32)

    def __len__(self):
        return len(self. ts_df)
    
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



def get_ts_txt_org(df, text_col = 'text'):
    # raise error if any of the columns are not in the dataframe
    df.columns = df.columns.astype(str)
    required_columns = ['VitalID', 'VitalTime', '1', '300', 'text', 'label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is required but not found in the dataframe.")
    
    df = df.assign(id_time='id_' + df['VitalID'].astype(str) + '_' + df['VitalTime'].astype(str))
    df = df.set_index('id_time')
    df = df.drop(columns=['VitalID', 'VitalTime'])
    
    # get normalized time series dataframe norm_ts_df
    ts_df = df.loc[:,'1':'300']
    # get text list txt_ls
    txt_ls = df.loc[:,text_col].tolist()
    labels = df.loc[:,'label'].tolist() # id of target matrix, indicating observations with the same id.

    return ts_df, txt_ls, labels


def get_features(df, 
                 config_dict,
                 text_col='text'):
    ts_df, txt_ls, labels = get_ts_txt_org(df, text_col = text_col)
    # --- ts_f ---
    ts_f = TSFeature(ts_df,
                    normalize_mean=config_dict['ts_normalize_mean'], 
                    normalize_std=config_dict['ts_normalize_std'], 
                    global_norm = config_dict['ts_global_normalize'], 
                    local_norm = config_dict['ts_local_normalize']).features

    # --- tx_f ---
    tx_f = TXTFeature(txt_ls, 
                      encoder_model_name=config_dict['text_encoder_name']).features

    # --- labels ---
    labels = torch.tensor(labels)
    
    return ts_f, tx_f, labels


def get_features3d(df, 
                 config_dict,
                 text_col_ls=['demo', 'cl_event', 'ts_description']):
    
    ts_df, _, labels = get_ts_txt_org(df)
    ts_f = TSFeature(ts_df, 
                     normalize_mean=config_dict['ts_normalize_mean'],
                     normalize_std=config_dict['ts_normalize_std'],
                     global_norm = config_dict['ts_global_normalize'], 
                     local_norm = config_dict['ts_local_normalize']).features
    labels = torch.tensor(labels)
    tx_f_list = []
    for text_col in text_col_ls:
        _, txt_ls, _ = get_ts_txt_org(df, text_col = text_col)
        tx_f = TXTFeature(txt_ls, 
                          encoder_model_name=config_dict['text_encoder_name']).features
        tx_f_list.append(tx_f)
    return ts_f, tx_f_list, labels



def gen_target(df, 
               cluster_cols = ['label']):
    targets = {}
    for cluster_col in cluster_cols:
        label_mapping = {cat: idx+1 for idx, cat in enumerate(sorted(df[cluster_col].unique()))}
        df['cluster'] = df[cluster_col].map(label_mapping).astype(int)
        labels = torch.tensor(df['cluster'].values).to(device)
        labels_equal = (labels.unsqueeze(0) == labels.unsqueeze(1))
        target = labels_equal.float()
        targets[cluster_col] = target

    # Sum all target matrices element-wise
    target_sum = sum(targets.values())
    # Method 2: Normalize by count
    target_normalized = target_sum / len(cluster_cols)
    return target_normalized


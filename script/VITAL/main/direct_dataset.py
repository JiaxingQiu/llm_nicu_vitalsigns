import numpy as np
from torch.utils.data import Dataset, DataLoader

class TEditDataset(Dataset):
    """
    A dataset class that initializes directly from dataframes without saving to local files.
    """
    def __init__(self, df_train, df_test, df_left, config_dict, split="train"):
        self.split = split
        
        # Build level maps and meta info
        level_maps = {c: self._build_level_map(df_train, c) for c in config_dict["txt2ts_y_cols"]}
        attr_list = [k for k in level_maps]      
        attr_n_ops = [len(v) for v in level_maps.values()]           
        self.attr_n_ops = np.array(attr_n_ops, dtype=int)
        
        # Calculate normalization stats
        train_mean_std = {
            'mean': np.nanmean(df_train[[str(i+1) for i in range(config_dict['seq_length'])]] .values), 
            'std': np.nanstd(df_train[[str(i+1) for i in range(config_dict['seq_length'])]] .values, ddof=0)
        }
        
        # Get arrays for each split
        train_ts, train_attrs_idx = self._get_arrays(df_train, config_dict, level_maps)
        valid_ts, valid_attrs_idx = self._get_arrays(df_test, config_dict, level_maps)
        test_ts, test_attrs_idx = self._get_arrays(df_left, config_dict, level_maps)
        
        # Normalize
        if config_dict["ts_global_normalize"]:
            m, s = train_mean_std["mean"], train_mean_std["std"]
            s = s if s > 0 else 1e-8
        else:
            m, s = 0, 1
        
        train_ts = (train_ts - m) / s
        valid_ts = (valid_ts - m) / s
        test_ts = (test_ts - m) / s
        
        # Set data based on split
        if split == "all":
            self.ts = np.concatenate([train_ts, valid_ts, test_ts], axis=0)
            self.attrs = np.concatenate([train_attrs_idx, valid_attrs_idx, test_attrs_idx], axis=0)
        elif split == "train":
            self.ts = train_ts
            self.attrs = train_attrs_idx
        elif split == "valid":
            self.ts = valid_ts
            self.attrs = valid_attrs_idx
        elif split == "test":
            self.ts = test_ts
            self.attrs = test_attrs_idx
            
        self.time_points = np.arange(self.ts.shape[1])
        print(f"Loaded {split} split – {len(self)} samples.")

    def _build_level_map(self, df_train, col, sort_levels=True):
        """Build mapping from categorical values to indices."""
        levels = df_train[col].dropna().unique()
        if sort_levels:
            levels = sorted(levels)
        level2idx = {lvl: i for i, lvl in enumerate(levels)}
        return level2idx

    def _encode_column(self, df, col, level2idx):
        """Encode categorical column to integer indices."""
        return (
            df[col]
            .map(level2idx)          # map strings → ints
            .fillna(-1)              # unseen level or NaN
            .astype(int)
            .to_numpy()
        )

    def _get_arrays(self, df, config_dict, level_maps):
        """Extract time series and attribute arrays from dataframe."""
        ts = df[[str(i) for i in range(1, config_dict['seq_length'] + 1)]].values
        attrs_idx = []
        for attr_col in config_dict['txt2ts_y_cols']:
            attr_arr = self._encode_column(df, attr_col, level_maps[attr_col])
            attrs_idx.append(attr_arr)
        attrs_idx = np.stack(attrs_idx, axis=1)
        return ts, attrs_idx

    def __len__(self):
        return self.ts.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.ts[idx][..., None],    # (L, 1)
            "attrs": self.attrs[idx],        # (n_attrs,)
            "tp": self.time_points           # (L,)
        }
    
    def get_loader(self, batch_size=128, shuffle=False, num_workers=0):
        """Create a DataLoader for this dataset instance."""
        return DataLoader(self,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers)

# # usage example
# loader = TEditDataset(df_train, df_test, df_left, config_dict, split="test").get_loader(batch_size=128)
# for batch in loader:
#     print(batch["x"].shape, batch["attrs"].shape, batch["tp"].shape)
#     break

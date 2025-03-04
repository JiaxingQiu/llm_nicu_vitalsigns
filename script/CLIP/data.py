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



# --------------------------text input processing----------------------------------
def extract_event1(text):
    """
    Extract the first event description from the text.
    """
    # Find start of Event 1
    start_marker = "\n Event 1: "
    end_marker = "\n Event 2: "
    
    # initialize event1_str
    event1_str = ""

    # Get start position
    start_pos = text.find(start_marker)
    if start_pos == -1:  # If Event 1 not found
        return event1_str
    start_pos += len(start_marker)
    # Get end position
    end_pos = text.find(end_marker)
    if end_pos == -1:  # If Event 2 not found, take until end
        return text[start_pos:].strip()
    
    # Return text between markers
    event1_str = text[start_pos:end_pos].strip()
    
    return event1_str

    
def summarize_brady(text):
    """
    Count events and identify the single event type in the text.
    
    Args:
        text (str): Input text containing event descriptions
        
    Returns:
        str: Summary like "2 Bradycardia90 (heart rate below 90) events happened."
              or "1 Bradycardia90 (heart rate below 90) event happened."
    """
    if not isinstance(text, str):
        return "No Bradycardia events."
    
    # Count events
    event_count = text.count("\n Event")
    if event_count == 0:
        return "No Bradycardia events."
    
    # Find the event name and threshold
    import re
    brady_match = re.search(r'Bradycardia(\d+)', text)

    if brady_match:
        event_name = brady_match.group(0)  # e.g., "Bradycardia90"
        threshold = brady_match.group(1)    # e.g., "90"
        event_word = "event" if event_count == 1 else "events"
        return f"{event_count} {event_name} {event_word} (heart rate below {threshold}) happened."
    else:
        event_word = "event" if event_count == 1 else "events"
        return f"{event_count} Bradycardia {event_word} happened."
    
def summarize_desat(text):
    """
    Count events and identify the single event type in the text.
    
    Args:
        text (str): Input text containing event descriptions
        
    Returns:
        str: Summary like "2 Desaturation90 (spo2 below 90) events happened." or "1 Desaturation90 (spo2 below 90) event happened."
    """
    if not isinstance(text, str):
        return "No Desaturation events."
    
    # Count events
    event_count = text.count("\n Event")
    if event_count == 0:
        return "No Desaturation events."
    
    # Find the event name and threshold
    import re
    desat_match = re.search(r'Desaturation(\d+)', text)

    if desat_match:
        event_name = desat_match.group(0)  # e.g., "Desaturation90"
        threshold = desat_match.group(1)    # e.g., "90"
        event_word = "event" if event_count == 1 else "events"
        return f"{event_count} {event_name} {event_word} (spo2 below {threshold}) happened."
    else:
        event_word = "event" if event_count == 1 else "events"
        return f"{event_count} Desaturation {event_word} happened."

def gen_demo(row,
             ga_bwt=True,
             gre=False, # gender_race_ethnicity
             apgar_mage=False):
    """
    Generate a demographic description string from a single row.
    
    Args:
        row (pd.Series): A row containing demographic information:
            - EGA: Gestational age in weeks
            - BWT: Birth weight in grams
            - Male: 0 for female, 1 for male
            - Black: 0 for white, 1 for black
            - Hispanic: 0 for non-Hispanic, 1 for Hispanic
            - Apgar5: APGAR score at 5 minutes
            - Maternal Age: Mother's age in years
            
    Returns:
        str: Formatted demographic description
    """

    def is_valid_number(value):
        """Check if value is a valid number (not NaN, None, or invalid)"""
        if value is None:
            return False
        if pd.isna(value):  # Catches pandas NA, NaN
            return False
        if np.isnan(value) if isinstance(value, float) else False:  # Catches numpy NaN
            return False
        try:
            float(value)  # Try converting to float
            return True
        except (ValueError, TypeError):
            return False
        
    ga_str = ""
    weight_str = ""
    gender_str = ""
    race_str = ""
    ethnicity_str = ""
    apgar_str = ""
    mother_str = ""

    # Format each component
    if ga_bwt:
        if is_valid_number(row['EGA']): 
            ga_str = f"This infant has gestational age {int(round(row['EGA'], 1))} weeks. "
        if is_valid_number(row['BWT']):
            weight_str = f"Birth weight is {int(round(row['BWT'], 1))} grams. "
    
    if gre:
        gender_str = "This infant is Male " if row['Male'] == 1 else "This infant is Female "
        race_str = "Black " if row['Black'] == 1 else "non-Black "
        ethnicity_str = "Hispanic. " if row['Hispanic'] == 1 else "non-Hispanic. "
    if apgar_mage:
        if is_valid_number(row['Apgar5']):
            apgar_str = f"The Apgar5 scores {int(round(row['Apgar5'], 1))}. "
        if is_valid_number(row['Maternal Age']):
            mother_str = f"Mother is {int(round(row['Maternal Age'], 1))} years old. "
    
    # if all _str are "", return empty string
    if not (ga_str or weight_str or gender_str or race_str or ethnicity_str or apgar_str or mother_str):
        return ""
    # Combine all parts
    return f"{ga_str}{weight_str}{gender_str}{race_str}{ethnicity_str}{apgar_str}{mother_str}"
    
    

def gen_cl_event(row,
                 die7d=True,
                 fio2=False):
    """
    Generate a clinical event description string from a single row.
    
    Args:
        - row (pd.Series): A row containing clinical information.
        - die7d (bool): if ture, return death in 7 days.
        - fio2 (bool): if ture, return the FiO2 event.
    """
    die7d_str = ""
    fio2_str = ""

    if die7d:
        die7d_str = 'This infant will die in 7 days. ' if row['Died'] == 1 else 'This infant will survive. '
    
    if fio2:
        fio2_str = f"The FIO2 is {int(round(row['FIO2']))}."
    
    return f"{die7d_str}{fio2_str}"


def gen_ts_event(row,
                 sumb=True, # sum_brady
                 sumd=False, # sum_desat
                 simple=True,
                 full=False,
                 event1=False):
    """
    Generate a time series event description string from a single row.

    Args:
        - row (pd.Series): A row containing time series information.
        - sumb (bool): if ture, return summarize of count and definition of Bradycardia event.
        - sumd (bool): if ture, return summarize of count and definition of Desaturation event.
        - simple (bool): if ture, return simplified time series event description.
        - full (bool): if ture, return full event description.
        - event1 (bool): if ture, return the first event description.
    """

    sum_str = ""
    simple_str = ""
    full_str = ""
    event1_str = ""

    if sumb:
        sum_str = summarize_brady(row['event_description'])
        sum_str = sum_str + " "
    if sumd:
        sum_str = summarize_desat(row['event_description'])
        sum_str = sum_str + " "
    if simple:
        x = row['event_description']
        simple_str = '\n'.join(x.split('\n')[1:]) if isinstance(x, str) else x
        simple_str = simple_str + " "
    if full:
        full_str = row['event_description']
        full_str = full_str + " "
    if event1:
        event1_str = extract_event1(row['event_description'])
        event1_str = event1_str + " "
    
    return f"{sum_str}{simple_str}{full_str}{event1_str}"


def gen_text_input_column(df, text_config):
    # demographic description
    df['demo'] = df.apply(gen_demo, axis=1, **text_config['demo'])
    # clinical events
    df['cl_event'] = df.apply(gen_cl_event, axis=1, **text_config['cl'])
    # time series events
    df['ts_event'] = df.apply(gen_ts_event, axis=1, **text_config['ts'])
    
    df['text'] = df['cl_event'] + ' ' + df['demo'] + ' ' + df['ts_event']
    print(df['text'][0])

    return df
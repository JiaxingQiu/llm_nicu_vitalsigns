import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary as nn_summary
from eval import *
from encoder import *



class GeneralBinaryClassifier(nn.Module):
    def __init__(self, encoder):
        """
        General binary classifier that accepts any encoder architecture.
        
        Args:
            encoder: Encoder module (nn.Module) that outputs fixed dimension
        """
        super().__init__()
        
        # Set encoder
        self.encoder = encoder
        
        # Get encoder output dimension
        self.encoder.eval()  # Set to eval mode temporarily
        with torch.no_grad():
            dummy = torch.zeros(2, 300)  # Assuming time series length of 300
            encoder_out_dim = self.encoder(dummy).shape[1]
        self.encoder.train()  # Set back to train mode
        
        # Classifier (outputs logits)
        self.classifier = nn.Linear(encoder_out_dim, 2)
        # Move model to device
        self.device = device
        self.to(device)
        print(nn_summary(self))
    
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)
    
    def get_features(self, x):
        """Get encoded features without classification"""
        return self.encoder(x)
    
    def get_probs(self, x):
        """Get probability predictions"""
        return torch.softmax(self(x), dim=1)


class TSSEDataset(Dataset):

    def __init__(self, ts, y):
        # y is the obs outcome, x is the obs by 300 time points
        if not isinstance(ts, torch.Tensor):
            ts = torch.FloatTensor(ts)
        if not isinstance(y, torch.Tensor):
            y = torch.LongTensor(y)
        assert len(ts) == len(y), "All inputs must have the same length"
        self.ts = ts
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            self.ts[idx],
            self.y[idx]
        )
    def dataloader(self, batch_size=32):
        return DataLoader(self, 
                          batch_size=batch_size, 
                          shuffle=False)


def prepare_basedata(df, y, batch_size = 128, normalize=True):

    data = df[[f'{i}' for i in range(1, 301)]].values
    if normalize:
        obs_mean = np.nanmean(data, axis=1)
        osb_std = np.nanstd(data, axis=1)
        osb_std = np.where(osb_std == 0, 1e-8, osb_std)
        data_t = (data.T - obs_mean.T) / osb_std.T
        data_scaled = data_t.T
    else:
        data_scaled = data

    data_scaled = data_scaled.astype(float)
    dataset = TSSEDataset(data_scaled, y)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def train_binary_classifier(model, train_dataloader, test_dataloader, optimizer, scheduler, num_epochs, device):
    """
    Train binary classifier with one-hot encoded outputs (batch_y shape: [obs, 2]).
    """
    train_losses = []
    test_losses = []
    train_metrics_list = []
    test_metrics_list = []

    # Keep track of best model
    best_loss = float('inf')
    best_model_state = None
    
    # Cross entropy loss for one-hot encoded outputs
    criterion = nn.CrossEntropyLoss()
    
    try:
        for epoch in range(num_epochs):
            # Training
            model.train()
            epoch_train_losses = []
            for batch_x, batch_y in train_dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Store loss
                epoch_train_losses.append(loss.item())
            
            # Calculate average training loss
            train_loss = np.mean(epoch_train_losses)
            
            # Testing
            model.eval()
            epoch_test_losses = []
            
            with torch.no_grad():
                for batch_x, batch_y in test_dataloader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    # Forward pass
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    
                    # Store loss
                    epoch_test_losses.append(loss.item())
            
            # Calculate average test loss
            test_loss = np.mean(epoch_test_losses)
            
            # Store losses
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            # Get evaluation metrics
            train_metrics, test_metrics = eval_binary_classifier(model, train_dataloader, test_dataloader, device)
            train_metrics_list.append(train_metrics)
            test_metrics_list.append(test_metrics)

            # Update learning rate based on test loss
            scheduler.step(test_loss)
            
            # Save best model based on test loss
            if test_loss < best_loss:
                best_loss = test_loss
                best_model_state = model.state_dict().copy()
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            # Print progress
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(
                f'\tTrain Loss: {train_loss:.4f}\t'
                f'Test Loss: {test_loss:.4f}\n'
                f'\tTrain F1: {train_metrics["per_class"][0]["f1"]:.4f}\t'  # Changed to index 1 for death class
                f'Test F1: {test_metrics["per_class"][0]["f1"]:.4f}\n'
                f'\tTrain AUROC: {train_metrics["auroc_per_class"][0]:.4f}\t'  # Changed to index 1 for death class
                f'Test AUROC: {test_metrics["auroc_per_class"][0]:.4f}\n'
                f'\tTrain AUPRC: {train_metrics["auprc_per_class"][0]:.4f}\t'  # Changed to index 1 for death class
                f'Test AUPRC: {test_metrics["auprc_per_class"][0]:.4f}\n'
                f'\tLearning Rate: {current_lr:.9f}'
            )
            # print(test_aucs(model, train_dataloader, test_dataloader, device)) # check
            

            # Early stopping if learning rate becomes too small
            if current_lr <= scheduler.min_lrs[0]:
                print("Learning rate too small. Stopping training.")
                break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
    finally:
        # Load best model if we found one
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        eval_dict = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_evals': train_metrics_list,
            'test_evals': test_metrics_list
        }
        
        return eval_dict, train_losses, test_losses
    

def eval_binary_classifier(model, train_dataloader, test_dataloader, device):
    model.eval()
    # Get evaluation metrics
    train_labels = []
    train_preds = []
    test_labels = []
    test_preds = []
    
    # Collect predictions and labels
    with torch.no_grad():
        for batch_x, batch_y in train_dataloader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            pred = torch.softmax(logits, dim=1)
            train_labels.extend(batch_y.detach().cpu().numpy())
            train_preds.extend(pred.detach().cpu().numpy())
        
        for batch_x, batch_y in test_dataloader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            pred = torch.softmax(logits, dim=1)
            test_labels.extend(batch_y.detach().cpu().numpy())
            test_preds.extend(pred.detach().cpu().numpy())

    # Convert to numpy arrays
    train_labels = np.array(train_labels)
    train_preds = np.array(train_preds)
    test_labels = np.array(test_labels)
    test_preds = np.array(test_preds)

    # Get evaluation metrics
    train_metrics = get_eval_metrics(train_labels, train_preds)
    test_metrics = get_eval_metrics(test_labels, test_preds)
    return train_metrics, test_metrics


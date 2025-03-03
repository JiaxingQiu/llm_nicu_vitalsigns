import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary as nn_summary
from config import *
from data import *
from evals import *
from tqdm import tqdm

class CLIPModel(nn.Module):
    def __init__(self, ts_dim, text_dim, projection_dim=128, temperature=0.01):
        super().__init__()
        self.W_ts = nn.Sequential(
            nn.Linear(ts_dim, 128, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(128, projection_dim, bias=True)
            # nn.Linear(ts_dim, projection_dim, bias=False)
        )
        # change to sequential
        self.W_t = nn.Sequential(
            nn.Linear(text_dim, 128, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256, bias=True),
            nn.LeakyReLU(0.2),
            # nn.Linear(256, 256, bias=True),
            # nn.LeakyReLU(0.2),
            nn.Linear(256, 256, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(128, projection_dim, bias=True)
            # nn.Linear(text_dim, projection_dim, bias=False)
        )
        
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        print(nn_summary(self))
        
    def forward(self, ts_features, text_features):
        # Project and normalize
        ts_embedded = F.normalize(self.W_ts(ts_features), dim=1)    # ts_embedded is a matrix of size (obs, projection_dim)
        text_embedded = F.normalize(self.W_t(text_features), dim=1)
        
        # Calculate similarity # logits is a matrix of size (obs, obs)
        logits = torch.matmul(ts_embedded, text_embedded.T) * torch.exp(self.temperature)
        # print(logits)
        return logits, ts_embedded, text_embedded
    


def cross_entropy(preds, targets):
    # Compute cross entropy loss manually
    batch_size = preds.shape[0]
    exp_preds = torch.exp(preds)
    softmax_probs = exp_preds / exp_preds.sum(dim=1, keepdim=True) # sum over the columns, keep the rows
    log_probs = torch.log(softmax_probs)
    loss = -torch.sum(targets * log_probs) / batch_size
    return loss

def cross_entropy2(preds, targets):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1).mean()
    return loss



def loss_block_diagonal(logits, labels): # symmetric cross-entropy loss calculation
    # labels = labels.detach().cpu().numpy()
    # labels = pd.factorize(labels)[0]
    # labels = torch.tensor(labels, device=logits.device)

    batch_size = logits.shape[0]
    # build block-diagonal target matrix based on labels
    targets = torch.zeros((batch_size, batch_size), device=logits.device)
    for i in range(batch_size):
        for j in range(batch_size):
            if labels[i] == labels[j]:
                targets[i,j] = 1
    loss_ts = cross_entropy(logits, targets)
    loss_tx = cross_entropy(logits.T, targets.T)
    loss = (loss_ts + loss_tx) / 2
    return loss

def loss_similarity(logits, ts_embedded, text_embedded):
    batch_size = logits.shape[0]
    # Compute similarities
    with torch.no_grad():  # Detach the similarity computation
        ts_similarity = ts_embedded @ ts_embedded.T
        texts_similarity = text_embedded @ text_embedded.T
        # calculate targets using average similarity
        targets = F.softmax( (ts_similarity + texts_similarity)/2,  dim=-1 )
    # assert targets is a matrix of size (batch_size, batch_size)
    assert targets.shape == (batch_size, batch_size)
    
    loss_ts = cross_entropy(logits, targets)
    loss_tx = cross_entropy(logits.T, targets.T)
    return (loss_ts + loss_tx) / 2


def train_epoch(model, train_dataloader, optimizer, device, loss_type='block_diagonal'):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (ts_features, text_features, labels) in enumerate(train_dataloader):
        
        ts_features = ts_features.to(device)
        text_features = text_features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, ts_embedded, text_embedded = model(ts_features, text_features)

        # Choose one loss function
        if loss_type == 'block_diagonal':
            loss = loss_block_diagonal(logits, labels)
        elif loss_type == 'similarity':
            loss = loss_similarity(logits, ts_embedded, text_embedded)
        loss.backward(retain_graph=True)
        optimizer.step()


        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def test_epoch(model, test_dataloader, device, loss_type='block_diagonal'):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (ts_features, text_features, labels) in enumerate(test_dataloader):
            ts_features = ts_features.to(device)
            text_features = text_features.to(device)
            labels = labels.to(device)
            
            logits, ts_embedded, text_embedded = model(ts_features, text_features)
            if loss_type == 'block_diagonal':
                loss = loss_block_diagonal(logits, labels)
            elif loss_type == 'similarity':
                loss = loss_similarity(logits, ts_embedded, text_embedded)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches



def train(model, train_dataloader, test_dataloader, optimizer, scheduler, num_epochs, device, loss_type='block_diagonal'):
    train_losses = []
    test_losses = []
    
    # Keep track of best model
    best_loss = float('inf')
    best_model_state = None
    
    try:
        for epoch in range(num_epochs):#tqdm()
            # Train for one epoch
            train_loss = train_epoch(model, train_dataloader, optimizer, device, loss_type)
            
            # Test for one epoch
            test_loss = test_epoch(model, test_dataloader, device, loss_type)
            
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
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Load best model if we found one
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return train_losses, test_losses




def eval_model(model, y_true, ts_df, txt_ls, ts_encoder_name, text_encoder_name):
    _, y_prob = get_logit(model, ts_df, txt_ls, ts_encoder_name, text_encoder_name)
    eval_metrics = evaluate_predictions(y_true, y_prob, txt_ls)
    return eval_metrics

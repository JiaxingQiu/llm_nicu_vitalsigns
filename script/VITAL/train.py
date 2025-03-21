import torch
import torch.nn.functional as F
import numpy as np

 
def compute_clip_loss(logits, labels): 

    batch_size = logits.shape[0]
    # build block-diagonal target matrix based on labels
    targets = torch.zeros((batch_size, batch_size), device=logits.device)
    for i in range(batch_size):
        for j in range(batch_size):
            if labels[i] == labels[j]:
                targets[i,j] = 1
    loss_ts = cross_entropy(logits, targets)
    loss_tx = cross_entropy(logits.T, targets.T)
    clip_loss = (loss_ts + loss_tx) / 2
    return clip_loss

def cross_entropy(preds, targets):
    # Compute cross entropy loss manually
    batch_size = preds.shape[0]
    exp_preds = torch.exp(preds)
    softmax_probs = exp_preds / exp_preds.sum(dim=1, keepdim=True) # sum over the columns, keep the rows
    log_probs = torch.log(softmax_probs)
    loss = -torch.sum(targets * log_probs) / batch_size
    return loss



class KLAnnealer:
    def __init__(self, start, end, epochs):
        self.start = start
        self.end = min(end,1)
        self.epochs = epochs
        
    def get_beta(self, epoch):
        beta = min(self.end, self.start + (self.end - self.start) * epoch / self.epochs)
        # beta = np.exp(beta)-1
        print(f'beta: {beta}')
        return beta

def compute_vae_loss(ts, ts_hat, mean, log_var, beta=1.0):
    reconstruction_loss = F.mse_loss(ts_hat, ts, reduction='sum') / ts.size(0)
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1).mean()
    # reconstruction_loss = F.mse_loss(ts_hat, ts)
    # kl_loss = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    vae_loss = reconstruction_loss + beta * kl_loss
    return vae_loss

  
        
def train_epoch(model, train_dataloader, optimizer, train_type='joint', alpha=1.0, beta=0.1): 
    # alpha controls the balance between vae loss and clip loss
    # beta controls the balance between reconstruction loss and kl loss
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (ts, text_features, labels) in enumerate(train_dataloader):
        
        # Clear gradients
        optimizer.zero_grad()

        # forward pass  
        logits, ts_hat, mean, log_var = model(ts, text_features)

        # compute losses
        if train_type == 'joint':
            clip_loss = compute_clip_loss(logits, labels)
            vae_loss = compute_vae_loss(ts, ts_hat, mean, log_var, beta)
            loss = clip_loss + alpha * vae_loss
        elif train_type == 'vae':
            loss = compute_vae_loss(ts, ts_hat, mean, log_var, beta)
        elif train_type == 'clip':
            loss = compute_clip_loss(logits, labels)
        else:
            raise ValueError(f"Invalid model type: {train_type}")
        
        # backward pass
        loss.backward(retain_graph=True)

        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()

        # update total loss and number of batches
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def test_epoch(model, test_dataloader, train_type='joint', alpha=1.0, beta=0.1):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (ts, text_features, labels) in enumerate(test_dataloader):
            
            logits, ts_hat, mean, log_var = model(ts, text_features)
            
            # compute losses
            if train_type == 'joint':
                clip_loss = compute_clip_loss(logits, labels)
                print(f'batch {batch_idx} test clip_loss: {clip_loss.item()}')
                vae_loss = compute_vae_loss(ts, ts_hat, mean, log_var, beta)
                print(f'batch {batch_idx} test vae_loss: {vae_loss.item()}')
                loss = clip_loss + alpha * vae_loss
            elif train_type == 'vae':
                loss = compute_vae_loss(ts, ts_hat, mean, log_var, beta)
            elif train_type == 'clip':
                loss = compute_clip_loss(logits, labels)
            else:
                raise ValueError(f"Invalid model type: {train_type}")
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_vital(model, train_dataloader, test_dataloader, optimizer, scheduler, kl_annealer, num_epochs, train_type='joint', alpha=1.0):
    train_losses = []
    test_losses = []
    

    # Keep track of best model
    best_loss = float('inf')
    best_model_state = None
    try:
        for epoch in range(num_epochs):
            if epoch < int(0.2 * num_epochs): # first 20% of num_epochs
                scheduler.patience = min(int(0.1*num_epochs), 100)
            elif epoch < int(0.5 * num_epochs): # second 30% of num_epochs
                scheduler.patience = min(int(0.05*num_epochs), 50)
            else:
                scheduler.patience = min(int(0.01*num_epochs), 10)
            

            beta = kl_annealer.get_beta(epoch)

            # Train for one epoch
            train_loss = train_epoch(model, train_dataloader, optimizer, train_type, alpha, beta)
            
            # Test for one epoch
            test_loss = test_epoch(model, test_dataloader, train_type, alpha, beta)
            
            # Store losses
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            # Update learning rate based on test loss
            average_loss = (train_loss + test_loss) / 2
            # scheduler.step(average_loss)
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
        raise e
    finally:
        # Load best model if we found one
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return train_losses, test_losses




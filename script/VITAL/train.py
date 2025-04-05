import torch
import torch.nn.functional as F

 
def compute_clip_loss(logits, labels, 
                      loss_type = 'block_diagonal', ts_embedded=None, text_embedded=None): # embedded or features
    if loss_type == 'block_diagonal':
        # # build block-diagonal target matrix based on labels
        # batch_size = logits.shape[0]
        # targets = torch.zeros((batch_size, batch_size), device=logits.device)
        # for i in range(batch_size):
        #     for j in range(batch_size):
        #         if labels[i] == labels[j]:
        #             targets[i,j] = 1
        # Vectorized operation on GPU!!
        labels_equal = (labels.unsqueeze(0) == labels.unsqueeze(1))
        targets = labels_equal.float()
        loss_ts = cross_entropy(logits, targets)
        loss_tx = cross_entropy(logits.T, targets.T)
        clip_loss = (loss_ts + loss_tx) / 2
    
    if loss_type == 'similarity':
        batch_size = logits.shape[0]
        with torch.no_grad():  # Detach the similarity computation
            targets = get_similarity_target(ts_embedded, text_embedded)
        assert targets.shape == (batch_size, batch_size)
        loss_ts = cross_entropy(logits, targets)
        loss_tx = cross_entropy(logits.T, targets.T)
        clip_loss = (loss_ts + loss_tx) / 2

    return clip_loss

def cross_entropy(preds, targets):
    # Compute cross entropy loss manually
    batch_size = preds.shape[0]
    exp_preds = torch.exp(preds)
    softmax_probs = exp_preds / exp_preds.sum(dim=1, keepdim=True) # sum over the columns, keep the rows
    log_probs = torch.log(softmax_probs + 1e-16)
    loss = -torch.sum(targets * log_probs) / batch_size
    return loss

def get_similarity_target(ts_embedded, text_embedded):
    ts_embedded = F.normalize(ts_embedded, p=2, dim=1)  # dim=1 for row-wise normalization
    text_embedded = F.normalize(text_embedded, p=2, dim=1)
    ts_similarity = torch.matmul(ts_embedded, ts_embedded.T) # cosine similarity
    texts_similarity = torch.matmul(text_embedded, text_embedded.T) # cosine similarity
    target = (ts_similarity + texts_similarity)/2
    # # scale each rwo to sum up to 1
    # # targets = F.softmax(targets, dim=-1)
    # # scale each row to [0,1]
    # row_min = targets.min(dim=1, keepdim=True)[0]
    # row_max = targets.max(dim=1, keepdim=True)[0]
    # targets = (targets - row_min) / (row_max - row_min)
    # targets = (targets + targets.T)/2  # re-symmetrize again after scaling
    return target

class KLAnnealer:
    def __init__(self, start, end, epochs):
        self.start = start
        self.end = min(end,1)
        self.epochs = epochs
        
    def get_beta(self, epoch):
        beta = min(self.end, self.start + (self.end - self.start) * epoch / self.epochs)
        # beta = np.exp(beta)-1
        return beta

def compute_vae_loss(ts, ts_hat, mean, log_var, beta=1.0):
    reconstruction_loss = F.mse_loss(ts_hat, ts, reduction='sum') / ts.size(0)
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1).mean()
    vae_loss = reconstruction_loss + beta * kl_loss
    # print(f'reconstruction: {reconstruction_loss.item()}, kl: {kl_loss.item()}')
    return vae_loss

  
def compute_loss(model, ts, text_features, labels, train_type='joint', alpha=1.0, beta=0.1):
    # initialized to return
    loss = 0
    clip_loss = 0
    vae_loss = 0

    logits, ts_hat, mean, log_var = model(ts, text_features)
    # compute losses
    if train_type == 'joint':
        clip_loss = compute_clip_loss(logits, labels)
        vae_loss = compute_vae_loss(ts, ts_hat, mean, log_var, beta)
        loss = clip_loss + alpha * vae_loss
    elif train_type == 'vae':
        loss = compute_vae_loss(ts, ts_hat, mean, log_var, beta)
        loss = alpha * loss
    elif train_type == 'clip':
        loss = compute_clip_loss(logits, labels)
    else:
        raise ValueError(f"Invalid model type: {train_type}")
    return loss, clip_loss, vae_loss



def train_epoch(model, train_dataloader, optimizer, train_type='joint', alpha=1.0, beta=0.1): 
    # alpha controls the balance between vae loss and clip loss
    # beta controls the balance between reconstruction loss and kl loss
    model.train()
    total_loss = 0
    total_clip_loss = 0 # only updated for joint training
    total_vae_loss = 0 # only updated for joint training
    num_batches = 0
    
    for _, (ts, text_features, labels) in enumerate(train_dataloader):
        
        loss, clip_loss, vae_loss = compute_loss(model, ts, text_features, labels, train_type, alpha, beta)

        # Clear gradients
        optimizer.zero_grad()
        # backward pass
        loss.backward(retain_graph=True)
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Update weights
        optimizer.step()

        # update total loss and number of batches
        total_loss += loss.item()
        if train_type == 'joint':
            total_clip_loss += clip_loss.item()
            total_vae_loss += vae_loss.item()
        
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_clip_loss = total_clip_loss / num_batches
    avg_vae_loss = total_vae_loss / num_batches
    
    return avg_loss, avg_clip_loss, avg_vae_loss



def test_epoch(model, test_dataloader, train_type='joint', alpha=1.0, beta=0.1):
    model.eval()
    total_loss = 0
    total_clip_loss = 0 # only updated for joint training
    total_vae_loss = 0 # only updated for joint training
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (ts, text_features, labels) in enumerate(test_dataloader):
            loss, clip_loss, vae_loss = compute_loss(model, ts, text_features, labels, train_type, alpha, beta)
            
            total_loss += loss.item()
            if train_type == 'joint':
                total_clip_loss += clip_loss.item()
                total_vae_loss += vae_loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_clip_loss = total_clip_loss / num_batches
    avg_vae_loss = total_vae_loss / num_batches
    
    return avg_loss, avg_clip_loss, avg_vae_loss


def train_vital(model, train_dataloader, test_dataloader, optimizer, scheduler, kl_annealer, num_epochs, train_type='joint', alpha=1.0):
    train_losses = []
    test_losses = []
    

    # Keep track of best model
    best_loss = float('inf')
    best_model_state = None
    try:
        for epoch in range(num_epochs):

            beta = kl_annealer.get_beta(epoch)

            # Train for one epoch
            train_loss, train_clip_loss, train_vae_loss = train_epoch(model, train_dataloader, optimizer, train_type, alpha, beta)
            
            # Test for one epoch
            test_loss, test_clip_loss, test_vae_loss = test_epoch(model, test_dataloader, train_type, alpha, beta)
            
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
            if epoch % 10 == 0:
                if train_type == 'joint':
                    print(f'Epoch [{epoch+1}/{num_epochs}]')
                    print(f'\tTraining Loss: {train_loss:.6f} (clip: {train_clip_loss:.6f}, vae: {train_vae_loss:.6f})')
                    print(f'\tTesting Loss: {test_loss:.6f} (clip: {test_clip_loss:.6f}, vae: {test_vae_loss:.6f})')
                    print(f'\tLearning Rate: {current_lr:.9f}')
                    print(f'beta: {beta}')
                else:
                    print(f'Epoch [{epoch+1}/{num_epochs}]')
                    print(f'\tTraining Loss: {train_loss:.6f}')
                    print(f'\tTesting Loss: {test_loss:.6f}')
                    print(f'\tLearning Rate: {current_lr:.9f}')
                    print(f'beta: {beta}')
            
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




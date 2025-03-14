from config import *
from data import *
from clip import *
import torch


class EvalInputs:
    def __init__(self, 
                 df_new,
                 ts_encoder_name, 
                 text_encoder_name, 
                 ts_normalize, 
                 ts_encode,
                 y_true_cols = ['true1', 'true2'],
                 y_pred_cols = ['text1', 'text2'],
                 y_pred_cols_ls = [['demo', 'text1', 'ts_description'], 
                                   ['demo', 'text2', 'ts_description']],
                 clip_type="2d"):
        # true1 column is one-hot indicator of true text of first level of outcome, "this infant will die in 7 days"
        # true2 column is one-hot indicator of true text of second level of outcome, "this infant will survive"
        # text1 column is text of first level of predicted outcome, "this infant will die in 7 days"
        # text2 column is text of second level of predicted outcome, "this infant will survive"

        self.y_true = torch.tensor(df_new[y_true_cols].values)
        if clip_type == "2d":
            self.ts_f_mat, self.tx_f_mat_ls, _ = get_features3d(df_new,
                                                                ts_encoder_name, 
                                                                text_encoder_name, 
                                                                ts_normalize, 
                                                                ts_encode,
                                                                text_col_ls=y_pred_cols) # text1 is a column of "will die", text2 is a column of "will survive"
        elif clip_type == "3d":
            tx_f_mat_ls_ls = []
            for y_pred_cols in y_pred_cols_ls:
                # replace cl_event with text1 / text2 in tx_df_mat_ls
                ts_f_mat, tx_f_mat_ls, _ = get_features3d(df_new,
                                                          ts_encoder_name, 
                                                          text_encoder_name, 
                                                          ts_normalize, 
                                                          ts_encode,
                                                          text_col_ls=y_pred_cols) # text1 is a column of "will die", text2 is a column of "will survive"
                tx_f_mat_ls_ls.append(tx_f_mat_ls)
            self.tx_f_mat_ls_ls = tx_f_mat_ls_ls
            self.ts_f_mat = ts_f_mat
     
            
        

@torch.no_grad() 
def eval_clip(model, 
              eval_inputs,
              return_probs=False):
    
    model.eval()
    y_true = eval_inputs.y_true
    ts_f_mat = eval_inputs.ts_f_mat
    tx_f_mat_ls = eval_inputs.tx_f_mat_ls
    
    model = model.to(device)
    ts_f_mat = ts_f_mat.to(device)
    tx_f_mat_ls = [tx_f_mat.to(device) for tx_f_mat in tx_f_mat_ls]
    _, y_prob = get_logit(model, ts_f_mat, tx_f_mat_ls)
    y_true = y_true.to(device) # y_true is a tensor of size (obs, num_classes)
    y_prob = y_prob.to(device) # y_prob is a tensor of size (obs, num_classes)
    eval_metrics = get_eval_metrics(y_true, y_prob)
    if return_probs:
        return eval_metrics, y_prob
    else:
        return eval_metrics



@torch.no_grad() 
def get_logit(model, 
              ts_f_mat, # ts tensor engineered by TSFeature
              tx_f_mat_ls # txt tensor engineered by TXTFeature 
              ):
    
    model.eval()
    # ts_f_mat = TSFeature(ts_df, encoder_model_name=ts_encoder_name, normalize=ts_normalize, encode_ts=ts_encode).features
    # tx_f_ls = TXTFeature(txt_ls, encoder_model_name=text_encoder_name).features   

    # calculate the logits for all observations and outcomes, one outcome all observations per each 
    obs_ys_logits = []
    # # for each outcome, get the logits for all observations
    # for tx_f in tx_f_ls:
    #     tx_f = tx_f.reshape(1, -1)
    #     tx_f_mat = torch.cat([tx_f] * ts_f_mat.shape[0], dim=0) # shape = (obs, dim_tx_f)
    for tx_f_mat in tx_f_mat_ls: # shape = (obs, dim_tx_f)
        logit, _, _ = model(ts_f_mat, tx_f_mat)
        # keep the diagonal of logit
        obs_logits = torch.diag(logit)
        obs_logits = obs_logits.reshape(-1, 1)
        obs_ys_logits.append(obs_logits)

    # concat by columns
    obs_ys_logits = torch.cat(obs_ys_logits, dim=1)
    exp_preds = torch.exp(obs_ys_logits)
    softmax_probs = exp_preds / exp_preds.sum(dim=1, keepdim=True)

    return obs_ys_logits, softmax_probs


@torch.no_grad() 
def eval_clip3d(model, # model of CLIP3DModel
                eval_inputs,
                return_probs=False):
    # true1 column is one-hot indicator of true text of first level of outcome, "this infant will die in 7 days"
    # true2 column is one-hot indicator of true text of second level of outcome, "this infant will survive"
    # text1 column is text of first level of predicted outcome, "this infant will die in 7 days"
    # text2 column is text of second level of predicted outcome, "this infant will survive"
    model.eval()
    y_true = eval_inputs.y_true
    ts_f_mat = eval_inputs.ts_f_mat
    tx_f_mat_ls_ls = eval_inputs.tx_f_mat_ls_ls
    obs_ys_logits = []
    for tx_f_mat_ls in tx_f_mat_ls_ls:
        ts_f_mat = ts_f_mat.to(device)
        tx_f_mat_ls = [tx_f_mat.to(device) for tx_f_mat in tx_f_mat_ls]
        logits, _, _ = model(ts_f_mat, tx_f_mat_ls)
        obs_logits = torch.diag(logits)
        obs_logits = obs_logits.reshape(-1, 1)
        obs_ys_logits.append(obs_logits)
    
    # concat by columns
    obs_ys_logits = torch.cat(obs_ys_logits, dim=1)
    exp_preds = torch.exp(obs_ys_logits)
    y_prob = exp_preds / exp_preds.sum(dim=1, keepdim=True)
    
    y_true = y_true.to(device) # y_true is a tensor of size (obs, num_classes)
    y_prob = y_prob.to(device) # y_prob is a tensor of size (obs, num_classes)
    eval_metrics = get_eval_metrics(y_true, y_prob)

    if return_probs:
        return eval_metrics, y_prob
    else:
        return eval_metrics






# @torch.no_grad() 
# def get_logit1(model, 
#               ts_f_mat, # ts tensor engineered by TSFeature
#               tx_f_ls # txt tensor engineered by TXTFeature 
#               ):
    
#     model.eval()
#     # ts_f_mat = TSFeature(ts_df, encoder_model_name=ts_encoder_name).features
#     # tx_f_ls = TXTFeature(txt_ls, encoder_model_name=text_encoder_name).features   

#     # calculate the logits for all observations and outcomes, one by one
#     obs_ys_logits = []
#     for i in range(ts_f_mat.shape[0]):
#         ts_f = ts_f_mat[i,:]
#         ts_f = ts_f.reshape(1, -1)
#         logits = [] 
#         for tx_f in tx_f_ls:
#             tx_f = tx_f.reshape(1, -1)
#             with torch.no_grad():
#                 logit, _, _ = model(ts_f, tx_f) # logit is a tensor of size (1, 1)
#                 logits.append(logit) 
#         logits = torch.cat(logits, dim=1)
#         obs_ys_logits.append(logits)
#     obs_ys_logits = torch.cat(obs_ys_logits, dim=0)
#     exp_preds = torch.exp(obs_ys_logits)
#     softmax_probs = exp_preds / exp_preds.sum(dim=1, keepdim=True)

#     return obs_ys_logits, softmax_probs




def get_eval_metrics(y_true, y_prob):
    """
    Evaluate multiclass classification predictions with multiple metrics.
    
    Args:
        y_true: Ground truth labels (tensor of size [obs, num_classes])
        y_prob: Predicted probabilities (tensor of size [obs, num_classes])
        class_names: Optional list of class names for display
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, average_precision_score
    
    # Convert tensors to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_prob):
        y_prob = y_prob.detach().cpu().numpy()
    
    # remove rows with nan in y_prob
    y_true = y_true[~np.isnan(y_prob).any(axis=1)]
    y_prob = y_prob[~np.isnan(y_prob).any(axis=1)]

    # Assert shapes match
    assert y_true.shape == y_prob.shape, f"Shape mismatch: y_true {y_true.shape} != y_prob {y_prob.shape}"
    
    
    # Get predicted class labels
    y_pred_labels = np.argmax(y_prob, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    
    # Calculate metrics
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true_labels, y_pred_labels)
    
    # Precision, Recall, F1 
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average='macro', zero_division=0 
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average='micro', zero_division=0 
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average='weighted', zero_division=0 
    )
    
    
    metrics.update({
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro
    })
    # Per-class metrics
    class_metrics = {}
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average=None, zero_division=0
    )

    for i in range(y_true.shape[1]):
        class_metrics[i] = {  # Use index directly as key
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }
    metrics['per_class'] = class_metrics
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    metrics['confusion_matrix'] = cm
    
    # ROC AUC
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        auc_scores.append(auc)
    metrics['auroc_per_class'] = auc_scores
    metrics['auroc_macro'] = roc_auc_score(y_true, y_prob, average='macro')
    metrics['auroc_weighted'] = roc_auc_score(y_true, y_prob, average='weighted')
    
    
    # PRC AUC
    prc_scores = []
    for i in range(y_true.shape[1]):
        auc = average_precision_score(y_true[:, i], y_prob[:, i])
        prc_scores.append(auc)
    metrics['auprc_per_class'] = prc_scores
    metrics['auprc_macro'] = average_precision_score(y_true, y_prob, average='macro')
    metrics['auprc_weighted'] = average_precision_score(y_true, y_prob, average='weighted')
    
    return metrics


# def calculate_f1_precision_recall_from_cm(confusion_matrix):
#     # Extract values from confusion matrix
#     tp, fn = confusion_matrix[0]
#     fp, tn = confusion_matrix[1]
    
#     # Calculate metrics
#     precision = tp / (tp + fp) if (tp + fp) != 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) != 0 else 0
#     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
#     return {'precision': precision, 'recall': recall, 'f1': f1}



def eng_eval_metrics(eval_dict, plot=True, binary=False, pos_class_index=0, plot_confusion_matrices=False):
    import matplotlib.pyplot as plt

    train_losses = eval_dict['train_losses']
    test_losses = eval_dict['test_losses']
    train_eval_metrics_list = eval_dict['train_evals']
    test_eval_metrics_list = eval_dict['test_evals']
    confusion_matrices_train = [eval_metrics['confusion_matrix'] for eval_metrics in train_eval_metrics_list]
    confusion_matrices_test = [eval_metrics['confusion_matrix'] for eval_metrics in test_eval_metrics_list]
    saves = range(1, len(train_eval_metrics_list) + 1)

    if binary:
        train_f1 = [eval_metrics['per_class'][pos_class_index]['f1'] for eval_metrics in train_eval_metrics_list]
        train_precision = [eval_metrics['per_class'][pos_class_index]['precision'] for eval_metrics in train_eval_metrics_list]
        train_recall = [eval_metrics['per_class'][pos_class_index]['recall'] for eval_metrics in train_eval_metrics_list]
        train_auroc = [eval_metrics['auroc_per_class'][pos_class_index] for eval_metrics in train_eval_metrics_list]
        train_auprc = [eval_metrics['auprc_per_class'][pos_class_index] for eval_metrics in train_eval_metrics_list]
        test_f1 = [eval_metrics['per_class'][pos_class_index]['f1'] for eval_metrics in test_eval_metrics_list]
        test_precision = [eval_metrics['per_class'][pos_class_index]['precision'] for eval_metrics in test_eval_metrics_list]
        test_recall = [eval_metrics['per_class'][pos_class_index]['recall'] for eval_metrics in test_eval_metrics_list]
        test_auroc = [eval_metrics['auroc_per_class'][pos_class_index] for eval_metrics in test_eval_metrics_list]
        test_auprc = [eval_metrics['auprc_per_class'][pos_class_index] for eval_metrics in test_eval_metrics_list]
        
    else:
        train_f1 = [eval_metrics['f1_macro'] for eval_metrics in train_eval_metrics_list]
        train_precision = [eval_metrics['precision_macro'] for eval_metrics in train_eval_metrics_list]
        train_recall = [eval_metrics['recall_macro'] for eval_metrics in train_eval_metrics_list]
        train_auroc = [eval_metrics['auroc_macro'] for eval_metrics in train_eval_metrics_list]
        train_auprc = [eval_metrics['auprc_macro'] for eval_metrics in train_eval_metrics_list]
        test_f1 = [eval_metrics['f1_macro'] for eval_metrics in test_eval_metrics_list]
        test_precision = [eval_metrics['precision_micro'] for eval_metrics in test_eval_metrics_list]
        test_recall = [eval_metrics['recall_micro'] for eval_metrics in test_eval_metrics_list]
        test_auroc = [eval_metrics['auroc_macro'] for eval_metrics in test_eval_metrics_list]
        test_auprc = [eval_metrics['auprc_macro'] for eval_metrics in test_eval_metrics_list]
        
    
    if plot:
        # Create figure with three subplots side by side
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

        # Plot losses on the left subplot
        ax1.plot(train_losses, 'b-', label='Train Loss')
        ax1.plot(test_losses, 'r-', label='Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Test Loss')

        # Set titles for other subplots
        # ax2.set_title('AUROC and Recall')
        ax2.set_xlabel('Saves')

        # ax3.set_title('F1, Precision, and AUPRC')
        ax3.set_xlabel('Saves')
        # Split metrics into two groups
        metrics_config = {
            'ax2_metrics': {
                'AUROC': {'color': 'purple', 'label': 'AUROC'},
                'AUPRC': {'color': 'orange', 'label': 'AUPRC'}
            },
            'ax3_metrics': {
                'F1': {'color': 'blue', 'label': 'F1'},
                'Precision': {'color': 'green', 'label': 'Precision'},
                'Recall': {'color': 'red', 'label': 'Recall'}
            }
        }

        # Plot lines for ax2 (AUROC and Recall)
        for metric, config in metrics_config['ax2_metrics'].items():
            # Train (solid line)
            ax2.plot(saves, eval(f'train_{metric.lower()}'), 
                    color=config['color'], 
                    linestyle='-',
                    label=f'{config["label"]} (Train)')
            # Test (dashed line)
            ax2.plot(saves, eval(f'test_{metric.lower()}'), 
                    color=config['color'], 
                    linestyle='--',
                    label=f'{config["label"]} (Test)')

        # Plot lines for ax3 (F1, Precision, AUPRC)
        for metric, config in metrics_config['ax3_metrics'].items():
            # Train (solid line)
            ax3.plot(saves, eval(f'train_{metric.lower()}'), 
                    color=config['color'], 
                    linestyle='-',
                    label=f'{config["label"]} (Train)')
            # Test (dashed line)
            ax3.plot(saves, eval(f'test_{metric.lower()}'), 
                    color=config['color'], 
                    linestyle='--',
                    label=f'{config["label"]} (Test)')
        
        ax2.set_ylim(0, 1)
        ax3.set_ylim(0, 1)
        # Add horizontal line at y=0.5 for ax2
        ax2.axhline(y=0.5, color='darkgray', linestyle='--', linewidth=2)
        ax2.axhline(y=0.1, color='darkgray', linestyle='--', linewidth=2)
        ax3.axhline(y=0.5, color='darkgray', linestyle='--', linewidth=2)
        ax3.axhline(y=0.1, color='darkgray', linestyle='--', linewidth=2)

        # Add legends
        ax2.legend(loc='upper right')
        ax3.legend(loc='upper right')
        
        # Add grid
        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)

    if plot_confusion_matrices:
        # plot confusion matrices
        n_matrices = len(confusion_matrices_train)
        n_cols = min(10, n_matrices)  # max 10 columns
        n_rows = (n_matrices + n_cols - 1) // n_cols  # ceiling division

        plt.figure(figsize=(20, 4*n_rows))
        for i, conf_matrix in enumerate(confusion_matrices_train):
            plt.subplot(n_rows, n_cols, i+1)
            plt.imshow(conf_matrix, cmap='Blues')
            plt.title(f'Epoch {i+1}', fontsize=12)
            plt.tick_params(axis='both', which='major', labelsize=2)
            for x in range(conf_matrix.shape[0]):
                for y in range(conf_matrix.shape[1]):
                    plt.text(y, x, str(conf_matrix[x, y]), ha='center', va='center', fontsize=10)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(20, 4*n_rows))
        for i, conf_matrix in enumerate(confusion_matrices_test):
            plt.subplot(n_rows, n_cols, i+1)
            plt.imshow(conf_matrix, cmap='Blues')
            plt.title(f'Epoch {i+1}', fontsize=12)
            plt.tick_params(axis='both', which='major', labelsize=2)
            # Add numbers to the cells
            for x in range(conf_matrix.shape[0]):
                for y in range(conf_matrix.shape[1]):
                    plt.text(y, x, str(conf_matrix[x, y]), ha='center', va='center', fontsize=10)
        plt.tight_layout()
        plt.show()

    
    
    return {'train_losses': train_losses,
            'test_losses': test_losses,
            'train_f1': train_f1,
            'train_precision': train_precision, 
            'train_recall': train_recall, 
            'train_auroc': train_auroc,
            'train_auprc': train_auprc,
            'test_f1': test_f1, 
            'test_precision': test_precision, 
            'test_recall': test_recall,
            'test_auroc': test_auroc,
            'test_auprc': test_auprc}



# ------- diagnostic plots -------
def diag_make_calibration_plot(y_prob, y_true, bins=15):
    """
    Create a calibration plot comparing predicted probabilities with actual outcomes.
    
    Parameters:
    -----------
    y_prob : torch.Tensor or numpy.ndarray
        Predicted probabilities with shape (n_samples, 2)
    y_true : torch.Tensor or numpy.ndarray
        True binary labels with shape (n_samples, 2)
        
    Returns:
    --------
    None, displays plot and prints Brier scores
    """
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt
    from sklearn.metrics import brier_score_loss
    
    # Convert tensors to numpy arrays if needed
    if hasattr(y_prob, 'cpu'):
        y_prob = y_prob.cpu().numpy()
    if hasattr(y_true, 'values'):
        y_true = y_true.values
        
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Calculate calibration curves for both classes
    prob_true1, prob_pred1 = calibration_curve(y_true[:, 0], y_prob[:, 0], n_bins=bins)
    prob_true2, prob_pred2 = calibration_curve(y_true[:, 1], y_prob[:, 1], n_bins=bins)
    
    # Plot calibration curves
    plt.plot(prob_pred1, prob_true1, "s-", label="Class 1")
    plt.plot(prob_pred2, prob_true2, "s-", label="Class 2")
    
    # Plot the perfectly calibrated line
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    # Customize the plot
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Plot")
    plt.legend()
    plt.grid(True)
    
    # Add margins to the plot
    plt.margins(0.02)
    plt.tight_layout()
    plt.show()
    
    # Calculate and print Brier scores
    brier_1 = brier_score_loss(y_true[:, 0], y_prob[:, 0])
    brier_2 = brier_score_loss(y_true[:, 1], y_prob[:, 1])
    
    print(f"Brier score for class1 prediction: {brier_1:.3f}")
    print(f"Brier score for class2 prediction: {brier_2:.3f}")




def diag_plot_top_k_predictions(y_true, y_prob, df, K=15):
    """
    Plot top K time series for true/false positives/negatives based on model predictions.
    
    Parameters:
    -----------
    y_true : numpy.ndarray or pandas.DataFrame
        True binary labels with shape (n_samples, 2)
    y_prob : torch.Tensor
        Predicted probabilities with shape (n_samples, 2)
    df : pandas.DataFrame
        DataFrame containing the time series data
    K : int
        Number of top samples to plot (default=15)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Convert tensors to numpy if needed
    if hasattr(y_prob, 'cpu'):
        y_prob = y_prob.cpu().numpy()
    if hasattr(y_true, 'values'):
        y_true = y_true.values
        
    # Get time series column names
    ts_cols = [str(i) for i in range(1, 301)]
    
    # Get predictions
    y_pred = (y_prob[:, 0] > 0.5).astype(int)
    
    # Separate indices by prediction type
    true_pos_idx = np.where((y_pred == 1) & (y_true[:, 0] == 1))[0]
    false_pos_idx = np.where((y_pred == 1) & (y_true[:, 0] == 0))[0]
    true_neg_idx = np.where((y_pred == 0) & (y_true[:, 0] == 0))[0]
    false_neg_idx = np.where((y_pred == 0) & (y_true[:, 0] == 1))[0]
    
    # Sort indices by prediction confidence
    true_pos_idx = true_pos_idx[np.argsort(y_prob[true_pos_idx, 0])[-K:]]
    false_pos_idx = false_pos_idx[np.argsort(y_prob[false_pos_idx, 0])[-K:]]
    true_neg_idx = true_neg_idx[np.argsort(y_prob[true_neg_idx, 1])[-K:]]
    false_neg_idx = false_neg_idx[np.argsort(y_prob[false_neg_idx, 1])[-K:]]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'Top {K} Predictions by Category', fontsize=16)
    
    # Plot true positives
    for idx in true_pos_idx:
        ts = df.iloc[idx].loc[ts_cols].values
        axes[0,0].plot(ts, alpha=0.7)
    axes[0,0].set_title(f'True Positives (Class 1)\nMean prob: {y_prob[true_pos_idx, 0].mean():.3f}')
    axes[0,0].set_xlabel('Time (seconds)')
    axes[0,0].set_ylabel('Heart Rate')
    axes[0,0].grid(True)
    
    # Plot false positives
    for idx in false_pos_idx:
        ts = df.iloc[idx].loc[ts_cols].values
        axes[0,1].plot(ts, alpha=0.7)
    axes[0,1].set_title(f'False Positives (Predicted Class 1, Actually Class 2)\nMean prob: {y_prob[false_pos_idx, 0].mean():.3f}')
    axes[0,1].set_xlabel('Time (seconds)')
    axes[0,1].set_ylabel('Heart Rate')
    axes[0,1].grid(True)
    
    # Plot true negatives
    for idx in true_neg_idx:
        ts = df.iloc[idx].loc[ts_cols].values
        axes[1,0].plot(ts, alpha=0.7)
    axes[1,0].set_title(f'True Negatives (Class 2)\nMean prob: {y_prob[true_neg_idx, 1].mean():.3f}')
    axes[1,0].set_xlabel('Time (seconds)')
    axes[1,0].set_ylabel('Heart Rate')
    axes[1,0].grid(True)
    
    # Plot false negatives
    for idx in false_neg_idx:
        ts = df.iloc[idx].loc[ts_cols].values
        axes[1,1].plot(ts, alpha=0.7)
    axes[1,1].set_title(f'False Negatives (Predicted Class 2, Actually Class 1)\nMean prob: {y_prob[false_neg_idx, 1].mean():.3f}')
    axes[1,1].set_xlabel('Time (seconds)')
    axes[1,1].set_ylabel('Heart Rate')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # # Print summary statistics
    # print("\nNumber of samples in each category:")
    # print(f"True Positives: {len(true_pos_idx)}")
    # print(f"False Positives: {len(false_pos_idx)}")
    # print(f"True Negatives: {len(true_neg_idx)}")
    # print(f"False Negatives: {len(false_neg_idx)}")

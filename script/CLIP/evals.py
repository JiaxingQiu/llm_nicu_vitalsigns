
from config import *
from data import *
from models import *


def eval_model(model, y_true, ts_df, txt_ls, ts_encoder_name, text_encoder_name):
    _, y_prob = get_logit(model, ts_df, txt_ls, ts_encoder_name, text_encoder_name)
    eval_metrics = get_eval_metrics(y_true, y_prob)
    return eval_metrics


def get_logit(model, 
              ts_df, # df of new time series
              txt_ls, # list of possible outcome texts
              ts_encoder_name,
              text_encoder_name):
    
    model.eval()
    ts_f_mat = TSFeature(ts_df, encoder_model_name=ts_encoder_name).features
    tx_f_ls = TXTFeature(txt_ls, encoder_model_name=text_encoder_name).features   

    # calculate the logits for all observations and outcomes, one outcome all observations per each 
    obs_ys_logits = []
    # for each outcome, get the logits for all observations
    for tx_f in tx_f_ls:
        tx_f = tx_f.reshape(1, -1)
        tx_f_mat = torch.cat([tx_f] * ts_f_mat.shape[0], dim=0) # shape = (obs, dim_tx_f)
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


def get_logit1(model, 
              ts_df, # df of new time series
              txt_ls, # list of possible outcome texts
              ts_encoder_name,
              text_encoder_name):
    
    model.eval()
    ts_f_mat = TSFeature(ts_df, encoder_model_name=ts_encoder_name).features
    tx_f_ls = TXTFeature(txt_ls, encoder_model_name=text_encoder_name).features   

    # calculate the logits for all observations and outcomes, one by one
    obs_ys_logits = []
    for i in range(ts_f_mat.shape[0]):
        ts_f = ts_f_mat[i,:]
        ts_f = ts_f.reshape(1, -1)
        logits = [] 
        for tx_f in tx_f_ls:
            tx_f = tx_f.reshape(1, -1)
            with torch.no_grad():
                logit, _, _ = model(ts_f, tx_f) # logit is a tensor of size (1, 1)
                logits.append(logit) 
        logits = torch.cat(logits, dim=1)
        obs_ys_logits.append(logits)
    obs_ys_logits = torch.cat(obs_ys_logits, dim=0)
    exp_preds = torch.exp(obs_ys_logits)
    softmax_probs = exp_preds / exp_preds.sum(dim=1, keepdim=True)

    return obs_ys_logits, softmax_probs




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
        y_true = y_true.detach().numpy()
    if torch.is_tensor(y_prob):
        y_prob = y_prob.detach().numpy()
    
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
        y_true_labels, y_pred_labels, average='macro'
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average='micro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average='weighted'
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
        y_true_labels, y_pred_labels, average=None
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
        
        # train_metrics = []
        # for cm in confusion_matrices_train:
        #     metrics = calculate_f1_precision_recall_from_cm(cm)
        #     train_metrics.append(metrics)
        #     test_metrics = []
        # for cm in confusion_matrices_test:
        #     metrics = calculate_f1_precision_recall_from_cm(cm)
        #     test_metrics.append(metrics)
        # train_f1 = [m['f1'] for m in train_metrics]
        # train_precision = [m['precision'] for m in train_metrics]
        # train_recall = [m['recall'] for m in train_metrics]
        # test_f1 = [m['f1'] for m in test_metrics]
        # test_precision = [m['precision'] for m in test_metrics]
        # test_recall = [m['recall'] for m in test_metrics]
        # epochs = range(1, len(train_f1) + 1)
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
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        # Plot losses on the left subplot
        ax1.plot(train_losses, 'b-', label='Train Loss')
        ax1.plot(test_losses, 'r-', label='Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Test Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot metrics on the right subplot
        # metrics_config = {
        #     'F1': {'color': 'navy', 'label': 'F1'},
        #     'Precision': {'color': 'darkgreen', 'label': 'Precision'},
        #     'Recall': {'color': 'darkred', 'label': 'Recall'},
        #     'AUROC': {'color': 'indigo', 'label': 'AUROC'},
        #     'AUPRC': {'color': 'darkorange', 'label': 'AUPRC'}
        # }

        metrics_config = {
            'F1': {'color': 'blue', 'label': 'F1'},
            'Precision': {'color': 'green', 'label': 'Precision'},
            'Recall': {'color': 'red', 'label': 'Recall'},
            'AUROC': {'color': 'purple', 'label': 'AUROC'},  # Changed from cyan
            'AUPRC': {'color': 'orange', 'label': 'AUPRC'}   # Changed from yellow
        }

        # Plot lines and create labels
        for metric, config in metrics_config.items():
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
        # Single legend
        ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)

        # # Create figure with two subplots side by side
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        # # Plot losses on the left subplot
        # ax1.plot(train_losses, 'b-', label='Train Loss')
        # ax1.plot(test_losses, 'r-', label='Test Loss')
        # ax1.set_xlabel('Epoch')
        # ax1.set_ylabel('Loss')
        # ax1.set_title('Training and Test Loss')
        # ax1.legend()
        # ax1.grid(True)

        # # Plot metrics on the right subplot
        # ax2.plot(saves, train_f1, 'b-', label='Train F1')
        # ax2.plot(saves, test_f1, 'b--', label='Test F1')
        # ax2.plot(saves, train_precision, 'g-', label='Train Precision')
        # ax2.plot(saves, test_precision, 'g--', label='Test Precision')
        # ax2.plot(saves, train_recall, 'r-', label='Train Recall')
        # ax2.plot(saves, test_recall, 'r--', label='Test Recall')
        # ax2.plot(saves, train_auroc, 'c-', label='Train AUROC')
        # ax2.plot(saves, test_auroc, 'c--', label='Test AUROC')
        # ax2.plot(saves, train_auprc, 'y-', label='Train AUPRC')
        # ax2.plot(saves, test_auprc, 'y--', label='Test AUPRC')
        # # legend -- as test, - as train
        # ax2.set_xlabel('Saves')
        # ax2.set_ylabel('Score')
        # ax2.set_title('Training and Test Metrics')
        # ax2.legend()
        # ax2.grid(True)

        # plt.tight_layout()
        # plt.show()

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


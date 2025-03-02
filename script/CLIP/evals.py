
from config import *
from data import *
from models import *


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




def evaluate_predictions(y_true, y_prob, class_names=None):
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
    
    # Assert shapes match
    assert y_true.shape == y_prob.shape, f"Shape mismatch: y_true {y_true.shape} != y_prob {y_prob.shape}"
    
    # Get predicted class labels
    y_pred_labels = np.argmax(y_prob, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    
    # Calculate metrics
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true_labels, y_pred_labels)
    
    # Precision, Recall, F1 (macro and weighted)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average='macro'
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
        'f1_weighted': f1_weighted
    })
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    metrics['confusion_matrix'] = cm
    
    # ROC AUC (one-vs-rest)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        metrics['roc_auc'] = auc
    except:
        metrics['roc_auc'] = None
    
    # PRC AUC
    try:
        auc = average_precision_score(y_true, y_prob, multi_class='ovr')
        metrics['prc_auc'] = auc
    except:
        metrics['prc_auc'] = None

    # Per-class metrics
    class_metrics = {}
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average=None
    )
    
    for i in range(y_true.shape[1]):
        class_name = class_names[i] if class_names else f"Class_{i}"
        class_metrics[class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }
    
    metrics['per_class'] = class_metrics
    
    # # Print summary
    # print("\nClassification Report:")
    # print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    # print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
    # print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
    # if metrics['roc_auc']:
    #     print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
    
    # # Plot confusion matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.title('Confusion Matrix')
    # plt.ylabel('True Label')
    # plt.xlabel('Predicted Label')
    # if class_names:
    #     plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
    #     plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
    # plt.tight_layout()
    # plt.show()
    
    return metrics

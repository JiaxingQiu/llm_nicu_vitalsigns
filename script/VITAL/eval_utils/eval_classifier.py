import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from config import device
import numpy as np
import pandas as pd
from tqdm import tqdm
from generation import interpolate_ts_tx
import copy


# ─────────────── define a classifer ──────────────────────────────
def _get_mean(enc, x):
    _, mean, *_ = enc(x)          # VITAL returns (z, mean, …)
    return mean

# Build the classifier: encoder + linear head
class TSClassifier(nn.Module):
    def __init__(self, encoder, n_classes):
        super().__init__()
        self.encoder = encoder                     # frozen or trainable
        self.head    = nn.LazyLinear(n_classes)    # infers in‑features on 1st use

    def forward(self, x):
        h = _get_mean(self.encoder, x)             # (B, emb_dim)
        return self.head(h)

# ───────────────────────── train_clf ──────────────────────────
def train_clf(
        clf,
        df_train,
        ts_cols,
        label_col,
        y_levels,
        *,
        device="cpu",
        epochs=100,
        batch_size=256,
        lr=1e-3,
        plot=False):
    """
    Fit `clf` (any nn.Module) on `df_train`.
    Returns the trained model plus (train_losses, val_losses).
    """
    # label ↔ index map
    label2idx = {lvl: i for i, lvl in enumerate(y_levels)}

    # tensors
    X_all = torch.tensor(df_train[ts_cols].values, dtype=torch.float32)
    y_all = torch.tensor(df_train[label_col].map(label2idx).values,
                         dtype=torch.long)
    ds_full = TensorDataset(X_all.to(device), y_all.to(device))

    train_len = int(0.8 * len(ds_full))
    val_len   = len(ds_full) - train_len
    train_ds, val_ds = random_split(ds_full, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False)

    # training
    clf.to(device).train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(clf.parameters(), lr=lr)

    train_losses, val_losses = [], []
    interval = max(1, epochs // 5)
    for ep in tqdm(range(epochs)):
        run = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(clf(xb), yb)
            loss.backward()
            optimizer.step()
            run += loss.item() * xb.size(0)
        train_loss = run / train_len
        train_losses.append(train_loss)

        # validation
        clf.eval(); run = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                run += criterion(clf(xb), yb).item() * xb.size(0)
        val_loss = run / val_len
        val_losses.append(val_loss)
        clf.train()
        if plot: # for debugging
            if (ep + 1) % interval == 0 or ep == 0:
                print(f"epoch {ep+1:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

    if plot:
        plt.plot(train_losses, label="train")
        plt.plot(val_losses,   label="val")
        plt.xlabel("epoch"); plt.ylabel("loss")
        plt.title("Training vs. validation loss")
        plt.grid(True); plt.legend(); plt.show()

    return clf, (train_losses, val_losses)

# ────────────────────── predict_with_clf ──────────────────────
def predict_with_clf(clf, df_pred, ts_cols, y_levels, *, device="cpu"):
    """
    Append soft‑max probabilities (one column per class) to `df_pred`.
    Returns the modified dataframe.
    """
    clf.eval()
    with torch.no_grad():
        X = torch.tensor(df_pred[ts_cols].values,
                         dtype=torch.float32, device=device)
        probs = torch.softmax(clf(X), dim=1).cpu().numpy()
    df_pred[[f"prob_{lvl}" for lvl in y_levels]] = probs
    return df_pred

# ────────────────────── add_rats_score ──────────────────────
def add_rats_score(df_aug, df_src, *, y_col, y_levels, eps=1e-12):
    """
    Vectorised RaTS score for aligned data‑frames.

    Parameters
    ----------
    df_aug : pd.DataFrame   (must contain prob_<y_level> columns)
    df_src : pd.DataFrame   (same length/order as df_aug)
    y_col  : str            column holding the *target* label in df_aug
    y_levels : list[str]    list of all class names
    eps    : float          numerical stabiliser

    Returns
    -------
    pd.DataFrame  -- df_aug with an extra 'RaTS' column
    """
    df_aug = df_aug.copy()          # keep originals intact
    df_src = df_src.reset_index(drop=True)
    df_aug = df_aug.reset_index(drop=True)

    # build a Series of probabilities p(a_tgt | x_hat) and p(a_tgt | x_src)
    p_aug = np.empty(len(df_aug))
    p_src = np.empty(len(df_src))

    # iterate over target classes once, but apply all rows at once (vectorised)
    for lvl in y_levels:
        mask = df_aug[y_col] == lvl
        col  = f"prob_{lvl}"
        p_aug[mask] = df_aug.loc[mask, col]
        p_src[mask] = df_src.loc[mask, col]

    # RaTS = log( p_aug / p_src )
    df_aug['p_aug'] = p_aug
    df_aug['p_src'] = p_src
    df_aug["RaTS"] = np.log((p_aug + eps) / (p_src + eps))

    return df_aug



# ────────────────────── main functions ──────────────────────
def _stratified_bootstrap(group, b):
    n = len(group)
    replace = n < b
    return group.sample(n=b, replace=replace, random_state=333)


def eval_ts_classifier(df, # df can be df_train / df_test
                      model, config_dict, w, y_col, 
                      conditions = None, # a list of tuples of (y_col, y_level) to filter the df (should not filter y_col)
                      b=None, # number to bootstrap
                      finetune=True):
    
    model.eval()
    y_levels = list(df[y_col].unique())
    ts_str_cols = [str(i+1) for i in range(config_dict['seq_length'])]


    #  --- define a classifer ---------------------------------------------------------------------------------------------------------------
    # ts_enc = model.ts_encoder.to(device).eval()        # reuse in eval mode
    ts_enc = copy.deepcopy(model.ts_encoder).to(device).eval()
    if not finetune:
        for p in ts_enc.parameters(): 
            p.requires_grad = False
    clf = TSClassifier(ts_enc, len(y_levels)).to(device)


    #  --- prepare dataframe to train clf (predictors: ts_str_cols, outcome (multi_class): y_col) --------------------------------------------
    if conditions is not None:
        for condition in conditions:
            df = df[df[condition[0]] == condition[1]]

    if b is not None:
        df2train = (
            df.groupby(y_col, group_keys=False)
            .apply(_stratified_bootstrap, b=b)
            .reset_index(drop=True)
        )
    else:
        df2train = df.copy()

    df2train = df2train[ts_str_cols + [y_col, 'text', 'label']].copy()

    # --- prepare dataframe to predict by clf a softmax prob for the given y_col ------------------------------------------------------------
    df2pred_aug = pd.DataFrame()
    df2pred_src = pd.DataFrame()
    for tgt_level in y_levels:                              
        # rows we want to *re‑write* toward tgt_level
        df2aug = df2train[df2train[y_col] != tgt_level].copy().reset_index(drop=True)
        df2aug['new_text'] = tgt_level
        df2aug_src = df2aug[ts_str_cols + [y_col, 'text', 'label']].copy()
        df2pred_src = pd.concat([df2pred_src, df2aug_src], ignore_index=True)

        # generate edited time series
        ts_hat_ls = interpolate_ts_tx(df2aug, model, config_dict, ['new_text'], w)
        tmp = pd.DataFrame(ts_hat_ls['new_text'], columns=['aug_text', 'ts_hat'])
        tmp['ts_hat'] = tmp['ts_hat'].apply(lambda x: x.cpu().detach().numpy() )

        # add y_col and update ts_str_cols 
        df2aug[y_col] = tmp['aug_text'].values               
        df2aug[ts_str_cols] = np.vstack(tmp['ts_hat'].to_numpy())

        # tidy‑up
        df2aug = df2aug[ts_str_cols + [y_col]].copy()
        df2pred_aug = pd.concat([df2pred_aug, df2aug], ignore_index=True)

    # --- train and predict ---------------------------------------------------------------------------------------------------------------
    clf, _ = train_clf(
        clf,
        df2train,                 # dataframe with predictors + label col
        ts_str_cols,
        y_col,
        y_levels,
        device=device,
        epochs=100,
        plot=False
    )
    df2pred_aug = predict_with_clf(
        clf,
        df2pred_aug,                  # dataframe to receive prob columns
        ts_str_cols,
        y_levels,
        device=device
    )
    df2pred_src = predict_with_clf(
        clf,
        df2pred_src,                  # dataframe to receive prob columns
        ts_str_cols,
        y_levels,
        device=device
    )
    df2pred_aug = add_rats_score(
        df2pred_aug,
        df2pred_src,
        y_col=y_col,
        y_levels=y_levels
    )
    # summary statistics of RaTS score
    def _summ(col):
        return {
            'mean': np.round(col.mean(), 4),
            'std' : np.round(col.std(),  4),
            'min' : np.round(col.min(),  4),
            'max' : np.round(col.max(),  4),
            **{f"q{p}": np.round(np.quantile(col, p/100), 4)
            for p in (5, 25, 50, 75, 95)}
        }  
     
    RaTS_summ = _summ(df2pred_aug['RaTS'])
    torch.cuda.empty_cache()

    return df2pred_aug, RaTS_summ

# usage:
# df2pred_aug, rats_summ = eval_ts_classifier(df_test, model, config_dict, 
#                                             w = 0.8, y_col = 'segment2', conditions = None)




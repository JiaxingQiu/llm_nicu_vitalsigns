from .eval_clip import *
import pandas as pd

# ---------------- functions to evaluate CLIP in terms of time series classification -------------------- 
class EvalCLIPTXT2TS:
    def __init__(self, 
                 df,
                 txt_tsid_mapping, 
                 config_dict):
        """
        df: the raw dataframe (i.e. df_train or df_test by main_preprocess)
        txt_tsid_mapping: a dictionary of {txt_caption(string): [ts_id1, ts_id2, ...]}, NOTE the first ts_id is always the currect time series (ground truth)
        """

        df_new, self.y_true, self.n_captions, self.n_ts_candidates, self.txt_tsid_mapping_new = self.prep_df_eval(df, txt_tsid_mapping)
        self.y_true = self.y_true.to(device)
        

        if not config_dict['3d']:
            self.ts_f_mat, self.tx_f_mat, _ = get_features(df_new, 
                                                            config_dict,
                                                            text_col='caption')
            
            # self.ts_f_mat.shape[0] and self.tx_f_mat.shape[0] should be self.n_captions*self.n_ts_candidates
            assert self.ts_f_mat.shape[0] == self.n_captions*self.n_ts_candidates, "ts_f_mat.shape[0] error"
            assert self.tx_f_mat.shape[0] == self.n_captions*self.n_ts_candidates, "tx_f_mat.shape[0] error"
            
            self.ts_f_mat = self.ts_f_mat.to(device)
            self.tx_f_mat = self.tx_f_mat.to(device)

        elif config_dict['3d']:
            text_cols = config_dict['text_col_ls']
            sub_text_col = config_dict['sub_text_col']
            # substitute sub_text_col with 'caption' in text_cols
            text_cols = [col if col != sub_text_col else 'caption' for col in text_cols]
            self.ts_f_mat, self.tx_f_mat_ls, _ = get_features3d(df_new, 
                                                                config_dict,
                                                                text_col_ls=text_cols) # text1 is a column of "will die", text2 is a column of "will survive"
            assert self.ts_f_mat.shape[0] == self.n_captions*self.n_ts_candidates, "ts_f_mat.shape[0] error"
            self.ts_f_mat = self.ts_f_mat.to(device)
            self.tx_f_mat_ls = [tx_f_mat.to(device) for tx_f_mat in self.tx_f_mat_ls]
    
    def prep_df_eval(self, df, txt_tsid_mapping):
        import numpy as np
        np.random.seed(333)
        """
        df: the raw dataframe
        txt_tsid_mapping: a dictionary of {txt_caption(string): [ts_id1, ts_id2, ...]} # the first ts_id is always the currect time series (ground truth)
        return: a dataframe of every n_ts rows are n_ts time series candiates for a text caption

        usage:
        txt_tsid_mapping = {'text1':[559, 560, 561],
                        'text2':[562, 563, 564], 
                        'text3':[565, 566, 832], 
                        'text4':[1153, 65148, 65149]}
        df_eval = prep_df_eval(df_train, txt_tsid_mapping)
        """
        
        assert len(set([len(ts_ids) for ts_ids in txt_tsid_mapping.values()])) == 1, "each ts_ids in txt_tsid_mapping must be the same length"
        assert 'caption' not in df.columns, "df must not have a column named 'caption'"

        n_ts_candidates = len(txt_tsid_mapping[list(txt_tsid_mapping.keys())[0]])
        n_captions = len(txt_tsid_mapping)

        y_true = torch.zeros(n_captions, n_ts_candidates)
        df_eval = pd.DataFrame()
        txt_tsid_mapping_new = {}
        for cap_id, (txt, ts_ids) in enumerate(txt_tsid_mapping.items()):
            true_ts_id = ts_ids[0]
            ts_ids = np.random.permutation(ts_ids)
            y_true[cap_id, ts_ids == true_ts_id] = 1
            txt_tsid_mapping_new[txt] = ts_ids

            df_txt = df.loc[ts_ids].copy() # ts_ids  is a list of row ids in df
            df_txt['caption'] = txt
            df_eval = pd.concat([df_eval, df_txt])
            
        return df_eval, y_true, n_captions, n_ts_candidates, txt_tsid_mapping_new





@torch.no_grad() 
def eval_clip_txt2ts(model, 
                    eval_inputs, # instance of EvalCLIPTXT2TS (2d)
                    return_probs=False):
    
    model.eval()
    with torch.no_grad(): 
        y_true = eval_inputs.y_true
        n_ts = eval_inputs.n_ts_candidates
        n_cap = eval_inputs.n_captions


        y_prob = torch.zeros(n_cap, n_ts)
        for cap_id in range(n_cap):
            ts_f_mat = eval_inputs.ts_f_mat[cap_id*n_ts:(cap_id+1)*n_ts,:] # get the cap_idTH n_ts_candidates rows
            tx_f_mat = eval_inputs.tx_f_mat[cap_id*n_ts:(cap_id+1)*n_ts,:]
            logits, _, _, _ = model(ts_f_mat, tx_f_mat)
            logit = torch.diag(logits).unsqueeze(0) # (1, n_ts)
            exp_preds = torch.exp(logit)
            y_prob_row = exp_preds / exp_preds.sum(dim=1, keepdim=True)
            y_prob[cap_id,:] = y_prob_row

    eval_metrics = get_eval_metrics(y_true, y_prob)
    if return_probs:
        return eval_metrics, y_prob
    else:
        return eval_metrics




@torch.no_grad() 
def eval_clip3d_txt2ts(model, 
                        eval_inputs, # instance of EvalCLIPTXT2TS (3d)
                        return_probs=False):
    
    model.eval()
    with torch.no_grad(): 
        y_true = eval_inputs.y_true
        n_ts = eval_inputs.n_ts_candidates
        n_cap = eval_inputs.n_captions


        y_prob = torch.zeros(n_cap, n_ts)
        for cap_id in range(n_cap):
            ts_f_mat = eval_inputs.ts_f_mat[cap_id*n_ts:(cap_id+1)*n_ts,:] # get the cap_idTH n_ts_candidates rows
            tx_f_mat_ls = [tx_f_mat[cap_id*n_ts:(cap_id+1)*n_ts,:] for tx_f_mat in eval_inputs.tx_f_mat_ls]
            logits, _, _, _ = model(ts_f_mat, tx_f_mat_ls)
            logit = torch.diag(logits).unsqueeze(0) # (1, n_ts)
            exp_preds = torch.exp(logit)
            y_prob_row = exp_preds / exp_preds.sum(dim=1, keepdim=True)
            y_prob[cap_id,:] = y_prob_row

    eval_metrics = get_eval_metrics(y_true, y_prob)
    if return_probs:
        return eval_metrics, y_prob
    else:
        return eval_metrics



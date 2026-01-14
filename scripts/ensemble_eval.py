import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from scripts.utils import *

import tensorflow as tf
import os

from scripts.checkpointing import load_model_checkpoint
from scripts.data import get_datasets, get_external_datasets
from scripts.loss_and_metrics import compute_frameshift_rate_numpy


def compute_metrics(fs_true, fs_pred, median):
    """
    fs_true, fs_pred: 1D arrays
    median: scalar (dataset-specific median threshold)
    Returns: pearson, spearman, auc, mse
    """

    # MSE
    mse = np.mean((fs_pred - fs_true)**2)

    # Pearson
    if np.std(fs_true) == 0 or np.std(fs_pred) == 0:
        pearson = np.nan
    else:
        pearson = np.corrcoef(fs_true, fs_pred)[0, 1]

    # Spearman
    try:
        spearman = spearmanr(fs_true, fs_pred).correlation
    except:
        spearman = np.nan

    # AUC
    y_true_bin = (fs_true >= median).astype(int)
    if len(np.unique(y_true_bin)) < 2:
        auc = np.nan
    else:
        auc = roc_auc_score(y_true_bin, fs_pred)

    return pearson, spearman, auc, mse




def predict_with_singleN_marginalization(
    model,
    tokens_sel,          # (B, L) np.int32
    forced_ids,          # (B,)  np.int32
    *,
    apply_forced_ds,     # int
    target_true_ds,      # int
    special_true_ds=21, # if LINDEL IS IN NO NEED THEN SET IT TO RANDOM NUMBER, OTHERISE 21
):
    """
    If target_true_ds == special_true_ds and each sample has exactly one 'N',
    replace that N with A/C/G/T -> predict -> average predictions.
    Otherwise predict normally.

    Returns:
        preds_mean: (B, C) float32
    """
    N_ID = VOCAB["N"]
    A_ID = VOCAB["A"]
    C_ID = VOCAB["C"]
    G_ID = VOCAB["G"]
    T_ID = VOCAB["T"]

    B, L = tokens_sel.shape

    # Only special-case dataset 13 (true dataset), everything else unchanged
    if target_true_ds != special_true_ds:
        return model({"tokens": tokens_sel, "dataset_id": forced_ids}).numpy()

    # Find N positions per sample
    n_pos_lists = [np.where(tokens_sel[i] == N_ID)[0] for i in range(B)]
    counts = np.array([len(p) for p in n_pos_lists], dtype=np.int32)

    # We only handle "exactly one N" as you described; others stay normal
    oneN_mask = (counts == 1)
    if not np.any(oneN_mask):
        return model({"tokens": tokens_sel, "dataset_id": forced_ids}).numpy()

    # ---- Predict for the "normal" subset (not exactly one N) ----
    preds_out = np.zeros((B, NUM_CLASSES), dtype=np.float32)

    normal_mask = ~oneN_mask
    if np.any(normal_mask):
        tok_n = tokens_sel[normal_mask]
        ids_n = forced_ids[normal_mask]
        preds_out[normal_mask] = model({"tokens": tok_n, "dataset_id": ids_n}).numpy()

    # ---- Predict for the "one N" subset by expanding to 4 variants ----
    tok_1 = tokens_sel[oneN_mask]
    ids_1 = forced_ids[oneN_mask]
    B1 = tok_1.shape[0]

    # positions of the single N
    n_positions = np.array([n_pos_lists[i][0] for i in np.where(oneN_mask)[0]], dtype=np.int32)  # (B1,)

    # Make 4 copies: (4*B1, L)
    tok_rep = np.repeat(tok_1, 4, axis=0)

    # Which row gets which base replacement
    base_ids = np.array([A_ID, C_ID, G_ID, T_ID], dtype=np.int32)
    rep_base = np.tile(base_ids, B1)  # length 4*B1

    # For each sample i, its 4 rows are at indices 4*i .. 4*i+3
    row_idx = np.arange(4 * B1, dtype=np.int32)
    sample_idx = row_idx // 4
    col_idx = n_positions[sample_idx]
    tok_rep[row_idx, col_idx] = rep_base

    ids_rep = np.repeat(ids_1, 4, axis=0)


    preds_rep = model({"tokens": tok_rep, "dataset_id": ids_rep}).numpy()

    # reshape to (B1, 4, C) and average over A/C/G/T
    C = preds_rep.shape[-1]
    preds_rep = preds_rep.reshape(B1, 4, C)
    preds_mean = preds_rep.mean(axis=1)  # (B1, C)

    preds_out[oneN_mask] = preds_mean
    return preds_out




# TODO: load models ONCE per fold, not per dataset
def evaluate_cross_dataset_ensemble(
        num_folds,
        seed=42,
        checkpoint_dir="saved_models/",
        save_dir="ensemble_results/",
        use_mh=True,
        use_legacy_split=True,
        filter_sprout=False):

    os.makedirs(save_dir, exist_ok=True)

    # Load test set ONCE
    _, _, test_ds, MASK_TABLE_TF, DATASET_WEIGHTS_TF, DATASET_MEDIANS_TF, _, NUM_DATASETS = \
        get_datasets(seed=seed, val_bucket=0, save_splits=False, filter_ids=[9] if filter_sprout else [], use_legacy_split=use_legacy_split) 
    # Load external test set ONCE




    # Extract all test batches into RAM 
    test_batches = []
    for inputs, y_true in test_ds:
        test_batches.append((
            inputs["tokens"].numpy(),
            inputs["dataset_id"].numpy(),
            y_true.numpy()
        ))

    if filter_sprout:
        external_test_ds = get_external_datasets()
        # Extract all test batches from external test set into RAM 
        for inputs, y_true in external_test_ds:
            test_batches.append((
                inputs["tokens"].numpy(),
                inputs["dataset_id"].numpy(),
                y_true.numpy()
            ))

    # -----------------------------------------------------------
    # Prepare output matrices
    # -----------------------------------------------------------
    pearson_mat  = np.zeros((NUM_DATASETS, NUM_DATASETS))
    spearman_mat = np.zeros((NUM_DATASETS, NUM_DATASETS))
    auc_mat      = np.zeros((NUM_DATASETS, NUM_DATASETS))
    mse_mat      = np.zeros((NUM_DATASETS, NUM_DATASETS))

    # Preload dataset medians
    dataset_medians = DATASET_MEDIANS_TF.numpy()

    # 1. Pre-load all models into a list once
    print(f"--- Pre-loading {num_folds} models into memory ---")
    loaded_models = []
    for fold in range(num_folds):
        model_data = load_model_checkpoint(fold, checkpoint_dir)
        # model_data is (model, MASK_TABLE, MEDIANS, SIZE_SCALE, meta)
        loaded_models.append(model_data)

    # -----------------------------------------------------------
    # Iterate over TRUE dataset k
    # -----------------------------------------------------------
    for true_ds in range(NUM_DATASETS):
        print(f"\n=== Evaluating TRUE dataset {index_to_DS_name[true_ds]} ===")

        fs_true_all = []

        # Collect sample indices belonging to true_ds (unchanged logic)
        for tokens, dsids, y_true in test_batches:
            mask = (dsids == true_ds)
            if mask.sum() == 0:
                continue

            masks = MASK_TABLE_TF.numpy()[dsids[mask]]
            y_norm = y_true[mask] * masks
            y_norm /= (y_norm.sum(axis=1, keepdims=True)) # should not have zero rows here so should be fine

            fs_true = compute_frameshift_rate_numpy(
                y_norm,
                np.arange(MIN_DELTA, MAX_DELTA+1)
            )
            fs_true_all.append(fs_true)

        fs_true_all = np.concatenate(fs_true_all)

        # -----------------------------------------------------------
        # Now evaluate all FORCED dataset embeddings j
        # -----------------------------------------------------------
        for forced_ds in range(NUM_DATASETS):
            print(f"   testing forced embedding {index_to_DS_name[forced_ds]}")

            fs_pred_fold_list = []

            # ------------------ Ensemble prediction ------------------
            # Use the pre-loaded models instead of calling load_model_checkpoint
            for fold in range(num_folds):
                # Unpack the pre-loaded model data
                model, MASK_TABLE, MEDIANS, SIZE_SCALE, meta = loaded_models[fold]

                fs_pred_all = []

                for tokens, dsids, y_true in test_batches:
                    mask = (dsids == true_ds)
                    if mask.sum() == 0:
                        continue

                    tokens_sel = tokens[mask]
                    forced_ids = np.full(len(tokens_sel), forced_ds, dtype=np.int32)

                    preds = predict_with_singleN_marginalization(
                        model,
                        tokens_sel,
                        forced_ids,
                        apply_forced_ds=forced_ds,
                        target_true_ds=true_ds,
                        special_true_ds=21, 
                    )

                    # apply mask & normalize
                    masks_arr = MASK_TABLE_TF.numpy()[forced_ds]
                    masks_arr = np.tile(masks_arr, (preds.shape[0], 1))

                    preds = preds * masks_arr
                    preds_norm = preds / (preds.sum(axis=1, keepdims=True) + 1e-9)

                    fs_pred = compute_frameshift_rate_numpy(
                        preds_norm,
                        np.arange(MIN_DELTA, MAX_DELTA+1)
                    )
                    fs_pred_all.append(fs_pred)

                fs_pred_fold_list.append(np.concatenate(fs_pred_all))

            # Average across folds and compute metrics (unchanged logic)
            fs_pred_mean = np.mean(fs_pred_fold_list, axis=0)

            pear, spear, auc, mse = compute_metrics(
                fs_true_all,
                fs_pred_mean,
                dataset_medians[true_ds]
            )

            pearson_mat[true_ds, forced_ds]  = pear
            spearman_mat[true_ds, forced_ds] = spear
            auc_mat[true_ds, forced_ds]      = auc
            mse_mat[true_ds, forced_ds]      = mse


    # -----------------------------------------------------------
    # Save CSVs with proper row/column labels
    # -----------------------------------------------------------
    labels = [index_to_DS_name[i] for i in range(NUM_DATASETS)]  

    def save_matrix(mat, filename):
        df = pd.DataFrame(mat, index=labels, columns=labels)
        df.index.name = "True Dataset \\ Forced Embedding"
        df.to_csv(os.path.join(save_dir, filename))

    save_matrix(pearson_mat,  "pearson_matrix.csv")
    save_matrix(spearman_mat, "spearman_matrix.csv")
    save_matrix(auc_mat,      "auc_matrix.csv")
    save_matrix(mse_mat,      "mse_matrix.csv")

    print("\nSaved matrices to:", save_dir)


    return pearson_mat, spearman_mat, auc_mat, mse_mat





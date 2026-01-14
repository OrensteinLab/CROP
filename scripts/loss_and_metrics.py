from scripts.utils import *
from scripts.data import get_datasets
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import numpy as np

########################################
# CUSTOM LOSS
########################################


def masked_balanced_mse(y_true, y_pred, dsid_batch,
                        MASK_TABLE_TF):
    """
    MSE over the full delta distribution, after masking + renormalizing.
    """

    # (B, C)
    masks = tf.gather(MASK_TABLE_TF, dsid_batch)

    # -------------------------
    # Mask + renormalize truth
    # -------------------------
    y_true_masked = y_true * masks
    true_sum = tf.reduce_sum(y_true_masked, axis=-1, keepdims=True)
    y_true_norm = tf.math.divide_no_nan(y_true_masked, true_sum)

    # -------------------------
    # Mask + renormalize pred
    # -------------------------
    y_pred_masked = y_pred * masks
    pred_sum = tf.reduce_sum(y_pred_masked, axis=-1, keepdims=True)
    y_pred_norm = tf.math.divide_no_nan(y_pred_masked, pred_sum)

    # -------------------------
    # Per-bin squared error
    # -------------------------
    se = tf.square(y_pred_norm - y_true_norm)     # (B, C)

    # Sum over bins
    mse_per_sample = tf.reduce_sum(se, axis=-1)  # (B,)

    # Dataset-level balancing
    #w = tf.gather(DATASET_WEIGHTS_TF, dsid_batch)
    #return tf.reduce_mean(w * mse_per_sample)
    return tf.reduce_mean(mse_per_sample)


def masked_balanced_cce(y_true, y_pred, dsid_batch, MASK_TABLE_TF, DATASET_WEIGHTS_TF):
    masks = tf.gather(MASK_TABLE_TF, dsid_batch)             # (B,C)

    y_pred_masked = y_pred * masks
    pred_sum = tf.reduce_sum(y_pred_masked, axis=-1, keepdims=True)
    y_pred_norm = tf.math.divide_no_nan(y_pred_masked, pred_sum)

    y_true_masked = y_true * masks
    true_sum = tf.reduce_sum(y_true_masked, axis=-1, keepdims=True)
    y_true_norm = tf.math.divide_no_nan(y_true_masked, true_sum)

    eps = 1e-9
    ce = -tf.reduce_sum(y_true_norm * tf.math.log(y_pred_norm + eps), axis=-1)

    #w = tf.gather(DATASET_WEIGHTS_TF, dsid_batch)
    #return tf.reduce_mean(w * ce)
    return tf.reduce_mean(ce)

def masked_balanced_kld(y_true, y_pred, dsid_batch, MASK_TABLE_TF):
    masks = tf.gather(MASK_TABLE_TF, dsid_batch)             # (B,C)

    # Mask + renormalize prediction
    y_pred_masked = y_pred * masks
    pred_sum = tf.reduce_sum(y_pred_masked, axis=-1, keepdims=True)
    y_pred_norm = tf.math.divide_no_nan(y_pred_masked, pred_sum)

    # Mask + renormalize truth
    y_true_masked = y_true * masks
    true_sum = tf.reduce_sum(y_true_masked, axis=-1, keepdims=True)
    y_true_norm = tf.math.divide_no_nan(y_true_masked, true_sum)

    eps = 1e-9

    # KL divergence = sum true * (log true - log pred)
    kl = tf.reduce_sum(
        y_true_norm * (tf.math.log(y_true_norm + eps) - tf.math.log(y_pred_norm + eps)),
        axis=-1
    )

    # Dataset weighting
    #w = tf.gather(DATASET_WEIGHTS_TF, dsid_batch)
    #return tf.reduce_mean(w * kl)

    # temp no class weight
    return tf.reduce_mean(kl)




########################################
# STREAMING METRICS FOR FS PEARSON & MSE
########################################

class FrameshiftPearsonMetric(tf.keras.metrics.Metric):
    def __init__(self, name="fs_pearson", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_x = self.add_weight(name="sum_x", initializer="zeros")
        self.sum_y = self.add_weight(name="sum_y", initializer="zeros")
        self.sum_x2 = self.add_weight(name="sum_x2", initializer="zeros")
        self.sum_y2 = self.add_weight(name="sum_y2", initializer="zeros")
        self.sum_xy = self.add_weight(name="sum_xy", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, mask):
        """
        y_true, y_pred: (B,) FS values
        mask: (B,) bool or 0/1 float indicating which samples belong to THIS dataset
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mask = tf.cast(mask, tf.float32)

        # Only count masked samples
        w = mask
        sx  = tf.reduce_sum(y_pred * w)
        sy  = tf.reduce_sum(y_true * w)
        sx2 = tf.reduce_sum((y_pred ** 2) * w)
        sy2 = tf.reduce_sum((y_true ** 2) * w)
        sxy = tf.reduce_sum((y_pred * y_true) * w)
        c   = tf.reduce_sum(w)

        self.sum_x.assign_add(sx)
        self.sum_y.assign_add(sy)
        self.sum_x2.assign_add(sx2)
        self.sum_y2.assign_add(sy2)
        self.sum_xy.assign_add(sxy)
        self.count.assign_add(c)

    def result(self):
        c = self.count

        def _compute():
            mean_x = self.sum_x / c
            mean_y = self.sum_y / c
            cov = self.sum_xy - c * mean_x * mean_y
            var_x = self.sum_x2 - c * mean_x**2
            var_y = self.sum_y2 - c * mean_y**2
            denom = tf.sqrt(var_x * var_y) + 1e-8
            return cov / denom

        return tf.cond(c > 1.0,
                       _compute,
                       lambda: tf.constant(float("nan"), tf.float32))

    def reset_state(self):
        for v in self.variables:
            v.assign(0.0)


class FrameshiftMSEMetric(tf.keras.metrics.Metric):
    def __init__(self, name="fs_mse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_sqerr = self.add_weight(name="sum_sqerr", initializer="zeros")
        self.count     = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, mask):
        """
        y_true, y_pred: (B,) FS values
        mask: (B,) bool or 0/1 float indicating which samples belong to THIS dataset
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mask   = tf.cast(mask, tf.float32)

        w = mask
        sqerr = (y_pred - y_true) ** 2
        self.sum_sqerr.assign_add(tf.reduce_sum(sqerr * w))
        self.count.assign_add(tf.reduce_sum(w))

    def result(self):
        def _compute():
            return self.sum_sqerr / (self.count + 1e-8)
        return tf.cond(self.count > 0.0,
                       _compute,
                       lambda: tf.constant(float("nan"), tf.float32))

    def reset_state(self):
        for v in self.variables:
            v.assign(0.0)

# class FrameshiftAUCMetric(tf.keras.metrics.Metric):
#     def __init__(self, name="fs_auc", **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.y_trues = []
#         self.y_scores = []

#     def update_state(self, y_true_binary, y_score, mask):
#         mask = tf.cast(mask, tf.bool)

#         # filter
#         y_true_binary = tf.boolean_mask(y_true_binary, mask)
#         y_score       = tf.boolean_mask(y_score, mask)

#         # DO NOT call .numpy() here — keep tensors in python list
#         self.y_trues.append(y_true_binary)
#         self.y_scores.append(y_score)

#     def result(self):
#         if len(self.y_trues) == 0:
#             return tf.constant(np.nan, tf.float32)

#         # At result() time, we are outside test_step graph.
#         # We can safely convert to numpy.
#         y_true  = tf.concat(self.y_trues, axis=0).numpy()
#         y_score = tf.concat(self.y_scores, axis=0).numpy()

#         if len(np.unique(y_true)) < 2:
#             return tf.constant(np.nan, tf.float32)

#         auc = roc_auc_score(y_true, y_score)
#         return tf.constant(float(auc), tf.float32)

#     def reset_state(self):
#         self.y_trues = []
#         self.y_scores = []


def compute_frameshift_rate(y, masks):
    """
    y:     (B, C) distribution (already masked + renormalized)
    masks: (B, C) mask for each sample
    """
    # The class index 0 = -100
    B = tf.shape(y)[0]
    delta_values = tf.range(MIN_DELTA, MAX_DELTA + 1, dtype=tf.int32)  # (C,)

    # Expand to (B, C)
    delta_values = tf.cast(delta_values[None, :], tf.int32)

    # Frameshift mask: 1 if abs(delta)%3 != 0
    fs_mask = tf.cast(tf.not_equal(tf.math.floormod(delta_values, 3), 0), tf.float32)

    # But also obey dataset mask
    fs_mask = fs_mask * masks  # (B, C)

    # Frameshift rate = sum(prob of frameshift deltas)
    fs_rate = tf.reduce_sum(y * fs_mask, axis=-1)  # (B,)
    return fs_rate


def pearson_corr(a, b):
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)
    am = tf.reduce_mean(a)
    bm = tf.reduce_mean(b)
    cov = tf.reduce_sum((a - am) * (b - bm))
    std_a = tf.sqrt(tf.reduce_sum((a - am)**2) + 1e-8)
    std_b = tf.sqrt(tf.reduce_sum((b - bm)**2) + 1e-8)
    return cov / (std_a * std_b + 1e-8)





def compute_frameshift_rate_numpy(y, deltas):
    """
    y:      (B, C) masked + renormalized distributions
    deltas: (C,)   array of delta values from MIN_DELTA to MAX_DELTA
    """
    deltas = np.asarray(deltas)

    # Frameshift mask: 1 where abs(delta)%3 != 0
    fs_mask = (np.abs(deltas) % 3 != 0).astype(np.float32)  # (C,)

    # Compute FS rate per sample
    fs_rate = (y * fs_mask).sum(axis=1)  # (B,)
    return fs_rate.astype(np.float32)




def report_fs_metrics(model, dataset, MASK_TABLE_TF, DATASET_MEDIANS_TF,
                      NUM_DATASETS, index_to_DS_name):
    print("\n==== Computing Final Validation Metrics per Dataset ====\n")

    all_fs_true = []
    all_fs_pred = []
    all_fs_binary = []
    all_dsids = []

    deltas = np.arange(MIN_DELTA, MAX_DELTA + 1)

    # ---------------------------------------------------------
    # GATHER ALL PREDICTIONS
    # ---------------------------------------------------------
    for batch in dataset:
        inputs, y_true = batch
        dsids = inputs["dataset_id"].numpy()          # (B,)
        y_pred = model(inputs, training=False).numpy()

        masks = MASK_TABLE_TF.numpy()[dsids]

        # Mask + renorm prediction
        y_pred_masked = y_pred * masks
        y_pred_norm = y_pred_masked / (y_pred_masked.sum(axis=1, keepdims=True) + 1e-9)

        # Mask + renorm truth
        y_true = y_true.numpy()
        y_true_masked = y_true * masks
        y_true_norm = y_true_masked / (y_true_masked.sum(axis=1, keepdims=True) + 1e-9)

        # FS values
        fs_pred = compute_frameshift_rate_numpy(y_pred_norm, deltas)
        fs_true = compute_frameshift_rate_numpy(y_true_norm, deltas)

        # Append all
        all_fs_pred.extend(fs_pred)
        all_fs_true.extend(fs_true)
        all_dsids.extend(dsids)

        # Binary labels using dataset medians
        medians = DATASET_MEDIANS_TF.numpy()[dsids]
        fs_bin = fs_true >= medians
        all_fs_binary.extend(fs_bin)

    # Convert to numpy
    all_fs_true = np.array(all_fs_true)
    all_fs_pred = np.array(all_fs_pred)
    all_fs_binary = np.array(all_fs_binary)
    all_dsids = np.array(all_dsids)

    # ---------------------------------------------------------
    # COMPUTE METRICS PER DATASET
    # ---------------------------------------------------------
    rows = []

    print("Final Results:\n")
    print(f"{'Dataset':25s} | {'AUC':6s} | {'Pearson':8s} | {'Spearman':9s} | {'MSE':8s} | N")

    for dsid in range(NUM_DATASETS):
        # if 0 samples -> skip
        if np.sum(all_dsids == dsid) == 0:
            continue
        name = index_to_DS_name[dsid]
        mask = (all_dsids == dsid)

        y_true = all_fs_true[mask]
        y_pred = all_fs_pred[mask]
        y_true_bin = all_fs_binary[mask]

        # Compute metrics safely
        # AUC
        if len(np.unique(y_true_bin)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y_true_bin, y_pred)

        # Pearson
        if len(y_true) > 1:
            pear, _ = pearsonr(y_true, y_pred)
        else:
            pear = np.nan

        # Spearman
        if len(y_true) > 1:
            spear, _ = spearmanr(y_true, y_pred)
        else:
            spear = np.nan

        # MSE
        mse = mean_squared_error(y_true, y_pred)

        print(f"{name:25s} | {auc:6.3f} | {pear:8.3f} | {spear:9.3f} | {mse:8.4f} | {mask.sum()}")

        rows.append({
            "dataset": name,
            "AUC": auc,
            "Pearson": pear,
            "Spearman": spear,
            "MSE": mse,
            "N_samples": mask.sum(),
        })

    print("\n=============================================\n")

    # Return DataFrame
    return pd.DataFrame(rows)

from scripts.utils import *
from scripts.loss_and_metrics import *
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split





########################################
# CUSTOM TRAINING LOOP MODEL
########################################

class CRISPRModel(tf.keras.Model):
    def __init__(self, base_model, NUM_DATASETS,MASK_TABLE_TF,DATASET_MEDIANS_TF,DATASET_SIZE_SCALE_TF, alpha_start=0.00, alpha_decay_epochs=20, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.num_datasets = NUM_DATASETS
        self.mask_table = MASK_TABLE_TF
        self.size_scale = DATASET_SIZE_SCALE_TF

        # for cosine decay on embedding regularizer
        self.alpha_start = alpha_start             # starting strength
        self.alpha_decay_epochs = alpha_decay_epochs        # reaches zero at epoch 20
        self.current_epoch = 0

        # Loss trackers
        self.train_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_loss_tracker   = tf.keras.metrics.Mean(name="val_loss")

        # Per-dataset FS metrics (Pearson + MSE)
        self.val_fs_pearson_ds = []
        self.val_fs_mse_ds     = []
        #self.val_fs_auc_ds     = []
        for dsid in range(NUM_DATASETS):
            name = index_to_DS_name[dsid]
            self.val_fs_pearson_ds.append(
                FrameshiftPearsonMetric(name=f"val_fs_pearson_{name}")
            )
            self.val_fs_mse_ds.append(
                FrameshiftMSEMetric(name=f"val_fs_mse_{name}")
            )
            #self.val_fs_auc_ds.append(
            #    FrameshiftAUCMetric(name=f"val_fs_auc_{name}")
            #)
        self.dataset_medians = DATASET_MEDIANS_TF

    # @property
    # def metrics(self):
    #     # Keras will reset these between epochs
    #     return (
    #         [self.train_loss_tracker, self.val_loss_tracker]
    #         + self.val_fs_pearson_ds
    #         + self.val_fs_mse_ds
    #     )
    @property
    def metrics(self):
        # Only basic metrics should be tracked by Keras itself
        # (we will log per-dataset FS metrics manually in CleanLogger)
        return [
            self.train_loss_tracker,
            self.val_loss_tracker,
        ]



    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    ###########################################################
    # TRAIN STEP
    ###########################################################
    def train_step(self, data):
        inputs, y_true = data
        dsids = inputs["dataset_id"]

        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            loss = masked_balanced_kld(y_true, y_pred, dsids, self.mask_table)


        grads = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))

        self.train_loss_tracker.update_state(loss)
        return {"loss": self.train_loss_tracker.result()}

    ###########################################################
    # TEST STEP — compute FS and update streaming metrics
    ###########################################################
    def test_step(self, data):
        inputs, y_true = data
        dsids = inputs["dataset_id"]
        y_pred = self(inputs, training=False)

        # Masks
        masks = tf.gather(self.mask_table, dsids)

        # Mask + renorm pred
        y_pred_masked = y_pred * masks
        y_pred_norm = tf.math.divide_no_nan(
            y_pred_masked, tf.reduce_sum(y_pred_masked, axis=-1, keepdims=True)
        )

        # Mask + renorm true
        y_true_masked = y_true * masks
        y_true_norm = tf.math.divide_no_nan(
            y_true_masked, tf.reduce_sum(y_true_masked, axis=-1, keepdims=True)
        )

        # FS values
        fs_pred = compute_frameshift_rate(y_pred_norm, masks)  # (B,)
        fs_true = compute_frameshift_rate(y_true_norm, masks)  # (B,)

        # Loss
        loss = masked_balanced_kld(y_true, y_pred, dsids, self.mask_table)
        self.val_loss_tracker.update_state(loss)

        # FOR AUC
        #medians = tf.gather(self.dataset_medians, dsids)
        #fs_true_binary = tf.cast(fs_true >= medians, tf.float32)  # (B,)        

        # Update per-dataset metrics (streaming over full val set)
        for dsid in range(self.num_datasets):
            mask_ds = tf.equal(dsids, dsid)  # (B,) bool
            self.val_fs_pearson_ds[dsid].update_state(fs_true, fs_pred, mask_ds)
            self.val_fs_mse_ds[dsid].update_state(fs_true, fs_pred, mask_ds)
            #self.val_fs_auc_ds[dsid].update_state(fs_true_binary, fs_pred, mask_ds)

        results = {
            "loss": self.val_loss_tracker.result()   # Keras will print as "val_loss"
        }

        # Add FS metrics
        # for dsid in range(self.num_datasets):
        #     name = index_to_DS_name[dsid]
        #     results[f"fs_pearson_{name}"] = self.val_fs_pearson_ds[dsid].result()
        #     results[f"fs_mse_{name}"]     = self.val_fs_mse_ds[dsid].result()
        #     results[f"fs_auc_{name}"]     = self.val_fs_auc_ds[dsid].result()
        # return results
        # Only return basic metrics per batch
        return {"val_loss": self.val_loss_tracker.result()}
    

class EpochSetter(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.current_epoch = float(epoch)


class EpochSetterMax(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.current_epoch = float(100000000) #so we dont decay anymore


class CleanLogger(tf.keras.callbacks.Callback):

    # def on_epoch_end(self, epoch, logs=None):
    #     logs = logs or {}
    #     print("\n================ EPOCH %d =================" % (epoch+1))
    #     #print("Loss:       %.4f" % logs.get("loss", float("nan")))
    #     #print("Val Loss:   %.4f" % logs.get("val_loss", float("nan")))
    #     print("-------------------------------------------")

    #     # Collect dataset metrics
    #     ds_metrics = {}

    #     for key, val in logs.items():
    #         # match metric names like:
    #         #   val_fs_pearson_FORECasT K562
    #         #   val_fs_mse_InDelphi mESC
    #         if key.startswith("val_fs_"):
    #             parts = key.split("_", 3)
    #             #   ['val', 'fs', 'pearson', 'FORECasT K562']
    #             metric_type = parts[2]      # pearson or mse
    #             ds_name = parts[3]          # everything after

    #             if ds_name not in ds_metrics:
    #                 ds_metrics[ds_name] = {}
    #             ds_metrics[ds_name][metric_type] = val

    #     # pretty print
    #     for ds, m in ds_metrics.items():
    #         pearson = m.get("pearson", float("nan"))
    #         mse     = m.get("mse", float("nan"))
    #         #auc     = m.get("auc", float("nan"))
    #         print(f"{ds:28s} |  Pearson: {pearson:6.3f}   MSE: {mse:7.4f}")
    #         #print(f"{ds:28s} |  Pearson: {pearson:6.3f}   MSE: {mse:7.4f}   AUC: {auc:6.3f}")


    #     print("===========================================\n")
    def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            model = self.model

            print("\n================ EPOCH %d =================" % (epoch+1))
            #print("Loss:       %.4f" % logs.get("loss", float("nan")))
            #print("Val Loss:   %.4f" % logs.get("val_loss", float("nan")))
            print("-------------------------------------------")

            for dsid in range(model.num_datasets):
                name = index_to_DS_name[dsid]

                pear = model.val_fs_pearson_ds[dsid].result().numpy()
                mse  = model.val_fs_mse_ds[dsid].result().numpy()
                #auc  = model.val_fs_auc_ds[dsid].result().numpy()

                #print(f"{name:28s} |  Pearson: {pear:6.3f}   MSE: {mse:7.4f}   AUC: {auc:6.3f}")
                print(f"{name:28s} |  Pearson: {pear:6.3f}   MSE: {mse:7.4f}   ")

            # reset for next epoch
            for m in model.val_fs_pearson_ds + model.val_fs_mse_ds: #+ model.val_fs_auc_ds:
                m.reset_state()

            print("===========================================\n")


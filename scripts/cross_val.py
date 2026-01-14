
from scripts.utils import *
from scripts.data import get_datasets
from scripts.model import build_transformer, freeze_all_non_dataset_embedding_layers
from scripts.training import CRISPRModel, CleanLogger, EpochSetter, EpochSetterMax
from scripts.loss_and_metrics import report_fs_metrics
from scripts.checkpointing import save_model_checkpoint

import numpy as np
import pandas as pd
import tensorflow as tf


def basic_cross_val(seed=42, use_mh=True, filter_ids=[], use_legacy_split=True, use_val = False, save_splits=False, use_ds_embedding = True):



    print(f"Starting basic cross-validation..., seed={seed}, use_mh={use_mh}, filter_ids={filter_ids}, save_splits={save_splits}")


    all_reports = []
    
    for val_bucket in range(N_TRAIN_VAL_FOLDS): # was N_TRAIN_VAL_FOLDS
        if use_val:
            print(f"\n\n=== SPLIT WITH VAL BUCKET {val_bucket} ===\n")
        train_ds, val_ds, test_ds,\
        MASK_TABLE_TF, DATASET_WEIGHTS_TF,DATASET_MEDIANS_TF,\
        DATASET_SIZE_SCALE_TF, NUM_DATASETS = get_datasets(
                                                    seed=seed,
                                                    val_bucket=val_bucket,
                                                    save_splits=save_splits,
                                                    filter_ids=filter_ids,
                                                    use_legacy_split=use_legacy_split,
                                                    use_val=use_val)  

        model_config = {
        "n_datasets": int(NUM_DATASETS),
        "use_mh": use_mh,
        "d_model": 128,
        "n_attention_layers": 3,
        "n_attention_heads": 4,
        "dropout_rate": 0.05,
        "use_weird_rope": False,
        "add_to_embedding": True,
        "add_ds_embedding": use_ds_embedding, 
        }

        # print model config
        print("MODEL CONFIG:")
        for k, v in model_config.items():
            print(f"  {k}: {v}")

        base_model = build_transformer(model_config)
        model = CRISPRModel(
            base_model,
            NUM_DATASETS,
            MASK_TABLE_TF,
            DATASET_MEDIANS_TF,
            DATASET_SIZE_SCALE_TF,
            alpha_start=ALPHA_START,
            alpha_decay_epochs=ALPHA_DECAY_EPOCHS) 
        

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3)
        )

        print("\nStarting training...\n")
        if use_val:
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=N_EPOCHS_FIRST_PHASE, 
                callbacks=[CleanLogger(), EpochSetter()],
            )
        else:
            history = model.fit(
                train_ds,
                epochs=N_EPOCHS_FIRST_PHASE, 
                callbacks=[],
            )
        if use_ds_embedding:
            freeze_all_non_dataset_embedding_layers(model)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-3)
            )
            if use_val:
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=N_EPOCHS_SECOND_PHASE, 
                    callbacks=[CleanLogger(), EpochSetterMax()],
                )
            else:
                history = model.fit(
                    train_ds,
                    epochs=N_EPOCHS_SECOND_PHASE, 
                    callbacks=[],
                )
        if use_val:
            report_df = report_fs_metrics(
                model,
                val_ds,
                MASK_TABLE_TF,
                DATASET_MEDIANS_TF,
                NUM_DATASETS,
                index_to_DS_name)
        

            all_reports.append(report_df)

        save_model_checkpoint(
            model,
            fold=val_bucket,
            MASK_TABLE=MASK_TABLE_TF.numpy(),
            DATASET_MEDIANS=DATASET_MEDIANS_TF.numpy(),
            DATASET_SIZE_SCALE=DATASET_SIZE_SCALE_TF.numpy(),
            dataset_names=index_to_DS_name,
            extra_metadata={
                "MODEL_CONFIG": model_config,  ### <<<<<< SAVE IT HERE
                "MIN_DELTA": MIN_DELTA,
                "MAX_DELTA": MAX_DELTA,
                "DELTA_OFFSET": DELTA_OFFSET,
            }
        )

        # discard the model and optimizer andd clear session to avoid GPU memory leak
        del model
        tf.keras.backend.clear_session()
    if use_val:
        combined = pd.concat(all_reports, ignore_index=True)

        df_mean = combined.groupby('dataset').mean()
        df_std = combined.groupby('dataset').std()

        print("\n\n====== CROSS-VALIDATION SUMMARY ======\n")
        print("Mean metrics:\n", df_mean)
        print("\nSTD metrics:\n", df_std)

        return df_mean, df_std, all_reports
    else:
        return None, None, None

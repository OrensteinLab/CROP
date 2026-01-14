import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 0 OR 1   
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
from scripts.utils import *
from scripts.cross_val import basic_cross_val
from scripts.ensemble_eval import evaluate_cross_dataset_ensemble
from scripts.interpertability import analyze_dataset_outcomes, analyze_dataset_outcomes_pairwise, sweep_synthetic_mh_efficiency

import numpy as np
import pandas as pd
import tensorflow as tf
#tf.debugging.enable_check_numerics()
#tf.debugging.set_log_device_placement(True)
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy("mixed_float16")
#tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")




def save_reports(df_mean, df_std, all_reports, subfolder="default"):
        if not os.path.exists("results"):
            os.makedirs("results")
        if not os.path.exists(f"results/{subfolder}"):
            os.makedirs(f"results/{subfolder}")
        df_mean.to_csv(f"results/{subfolder}/cross_val_report_mean.csv", index=False)
        df_std.to_csv(f"results/{subfolder}/cross_val_report_std.csv", index=False)
        # save all reports
        for i, report in enumerate(all_reports):
            report.to_csv(f"results/{subfolder}/cross_val_report_fold_{i}.csv", index=False)


# Show no overfitting on these datasets
def experiment_zero():
    print("Starting experiment zero...")
    for seed in [1]:
        use_mh = True
        filter_ids = [9]  
        use_legacy_split = True 

        df_mean, df_std, all_reports = basic_cross_val(
            seed=seed,
            use_mh=use_mh,
            filter_ids=filter_ids,
            use_legacy_split=use_legacy_split,
            use_val = True) 
        
        save_reports(df_mean, df_std, all_reports, subfolder="experiment_zero/seed_"+str(seed))


# Show vs all models that can run
def experiment_one():
    print("Starting experiment one...")
    for seed in [1]:
        use_mh = True
        filter_ids = []  # was [9]
        use_legacy_split = True 

        df_mean, df_std, all_reports = basic_cross_val(
            seed=seed,
            use_mh=use_mh,
            filter_ids=filter_ids,
            use_legacy_split=use_legacy_split,
            use_val=False,
            save_splits=True) 
        
        #save_reports(df_mean, df_std, all_reports, subfolder="experiment_one/seed_"+str(seed)) used when having a val set

        pearson, spearman, auc, mse = evaluate_cross_dataset_ensemble(
            num_folds=N_TRAIN_VAL_FOLDS,
            seed=seed,
            checkpoint_dir="saved_models/",
            save_dir=f"ensemble_results/experiment_one/seed_{seed}/",
            use_mh=use_mh,
            use_legacy_split=use_legacy_split,
            filter_sprout=False
            )
        
# Show best parameters
def experiment_two():
    print("Starting experiment two...")
    for seed in [1]:
        filter_ids = [9]  
        use_legacy_split = True 

        df_mean, df_std, all_reports = basic_cross_val(
            seed=seed,
            use_mh=False,
            filter_ids=filter_ids,
            use_legacy_split=use_legacy_split,
            use_val=True,
            save_splits=True,
            use_ds_embedding=True) 
        
        save_reports(df_mean, df_std, all_reports, subfolder="experiment_twoA/seed_"+str(seed))  # reports are on val set

        

    for seed in [1]:
        filter_ids = [9]  
        use_legacy_split = True 

        df_mean, df_std, all_reports = basic_cross_val(
            seed=seed,
            use_mh=True,
            filter_ids=filter_ids,
            use_legacy_split=use_legacy_split,
            use_val=True,
            save_splits=True,
            use_ds_embedding=False) 
        
        save_reports(df_mean, df_std, all_reports, subfolder="experiment_twoB/seed_"+str(seed)) # reports are on val set
    
    for seed in [1]:
        filter_ids = [9]  
        use_legacy_split = True 

        df_mean, df_std, all_reports = basic_cross_val(
            seed=seed,
            use_mh=True,
            filter_ids=filter_ids,
            use_legacy_split=use_legacy_split,
            use_val=True,
            save_splits=True,
            use_ds_embedding=True) 
        
        save_reports(df_mean, df_std, all_reports, subfolder="experiment_twoORIG/seed_"+str(seed)) # reports are on val set


# Show beatin croton + Apindel
# FOR THIS WE NEED TO USE SPECIFIC DATASETS - DONT RUN THIS ON OG DATASETS-> need to edit them to match CROTON/Apindel premise
# Like max ins =3 and remove mixed events
def experiment_three():
    print("Starting experiment three...")
    for seed in [1]:
        use_mh = True
        filter_ids = [9]  
        use_legacy_split = True 

        df_mean, df_std, all_reports = basic_cross_val(
            seed=seed,
            use_mh=use_mh,
            filter_ids=filter_ids,
            use_legacy_split=use_legacy_split,
            use_val=False,
            save_splits=True) 
        
        #save_reports(df_mean, df_std, all_reports, subfolder="experiment_three/seed_"+str(seed))

        pearson, spearman, auc, mse = evaluate_cross_dataset_ensemble(
            num_folds=N_TRAIN_VAL_FOLDS,
            seed=seed,
            checkpoint_dir="saved_models/", # These are the models we actually use
            save_dir=f"ensemble_results/experiment_three/seed_{seed}/",
            use_mh=use_mh,
            use_legacy_split=use_legacy_split,
            filter_sprout=True
            )


def experiment_interpertability():
    og_models = True
    print("Starting interpretability experiments...")

    # create folder of results/interpretability if not exists
    if not os.path.exists("results/interpretability"):
        os.makedirs("results/interpretability")

    for dataset_id in [0, 10, 9]:
        sweep_synthetic_mh_efficiency(
            dataset_id=dataset_id,
            gap_max=10,
            mh_max=20,
            n_sequences=200,
            checkpoint_dir="saved_models/",
            num_folds=N_TRAIN_VAL_FOLDS,
            og_models=og_models 
        )

    for target_delta in [-20, -15, -10, -7, -5, -3, -2, -1, 1 , 2 ,3]:
        analyze_dataset_outcomes(
            dataset_id_to_analyze=0,
            target_delta=target_delta,
            k=3,
            checkpoint_dir="saved_models/",
            num_folds=N_TRAIN_VAL_FOLDS,
            og_models=og_models 
        )

        analyze_dataset_outcomes_pairwise(
            dataset_id_to_analyze= 0,
            target_delta=target_delta,
            k=3,
            checkpoint_dir="saved_models/",
            num_folds=N_TRAIN_VAL_FOLDS,
            og_models=og_models 
        )





def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("GPUs:", tf.config.list_physical_devices('GPU'))

    #experiment_zero() - #DONE -> HP search using legacy split, without sprout  (Doesn't matter whichj sprout version)
    #experiment_one()#- DONE  -> best model (with train-test) legacy split, with sprout (good sprout)
    #experiment_two() # -> DONE  -> train-test-val only, no val, legacy split. We need to DISCARD original test, and then use the val split as the "real" test. Also use no sprout at all 
    #experiment_three() - DOING -> best model (with train-test) legacy split, without sprout (for CROTON+Apindel) + bad sprout test
    experiment_interpertability() #- DONE -> doing it with experiment one model (with good sprout?)

   

if __name__ == "__main__":
    main()




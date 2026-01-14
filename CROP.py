import os
import sys
import argparse
import pandas as pd
import numpy as np

# Set environment variables BEFORE importing tensorflow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

import tensorflow as tf

# Import from your original files
from scripts.data import center_sequence_around_pam, seq_to_token_ids
from scripts.checkpointing import load_model_checkpoint
from scripts.utils import MIN_DELTA, MAX_DELTA, N_TRAIN_VAL_FOLDS

# Define delta values globally for mapping based on project constants
DELTA_VALUES = np.arange(MIN_DELTA, MAX_DELTA + 1)

def run_crop_inference(input_csv, output_subfolder, vs_sota=False):
    # 1. Determine checkpoint directory based on flag
    checkpoint_base = "saved_models_vs_sota" if vs_sota else "saved_models" #
    print(f"Using models from: {checkpoint_base}")

    # 2. Setup Output Directory
    output_dir = os.path.join("results", output_subfolder)
    os.makedirs(output_dir, exist_ok=True)

    # 3. Load and Clean Input Data
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Map column names case-insensitively and remove spaces for matching
    col_map = {c.lower().replace(" ", ""): c for c in df.columns}
    seq_col = col_map.get("targetsequence")
    pam_col = col_map.get("pamposition")

    if not seq_col or not pam_col:
        print(f"Error: Required columns not found. Ensure CSV has 'Target sequence' and 'PAM position'.")
        return

    print(f"Loaded {len(df)} sequences. Preparing tokens...")

    # 4. Tokenization using the original centering and token ID logic
    tokens_list = []
    for _, row in df.iterrows():
        centered = center_sequence_around_pam(str(row[seq_col]), int(row[pam_col]))
        tokens_list.append(seq_to_token_ids(centered))
    X_tokens = np.array(tokens_list)

    # 5. Ensemble Prediction Setup
    # Load metadata using dynamic base_dir
    _, _, _, _, metadata = load_model_checkpoint(0, base_dir=checkpoint_base, og_model=True)
    index_to_name = metadata["dataset_names"]
    
    # Initialize the summary dataframe with the original input data
    fs_summary_df = df.copy()

    # Define the mask for calculating frameshift rates (deltas not divisible by 3)
    fs_mask = (np.abs(DELTA_VALUES) % 3 != 0).astype(np.float32)

    for ds_id_str, ds_name in index_to_name.items():
        ds_id = int(ds_id_str)
        print(f"Processing Cell Type: {ds_name}...")
        
        all_fold_probs = []
        mask = None

        # Predict across the defined number of folds for an ensemble result
        for fold in range(N_TRAIN_VAL_FOLDS):
            model, mask_table, _, _, _ = load_model_checkpoint(fold, base_dir=checkpoint_base, og_model=True)
            
            ds_ids = np.full((len(X_tokens),), ds_id, dtype=np.int32)
            probs = model.base_model.predict({"tokens": X_tokens, "dataset_id": ds_ids}, verbose=0)
            
            # Use the dataset-specific mask to zero out invalid repair outcomes
            mask = mask_table[ds_id] 
            probs_masked = probs * mask 
            
            # Re-normalize probabilities after masking
            row_sums = probs_masked.sum(axis=1, keepdims=True)
            probs_masked = np.divide(probs_masked, row_sums, out=np.zeros_like(probs_masked), where=row_sums!=0)
            
            all_fold_probs.append(probs_masked)

        # Average ensemble results
        mean_probs = np.mean(all_fold_probs, axis=0)

        # Calculate specific Frameshift Rate for this cell type/model
        fs_rates = (mean_probs * fs_mask).sum(axis=1)

        # 6. Create Individual Cell Type CSV
        # Identify indices where the mask is active for this dataset
        active_indices = np.where(mask == 1.0)[0]
        active_deltas = DELTA_VALUES[active_indices]
        
        # Filter mean probabilities to only include active repair outcome columns
        filtered_probs_df = pd.DataFrame(mean_probs[:, active_indices], 
                                         columns=[f"delta_{d}" for d in active_deltas])

        # Individual file includes: Original DF + Masked Outcomes + Cell Type FS Rate
        cell_type_df = pd.concat([df.reset_index(drop=True), 
                                  filtered_probs_df, 
                                  pd.Series(fs_rates, name=f"{ds_name}_frameshift_rate")], axis=1)
        
        safe_name = ds_name.replace(" ", "_").replace("/", "_")
        cell_type_df.to_csv(os.path.join(output_dir, f"{safe_name}_outcomes.csv"), index=False)

        # 7. Append this cell type's FS rate to the master summary
        fs_summary_df[f"{ds_name}_frameshift_rate"] = fs_rates

    # Save final aggregate summary (Original DF + All FS Rates)
    fs_summary_df.to_csv(os.path.join(output_dir, "frameshift_rates_summary.csv"), index=False)
    print(f"Success! Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CRISPR Repair Outcome Prediction (CROP)")
    parser.add_argument("input_csv", help="Path to the input .csv file")
    parser.add_argument("output_folder", help="Subfolder name under results/")
    parser.add_argument("-vs_sota_model", action="store_true", help="Use the model used to compare CROP with SOTA models, instead of the model trained on all datasets")
    
    args = parser.parse_args()
    
    run_crop_inference(args.input_csv, args.output_folder, vs_sota=args.vs_sota_model)
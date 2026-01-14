import numpy as np
import tensorflow as tf
import pandas as pd
import os
#import matplotlib.pyplot as plt
#import seaborn as sns

# Import configuration and utils
from scripts.utils import * 
from scripts.data import get_datasets, load_and_prepare_data, tokens_to_sequence
from scripts.checkpointing import load_model_checkpoint


def get_top_k_sequences_for_outcome(dataset_id, target_delta, k, X, y, dsids, min_reads=50):
    """
    Finds the top k sequences in a specific dataset where the target_delta 
    has the highest normalized probability, filtered by a minimum read count.
    """
    # Filter by Dataset ID
    ds_mask = (dsids == dataset_id)
    X_ds = X[ds_mask]
    y_ds = y[ds_mask]
    
    # 1. Calculate total reads per sequence
    row_sums = np.sum(y_ds, axis=1) # Shape (N,)
    
    # 2. Apply the min_reads filter
    read_mask = row_sums >= min_reads
    X_filtered = X_ds[read_mask]
    y_filtered = y_ds[read_mask]
    reads_filtered = row_sums[read_mask]
    
    if len(X_filtered) == 0:
        print(f"Warning: No sequences in Dataset {dataset_id} meet the {min_reads} read threshold.")
        return np.array([]), np.array([])

    # 3. Normalize the filtered counts to probabilities
    y_norm = y_filtered / reads_filtered[:, np.newaxis]
    
    target_class_idx = target_delta + DELTA_OFFSET #
    
    if target_class_idx < 0 or target_class_idx >= NUM_CLASSES: #
        raise ValueError(f"Delta {target_delta} is out of bounds (Class index {target_class_idx})")

    target_probs = y_norm[:, target_class_idx]
    
    # 4. Handle cases where fewer than k sequences remain
    actual_k = min(k, len(target_probs))
    
    # Get top k indices
    top_indices = np.argsort(target_probs)[-actual_k:][::-1]
    
    top_sequences = X_filtered[top_indices]
    top_probs = target_probs[top_indices]
    
    print(f"\nFound top {actual_k} sequences (>= {min_reads} reads) in Dataset {dataset_id} "
          f"for event Delta={target_delta}")
    
    return top_sequences, top_probs

def perform_in_silico_mutagenesis(model, tokens, dataset_id, target_class_idx):
    """
    Performs comprehensive In-Silico Mutagenesis (ISM).
    Returns a matrix (L, VOCAB_SIZE) of Delta Scores (P_mutant - P_wt).
    """
    # 1. Baseline Prediction
    tokens_tf = tf.convert_to_tensor([tokens], dtype=tf.int32)
    dsid_tf = tf.convert_to_tensor([dataset_id], dtype=tf.int32)
    
    preds_base = model({"tokens": tokens_tf, "dataset_id": dsid_tf}, training=False)
    p_base = preds_base[0, target_class_idx].numpy()
    
    seq_len = len(tokens)
    mutated_seqs = []
    
    # Define valid mutation targets from VOCAB
    # We only want to mutate TO these bases, and FROM these bases.
    # We do NOT mutate PAD or N, and we do not mutate TO PAD or N.
    valid_bases = [VOCAB["A"], VOCAB["C"], VOCAB["G"], VOCAB["T"]]
    
    # Map mutation index back to (pos, base_idx)
    mutation_map = [] 
    
    for pos in range(seq_len):
        original_token = tokens[pos]
        
        # SKIP if the original position is PAD or N
        if original_token not in valid_bases: 
            continue
            
        for base in valid_bases:
            # SKIP mutation to self
            if base == original_token: 
                continue
            
            # Create Mutant
            mut_tokens = tokens.copy()
            mut_tokens[pos] = base
            mutated_seqs.append(mut_tokens)
            mutation_map.append((pos, base))
            
    if not mutated_seqs:
        # If sequence is all Ns or PADs, return empty grid
        return np.zeros((seq_len, len(VOCAB)))

    # 3. Batch Prediction
    batch_tokens = tf.convert_to_tensor(np.array(mutated_seqs), dtype=tf.int32)
    batch_dsids = tf.tile(dsid_tf, [len(mutated_seqs)])
    
    preds_mut = model({"tokens": batch_tokens, "dataset_id": batch_dsids}, training=False)
    p_muts = preds_mut[:, target_class_idx].numpy()
    
    # 4. Construct ISM Matrix (L, VOCAB_SIZE)
    # We use VOCAB_SIZE so indices match exactly (e.g. column 4 is T)
    ism_matrix = np.zeros((seq_len, len(VOCAB)))
    
    for idx, (pos, base) in enumerate(mutation_map):
        # Delta = P(Mutant) - P(WT)
        ism_matrix[pos, base] = p_muts[idx] - p_base
        
    return ism_matrix

def analyze_dataset_outcomes(
    dataset_id_to_analyze=0, 
    target_delta=-10, 
    k=3, 
    checkpoint_dir="saved_models/",
    num_folds=N_TRAIN_VAL_FOLDS, 
    seed=42,
    og_models=False
):
    # --- Load Data ---
    print(f"Loading Data (Seed {seed})...")
    # Using your config params from utils implicitly via imports
    train_ds, val_ds, test_ds, _, _, _, _, _ = get_datasets(seed=seed, val_bucket=0, save_splits=False, filter_ids=[], use_legacy_split=False, normalize=False) #Using legacy split here doesn't matter? TODO: check
    
    # Helper to concat
    def extract_all(ds_list):
        X_all, y_all, id_all = [], [], []
        for d in ds_list:
            for inputs, y in d:
                X_all.append(inputs['tokens'].numpy())
                id_all.append(inputs['dataset_id'].numpy())
                y_all.append(y.numpy())
        return np.concatenate(X_all), np.concatenate(y_all), np.concatenate(id_all)

    X, y, dsids = extract_all([train_ds, val_ds, test_ds])
    
    # Reuse previous helper (ensure it is defined in your file)
    top_seqs, top_probs = get_top_k_sequences_for_outcome(
        dataset_id=dataset_id_to_analyze, 
        target_delta=target_delta, 
        k=k, X=X, y=y, dsids=dsids
    )

    target_class_idx = target_delta + DELTA_OFFSET
    
    # Reverse Vocab for display: {1: 'A', 2: 'C', ...}
    reverse_vocab = {v: k for k, v in VOCAB.items()}

    # --- Load Models ---
    print(f"\nLoading {num_folds} ensemble models...")
    models = []
    for fold in range(num_folds):
        try:
            model, _, _, _, _ = load_model_checkpoint(fold, checkpoint_dir, og_model=og_models)
            models.append(model)
        except Exception:
            pass

    if not models: return

    # --- ISM Analysis ---
    print(f"\nPerforming ISM with VOCAB mapping: {VOCAB}")
    print("-" * 65)
    
    for i, (tokens, prob) in enumerate(zip(top_seqs, top_probs)):
        
        # Calculate ISM
        ensemble_ism = np.zeros((len(tokens), len(VOCAB)), dtype=np.float32)
        for model in models:
            ism_matrix = perform_in_silico_mutagenesis(model, tokens, dataset_id_to_analyze, target_class_idx)
            ensemble_ism += ism_matrix
        ensemble_ism /= len(models)
        
        # Build DataFrame
        data = []
        for pos, token in enumerate(tokens):
            
            # Check for PAD or N using your VOCAB
            if token == VOCAB["PAD"] or token == VOCAB["N"]:
                continue 
            
            wt_base = reverse_vocab.get(token, '?')
            
            # Extract scores for ACGT only
            row_data = {'Pos': pos, 'WT': wt_base}
            
            # Add columns for A, C, G, T dynamically
            for base_char in ["A", "C", "G", "T"]:
                base_idx = VOCAB[base_char]
                score = ensemble_ism[pos, base_idx]
                row_data[f"> {base_char}"] = score
            
            data.append(row_data)
            
        df = pd.DataFrame(data)
        
        # Display
        print(f"\nSequence Rank {i+1} | Base Prob: {prob:.4f}")
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.float_format', '{:+.4f}'.format)
        
        # Center view around middle of valid sequence
        # if len(df) > 0:
        #     center = len(df) // 2
        #     # Adjust window size as needed
        #     view_df = df.iloc[max(0, center-10) : min(len(df), center+10)]
        #     print(view_df.to_string(index=False))
            
        #     # Save
        #     csv_name = f"ism_seq_{i+1}_delta_{target_delta}.csv"
        #     df.to_csv(csv_name, index=False)


        if len(df) > 0:
            print(df.to_string(index=False))
            
            # Save
            csv_name = f"results/interpretability/ism_{index_to_DS_name[dataset_id_to_analyze]}_delta_{target_delta}_{i+1}.csv"
            df.to_csv(csv_name, index=False)
            print(f"\nSaved full matrix to {csv_name}")


def compute_full_ism_matrix(model, tokens, dataset_id, target_class_idx):
    """
    Helper: Computes the standard (L, 4) ISM matrix for a single sequence.
    Returns a matrix where Entry[pos, base] = Prob(mutant) - Prob(WT).
    """
    seq_len = len(tokens)
    valid_bases = [VOCAB["A"], VOCAB["C"], VOCAB["G"], VOCAB["T"]]
    
    # 1. Prepare Batch of all 3*L single mutants
    mutants = []
    mutant_map = [] # (pos, base_idx_in_vocab)
    
    # We also need the WT probability for delta calculation
    # We can include WT in the batch or run it separately.
    # Let's run it separately for clarity.
    tokens_tf = tf.convert_to_tensor([tokens], dtype=tf.int32)
    dsid_tf = tf.convert_to_tensor([dataset_id], dtype=tf.int32)
    
    p_wt = model({"tokens": tokens_tf, "dataset_id": dsid_tf}, training=False)[0, target_class_idx].numpy()
    
    for i in range(seq_len):
        if tokens[i] not in valid_bases: continue
        for b in valid_bases:
            if b == tokens[i]: continue
            
            mut = tokens.copy()
            mut[i] = b
            mutants.append(mut)
            mutant_map.append((i, b))
            
    if not mutants: return np.zeros((seq_len, len(VOCAB)))

    # 2. Batch Predict
    # Chunking is good practice if L is very large, but for L=100, 300 samples is tiny.
    batch_tokens = tf.convert_to_tensor(np.array(mutants), dtype=tf.int32)
    batch_dsids = tf.tile(dsid_tf, [len(mutants)])
    
    preds = model({"tokens": batch_tokens, "dataset_id": batch_dsids}, training=False)
    probs = preds[:, target_class_idx].numpy()
    
    # 3. Fill Matrix (L, VOCAB_SIZE)
    # We use raw probability differences (Delta)
    ism_matrix = np.zeros((seq_len, len(VOCAB)))
    
    for idx, (pos, base) in enumerate(mutant_map):
        ism_matrix[pos, base] = probs[idx] - p_wt
        
    return ism_matrix

def get_influence_map_shifts(model, tokens, dataset_id, target_class_idx):
    """
    Calculates the LxL 'Influence Map' based on ISM Shifts.
    
    Algorithm:
    1. Base_ISM = ISM(WT)
    2. For each pos 'i':
        For each mutation 'm' at 'i':
            Mut_ISM = ISM(Seq with i->m)
            Diff_Matrix = Abs(Mut_ISM - Base_ISM)
            Impact_Vector = Sum(Diff_Matrix, axis=nucleotides) -> (L,)
        Avg_Impact = Mean(Impact_Vectors of all m)
        Influence_Matrix[i] = Avg_Impact
    """
    seq_len = len(tokens)
    valid_bases = [VOCAB["A"], VOCAB["C"], VOCAB["G"], VOCAB["T"]]
    
    # 1. Baseline ISM (L, Vocab)
    # We use the raw matrix for comparison
    base_ism = compute_full_ism_matrix(model, tokens, dataset_id, target_class_idx)
    
    influence_matrix = np.zeros((seq_len, seq_len))
    
    # 2. Iterate Source Positions (i)
    # We need to run this efficiently. 
    # Total inferences = L * 3 (source mutations) * (L * 3 single mutants for ISM).
    # This is ~9 * L^2. For L=100 -> ~90,000 inferences.
    # We will loop 'i' but batch the inner ISM calculation.
    
    for i in range(seq_len):
        if tokens[i] not in valid_bases: continue
        
        # Accumulate the impact vector for position i across its 3 mutations
        sum_impact_vector = np.zeros(seq_len)
        mutation_count = 0
        
        # For each possible mutation at i
        for b_i in valid_bases:
            if b_i == tokens[i]: continue
            
            # Create the "Source Mutant"
            source_mut_seq = tokens.copy()
            source_mut_seq[i] = b_i
            
            # Run ISM on this new background
            # Note: This returns (L, Vocab)
            curr_ism = compute_full_ism_matrix(model, source_mut_seq, dataset_id, target_class_idx)
            
            # Calculate Absolute Difference between New ISM and Old ISM
            # "How much did the map change?"
            # Shape: (L, Vocab)
            diff_matrix = np.abs(curr_ism - base_ism)
            
            # Sum absolute diffs per position j
            # "Total shift for position j"
            # Shape: (L,)
            impact_vector = np.sum(diff_matrix, axis=1)
            
            # Zero out the source position itself (i) because its ISM is 0 by definition 
            # (since we fixed it to a specific base, we can't mutate it away from itself in the same way)
            impact_vector[i] = 0.0
            
            sum_impact_vector += impact_vector
            mutation_count += 1
            
        # Average over the 3 mutations
        if mutation_count > 0:
            avg_impact = sum_impact_vector / mutation_count
            influence_matrix[i, :] = avg_impact

    return influence_matrix, 0.0 # Return 0.0 dummy prob since we don't use it for plot title scaling

# def plot_influence_heatmap(matrix, tokens, prob, title_prefix="", save_path=None):
#     """
#     Plots the asymmetric influence matrix.
#     Rows = Source (Influencer), Cols = Target (Influenced).
#     """
#     rev_vocab = {v: k for k, v in VOCAB.items()}
#     seq_chars = [rev_vocab.get(t, '?') for t in tokens]
#     labels = [f"{c}\n{i}" for i, c in enumerate(seq_chars)]
    
#     plt.figure(figsize=(22, 18))
    
#     # No masking needed for asymmetric map (unless you want to mask diagonal)
#     mask = np.eye(len(matrix), dtype=bool) 

#     sns.heatmap(
#         matrix, 
#         mask=mask,
#         cmap="magma", # Magma/Inferno are great for "Intensity/Impact"
#         vmin=0.0,
#         square=True,
#         xticklabels=labels, 
#         yticklabels=labels,
#         cbar_kws={'label': 'Avg ISM Shift (Influence Score)', 'shrink': 0.7}
#     )
    
#     plt.xlabel("Target Position (Influenced)")
#     plt.ylabel("Source Position (Influencer)")
#     plt.title(f"{title_prefix} Positional Influence Map (ISM Shift) | Base Prob: {prob:.4f}", fontsize=16)
#     plt.xticks(fontsize=8)
#     plt.yticks(fontsize=8, rotation=0)
    
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=150)
#         print(f"Saved heatmap to {save_path}")
#         plt.close()
#     else:
#         plt.show()

def analyze_dataset_outcomes_pairwise(
    dataset_id_to_analyze=0, 
    target_delta=-10, 
    k=1, 
    checkpoint_dir="saved_models/",
    num_folds=N_TRAIN_VAL_FOLDS, 
    seed=42,
    og_models=False
):
    # 1. Load Data
    print(f"Loading Data (Seed {seed})...")
    results = get_datasets(seed=seed, val_bucket=0, save_splits=False, filter_ids=[], use_legacy_split=False, normalize=False)
    
    if len(results) == 11:
        # Optimized unpacking
        train_ds, val_ds, test_ds, _, _, _, _, _, (X_tr, y_tr, id_tr), (X_val, y_val, id_val), (X_te, y_te, id_te) = results
        X = np.concatenate([X_tr, X_val, X_te])
        y = np.concatenate([y_tr, y_val, y_te])
        dsids = np.concatenate([id_tr, id_val, id_te])
    else:
        # Fallback unpacking
        print("Extracting data (fallback)...")
        train_ds, val_ds, test_ds, _, _, _, _, _ = results
        X_all, y_all, id_all = [], [], []
        for ds in [train_ds, val_ds, test_ds]:
            for inputs, y_batch in ds:
                X_all.append(inputs['tokens'].numpy())
                id_all.append(inputs['dataset_id'].numpy())
                y_all.append(y_batch.numpy())
        X = np.concatenate(X_all)
        y = np.concatenate(y_all)
        dsids = np.concatenate(id_all)

    # 2. Get Top K Sequences
    print(f"\nFinding Top {k} sequences for Delta {target_delta}...")
    
    mask = (dsids == dataset_id_to_analyze)
    X_ds = X[mask]
    y_ds = y[mask]
    
    row_sums = np.sum(y_ds, axis=1, keepdims=True)
    y_norm = np.divide(y_ds, row_sums, out=np.zeros_like(y_ds), where=row_sums!=0)
    
    target_class_idx = target_delta + DELTA_OFFSET
    target_probs = y_norm[:, target_class_idx]
    
    top_indices = np.argsort(target_probs)[-k:][::-1]
    top_seqs = X_ds[top_indices]
    top_probs = target_probs[top_indices]

    # 3. Load Models
    print(f"\nLoading {num_folds} ensemble models...")
    models = []
    for fold in range(num_folds):
        try:
            model, _, _, _, _ = load_model_checkpoint(fold, checkpoint_dir, og_model=og_models)
            models.append(model)
        except: pass
    
    if not models:
        print("No models loaded!")
        return

    # 4. Perform Influence Analysis
    print(f"\nPerforming Influence Analysis (ISM Shift Method)...")
    print("Note: This performs ~90k inferences per sequence. It may take a minute.")
    
    for i, (tokens, prob) in enumerate(zip(top_seqs, top_probs)):
        print(f"  Analyzing Sequence Rank {i+1} (Prob: {prob:.4f})...")
        
        ensemble_matrix = np.zeros((len(tokens), len(tokens)))
        
        for model in models:
            matrix, _ = get_influence_map_shifts(model, tokens, dataset_id_to_analyze, target_class_idx)
            ensemble_matrix += matrix
                
        ensemble_matrix /= len(models)
        
        # Save Plot
        ds_name = index_to_DS_name.get(dataset_id_to_analyze, f"DS{dataset_id_to_analyze}")


        # save all to files
        csv_filename = f"results/interpretability/influence_map_{ds_name}_delta_{target_delta}_rank_{i+1}.csv"
        pd.DataFrame(ensemble_matrix).to_csv(csv_filename, index=False)
        print(f"Saved influence matrix to {csv_filename}")

        # print the sequence for reference
        seq_str = tokens_to_sequence(tokens)
        print(f"Sequence: {seq_str}")






        
        # filename = f"influence_map_{ds_name}_delta_{target_delta}_rank_{i+1}.png"
        
        # plot_influence_heatmap(
        #     ensemble_matrix, 
        #     tokens, 
        #     prob, 
        #     title_prefix=f"Seq Rank {i+1}", 
        #     save_path=filename
        # )


def analyze_synthetic_mh_efficiency(
    dataset_id=0,
    gap=0,
    mh_length=0,
    n_sequences=10,
    models=None,
    mask_table=None,
    seed=42
):
    """
    Returns the average ensemble probability for class -(2*gap + mh_length).
    Expects pre-loaded models and mask_table for performance.
    """
    if not models or mask_table is None:
        print("Error: Models and mask_table must be provided.")
        return 0.0

    np.random.seed(seed)
    
    # 1. Target Delta and Class Index
    target_delta = -(2 * gap + mh_length)
    target_class_idx = target_delta + DELTA_OFFSET
    
    if target_class_idx < 0 or target_class_idx >= NUM_CLASSES:
        return 0.0

    dataset_mask = mask_table[dataset_id]
    total_probs = []
    
    # 2. Sequence Generation and Mirroring
    for _ in range(n_sequences):
        # Create entire random sequence (A, C, G, T)
        tokens = np.random.randint(1, 5, size=(MAX_SEQ_LEN,))
        
        # Put GG in place (PAM site is NGG at 50, 51, 52)
        tokens[51] = VOCAB["G"]
        tokens[52] = VOCAB["G"]

        # Define indices relative to cut site (47)
        r_start = 47 + gap
        r_end = r_start + mh_length
        l_end = 47 - gap
        l_start = l_end - mh_length
        
        if l_start < 0 or r_end > MAX_SEQ_LEN:
            continue

        
        # Mirroring: Copy right-side sequence to the left side
        if mh_length > 0:
            tokens[l_start : l_end] = tokens[r_start : r_end]
            
            # --- Ensure MH isn't accidentally longer ---
            
            # Check the nucleotide 1 bp UPSTREAM (left) of the MH
            if l_start > 0 and r_start > 0:
                if tokens[l_start - 1] == tokens[r_start - 1]:
                    # Change tokens[l_start - 1] to something different
                    choices = [v for v in [1, 2, 3, 4] if v != tokens[r_start - 1]]
                    tokens[l_start - 1] = np.random.choice(choices)
            
            # Check the nucleotide 1 bp DOWNSTREAM (right) of the MH
            if l_end < MAX_SEQ_LEN and r_end < MAX_SEQ_LEN:
                # Note: l_end is the index right after the MH on the left side
                if tokens[l_end] == tokens[r_end]:
                    # Change tokens[l_end] to something different
                    choices = [v for v in [1, 2, 3, 4] if v != tokens[r_end]]
                    tokens[l_end] = np.random.choice(choices)

        # 3. Prediction and Masking
        tokens_tf = tf.convert_to_tensor([tokens], dtype=tf.int32)
        dsid_tf = tf.convert_to_tensor([dataset_id], dtype=tf.int32)
        
        ensemble_p = []
        for model in models:
            raw_preds = model({"tokens": tokens_tf, "dataset_id": dsid_tf}, training=False)[0]
            
            # Apply mask and re-normalize
            masked_preds = raw_preds * dataset_mask
            sum_p = tf.reduce_sum(masked_preds)
            
            if sum_p > 0:
                p_norm = (masked_preds / sum_p)[target_class_idx].numpy()
                ensemble_p.append(p_norm)
        
        if ensemble_p:
            total_probs.append(np.mean(ensemble_p))

    return np.mean(total_probs) if total_probs else 0.0

# TODO: make sequences that are similar to those in FORECasT K562
def sweep_synthetic_mh_efficiency(
    dataset_id=0,
    gap_max=10,
    mh_max=20,
    n_sequences=100,
    checkpoint_dir="saved_models/",
    num_folds=N_TRAIN_VAL_FOLDS,
    models=None,
    mask_table=None,
    seed=42,
    og_models=False
):
    """
    Sweeps gap and MH parameters while loading models once.
    """
    # Load models once if they are not passed in
    if models is None or mask_table is None:
        print(f"Loading {num_folds} models for sweep...")
        models = []
        mask_table = None
        for fold in range(num_folds):
            try:
                # checkpoint returns: model, optimizer, MASK_TABLE_TF, etc.
                m, _, mt, _, _ = load_model_checkpoint(fold, checkpoint_dir, og_model=og_models)
                models.append(m)
                if mask_table is None:
                    mask_table = mt
            except:
                pass

    if not models or mask_table is None:
        print("Failed to load models.")
        return None

    gaps = range(gap_max + 1)
    mhs = range(1, mh_max + 1)
    matrix = np.zeros((len(gaps), len(mhs)))
    
    print(f"Starting sweep for Dataset {dataset_id}...")
    for i, g in enumerate(gaps):
        for j, m in enumerate(mhs):
            print(f"  Gap: {g}, MH Length: {m}...", end=' ')
            # Pass pre-loaded models and mask_table
            matrix[i, j] = analyze_synthetic_mh_efficiency(
                dataset_id=dataset_id,
                gap=g,
                mh_length=m,
                n_sequences=n_sequences,
                models=models,
                mask_table=mask_table,
                seed=seed
            )
            print(f"Prob: {matrix[i, j]:.4f}")
            
    # Save to CSV
    df = pd.DataFrame(
        matrix, 
        index=[f"gap_{g}" for g in gaps], 
        columns=[f"mh_{m}" for m in mhs]
    )
    filename = f"results/interpretability/mh_sweep_matrix_ds{dataset_id}.csv"
    df.to_csv(filename)
    print(f"Sweep complete. Matrix saved to {filename}")
    
    return df
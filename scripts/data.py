import os
from scripts.utils import *
#from scripts.mh import compute_mh_grid
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split






def compute_frameshift_rate_numpy(y, deltas):
    fs = []
    for row in y:
        row_sum = np.sum(row)

        # Normalize safely: if the row sums to 0, treat it as all zeros
        if row_sum > 0:
            norm_row = row / row_sum
        else:
            norm_row = row  # leave as zeros

        rate = 0.0
        for i, prob in enumerate(norm_row):
            d = i + MIN_DELTA  # or i - DELTA_OFFSET  (same thing)

            if abs(d) % 3 != 0:
                rate += prob

        fs.append(rate)

    return np.array(fs, dtype=np.float32)



########################################
# SEQUENCE PROCESSING
########################################

def center_sequence_around_pam(seq, pam_index):
    """Center around PAM and pad to length MAX_SEQ_LEN."""
    PAD_CHAR = "P"
    start = pam_index - LEFT_CONTEXT
    end = pam_index + PAM_LEN + RIGHT_CONTEXT

    result = []
    for i in range(start, end):
        if 0 <= i < len(seq):
            base = seq[i].upper()
            if base not in ["A", "C", "G", "T", "N"]:
                base = "N"
            result.append(base)
        else:
            result.append(PAD_CHAR)
    return "".join(result)


def seq_to_token_ids(centered_seq):
    """Convert sequence (A/C/G/T/N/P) to token IDs."""
    token_ids = []
    for ch in centered_seq:
        if ch == "P":
            token_ids.append(VOCAB["PAD"])
        else:
            token_ids.append(VOCAB.get(ch, VOCAB["N"]))
    return np.array(token_ids, dtype=np.int32)

def tokens_to_sequence(tokens):
    """Convert token IDs back into sequence with P for padding."""
    inv_vocab = {v: k for k, v in VOCAB.items()}
    # Make the padding character 'P' for PAD token
    inv_vocab[VOCAB["PAD"]] = "P"
    return "".join(inv_vocab.get(t, "N") for t in tokens)

########################################
# LOADING DATASETS
########################################

def is_integer_string(s: str) -> bool:
    try:
        int(s)
        return True
    except:
        return False


def load_crispr_dataset(csv_path, dataset_id, sequence_col, pam_index_col, normalize=True):
    """Load dataset where label columns are integer strings."""
    df = pd.read_csv(csv_path)

    # remove all columns (aside from sequence) where the sum is 0
    removed = 0
    label_columns = [c for c in df.columns if is_integer_string(c)]
    for col in label_columns:
        if df[col].sum() == 0:
            del df[col]
            removed += 1
    print(f"Removed {removed} columns with sum 0")





    # get the sum per row for the label columns
    label_columns = [c for c in df.columns if is_integer_string(c)]
    sums = df[label_columns].sum(axis=1) 

    # remove all rows where the sum across label columns is 0
    initial_len = len(df)
    df = df[sums > 0]
    print(f"Removed {initial_len - len(df)} rows with sum 0 across label columns")


    percentage_to_remove = index_to_percentage_removed[dataset_id]
    if percentage_to_remove > 0.0:
        # remove bottom X% of rows based on sum
        threshold = np.percentile(sums, percentage_to_remove)
        df = df[sums > threshold]
        print(f"Removed {percentage_to_remove}% of rows with sum <= {threshold}")


    # Detect label columns
    label_cols = [c for c in df.columns if is_integer_string(c)]
    deltas = [int(c) for c in label_cols]

    print(f"\nLoading dataset: {csv_path}")
    #print("Label columns:", label_cols)
    #print("min_delta:", min(deltas), "max_delta:", max(deltas))
    print("Number of samples:", len(df))


    # DEBUG: Check sums, min/max, and frameshift proportion
    raw_sums = df[label_cols].sum(axis=1).values
    #print(f"  Sum range of raw label rows: min={raw_sums.min()}, max={raw_sums.max()}")

    # DEBUG: Frameshift percentage distribution
    delta_vals = np.array(deltas)
    fs_mask_local = (np.abs(delta_vals) % 3 != 0)
    fs_cols = [label_cols[i] for i in range(len(deltas)) if fs_mask_local[i]]
    # if len(fs_cols) > 0:
    #     fs_sum = df[fs_cols].sum(axis=1).values
    #     print(f"  Frameshift rate min/max before normalization: {fs_sum.min()}, {fs_sum.max()}")
    # else:
    #     print("  WARNING: Dataset has *no* frameshift deltas in allowed range!")


    # Build dataset-specific mask
    mask = np.zeros(NUM_CLASSES, dtype=np.float32)
    for delta in deltas:
        if MIN_DELTA <= delta <= MAX_DELTA:
            mask[delta + DELTA_OFFSET] = 1.0

    X_tokens_list = []
    y_list = []
    dsid_list = []

    for _, row in df.iterrows():
        seq = str(row[sequence_col])
        pam = int(row[pam_index_col])

        centered = center_sequence_around_pam(seq, pam)
        tokens = seq_to_token_ids(centered)

        # Build global y vector
        y = np.zeros(NUM_CLASSES, dtype=np.float32)
        for col, delta in zip(label_cols, deltas):
            if MIN_DELTA <= delta <= MAX_DELTA:
                y[delta + DELTA_OFFSET] = float(row[col])

        # Normalize 
        if normalize:
            s = y.sum()
            if s > 0:
                y /= s
        


        X_tokens_list.append(tokens)
        y_list.append(y)
        dsid_list.append(dataset_id)
        #print("\n\n")   
        #print(y_list[-1])
        #print(deltas)
        

    # Compute frameshift rates for this dataset
    fs_rates = compute_frameshift_rate_numpy(np.stack(y_list), deltas)
    # print all the fs rates
    #for i, fs in enumerate(fs_rates):
    #    print(f"Sample {i} frameshift rate: {fs:.4f}")
    median_fs = np.median(fs_rates)
    print(f"Dataset median frameshift rate: {median_fs:.4f}")


    return {
        "X_tokens": np.stack(X_tokens_list),
        "y": np.stack(y_list),
        "dataset_ids": np.array(dsid_list, dtype=np.int32),
        "mask": mask,
        "min_delta": min(deltas),
        "max_delta": max(deltas),
        "median_fs": median_fs, 
        "name": csv_path,
    }

def save_datasets_to_csv_human_readable(X_train, y_train, dsid_train,
                                        X_val,   y_val,   dsid_val,
                                        X_test,  y_test,  dsid_test,
                                        out_dir):

    os.makedirs(out_dir, exist_ok=True)

    delta_range = np.arange(MIN_DELTA, MAX_DELTA + 1)
    delta_cols = [f"delta_{d}" for d in delta_range]

    def _compute_fs(y):
        delta_vals = delta_range
        fs_mask = (np.abs(delta_vals) % 3 != 0).astype(np.float32)
        norm = y / (y.sum(axis=1, keepdims=True) + 1e-9)
        return (norm * fs_mask).sum(axis=1)

    def _pack(tokens, y, dsid):
        sequences = [tokens_to_sequence(row) for row in tokens]
        dataset_names = [index_to_DS_name[int(d)] for d in dsid]

        base_df = pd.DataFrame({
            "dataset_name": dataset_names,
            "sequence": sequences,
            "frameshift_rate": _compute_fs(y)
        })

        # Build delta columns in one vectorized shot (fast, clean)
        delta_df = pd.DataFrame(y, columns=delta_cols)

        # Concatenate horizontally
        df = pd.concat([base_df, delta_df], axis=1)

        return df

    _pack(X_train, y_train, dsid_train).to_csv(os.path.join(out_dir, "train.csv"), index=False)
    _pack(X_val,   y_val,   dsid_val).to_csv(os.path.join(out_dir, "val.csv"),   index=False)
    _pack(X_test,  y_test,  dsid_test).to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print(f"\nSaved human-readable splits to: {out_dir}")





########################################
# SEQUENCE GROUPING UTILITIES
########################################

def seq_key(tokens):
    """Convert token array to hashable key."""
    return tuple(tokens.tolist())


def build_groups(indices, X_tokens_all):
    """Group sample indices by sequence."""
    groups = {}
    for idx in indices:
        key = seq_key(X_tokens_all[idx])
        if key not in groups:
            groups[key] = []
        groups[key].append(idx)
    return groups



def split_groups(groups, seed_train_test=42, val_bucket=0):
    # this results in train 80% val 10% test 10%


    keys = list(groups.keys())
    np.random.seed(seed_train_test)
    np.random.shuffle(keys)

    n = len(keys)
    n_train = int(TRAIN_TEST_RATIO * n)

    train_keys = keys[:n_train]
    test_keys  = keys[n_train:]

    # ----- Split train into train/val by bucket -----
    val_keys = [k for i, k in enumerate(train_keys) if i % N_TRAIN_VAL_FOLDS == val_bucket]
    train_keys_final = [k for i, k in enumerate(train_keys) if i % N_TRAIN_VAL_FOLDS != val_bucket]

    # ----- Convert back to sample indices -----
    train_idx = [idx for k in train_keys_final for idx in groups[k]]
    val_idx   = [idx for k in val_keys         for idx in groups[k]]
    test_idx  = [idx for k in test_keys        for idx in groups[k]]

    return train_idx, val_idx, test_idx



def build_dataset_groups(selected_ids, grouping_rules):
    """
    selected_ids    – the user chooses which dataset IDs to use (subset of all)
    grouping_rules  – a list of sets, e.g. [{0,1,2,3,4}, {6,7}]
    
    Returns:
        groups – list of groups, each group is a set of dataset IDs
    """

    selected_ids = set(selected_ids)

    # 1) Filter grouping rules to selected IDs only
    groups = []
    for g in grouping_rules:
        g_filtered = set(g) & selected_ids
        if len(g_filtered) > 0:
            groups.append(g_filtered)

    # 2) Some datasets might not appear in any group → add as 1-dataset groups
    grouped_already = set().union(*groups) if groups else set()
    remaining = selected_ids - grouped_already

    for dsid in remaining:
        groups.append({dsid})

    return groups

def get_legacy_split_map(X_all, ids_all, target_ds_id=0, seed=42, val_bucket=0):
    """
    1. Replicates the exact Test split from the previous paper (Seed 777).
    2. Takes the remaining sequences (Legacy Train + Val) as a pool.
    3. Splits that pool into Train/Val using our modulo bucket logic (Seed 42).
    
    Returns: { sequence_tuple: 'train' | 'val' | 'test' }
    """
    print(f"Generating Legacy Split Map based on Dataset {target_ds_id}...")
    
    # --- Step 1: Get Global Indices for Dataset 0 ---
    ds_indices = np.where(ids_all == target_ds_id)[0]
    n_sample = len(ds_indices)
    
    if n_sample == 0:
        return {}

    # --- Step 2: Separate Test Set (Legacy Logic: Seed 777) ---
    test_prop = 0.1
    test_num = int(n_sample * test_prop)
    
    np.random.seed(777)
    idx_perm = np.random.choice(np.arange(n_sample), n_sample, replace=False)
    
    # First 10% are strictly TEST
    test_local_indices = idx_perm[0 : test_num]
    test_global_idxs = ds_indices[test_local_indices]
    
    # Remaining 90% are the POOL for Train/Val
    pool_local_indices = idx_perm[test_num :]
    pool_global_idxs = ds_indices[pool_local_indices]

    # --- Step 3: Convert Indices to Sequence Keys ---
    # We must group by sequence key to ensure duplicates are handled consistently
    legacy_map = {}
    
    # Mark Test Keys
    for idx in test_global_idxs:
        key = seq_key(X_all[idx])
        legacy_map[key] = 'test'
        
    # Collect Pool Keys
    pool_keys_list = []
    seen_pool_keys = set()
    for idx in pool_global_idxs:
        key = seq_key(X_all[idx])
        if key not in seen_pool_keys:
            # Important: If a sequence was already marked 'test' (e.g. duplicate in DS0),
            # it stays 'test'. Only add if not in map/seen.
            if key not in legacy_map: 
                pool_keys_list.append(key)
                seen_pool_keys.add(key)

    # --- Step 4: Split the Pool using Bucket Logic (Your Logic: Seed 42) ---
    np.random.seed(seed) 
    np.random.shuffle(pool_keys_list)
    
    train_count = 0
    val_count = 0
    
    for i, key in enumerate(pool_keys_list):
        if i % N_TRAIN_VAL_FOLDS == val_bucket:
            legacy_map[key] = 'val'
            val_count += 1
        else:
            legacy_map[key] = 'train'
            train_count += 1
            
    print(f"Legacy Map Stats -> Test: {len(test_global_idxs)} (indices), Train Keys: {train_count}, Val Keys: {val_count}")
    return legacy_map


def split_groups_standard(keys, groups_dict, seed=42, val_bucket=0):
    """Standard random split logic with rotating validation bucket."""
    np.random.seed(seed)
    # We shuffle a copy of keys list
    shuffled_keys = list(keys)
    np.random.shuffle(shuffled_keys)

    n = len(shuffled_keys)
    n_train_total = int(TRAIN_TEST_RATIO * n) # e.g. 90%

    # Top 90% is Train+Val, Bottom 10% is Test
    train_val_keys = shuffled_keys[:n_train_total]
    test_keys      = shuffled_keys[n_train_total:]

    # Split Train/Val based on modulo bucket
    val_keys_final   = [k for i, k in enumerate(train_val_keys) if i % N_TRAIN_VAL_FOLDS == val_bucket]
    train_keys_final = [k for i, k in enumerate(train_val_keys) if i % N_TRAIN_VAL_FOLDS != val_bucket]

    return train_keys_final, val_keys_final, test_keys



def perform_splitting(X_all, y_all, ids_all, groups, 
                      use_legacy_split=False, seed=42, val_bucket=0):
    
    train_idx, val_idx, test_idx = [], [], []

    for group_set in groups:
        group_list = list(group_set)
        
        # indices for this group of datasets
        group_mask = np.isin(ids_all, group_list)
        group_global_idxs = np.where(group_mask)[0]

        # Group by sequence content
        groups_dict = build_groups(group_global_idxs, X_all)
        all_keys = list(groups_dict.keys())

        # --- LOGIC BRANCHING ---
        current_train_keys = []
        current_val_keys = []
        current_test_keys = []

        # Check if we should use legacy split
        # Condition: Flag is True AND Dataset 0 is in this group
        if use_legacy_split and (0 in group_set):
            print(f"Applying Legacy Split (Seed 777) for group containing {group_set}")
            
            # Generate the fixed map from Dataset 0
            legacy_map = get_legacy_split_map(X_all, ids_all, target_ds_id=0, seed=seed, val_bucket=val_bucket)
            
            legacy_keys = []
            remaining_keys = []
            
            for k in all_keys:
                if k in legacy_map:
                    legacy_keys.append(k)
                    split = legacy_map[k]
                    if split == 'train': current_train_keys.append(k)
                    elif split == 'val': current_val_keys.append(k)
                    elif split == 'test': current_test_keys.append(k)
                else:
                    remaining_keys.append(k)
            
            print(f"  Legacy sequences matched: {len(legacy_keys)}")
            print(f"  Remaining sequences to split normally: {len(remaining_keys)}")
            
            # Split the remainder normally
            if remaining_keys:
                tr, va, te = split_groups_standard(remaining_keys, groups_dict, seed, val_bucket)
                current_train_keys.extend(tr)
                current_val_keys.extend(va)
                current_test_keys.extend(te)

        else:
            # Standard path for groups without DS 0 or if flag is False
            tr, va, te = split_groups_standard(all_keys, groups_dict, seed, val_bucket)
            current_train_keys.extend(tr)
            current_val_keys.extend(va)
            current_test_keys.extend(te)

        # Expand back to indices
        for k in current_train_keys: train_idx.extend(groups_dict[k])
        for k in current_val_keys:   val_idx.extend(groups_dict[k])
        for k in current_test_keys:  test_idx.extend(groups_dict[k])

    # Convert to numpy and sort
    train_idx = np.sort(np.array(train_idx))
    val_idx   = np.sort(np.array(val_idx))
    test_idx  = np.sort(np.array(test_idx))

    return (X_all[train_idx], y_all[train_idx], ids_all[train_idx],
            X_all[val_idx],   y_all[val_idx],   ids_all[val_idx],
            X_all[test_idx],  y_all[test_idx],  ids_all[test_idx])



def load_and_prepare_data(seed=42, val_bucket=0, save_splits=False, use_legacy_split=True, normalize=True):



    all_datasets = []
    for dataset_id, cfg in enumerate(DATASET_CONFIGS):


        print(f"\n=== Loading dataset ID {dataset_id} ===")
        ds = load_crispr_dataset(
            cfg["csv_path"],
            dataset_id,
            cfg["sequence_col"],
            cfg["pam_index_col"],
            normalize=normalize
        )
        all_datasets.append(ds)

    dataset_medians = np.array([ds["median_fs"] for ds in all_datasets], dtype=np.float32)
    DATASET_MEDIANS_TF = tf.constant(dataset_medians, dtype=tf.float32)

    dataset_sizes = [len(ds["X_tokens"]) for ds in all_datasets]
    # normalize by max size
    max_size = max(dataset_sizes)
    dataset_sizes_normalized = np.array([size / max_size for size in dataset_sizes], dtype=np.float32)
    DATASET_SIZE_SCALE_TF = tf.constant(dataset_sizes_normalized, dtype=tf.float32)


    NUM_DATASETS = len(all_datasets)

    ########################################
    # CONCATENATE DATASETS
    ########################################

    X_tokens_all = np.concatenate([ds["X_tokens"] for ds in all_datasets])
    y_all = np.concatenate([ds["y"] for ds in all_datasets])
    dataset_ids_all = np.concatenate([ds["dataset_ids"] for ds in all_datasets])


    dataset_mask_table = np.stack([ds["mask"] for ds in all_datasets])  # (D,201)


    print("\n=== MASK TABLE CHECK ===")
    for i, name in index_to_DS_name.items():
        mask = dataset_mask_table[i]
        print(f"\nDataset {i} ({name}):")
        #print("  Mask active deltas count:", mask.sum())
        print("  First active delta:", np.where(mask==1)[0][0] + MIN_DELTA)
        print("  Last active delta:", np.where(mask==1)[0][-1] + MIN_DELTA)

        # check if frameshift allowed at all
        delta_vals = np.arange(MIN_DELTA, MAX_DELTA+1)
        fs_mask = ((np.abs(delta_vals) % 3 != 0) & (mask == 1))
        #print("  Frameshift deltas allowed:", fs_mask.sum())
        if fs_mask.sum() == 0:
            print("  !!! WARNING: Dataset mask blocks ALL frameshift deltas !!!")


    # Compute dataset balancing weights
    dataset_sizes = np.array([len(ds["X_tokens"]) for ds in all_datasets])
    dataset_weights = 1.0 / np.sqrt(dataset_sizes)
    dataset_weights = dataset_weights / dataset_weights.mean()

    MASK_TABLE_TF = tf.constant(dataset_mask_table, dtype=tf.float32)
    DATASET_WEIGHTS_TF = tf.constant(dataset_weights, dtype=tf.float32)



    # ============================================
    # BUILD GROUPS
    # ============================================

    groups = build_dataset_groups(SELECTED_IDS, GROUPING_RULES)

    (X_train, y_train, dsid_train,
    X_val, y_val, dsid_val,
    X_test, y_test, dsid_test)= perform_splitting(
    X_tokens_all, y_all, dataset_ids_all,
    groups,
    use_legacy_split=use_legacy_split,
    seed=seed,
    val_bucket=val_bucket,
    )

    if save_splits:
        path = "data/saved_splits/"
        path = os.path.join(path, f"train_val_split_{val_bucket}/")
        save_datasets_to_csv_human_readable(X_train, y_train, dsid_train,
                            X_val, y_val, dsid_val,
                            X_test, y_test, dsid_test,
                            path)
    

    

    return (X_train, y_train, dsid_train,
            X_val, y_val, dsid_val,
            X_test, y_test, dsid_test,
            MASK_TABLE_TF, DATASET_WEIGHTS_TF,DATASET_MEDIANS_TF,DATASET_SIZE_SCALE_TF, len(SELECTED_IDS))



########################################
# TF.DATA PIPELINES
########################################


def make_dataset(X, y, ds_ids, batch_size=DEFAULT_BATCH_SIZE, shuffle=True):


    ds = tf.data.Dataset.from_tensor_slices(
        (
            {
                "tokens": X,
                "dataset_id": ds_ids,
            },
            y,
        )
    )
    if shuffle:
        # full shuffle each epoch, pure TF
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds




def get_datasets(seed=42, val_bucket=0, save_splits=False, filter_ids=[], use_legacy_split=True, use_val=False, normalize=True):
    (
        X_train, y_train, dsid_train, 
        X_val, y_val, dsid_val, 
        X_test, y_test, dsid_test, 
        MASK_TABLE_TF, DATASET_WEIGHTS_TF, DATASET_MEDIANS_TF,
        DATASET_SIZE_SCALE_TF, NUM_DATASETS
    ) = load_and_prepare_data(seed, val_bucket, save_splits, use_legacy_split, normalize=normalize)

    # (Optional but nice) convert to TF tensors once
    X_train = tf.convert_to_tensor(X_train, dtype=tf.int32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    dsid_train = tf.convert_to_tensor(dsid_train, dtype=tf.int32)

    X_val = tf.convert_to_tensor(X_val, dtype=tf.int32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
    dsid_val = tf.convert_to_tensor(dsid_val, dtype=tf.int32)

    X_test = tf.convert_to_tensor(X_test, dtype=tf.int32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    dsid_test = tf.convert_to_tensor(dsid_test, dtype=tf.int32)

    # if use_val is False, combine val into train
    if not use_val:
        X_train = tf.concat([X_train, X_val], axis=0)
        y_train = tf.concat([y_train, y_val], axis=0)
        dsid_train = tf.concat([dsid_train, dsid_val], axis=0)
        
        # Clear val sets
        X_val = tf.zeros((0, X_train.shape[1]), dtype=tf.int32)
        y_val = tf.zeros((0, y_train.shape[1]), dtype=tf.float32)
        dsid_val = tf.zeros((0,), dtype=tf.int32)



    # We filter ids after, as to keep the correct masks for all the datasets
    for id in filter_ids:
        mask_train = (dsid_train != id)
        X_train = tf.boolean_mask(X_train, mask_train)
        y_train = tf.boolean_mask(y_train, mask_train)
        dsid_train = tf.boolean_mask(dsid_train, mask_train)


        mask_val = (dsid_val != id)
        X_val = tf.boolean_mask(X_val, mask_val)
        y_val = tf.boolean_mask(y_val, mask_val)
        dsid_val = tf.boolean_mask(dsid_val, mask_val)


        mask_test = (dsid_test != id)
        X_test = tf.boolean_mask(X_test, mask_test)
        y_test = tf.boolean_mask(y_test, mask_test)
        dsid_test = tf.boolean_mask(dsid_test, mask_test)


    # Build datasets
    train_ds = make_dataset(X_train, y_train, dsid_train, shuffle=True)
    val_ds   = make_dataset(X_val,   y_val,   dsid_val,  shuffle=False)
    test_ds  = make_dataset(X_test,  y_test,  dsid_test,  shuffle=False)

    return (
        train_ds, val_ds, test_ds,
        MASK_TABLE_TF, DATASET_WEIGHTS_TF,
        DATASET_MEDIANS_TF, DATASET_SIZE_SCALE_TF,
        NUM_DATASETS
    )


def get_external_datasets():
    seed = 42
    val_bucket = 0
    save_splits = False
    (
        X_train, y_train, dsid_train, 
        X_val, y_val, dsid_val, 
        X_test, y_test, dsid_test, 
        MASK_TABLE_TF, DATASET_WEIGHTS_TF, DATASET_MEDIANS_TF,
        DATASET_SIZE_SCALE_TF, NUM_DATASETS
    ) = load_and_prepare_data(seed, val_bucket, save_splits)

    ids_to_keep = [9]  # SPROUT only
    mask_train = tf.reduce_any([dsid_train == id for id in ids_to_keep], axis=0)
    X_train = tf.boolean_mask(X_train, mask_train)
    y_train = tf.boolean_mask(y_train, mask_train)
    dsid_train = tf.boolean_mask(dsid_train, mask_train)
    X_val = tf.boolean_mask(X_val, tf.reduce_any([dsid_val == id for id in ids_to_keep], axis=0))
    y_val = tf.boolean_mask(y_val, tf.reduce_any([dsid_val == id for id in ids_to_keep], axis=0))
    dsid_val = tf.boolean_mask(dsid_val, tf.reduce_any([dsid_val == id for id in ids_to_keep], axis=0))
    X_test = tf.boolean_mask(X_test, tf.reduce_any([dsid_test == id for id in ids_to_keep], axis=0))
    y_test = tf.boolean_mask(y_test, tf.reduce_any([dsid_test == id for id in ids_to_keep], axis=0))
    dsid_test = tf.boolean_mask(dsid_test, tf.reduce_any([dsid_test == id for id in ids_to_keep], axis=0))

    # Combine them all to one dataset
    X_external = tf.concat([X_train, X_val, X_test], axis=0)
    y_external = tf.concat([y_train, y_val, y_test], axis=0)
    dsid_external = tf.concat([dsid_train, dsid_val, dsid_test], axis=0)

    external_ds = make_dataset(X_external, y_external, dsid_external, shuffle=False)
    return external_ds



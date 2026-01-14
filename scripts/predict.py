from scripts.utils import *
from scripts.data import center_sequence_around_pam, seq_to_token_ids
import numpy as np



DELTA_VALUES = np.arange(MIN_DELTA, MAX_DELTA + 1)

def predict_distribution(sequence, pam_index, dataset_id, base_model, MASK_TABLE_TF):
    centered = center_sequence_around_pam(sequence, pam_index)
    tokens = seq_to_token_ids(centered)

    tokens_batch = np.expand_dims(tokens, 0)
    dsid_batch = np.array([dataset_id])

    probs = base_model({"tokens": tokens_batch, "dataset_id": dsid_batch}, training=False)[0].numpy()
    mask = MASK_TABLE_TF[dataset_id]

    probs_masked = probs * mask
    s = probs_masked.sum()
    if s > 0:
        probs_masked /= s

    return DELTA_VALUES, probs_masked

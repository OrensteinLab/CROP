import os
import json
import numpy as np
import tensorflow as tf

from scripts.model import RoPE_MHA, AddDatasetEmbedding, CorrectRoPE_MHA, build_transformer  # add at the top or here
from scripts.training import CRISPRModel




import os
import json
import numpy as np
import tensorflow as tf

from scripts.training import CRISPRModel


import os
import json
import numpy as np
import tensorflow as tf

from scripts.training import CRISPRModel

def save_model_checkpoint(model, fold, *,
                          MASK_TABLE=None,
                          DATASET_MEDIANS=None,
                          DATASET_SIZE_SCALE=None,
                          dataset_names=None,
                          extra_metadata=None,
                          base_dir="saved_models"):

    save_dir = os.path.join(base_dir, f"fold_{fold}")
    os.makedirs(save_dir, exist_ok=True)

    # ---- Save weights only ----
    model.base_model.save_weights(os.path.join(save_dir, "base_weights.h5"))

    # ---- Save arrays ----
    if MASK_TABLE is not None:
        np.save(os.path.join(save_dir, "MASK_TABLE.npy"), MASK_TABLE)
    if DATASET_MEDIANS is not None:
        np.save(os.path.join(save_dir, "DATASET_MEDIANS.npy"), DATASET_MEDIANS)
    if DATASET_SIZE_SCALE is not None:
        np.save(os.path.join(save_dir, "DATASET_SIZE_SCALE.npy"), DATASET_SIZE_SCALE)

    # ---- Save metadata (including MODEL_CONFIG if provided) ----
    meta = {"dataset_names": dataset_names}
    if extra_metadata:
        meta.update(extra_metadata)

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print(f"[✓] Saved weights + metadata for fold {fold}")



from scripts.model import build_transformer
from scripts.training import CRISPRModel


def load_model_checkpoint(fold, base_dir="saved_models", og_model = False):
    save_dir = os.path.join(base_dir, f"fold_{fold}")

    # ---- Load arrays ----
    MASK_TABLE = np.load(os.path.join(save_dir, "MASK_TABLE.npy"))
    MEDIANS    = np.load(os.path.join(save_dir, "DATASET_MEDIANS.npy"))
    SIZE_SCALE = np.load(os.path.join(save_dir, "DATASET_SIZE_SCALE.npy"))

    # ---- Load metadata (including config) ----
    metadata = json.load(open(os.path.join(save_dir, "metadata.json")))
    MODEL_CONFIG = metadata["MODEL_CONFIG"]     # <<<<<<<<<< HERE

    if og_model:
        MODEL_CONFIG["add_ds_embedding"] = True
        MODEL_CONFIG["use_mh"] = True

    # ---- Rebuild base transformer ----
    base_model = build_transformer(MODEL_CONFIG)

    # ---- Load weights ----
    base_model.load_weights(os.path.join(save_dir, "base_weights.h5"))

    # ---- Wrap in CRISPRModel ----
    crispr_model = CRISPRModel(
        base_model,
        MODEL_CONFIG["n_datasets"],
        tf.constant(MASK_TABLE, dtype=tf.float32),
        tf.constant(MEDIANS, dtype=tf.float32),
        tf.constant(SIZE_SCALE, dtype=tf.float32),
    )
    crispr_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    print(f"[✓] Loaded model for fold {fold}")

    return crispr_model, MASK_TABLE, MEDIANS, SIZE_SCALE, metadata

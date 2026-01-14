from scripts.utils import *
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models


def apply_rope(x):
    seq_len = tf.shape(x)[1]
    dim = x.shape[-1]
    half = dim // 2

    pos = tf.range(seq_len, dtype=tf.float32)[:, None]
    freq = 10000.0 ** (-tf.range(half, dtype=tf.float32) / half)
    angles = pos * freq[None, :]

    sin = tf.sin(angles)[None, :, :]
    cos = tf.cos(angles)[None, :, :]

    x1 = x[:, :, :half]
    x2 = x[:, :, half:2*half]
    x_rest = x[:, :, 2*half:]

    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    return tf.concat([x1_rot, x2_rot, x_rest], axis=-1)



class RoPE_MHA(layers.Layer):
    def __init__(self,
                 embed_dim=128,
                 num_heads=4,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.attn = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.dropout_rate,
        )


    def call(self, x, mask=None):
        return self.attn(
            apply_rope(x),
            apply_rope(x),
            attention_mask=mask
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # for future checkpoints, config will actually contain these keys
        return cls(**config)


def apply_rope_v2(x):
    """
    x: (B, L, H, D) where D is even.
    Applies RoPE rotation ONLY on last dim.
    """
    B, L = tf.shape(x)[0], tf.shape(x)[1]
    H = x.shape[2]
    D = x.shape[3]
    half = D // 2

    # Slice into even & odd halves
    x1 = x[:, :, :, :half]   # (B, L, H, half)
    x2 = x[:, :, :, half:]   # (B, L, H, half)

    # Build rotary frequencies
    pos = tf.range(L, dtype=tf.float32)[:, None]                # (L,1)
    freq = 10000.0 ** (-tf.range(half, dtype=tf.float32) / half) # (half,)
    angles = pos * freq[None, :]                                 # (L,half)

    sin = tf.sin(angles)[None, :, None, :]  # (1, L, 1, half)
    cos = tf.cos(angles)[None, :, None, :]  # (1, L, 1, half)

    # Rotate
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    return tf.concat([x1_rot, x2_rot], axis=-1)


class CorrectRoPE_MHA(tf.keras.layers.Layer):
    def __init__(self,
                 embed_dim=128,
                 num_heads=4,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.key_dim = embed_dim // num_heads


        self.Wq = layers.Dense(embed_dim, use_bias=False)
        self.Wk = layers.Dense(embed_dim, use_bias=False)
        self.Wv = layers.Dense(embed_dim, use_bias=False)
        self.out = layers.Dense(embed_dim)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, mask=None):
        """
        x:    (B, L, D)
        mask: (B, 1, 1, L_k) with 1 = keep, 0 = pad
        """
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]
        D = x.shape[-1]

        # Linear projections
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # Reshape into heads (B, L, H, head_dim)
        q = tf.reshape(q, (B, L, self.num_heads, self.key_dim))
        k = tf.reshape(k, (B, L, self.num_heads, self.key_dim))
        v = tf.reshape(v, (B, L, self.num_heads, self.key_dim))

        # Apply rotary on q,k
        q = apply_rope_v2(q)
        k = apply_rope_v2(k)

        # Scaled dot-product attention
        scores = tf.einsum("blhd,bshd->bhls", q, k) \
                 / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))

        if mask is not None:
            mask = tf.cast(mask, scores.dtype)
            scores += (1.0 - mask) * -1e9

        attn = tf.nn.softmax(scores, axis=-1)
        attn = self.dropout(attn)

        out = tf.einsum("bhls,bshd->blhd", attn, v)
        out = tf.reshape(out, (B, L, D))

        return self.out(out)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



class MH_BiRNN_Encoder(layers.Layer):
    def __init__(self, d_model, rnn_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.rnn = layers.Bidirectional(
            layers.GRU(
                rnn_dim,
                return_sequences=True,
                reset_after=True,   # cuDNN-compatible
            )
        )
        self.proj = layers.Dense(d_model)

    def call(self, x, mask=None, training=None):
        # mask: (B, T)
        h = self.rnn(x, mask=mask, training=training)
        return self.proj(h)

class MH_StringMatch_BiRNN(layers.Layer):
    def __init__(self, cut_index, d_model, rnn_dim=64,
                 max_gap=60, **kwargs):
        super().__init__(**kwargs)
        self.cut = int(cut_index)
        self.max_gap = int(max_gap)

        self.shared_enc = MH_BiRNN_Encoder(d_model, rnn_dim)
        self.mix = layers.Dense(d_model, activation="gelu")
        self.out = layers.Dense(d_model)

        self.gate = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True,
            name="mh_rnn_gate"
        )

    def call(self, x, pad_mask, training=None):
        """
        x:        (B, L, D)
        pad_mask: (B, L) boolean or 0/1
        """

        # -------- split sequence --------
        left  = x[:, :self.cut, :]
        right = x[:, self.cut:, :]

        left_mask  = pad_mask[:, :self.cut]
        right_mask = pad_mask[:, self.cut:]

        # -------- reverse left (time = distance from cut) --------
        left = tf.reverse(left, axis=[1])
        left_mask = tf.reverse(left_mask, axis=[1])

        # -------- truncate to max_gap --------
        if self.max_gap is not None:
            left  = left[:, :self.max_gap, :]
            right = right[:, :self.max_gap, :]

            left_mask  = left_mask[:, :self.max_gap]
            right_mask = right_mask[:, :self.max_gap]

        # -------- encode with mask --------
        hL = self.shared_enc(left,  mask=left_mask,  training=training)
        hR = self.shared_enc(right, mask=right_mask, training=training)

        # -------- similarity --------
        scale = tf.math.sqrt(tf.cast(tf.shape(hL)[-1], tf.float32))
        sim = tf.matmul(hL, hR, transpose_b=True) / scale

        # mask invalid right positions in similarity
        sim += (1.0 - tf.cast(right_mask, sim.dtype))[:, None, :] * (-1e9)

        # TODO: mask invalid left positions in similarity?

        aL = tf.nn.softmax(sim, axis=-1)
        aR = tf.nn.softmax(sim, axis=-2)

        ctxL = tf.matmul(aL, hR)
        ctxR = tf.matmul(tf.transpose(aR, [0,2,1]), hL)

        # -------- explicit match channels --------
        diffL = hL - ctxL
        simL  = hL * ctxL

        diffR = hR - ctxR
        simR  = hR * ctxR

        featL = self.mix(tf.concat([hL, ctxL, diffL, simL], axis=-1))
        featR = self.mix(tf.concat([hR, ctxR, diffR, simR], axis=-1))

        # -------- restore order --------
        featL = tf.reverse(featL, axis=[1])

        # -------- stitch back --------
        B, L, D = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        Lm = tf.shape(featL)[1]
        Rm = tf.shape(featR)[1]

        res_left  = tf.pad(featL, [[0,0], [self.cut - Lm, L - self.cut], [0,0]])
        res_right = tf.pad(featR, [[0,0], [self.cut, L - (self.cut + Rm)], [0,0]])

        res = self.out(res_left + res_right)
        return tf.tanh(self.gate) * res

class MH_Simple_BiRNN(layers.Layer):
    """
    Simple MH baseline:
    - Encode left and right of cut with shared BiRNN
    - Project back to d_model
    - Inject as gated residual
    """
    def __init__(self, cut_index, d_model, rnn_dim=64, max_gap=60, **kwargs):
        super().__init__(**kwargs)
        self.cut = int(cut_index)
        self.max_gap = int(max_gap)

        self.encoder = MH_BiRNN_Encoder(d_model, rnn_dim)
        self.proj = layers.Dense(d_model)

        # start at zero so model behaves like no-MH at init
        self.gate = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True,
            name="mh_simple_gate"
        )

    def call(self, x, pad_mask, training=None):
        """
        x:        (B, L, D)
        pad_mask: (B, L) boolean
        """

        # ---- split ----
        left  = x[:, :self.cut, :]
        right = x[:, self.cut:, :]

        left_mask  = pad_mask[:, :self.cut]
        right_mask = pad_mask[:, self.cut:]

        # ---- reverse left so time = distance from cut ----
        left = tf.reverse(left, axis=[1])
        left_mask = tf.reverse(left_mask, axis=[1])

        # ---- truncate to max_gap ----
        if self.max_gap is not None:
            left  = left[:, :self.max_gap, :]
            right = right[:, :self.max_gap, :]

            left_mask  = left_mask[:, :self.max_gap]
            right_mask = right_mask[:, :self.max_gap]

        # ---- encode ----
        hL = self.encoder(left,  mask=left_mask,  training=training)
        hR = self.encoder(right, mask=right_mask, training=training)

        # ---- restore left order ----
        hL = tf.reverse(hL, axis=[1])

        # ---- stitch back into sequence ----
        B, L, D = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        Lm = tf.shape(hL)[1]
        Rm = tf.shape(hR)[1]

        res_left  = tf.pad(hL, [[0,0], [self.cut - Lm, L - self.cut], [0,0]])
        res_right = tf.pad(hR, [[0,0], [self.cut, L - (self.cut + Rm)], [0,0]])

        res = self.proj(res_left + res_right)

        return tf.tanh(self.gate) * res




class AddDatasetEmbedding(layers.Layer):
    def __init__(self, seq_len=MAX_SEQ_LEN, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len

    def call(self, ds_emb_raw):
        return tf.tile(tf.expand_dims(ds_emb_raw, 1),
                       [1, self.seq_len, 1])

    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def freeze_all_non_dataset_embedding_layers(model):
    print("\nFreezing all non-dataset-embedding layers...\n")
    for layer in model.base_model.layers:
        layer.trainable = False
    ds_emb_layer = model.base_model.get_layer("dataset_embedding")
    ds_emb_layer.trainable = True

    
def build_transformer(model_config):
    tokens_input = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="tokens")
    dsid_input   = layers.Input(shape=(), dtype=tf.int32, name="dataset_id")
   # if model_config["use_mh"]:
    #    mh_input = layers.Input(shape=(abs(MIN_DELTA), 4), dtype=tf.float32, name="mh_grid")
    CUT_INDEX = LEFT_CONTEXT - 3
      

    # ---- padding mask ----
    padding_mask = tf.cast(tokens_input != VOCAB["PAD"], tf.float32)
    padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :] 


    token_emb = layers.Embedding(VOCAB_SIZE, model_config["d_model"])(tokens_input)
    
    
    
    # ======== Stable Dataset Embedding ========
    raw_emb = layers.Embedding(
        model_config["n_datasets"], model_config["d_model"], name="dataset_embedding"
    )(dsid_input)

    if model_config["add_ds_embedding"]:
        # 1) Normalize (removes random-scale issues)
        ds_emb_raw = tf.nn.l2_normalize(raw_emb, axis=-1)

        # 2) Learnable scale (how much dataset identity influences model)
        scale = layers.Dense(1, use_bias=False, name="ds_scale")
        ds_emb_raw = scale(ds_emb_raw)

        # 3) Smoothing MLP (makes it less random / more structured)
        ds_emb_raw = layers.Dense(model_config["d_model"], activation="gelu")(ds_emb_raw)
        ds_emb_raw = layers.Dense(model_config["d_model"])(ds_emb_raw)

        add_ds = AddDatasetEmbedding(seq_len=MAX_SEQ_LEN)

    if model_config["use_mh"]:
        pad_mask_1d = tf.cast(tokens_input != VOCAB["PAD"], tf.bool)

        if model_config["add_to_embedding"]:
            x = token_emb + MH_Simple_BiRNN(
                cut_index=CUT_INDEX,
                d_model=model_config["d_model"],
                rnn_dim=64,
                max_gap=60,
                name="mh_simple_rnn"
            )(token_emb, pad_mask_1d)
        else:
            x = MH_Simple_BiRNN(
                cut_index=CUT_INDEX,
                d_model=model_config["d_model"],
                rnn_dim=64,
                max_gap=60,
                name="mh_simple_rnn"
            )(token_emb, pad_mask_1d)
    else:
        x = token_emb

    for _ in range(model_config["n_attention_layers"]):

        h = layers.LayerNormalization()(x)

        if model_config["add_ds_embedding"]:
            h = h + add_ds(ds_emb_raw)

        #attn_out = RoPE_MHA(d_model, num_heads, dropout_rate)(h)
        if  model_config["use_weird_rope"]:
            attn_layer = RoPE_MHA(model_config["d_model"], model_config["n_attention_heads"], model_config["dropout_rate"])
        else:
            attn_layer = CorrectRoPE_MHA(model_config["d_model"], model_config["n_attention_heads"], model_config["dropout_rate"])

        attn_out = attn_layer(h, mask=padding_mask)

        attn_out = layers.Dropout(model_config["dropout_rate"])(attn_out)

        x = x + attn_out   

        h2 = layers.LayerNormalization()(x)
        if model_config["add_ds_embedding"]:
            h2 = h2 + add_ds(ds_emb_raw)

        ff = layers.Dense(4 * model_config["d_model"], activation="gelu")(h2)
        ff = layers.Dropout(model_config["dropout_rate"])(ff)
        ff = layers.Dense(model_config["d_model"])(ff)
        ff = layers.Dropout(model_config["dropout_rate"])(ff)

        x = x + ff  # residual

 
    cut_rep = x[:, CUT_INDEX, :]   # (B, d_model)

    # Final head
    h = layers.LayerNormalization()(cut_rep)
    h = layers.Dense(256, activation="gelu")(h)
    h = layers.Dropout(model_config["dropout_rate"])(h)

    logits = layers.Dense(NUM_CLASSES)(h)
    probs = layers.Softmax()(logits)

    return models.Model([tokens_input, dsid_input], probs)



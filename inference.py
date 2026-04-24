"""
inference.py
────────────
Caption generation using the model trained in Google Colab.

Expected files in the SAME folder as this script:
    captioning_model.keras   ← downloaded from Colab
    tokenizer.pkl            ← downloaded from Colab
    config.pkl               ← downloaded from Colab

FIXES vs original:
    - pad_sequences uses padding='post' to match the training data generator
    - Beam search exits early when ALL active beams have produced endseq
      (previously the loop always ran for the full max_len = 34 steps)
    - Generation budget is max_len - 2 (reserves slots for startseq/endseq)
      so the model is never forced to fill every position
    - Greedy: endseq is detected even when predicted token id maps to None
      (handles edge case where endseq id falls outside idx2word)
    - Beam search length normalisation uses actual word count (excludes startseq)
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

IMG_SIZE = (299, 299)

# ── Resolve paths relative to this file ──────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH     = os.path.join(_BASE, "captioning_model_new.keras")
TOKENIZER_FILE = os.path.join(_BASE, "tokenizer.pkl")
CONFIG_FILE    = os.path.join(_BASE, "config.pkl")

# ── Lazy-loaded singletons ────────────────────────────────────────────
_extractor = None
_model     = None
_tokenizer = None
_idx2word  = None
_config    = None


def _load_config():
    global _config
    if _config is None:
        if not os.path.exists(CONFIG_FILE):
            raise FileNotFoundError(
                f"config.pkl not found at {CONFIG_FILE}\n"
                "Download it from Google Colab and place it next to app.py"
            )
        with open(CONFIG_FILE, "rb") as f:
            _config = pickle.load(f)
    return _config


def _load_extractor():
    global _extractor
    if _extractor is None:
        _extractor = tf.keras.applications.Xception(
            weights="imagenet", include_top=False,
            pooling="avg", input_shape=(*IMG_SIZE, 3)
        )
        _extractor.trainable = False
    return _extractor


def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"captioning_model.keras not found at {MODEL_PATH}\n"
                "Download it from Google Colab and place it next to app.py"
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def _load_tokenizer():
    global _tokenizer, _idx2word
    if _tokenizer is None:
        if not os.path.exists(TOKENIZER_FILE):
            raise FileNotFoundError(
                f"tokenizer.pkl not found at {TOKENIZER_FILE}\n"
                "Download it from Google Colab and place it next to app.py"
            )
        with open(TOKENIZER_FILE, "rb") as f:
            _tokenizer = pickle.load(f)
        _idx2word = {v: k for k, v in _tokenizer.word_index.items()}
    return _tokenizer, _idx2word


def extract_feature(image):
    """Extract a 2048-d Xception feature vector from a PIL Image."""
    img = image.convert("RGB").resize(IMG_SIZE)
    x   = img_to_array(img)
    x   = preprocess_input(x[np.newaxis])
    return _load_extractor().predict(x, verbose=0).squeeze()


def _greedy(feature):
    model               = _load_model()
    tokenizer, idx2word = _load_tokenizer()
    max_len             = _load_config()["max_len"]

    end_id = tokenizer.word_index.get("endseq", 2)
    # Reserve 2 slots for startseq + endseq — never force-fill all max_len steps
    budget = max_len - 2

    seq     = [tokenizer.word_index.get("startseq", 1)]
    caption = []

    for _ in range(budget):
        # padding='post' matches the training data generator
        padded  = pad_sequences([seq], maxlen=max_len, padding="post")
        preds   = model.predict([feature[np.newaxis], padded], verbose=0)
        next_id = int(np.argmax(preds))

        # Check by id AND by word to handle edge cases in idx2word
        if next_id == end_id:
            break
        word = idx2word.get(next_id)
        if word is None or word == "endseq":
            break

        seq.append(next_id)
        caption.append(word)

    return " ".join(caption)


def _beam(feature, beam_width=3):
    model               = _load_model()
    tokenizer, idx2word = _load_tokenizer()
    max_len             = _load_config()["max_len"]

    start_id = tokenizer.word_index.get("startseq", 1)
    end_id   = tokenizer.word_index.get("endseq", 2)

    # Reserve 2 slots for startseq + endseq
    budget = max_len - 2

    # Each beam: (token_id_list, cumulative_log_prob)
    beams     = [([start_id], 0.0)]
    completed = []

    for _ in range(budget):
        all_cands = []

        for seq, score in beams:
            # FIX: check by id so endseq is caught even if idx2word is offset
            if seq[-1] == end_id:
                completed.append((seq, score))
                continue

            # padding='post' matches the training data generator
            padded = pad_sequences([seq], maxlen=max_len, padding="post")
            preds  = model.predict([feature[np.newaxis], padded], verbose=0)[0]
            top_k  = np.argsort(preds)[-beam_width:]
            for idx in top_k:
                all_cands.append((
                    seq + [int(idx)],
                    score + float(np.log(preds[idx] + 1e-9))
                ))

        # FIX: exit early when every active beam has finished
        if not all_cands:
            break

        # Normalise by actual generated word count (excludes startseq token)
        all_cands.sort(
            key=lambda x: x[1] / max(len(x[0]) - 1, 1),
            reverse=True
        )
        beams = all_cands[:beam_width]

    # Collect any beams still active at budget exhaustion
    completed += beams

    # Pick the best sequence
    best = max(
        completed,
        key=lambda x: x[1] / max(len(x[0]) - 1, 1)
    )[0]

    return " ".join(
        idx2word[i] for i in best
        if idx2word.get(i) not in (None, "startseq", "endseq")
    )


def caption_image(image, method="beam", beam_width=3):
    """
    Generate a caption for a PIL Image.

    Args:
        image:      PIL.Image object
        method:     'beam' (recommended) or 'greedy'
        beam_width: number of beams (only used when method='beam')

    Returns:
        str — the generated caption
    """
    feature = extract_feature(image)
    if method == "beam":
        return _beam(feature, beam_width)
    return _greedy(feature)


def check_model_files():
    """Return a dict showing which required files exist."""
    return {
        "captioning_model.keras": os.path.exists(MODEL_PATH),
        "tokenizer.pkl":          os.path.exists(TOKENIZER_FILE),
        "config.pkl":             os.path.exists(CONFIG_FILE),
    }

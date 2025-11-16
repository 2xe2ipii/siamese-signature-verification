import os
import random
import json
import gc

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Lambda,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Lambda

from sklearn.metrics import roc_curve, auc, confusion_matrix

mixed_precision.set_global_policy("mixed_float16")
print("[DEBUG] Using mixed precision policy:", mixed_precision.global_policy())
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[DEBUG] Enabled memory growth for {len(gpus)} GPU(s)")
    except Exception as e:
        print(f"[WARN] Could not set memory growth: {e}")
else:
    print("[DEBUG] No GPU detected, running on CPU.")


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

BASE_CONFIG = {
    "dataset_root": "./dataset",
    "img_height": 120,
    "img_width": 200,
    "batch_size": 8,          
    "epochs": 40,             
    "learning_rate": 1e-4,
    "margin": 1.0,

    "num_train_writers": 80,  
    "num_test_writers": 10,
    "max_pairs_per_writer": None,

    "models_root": "./models",
    "experiment_name": "siamese_experiment",
    "random_seed": 42,
}



# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------

def seed_everything(seed: int = 42):
    print(f"[DEBUG] Seeding everything with {seed}")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: str):
    if not os.path.exists(path):
        print(f"[DEBUG] Creating directory: {path}")
        os.makedirs(path, exist_ok=True)


def create_experiment_dir(root: str, name: str) -> str:
    ensure_dir(root)
    idx = 1
    while True:
        p = os.path.join(root, f"{name}_{idx:02d}")
        if not os.path.exists(p):
            os.makedirs(p)
            print(f"[DEBUG] Created experiment directory: {p}")
            return p
        idx += 1


VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_images_from_folder(path: str, H: int, W: int):
    imgs = []
    print(f"[DEBUG] Loading images from folder: {path}")
    for fname in sorted(os.listdir(path)):
        if not fname.lower().endswith(VALID_EXTS):
            continue
        full = os.path.join(path, fname)
        img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Cannot read {full}, skipping.")
            continue
        img = cv2.resize(img, (W, H))
        img = img.astype("float32") / 255.0
        imgs.append(img)
    print(f"[DEBUG] Loaded {len(imgs)} images from {path}")
    return imgs


def load_dataset(root: str, H: int, W: int):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset not found: {root}")

    print(f"[INFO] Loading dataset from {root}")
    writers = {}

    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue

        if name.endswith("_forg"):
            writer_id = name[:-5]
            mode = "forg"
        else:
            writer_id = name
            mode = "genuine"

        imgs = load_images_from_folder(folder, H, W)
        if not imgs:
            continue

        if writer_id not in writers:
            writers[writer_id] = {"genuine": [], "forg": []}

        writers[writer_id][mode].extend(imgs)
        print(
            f"[DEBUG] Writer {writer_id}, mode={mode}, total now: "
            f"{len(writers[writer_id]['genuine'])} genuine, "
            f"{len(writers[writer_id]['forg'])} forged"
        )

    valid = {}
    for w, d in writers.items():
        g_count = len(d["genuine"])
        f_count = len(d["forg"])
        if g_count >= 2 and f_count >= 1:
            valid[w] = d
        else:
            print(f"[WARN] Skipping writer {w}: {g_count} genuine, {f_count} forged")

    print(f"[INFO] Total writers found: {len(writers)}")
    print(f"[INFO] Writers usable (>=2 genuine, >=1 forged): {len(valid)}")

    if len(valid) < 2:
        raise RuntimeError("Not enough valid writers for writer-independent training.")

    return valid


def split_writers(
    writers: dict,
    num_train: int,
    num_test: int | None,
    seed: int = 42,
):
    all_ids = list(writers.keys())
    total = len(all_ids)

    rng = random.Random(seed)
    rng.shuffle(all_ids)

    # ensure we have at least 1 test writer
    if num_train >= total:
        num_train = max(total - 1, 1)

    if num_test is None:
        num_test = total - num_train
    else:
        num_test = min(num_test, total - num_train)
        if num_test <= 0:
            raise ValueError(
                f"num_test_writers={num_test} is not valid for total={total}, num_train={num_train}"
            )

    train_ids = all_ids[:num_train]
    test_ids = all_ids[num_train:num_train + num_test]

    print(f"[INFO] Writer split:")
    print(f"       Total writers : {total}")
    print(f"       Train writers : {len(train_ids)}")
    print(f"       Test writers  : {len(test_ids)}")

    return train_ids, test_ids


def generate_pairs(
    writers: dict,
    ids,
    max_pairs_per_writer: int | None,
    rng: random.Random
):
    X1, X2, y = [], [], []

    print("[INFO] Generating pairs...")
    for wid in ids:
        g = writers[wid]["genuine"]
        f = writers[wid]["forg"]
        print(f"[DEBUG] Writer {wid}: {len(g)} genuine, {len(f)} forged")

        # Positive pairs: all distinct genuine-genuine pairs
        pos = [(g[i], g[j]) for i in range(len(g)) for j in range(i + 1, len(g))]
        # Negative pairs: genuine vs forged
        neg = [(gi, fj) for gi in g for fj in f]

        rng.shuffle(pos)
        rng.shuffle(neg)

        if not pos or not neg:
            print(f"[WARN] Writer {wid} has no pos or neg pairs, skipping.")
            continue

        if max_pairs_per_writer is None:
            use = min(len(pos), len(neg))
        else:
            use = min(max_pairs_per_writer, len(pos), len(neg))

        pos = pos[:use]
        neg = neg[:use]

        for a, b in pos:
            X1.append(a)
            X2.append(b)
            y.append(1.0)

        for a, b in neg:
            X1.append(a)
            X2.append(b)
            y.append(0.0)

        print(f"[DEBUG] Writer {wid}: using {use} positive and {use} negative pairs")

    if not X1:
        raise RuntimeError("No pairs generated. Check dataset and writer IDs.")

    # store as float16 to reduce memory (with mixed precision this is fine)
    X1 = np.array(X1, dtype="float16")[..., np.newaxis]
    X2 = np.array(X2, dtype="float16")[..., np.newaxis]
    y = np.array(y, dtype="float32")

    print(f"[INFO] Total pairs generated: {len(y)}")
    unique, counts = np.unique(y, return_counts=True)
    print(
        f"[INFO] Label distribution (1=genuine, 0=forged): "
        f"{dict(zip(unique.tolist(), counts.tolist()))}"
    )
    print(f"[DEBUG] X1 shape: {X1.shape}, X2 shape: {X2.shape}, y shape: {y.shape}")

    approx_mem = (X1.nbytes + X2.nbytes + y.nbytes) / (1024 ** 2)
    print(f"[DEBUG] Approx dataset memory (X1+X2+y): {approx_mem:.2f} MB")

    return X1, X2, y


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------

def lrn(x):
    return tf.nn.local_response_normalization(
        x,
        depth_radius=2,
        bias=2.0,
        alpha=1e-4,
        beta=0.75
    )


def build_base_network(input_shape):
    print(f"[DEBUG] Building base network with input shape: {input_shape}")
    inp = Input(shape=input_shape)

    # Block 1
    x = Conv2D(32, (7, 7), padding="same", activation="relu")(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 2
    x = Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # Block 3
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)

    # Dense embedding
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)

    # Final L2 normalization ONLY here
    x = Lambda(lambda t: tf.nn.l2_normalize(t, axis=1))(x)

    base = Model(inp, x, name="BaseNetwork")
    print("[DEBUG] Base network summary:")
    base.summary(print_fn=lambda s: print("[DEBUG] " + s))
    return base


def euclidean_distance(vects):
    a, b = vects
    # do distance in float32 to keep numerics stable with mixed precision
    diff = tf.cast(a, tf.float32) - tf.cast(b, tf.float32)
    sq = K.square(diff)
    dist = K.sqrt(K.maximum(K.sum(sq, axis=1, keepdims=True), K.epsilon()))
    return dist


def contrastive_loss(margin: float):
    m = tf.cast(margin, tf.float32)

    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)

        pos = y_true_f * K.square(y_pred_f)
        neg = (1.0 - y_true_f) * K.square(K.maximum(m - y_pred_f, 0.0))
        return K.mean(pos + neg)

    return loss


def build_siamese(input_shape, margin: float, lr: float):
    print(f"[DEBUG] Building siamese model with margin={margin}, lr={lr}")
    base = build_base_network(input_shape)

    i1 = Input(shape=input_shape, name="input_1")
    i2 = Input(shape=input_shape, name="input_2")

    e1 = base(i1)
    e2 = base(i2)

    # distance output as float32 to keep loss numerically stable in mixed precision
    dist = Lambda(
        euclidean_distance,
        name="distance",
        dtype="float32"
    )([e1, e2])

    siamese = Model([i1, i2], dist, name="SiameseNetwork")

    opt = RMSprop(learning_rate=lr)
    # Keras will wrap with LossScaleOptimizer automatically for mixed_float16

    siamese.compile(
        loss=contrastive_loss(margin),
        optimizer=opt,
    )

    print("[DEBUG] Siamese model summary:")
    siamese.summary(print_fn=lambda s: print("[DEBUG] " + s))

    return siamese, base


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def plot_curve(history, path: str):
    print(f"[DEBUG] Saving learning curve to {path}")
    plt.figure()
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_distances(dists, labels, path: str):
    print(f"[DEBUG] Saving distance histogram to {path}")
    dists = np.asarray(dists).reshape(-1)
    labels = np.asarray(labels).astype(int)

    pos = dists[labels == 1]
    neg = dists[labels == 0]

    plt.figure()
    plt.hist(pos, bins=30, alpha=0.6, label="genuine")
    plt.hist(neg, bins=30, alpha=0.6, label="forged")
    plt.legend()
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.grid()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_roc(fpr, tpr, aucv, path: str):
    print(f"[DEBUG] Saving ROC curve to {path}")
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={aucv:.3f}")
    plt.plot([0, 1], [0, 1], "--", label="random")
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_cm(cm, path: str):
    print(f"[DEBUG] Saving confusion matrix to {path}")
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xticks([0, 1], ["forged", "genuine"])
    plt.yticks([0, 1], ["forged", "genuine"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# EXPERIMENT
# ---------------------------------------------------------------------

def run_experiment(cfg: dict):
    tf.keras.backend.clear_session()
    seed_everything(cfg["random_seed"])

    exp_dir = create_experiment_dir(cfg["models_root"], cfg["experiment_name"])
    print(f"[INFO] Experiment directory: {exp_dir}")

    writers = load_dataset(cfg["dataset_root"], cfg["img_height"], cfg["img_width"])
    train_ids, test_ids = split_writers(
        writers,
        cfg["num_train_writers"],
        cfg["num_test_writers"],
        seed=cfg["random_seed"],
    )

    rng = random.Random(cfg["random_seed"])

    print("[INFO] Generating training pairs...")
    X1_train, X2_train, y_train = generate_pairs(
        writers, train_ids, cfg["max_pairs_per_writer"], rng
    )

    print("[INFO] Generating test pairs...")
    X1_test, X2_test, y_test = generate_pairs(
        writers, test_ids, cfg["max_pairs_per_writer"], rng
    )

    # free raw images to reduce CPU RAM usage
    del writers
    gc.collect()

    input_shape = (cfg["img_height"], cfg["img_width"], 1)
    siamese, base = build_siamese(input_shape, cfg["margin"], cfg["learning_rate"])

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    print("[INFO] Starting training...")
    history = siamese.fit(
        [X1_train, X2_train],
        y_train,
        validation_data=([X1_test, X2_test], y_test),
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        callbacks=callbacks,
    )

    print("[INFO] Predicting distances on test pairs...")
    dists = siamese.predict([X1_test, X2_test]).reshape(-1)
    # move to float32 on CPU for metrics
    dists = dists.astype("float32")
    print(
        f"[DEBUG] Distances stats -> "
        f"min: {dists.min():.4f}, max: {dists.max():.4f}, "
        f"mean: {dists.mean():.4f}, std: {dists.std():.4f}"
    )

    y_true = y_test.astype(int)

    # ROC (note: smaller distance => more likely genuine)
    fpr, tpr, roc_thresholds = roc_curve(y_true, -dists)
    aucv = auc(fpr, tpr)
    print(f"[INFO] ROC AUC on test pairs: {aucv:.4f}")

    # Threshold sweep for balanced accuracy
    thresholds = np.linspace(dists.min(), dists.max(), 500)
    best_acc = -1.0
    best_th = None

    print("[INFO] Sweeping thresholds for balanced accuracy...")
    for th in thresholds:
        pred = (dists < th).astype(int)
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # degenerate case; skip
            continue
        tpr_val = tp / (tp + fn + 1e-8)
        tnr_val = tn / (tn + fp + 1e-8)
        acc = 0.5 * (tpr_val + tnr_val)
        if acc > best_acc:
            best_acc = acc
            best_th = th

    print(
        f"[INFO] Best threshold: {best_th:.4f}, "
        f"best balanced accuracy: {best_acc:.4f}"
    )

    final_pred = (dists < best_th).astype(int)
    final_cm = confusion_matrix(y_true, final_pred, labels=[0, 1])

    # Save models
    siamese_path = os.path.join(exp_dir, "siamese_model.h5")
    base_path = os.path.join(exp_dir, "embedding_model.h5")
    print(f"[DEBUG] Saving siamese model to {siamese_path}")
    print(f"[DEBUG] Saving base model to {base_path}")
    siamese.save(siamese_path)
    base.save(base_path)

    # Save metrics
    metrics = {
        "best_threshold": float(best_th),
        "best_balanced_accuracy": float(best_acc),
        "roc_auc": float(aucv),
        "confusion_matrix": final_cm.tolist(),
        "num_train_pairs": int(len(y_train)),
        "num_test_pairs": int(len(y_test)),
    }

    metrics_path = os.path.join(exp_dir, "metrics.json")
    print(f"[DEBUG] Saving metrics to {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save plots
    plot_curve(history, os.path.join(exp_dir, "learning_curve.png"))
    plot_distances(dists, y_true, os.path.join(exp_dir, "distance_hist.png"))
    plot_roc(fpr, tpr, aucv, os.path.join(exp_dir, "roc_curve.png"))
    plot_cm(final_cm, os.path.join(exp_dir, "confusion_matrix.png"))

    print("[INFO] Finished experiment.")
    print("[INFO] Metrics:", metrics)
    return metrics

if __name__ == "__main__":
    run_experiment(BASE_CONFIG)
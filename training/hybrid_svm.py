import os, time, glob, pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
from scipy.stats import entropy as shannon_entropy
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type != "cuda":
    print("WARNING: no GPU. Runtime > Change runtime type > T4 GPU.")

"""## 1. Upload dataset

Structure: `dataset/train/{real,screen}` and `dataset/val/{real,screen}`.

"""

# from google.colab import files
import zipfile
print("Upload dataset.zip...")
uploaded = files.upload()
zip_name = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_name, 'r') as z:
    z.extractall('.')
print("Extracted.")

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validation"

def list_images(folder):
    out = []
    for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"):
        out += glob.glob(os.path.join(folder, ext))
    return sorted(set(out))

train_real = list_images(os.path.join(TRAIN_DIR, "real"))
train_screen = list_images(os.path.join(TRAIN_DIR, "screen"))
val_real = list_images(os.path.join(VAL_DIR, "real"))
val_screen = list_images(os.path.join(VAL_DIR, "screen"))
print(f"Train: {len(train_real)} real, {len(train_screen)} screen")
print(f"Val:   {len(val_real)} real, {len(val_screen)} screen")

"""## 2. Handcrafted features (6, identical to predict.py)"""

PATCH_SIZE = 64
PATCH_STRIDE = 32
PYRAMID_LEVELS = 3
HIGH_RES_THRESHOLD = 900
MAX_LONGEST_SIDE = 2200

def load_image_gray_and_bgr(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest > MAX_LONGEST_SIDE:
        scale = MAX_LONGEST_SIDE / longest
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray, bgr

def build_gaussian_pyramid(gray_img, levels=PYRAMID_LEVELS):
    pyramid = [gray_img]; current = gray_img
    for _ in range(levels - 1):
        if min(current.shape) < PATCH_SIZE * 2:
            break
        current = cv2.pyrDown(current); pyramid.append(current)
    return pyramid

def extract_patches(img, patch_size=PATCH_SIZE, stride=PATCH_STRIDE):
    h, w = img.shape
    if h < patch_size or w < patch_size:
        return
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            yield img[y:y + patch_size, x:x + patch_size]

def _fft_magnitude(patch):
    f = np.fft.fft2(patch); fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    cy, cx = magnitude.shape[0]//2, magnitude.shape[1]//2
    magnitude[cy, cx] = 0
    return magnitude

def _peak_prominence(magnitude, ring_radius=4):
    py, px = np.unravel_index(magnitude.argmax(), magnitude.shape)
    peak_val = magnitude[py, px]
    h, w = magnitude.shape
    y, x = np.indices((h, w))
    dist = np.sqrt((y - py)**2 + (x - px)**2)
    ring_mask = (dist > 1.5) & (dist <= ring_radius)
    if ring_mask.sum() == 0:
        return 0.0
    baseline = magnitude[ring_mask].mean()
    if baseline < 1e-6:
        return float(peak_val)
    return float(peak_val / baseline)

def _harmonic_regularity_score(magnitude, n_harmonics=3, search_tolerance=2, min_fundamental_dist=4):
    h, w = magnitude.shape
    cy, cx = h//2, w//2
    y_idx, x_idx = np.indices((h, w))
    dist_from_center = np.sqrt((y_idx - cy)**2 + (x_idx - cx)**2)
    valid_mask = dist_from_center >= min_fundamental_dist
    masked_mag = magnitude.copy(); masked_mag[~valid_mask] = 0
    if masked_mag.max() < 1e-6:
        return 0.0
    py, px = np.unravel_index(masked_mag.argmax(), masked_mag.shape)
    dy, dx = py - cy, px - cx
    fundamental_val = magnitude[py, px]
    harmonic_energies = []
    for k in range(2, n_harmonics + 2):
        hy = int(round(cy + dy*k)); hx = int(round(cx + dx*k))
        if not (0 <= hy < h and 0 <= hx < w):
            continue
        y0, y1 = max(0, hy - search_tolerance), min(h, hy + search_tolerance + 1)
        x0, x1 = max(0, hx - search_tolerance), min(w, hx + search_tolerance + 1)
        local_patch = magnitude[y0:y1, x0:x1]
        if local_patch.size == 0:
            continue
        harmonic_energies.append(local_patch.max())
    if not harmonic_energies:
        return 0.0
    return float(np.mean(harmonic_energies) / (fundamental_val + 1e-6))

def _frequency_entropy(magnitude):
    flat = magnitude.flatten(); total = flat.sum()
    if total < 1e-6:
        return 0.0
    p = flat / total; p = p[p > 1e-12]
    return float(shannon_entropy(p))

def _radial_energy_anomaly(magnitude):
    h, w = magnitude.shape
    cy, cx = h//2, w//2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy)**2 + (x - cx)**2).astype(int)
    max_r = r.max()
    radial_profile = np.zeros(max_r + 1); counts = np.zeros(max_r + 1)
    np.add.at(radial_profile, r.flatten(), magnitude.flatten())
    np.add.at(counts, r.flatten(), 1)
    counts[counts == 0] = 1
    radial_profile = radial_profile / counts
    n = len(radial_profile)
    band = radial_profile[max(2, n//10): n - 2]
    if len(band) < 3:
        return 0.0
    window = max(3, len(band)//10)
    kernel = np.ones(window) / window
    smooth = np.convolve(band, kernel, mode="same")
    smooth[smooth < 1e-6] = 1e-6
    deviation = np.abs(band - smooth) / smooth
    return float(deviation.max())

def patch_frequency_features(gray_img):
    pyramid = build_gaussian_pyramid(gray_img)
    all_prom, all_ent, all_rad, all_harm = [], [], [], []
    for scale_img in pyramid:
        for patch in extract_patches(scale_img):
            magnitude = _fft_magnitude(patch)
            all_prom.append(_peak_prominence(magnitude))
            all_ent.append(_frequency_entropy(magnitude))
            all_rad.append(_radial_energy_anomaly(magnitude))
            all_harm.append(_harmonic_regularity_score(magnitude))
    if len(all_prom) == 0:
        return None
    return {"prominence_patches": np.array(all_prom), "entropy_patches": np.array(all_ent),
            "radial_patches": np.array(all_rad), "harmonic_patches": np.array(all_harm)}

def laplacian_consistency(gray_img, grid_patch=64):
    sharpness_vals = []
    for patch in extract_patches(gray_img, patch_size=grid_patch, stride=grid_patch):
        lap = cv2.Laplacian(patch.astype(np.float64), cv2.CV_64F)
        sharpness_vals.append(lap.var())
    sharpness_vals = np.array(sharpness_vals)
    if len(sharpness_vals) < 2 or sharpness_vals.mean() < 1e-6:
        return 0.0
    return float(sharpness_vals.std() / (sharpness_vals.mean() + 1e-8))

def rgb_fringe_score(bgr_img, min_resolution=HIGH_RES_THRESHOLD):
    h, w = bgr_img.shape[:2]
    if min(h, w) < min_resolution:
        return None
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    ys, xs = np.where(edges > 0)
    if len(ys) == 0:
        return 0.0
    if len(ys) > 500:
        idx = np.random.RandomState(0).choice(len(ys), 500, replace=False)
        ys, xs = ys[idx], xs[idx]
    b, g, r = cv2.split(bgr_img.astype(np.float32))
    fringe_scores = []
    for y, x in zip(ys, xs):
        x0, x1 = max(0, x - 3), min(w, x + 4)
        if x1 - x0 < 4:
            continue
        channel_std = np.std([r[y, x0:x1], g[y, x0:x1], b[y, x0:x1]], axis=0)
        fringe_scores.append(channel_std.mean())
    if not fringe_scores:
        return 0.0
    return float(np.mean(fringe_scores))

FEATURE_NAMES_HC = [
    "prominence_top10_mean", "entropy_top5_mean", "radial_p90",
    "harmonic_mean", "laplacian_cov", "rgb_fringe",
]

def extract_handcrafted_features(path):
    gray, bgr = load_image_gray_and_bgr(path)
    if gray is None:
        return None
    freq_feats = patch_frequency_features(gray)
    if freq_feats is None:
        return None
    def top10_mean(a): return np.mean(np.sort(a)[-10:]) if len(a) >= 10 else np.mean(a)
    def top5_mean(a): return np.mean(np.sort(a)[-5:]) if len(a) >= 5 else np.mean(a)
    fringe = rgb_fringe_score(bgr)
    return np.array([
        top10_mean(freq_feats["prominence_patches"]),
        top5_mean(freq_feats["entropy_patches"]),
        np.percentile(freq_feats["radial_patches"], 90),
        np.mean(freq_feats["harmonic_patches"]),
        laplacian_consistency(gray),
        fringe if fringe is not None else np.nan,
    ])

print(f"Handcrafted features ready: {len(FEATURE_NAMES_HC)} ->", FEATURE_NAMES_HC)

"""## 3. EfficientNet-B0 embeddings (frozen, classifier stripped)"""

effnet = models.efficientnet_b0(weights="DEFAULT")
effnet.classifier = nn.Identity()
effnet.eval().to(device)

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
embed_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def extract_cnn_embedding(path):
    img = Image.open(path).convert("RGB")
    x = embed_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return effnet(x).cpu().numpy().flatten()

print("Embedding extractor ready (1280-dim).")

"""## 4. Extract features for all images"""

def build_dataset(real_paths, screen_paths, desc=""):
    hc_list, emb_list, labels, fnames = [], [], [], []
    all_paths = [(p, 0) for p in real_paths] + [(p, 1) for p in screen_paths]
    for i, (path, label) in enumerate(all_paths):
        hc = extract_handcrafted_features(path)
        if hc is None:
            print(f"  [skip] {path}"); continue
        emb = extract_cnn_embedding(path)
        hc_list.append(hc); emb_list.append(emb); labels.append(label)
        fnames.append(os.path.basename(path))
        if (i + 1) % 20 == 0:
            print(f"  {desc}: {i+1}/{len(all_paths)}")
    return np.array(hc_list), np.array(emb_list), np.array(labels), fnames

print("Extracting TRAIN features...")
t0 = time.time()
X_hc_train, X_emb_train, y_train, fnames_train = build_dataset(train_real, train_screen, "train")
print(f"Done in {time.time()-t0:.1f}s. HC: {X_hc_train.shape}, Emb: {X_emb_train.shape}")

print("Extracting VAL features...")
t0 = time.time()
X_hc_val, X_emb_val, y_val, fnames_val = build_dataset(val_real, val_screen, "val")
print(f"Done in {time.time()-t0:.1f}s.")

"""## 5. Combine features: impute, PCA on embeddings, concatenate, scale"""

N_PCA_COMPONENTS = 20

col_medians = np.nanmedian(X_hc_train, axis=0)
def impute(X):
    X = X.copy(); m = np.isnan(X)
    X[m] = np.take(col_medians, np.where(m)[1]); return X

X_hc_train_imp = impute(X_hc_train)
X_hc_val_imp = impute(X_hc_val)

pca = PCA(n_components=min(N_PCA_COMPONENTS, X_emb_train.shape[0]-1), random_state=42)
X_emb_train_pca = pca.fit_transform(X_emb_train)
X_emb_val_pca = pca.transform(X_emb_val)
print(f"PCA explained variance ({N_PCA_COMPONENTS} comps): {pca.explained_variance_ratio_.sum():.3f}")

X_combined_train = np.concatenate([X_hc_train_imp, X_emb_train_pca], axis=1)
X_combined_val = np.concatenate([X_hc_val_imp, X_emb_val_pca], axis=1)

scaler = StandardScaler().fit(X_combined_train)
X_train_s = scaler.transform(X_combined_train)
X_val_s = scaler.transform(X_combined_val)
print(f"Combined feature dim: {X_train_s.shape[1]} (6 handcrafted + {N_PCA_COMPONENTS} PCA)")

"""## 6. Train SVM-RBF (hyperparameters via 5-fold CV on train only)"""

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

svm_grid = GridSearchCV(
    SVC(kernel="rbf", probability=True, random_state=42),
    {"C": [0.5, 1.0, 5.0, 10.0], "gamma": ["scale", "auto", 0.01, 0.1]},
    cv=cv, scoring="accuracy")
svm_grid.fit(X_train_s, y_train)

clf = svm_grid.best_estimator_
print(f"Best SVM params: {svm_grid.best_params_}")
print(f"5-fold CV accuracy on train: {svm_grid.best_score_:.3f}")

"""## 7. Validation evaluation"""

probs_val = clf.predict_proba(X_val_s)[:, 1]
preds_val = (probs_val >= 0.5).astype(int)

acc = accuracy_score(y_val, preds_val)
prec = precision_score(y_val, preds_val, zero_division=0)
rec = recall_score(y_val, preds_val, zero_division=0)
f1 = f1_score(y_val, preds_val, zero_division=0)

print(f"=== Final Hybrid (6 features + EfficientNet PCA + SVM-RBF) ===")
print(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
print(f"({int((preds_val==y_val).sum())}/{len(y_val)} correct)")

errors = [(f,l,p,pr) for f,l,p,pr in zip(fnames_val, y_val, preds_val, probs_val) if l != p]
print(f"\nErrors: {len(errors)}/{len(y_val)}")
for fname, t, p, pr in sorted(errors, key=lambda e: e[3]):
    ts = "Real" if t==0 else "Screen"; ps = "Real" if p==0 else "Screen"
    print(f"  {fname:35s} GT={ts:7s} Pred={ps:7s} P(screen)={pr:.3f}")

cm = confusion_matrix(y_val, preds_val, labels=[0,1])
fig, ax = plt.subplots(figsize=(5,5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Real","Screen"]); ax.set_yticklabels(["Real","Screen"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Final Hybrid Confusion Matrix (acc={acc:.3f})")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=16, fontweight="bold",
                color="white" if cm[i,j] > cm.max()/2 else "black")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout(); plt.savefig("final_confusion_matrix.png", dpi=130); plt.show()


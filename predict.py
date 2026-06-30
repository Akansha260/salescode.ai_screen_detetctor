"""
predict.py — Real Photo vs Photo-of-Screen Classifier
Usage:
    python predict.py image.jpg

Prints a single probability in [0, 1] to stdout:
    1 = photo of a screen
    0 = real photo"""

import sys
import os
import glob
import pickle
import time
import warnings

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from scipy.stats import entropy as shannon_entropy

warnings.filterwarnings("ignore")

MODEL_PATH = "hybrid_screen_classifier.pkl"

PATCH_SIZE = 64
PATCH_STRIDE = 32
PYRAMID_LEVELS = 3
HIGH_RES_THRESHOLD = 900

FEATURE_NAMES_HC = [
    "prominence_top10_mean", "entropy_top5_mean", "radial_p90",
    "harmonic_mean", "laplacian_cov", "rgb_fringe",
]

MAX_LONGEST_SIDE = 2200


def load_image_gray_and_bgr(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None

    h, w = bgr.shape[:2]
    longest_side = max(h, w)
    if longest_side > MAX_LONGEST_SIDE:
        scale = MAX_LONGEST_SIDE / longest_side
        new_w, new_h = int(w * scale), int(h * scale)
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray, bgr


def build_gaussian_pyramid(gray_img, levels=PYRAMID_LEVELS):
    pyramid = [gray_img]
    current = gray_img
    for _ in range(levels - 1):
        if min(current.shape) < PATCH_SIZE * 2:
            break
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def extract_patches(img, patch_size=PATCH_SIZE, stride=PATCH_STRIDE):
    h, w = img.shape
    if h < patch_size or w < patch_size:
        return
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            yield img[y:y + patch_size, x:x + patch_size]


def _fft_magnitude(patch):
    f = np.fft.fft2(patch)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    cy, cx = magnitude.shape[0] // 2, magnitude.shape[1] // 2
    magnitude[cy, cx] = 0
    return magnitude


def _peak_prominence(magnitude, ring_radius=4):
    py, px = np.unravel_index(magnitude.argmax(), magnitude.shape)
    peak_val = magnitude[py, px]
    h, w = magnitude.shape
    y, x = np.indices((h, w))
    dist = np.sqrt((y - py) ** 2 + (x - px) ** 2)
    ring_mask = (dist > 1.5) & (dist <= ring_radius)
    if ring_mask.sum() == 0:
        return 0.0
    baseline = magnitude[ring_mask].mean()
    if baseline < 1e-6:
        return float(peak_val)
    return float(peak_val / baseline)


def _harmonic_regularity_score(magnitude, n_harmonics=3, search_tolerance=2,
                                min_fundamental_dist=4):
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.indices((h, w))
    dist_from_center = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    valid_mask = dist_from_center >= min_fundamental_dist
    masked_mag = magnitude.copy()
    masked_mag[~valid_mask] = 0
    if masked_mag.max() < 1e-6:
        return 0.0
    py, px = np.unravel_index(masked_mag.argmax(), masked_mag.shape)
    dy, dx = py - cy, px - cx
    fundamental_val = magnitude[py, px]
    harmonic_energies = []
    for k in range(2, n_harmonics + 2):
        hy = int(round(cy + dy * k))
        hx = int(round(cx + dx * k))
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
    flat = magnitude.flatten()
    total = flat.sum()
    if total < 1e-6:
        return 0.0
    p = flat / total
    p = p[p > 1e-12]
    return float(shannon_entropy(p))


def _radial_energy_anomaly(magnitude):
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    max_r = r.max()
    radial_profile = np.zeros(max_r + 1)
    counts = np.zeros(max_r + 1)
    np.add.at(radial_profile, r.flatten(), magnitude.flatten())
    np.add.at(counts, r.flatten(), 1)
    counts[counts == 0] = 1
    radial_profile = radial_profile / counts
    n = len(radial_profile)
    band = radial_profile[max(2, n // 10): n - 2]
    if len(band) < 3:
        return 0.0
    window = max(3, len(band) // 10)
    kernel = np.ones(window) / window
    smooth = np.convolve(band, kernel, mode="same")
    smooth[smooth < 1e-6] = 1e-6
    deviation = np.abs(band - smooth) / smooth
    return float(deviation.max())


def patch_frequency_features(gray_img):
    pyramid = build_gaussian_pyramid(gray_img)
    all_prominence, all_entropy, all_radial, all_harmonic = [], [], [], []
    for scale_img in pyramid:
        for patch in extract_patches(scale_img):
            magnitude = _fft_magnitude(patch)
            all_prominence.append(_peak_prominence(magnitude))
            all_entropy.append(_frequency_entropy(magnitude))
            all_radial.append(_radial_energy_anomaly(magnitude))
            all_harmonic.append(_harmonic_regularity_score(magnitude))
    if len(all_prominence) == 0:
        return None
    return {
        "prominence_patches": np.array(all_prominence),
        "entropy_patches": np.array(all_entropy),
        "radial_patches": np.array(all_radial),
        "harmonic_patches": np.array(all_harmonic),
    }


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


def extract_handcrafted_features(path):
    gray, bgr = load_image_gray_and_bgr(path)
    if gray is None:
        return None
    freq_feats = patch_frequency_features(gray)
    if freq_feats is None:
        return None

    def top10_mean(a):
        return np.mean(np.sort(a)[-10:]) if len(a) >= 10 else np.mean(a)

    def top5_mean(a):
        return np.mean(np.sort(a)[-5:]) if len(a) >= 5 else np.mean(a)

    fringe = rgb_fringe_score(bgr)
    return np.array([
        top10_mean(freq_feats["prominence_patches"]),
        top5_mean(freq_feats["entropy_patches"]),
        np.percentile(freq_feats["radial_patches"], 90),
        np.mean(freq_feats["harmonic_patches"]),
        laplacian_consistency(gray),
        fringe if fringe is not None else np.nan,
    ])


# ---------------------------------------------------------------------------
# EfficientNet-B0 embedding extractor (frozen, classifier stripped)
# ---------------------------------------------------------------------------
def build_efficientnet_embedder():
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Identity()
    return model


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(
            f"Error: {MODEL_PATH} not found. Export it from "
            "hybrid_classifier.ipynb first (the export cell saves this file).",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(MODEL_PATH, "rb") as f:
        original_load = torch.load
        torch.load = lambda *a, **kw: original_load(*a, **{**kw, "map_location": "cpu"})
        try:
            bundle = pickle.load(f)
        finally:
            torch.load = original_load

    effnet = build_efficientnet_embedder()
    effnet.load_state_dict(bundle["effnet_state_dict"])
    effnet.eval()

    embed_transform = transforms.Compose([
        transforms.Resize((bundle["img_size"], bundle["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=bundle["normalize_mean"], std=bundle["normalize_std"]),
    ])

    return bundle, effnet, embed_transform


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_probability(image_path, bundle, effnet, embed_transform):
    hc_feats = extract_handcrafted_features(image_path)
    if hc_feats is None:
        raise ValueError(
            f"Could not extract handcrafted features from {image_path} "
            "(unreadable, corrupt, or below minimum patch size)"
        )

    nan_mask = np.isnan(hc_feats)
    if nan_mask.any():
        hc_feats = hc_feats.copy()
        hc_feats[nan_mask] = bundle["col_medians"][nan_mask]

    img = Image.open(image_path).convert("RGB")
    x = embed_transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = effnet(x).numpy()  # (1, 1280)

    embedding_pca = bundle["pca"].transform(embedding)  # (1, n_pca_components)
    combined = np.concatenate([hc_feats.reshape(1, -1), embedding_pca], axis=1)
    combined_scaled = bundle["scaler"].transform(combined)

    prob_screen = bundle["clf"].predict_proba(combined_scaled)[0, 1]
    return float(prob_screen)


def collect_image_paths(args):
    paths = []
    for arg in args:
        if os.path.isdir(arg):
            for f in sorted(glob.glob(os.path.join(arg, "*"))):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(f)
        elif os.path.isfile(arg):
            paths.append(arg)
        else:
            print(f"Error: path not found: {arg}", file=sys.stderr)
            sys.exit(1)
    return paths


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>", file=sys.stderr)
        print("       python predict.py <folder_path>          (batch mode)", file=sys.stderr)
        print("       python predict.py <img1> <img2> ...       (batch mode)", file=sys.stderr)
        sys.exit(1)

    image_paths = collect_image_paths(sys.argv[1:])
    if not image_paths:
        print("Error: no valid images found in the given argument(s).", file=sys.stderr)
        sys.exit(1)

    bundle, effnet, embed_transform = load_model()

    is_batch = len(image_paths) > 1
    per_image_latencies_ms = []
    t_total_start = time.time()

    for path in image_paths:
        try:
            t0 = time.time()
            prob = predict_probability(path, bundle, effnet, embed_transform)
            latency = time.time() - t0
        except Exception as e:
            print(f"Error: could not process image {path}: {e}", file=sys.stderr)
            if not is_batch:
                sys.exit(1)
            continue

        per_image_latencies_ms.append(latency * 1000)

        if is_batch:
            label = "SCREEN" if prob >= 0.5 else "REAL"
            print(f"{os.path.basename(path)}: {prob:.4f}  [{label}]")
        else:
            print(f"{prob:.4f}")
            print(f"[debug] label = {'SCREEN' if prob >= 0.5 else 'REAL'}", file=sys.stderr)
            print(f"[debug] inference latency = {latency*1000:.1f} ms",
                  file=sys.stderr)

    if is_batch:
        total_time_s = time.time() - t_total_start
        n = len(per_image_latencies_ms)
        if n > 0:
            mean_ms = sum(per_image_latencies_ms) / n
            print(f"\n[summary] processed {n}/{len(image_paths)} images "
                  f"in {total_time_s:.2f}s total "
                  f"(mean per-image latency: {mean_ms:.1f} ms, "
                  f"min: {min(per_image_latencies_ms):.1f} ms, "
                  f"max: {max(per_image_latencies_ms):.1f} ms)", file=sys.stderr)


if __name__ == "__main__":
    main()

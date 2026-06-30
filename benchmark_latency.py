#!/usr/bin/env python3
"""
benchmark_latency.py — Measure per-image inference latency
==============================================================================
Usage:
    python benchmark_latency.py image.jpg
"""

import sys
import os
import glob
import time
import statistics
import warnings

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from scipy.stats import entropy as shannon_entropy
import pickle

warnings.filterwarnings("ignore")

MODEL_PATH = "hybrid_screen_classifier.pkl"
N_TIMED_RUNS = 100

PATCH_SIZE = 64
PATCH_STRIDE = 32
PYRAMID_LEVELS = 3
HIGH_RES_THRESHOLD = 900

# MUST match predict.py exactly, or the benchmark measures a different
# pipeline than the one actually deployed.
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


def build_efficientnet_embedder():
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Identity()
    return model


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.", file=sys.stderr)
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        # Bundle was saved from a Colab GPU training session, so it
        # contains CUDA tensors. pickle.load() internally calls
        # torch.load() when it hits them; on a CPU-only machine this
        # fails unless map_location="cpu" is forced. Temporarily
        # monkey-patch torch.load for the duration of this load.
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


def single_inference(image_path, bundle, effnet, embed_transform):
    t0 = time.perf_counter()

    hc_feats = extract_handcrafted_features(image_path)
    nan_mask = np.isnan(hc_feats)
    if nan_mask.any():
        hc_feats = hc_feats.copy()
        hc_feats[nan_mask] = bundle["col_medians"][nan_mask]

    img = Image.open(image_path).convert("RGB")
    x = embed_transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = effnet(x).numpy()

    embedding_pca = bundle["pca"].transform(embedding)
    combined = np.concatenate([hc_feats.reshape(1, -1), embedding_pca], axis=1)
    combined_scaled = bundle["scaler"].transform(combined)
    _ = bundle["clf"].predict_proba(combined_scaled)[0, 1]

    return time.perf_counter() - t0


def main():
    if len(sys.argv) != 2:
        print("Usage: python benchmark_latency.py <image_path>          "
              "(repeats N_TIMED_RUNS times on one image)", file=sys.stderr)
        print("       python benchmark_latency.py <folder_path>         "
              "(measures each image once, reports per-image + totals)", file=sys.stderr)
        sys.exit(1)

    target = sys.argv[1]

    print("Loading model (untimed)...")
    bundle, effnet, embed_transform = load_model()

    if os.path.isdir(target):
        # Folder mode: one timing per image, report per-image AND total.
        image_paths = sorted(
            f for f in glob.glob(os.path.join(target, "*"))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not image_paths:
            print(f"No images found in {target}", file=sys.stderr)
            sys.exit(1)

        print("Warming up (1 untimed inference on first image)...")
        single_inference(image_paths[0], bundle, effnet, embed_transform)

        print(f"Timing {len(image_paths)} images (1 run each)...")
        per_image_ms = []
        with open("latency_per_image.csv", "w") as csv_f:
            csv_f.write("filename,latency_ms\n")
            for path in image_paths:
                elapsed_ms = single_inference(path, bundle, effnet, embed_transform) * 1000
                per_image_ms.append(elapsed_ms)
                fname = os.path.basename(path)
                print(f"  {fname}: {elapsed_ms:.1f} ms")
                csv_f.write(f"{fname},{elapsed_ms:.2f}\n")

        total_ms = sum(per_image_ms)
        print("\n" + "=" * 50)
        print(f"LATENCY REPORT -- {len(image_paths)} images, 1 run each "
              f"(hybrid pipeline: handcrafted features + CNN embedding + "
              f"PCA + classify)")
        print("=" * 50)
        print(f"Total (sum of all images):  {total_ms:.1f} ms  ({total_ms/1000:.2f} s)")
        print(f"Mean per image:              {statistics.mean(per_image_ms):.2f} ms")
        print(f"Median per image:            {statistics.median(per_image_ms):.2f} ms")
        print(f"Min:                         {min(per_image_ms):.2f} ms")
        print(f"Max:                         {max(per_image_ms):.2f} ms")
        if len(per_image_ms) > 1:
            print(f"Std:                         {statistics.stdev(per_image_ms):.2f} ms")

        with open("latency_hybrid.txt", "w") as f:
            f.write(f"Hybrid model inference latency (CPU)\n")
            f.write(f"Protocol: model loaded once, 1 warm-up run discarded, "
                    f"1 timed run per image across {len(image_paths)} images "
                    f"in {target}\n\n")
            f.write(f"Total (sum):       {total_ms:.1f} ms ({total_ms/1000:.2f} s)\n")
            f.write(f"Mean per image:    {statistics.mean(per_image_ms):.2f} ms\n")
            f.write(f"Median per image:  {statistics.median(per_image_ms):.2f} ms\n")
            f.write(f"Min:               {min(per_image_ms):.2f} ms\n")
            f.write(f"Max:               {max(per_image_ms):.2f} ms\n")
            if len(per_image_ms) > 1:
                f.write(f"Std:               {statistics.stdev(per_image_ms):.2f} ms\n")
            f.write(f"\nPer-image breakdown saved separately -> latency_per_image.csv\n")

        print(f"\nSaved -> latency_hybrid.txt (summary)")
        print(f"Saved -> latency_per_image.csv (per-image breakdown)")

    else:
        # Single-image mode (original behavior): repeat N_TIMED_RUNS times
        # on the same image to characterize steady-state per-call latency.
        if not os.path.exists(target):
            print(f"Error: file not found: {target}", file=sys.stderr)
            sys.exit(1)

        print("Warming up (1 untimed inference)...")
        single_inference(target, bundle, effnet, embed_transform)

        print(f"Timing {N_TIMED_RUNS} repeated inferences on {os.path.basename(target)}...")
        times = []
        for i in range(N_TIMED_RUNS):
            elapsed = single_inference(target, bundle, effnet, embed_transform)
            times.append(elapsed * 1000)

        times_sorted = sorted(times)
        print("\n" + "=" * 50)
        print(f"LATENCY REPORT (n={N_TIMED_RUNS} runs, hybrid pipeline: "
              f"handcrafted features + CNN embedding + PCA + classify)")
        print("=" * 50)
        print(f"Total (sum of all {N_TIMED_RUNS} runs): {sum(times):.1f} ms "
              f"({sum(times)/1000:.2f} s)")
        print(f"Mean:    {statistics.mean(times):.2f} ms")
        print(f"Median:  {statistics.median(times):.2f} ms")
        print(f"Min:     {min(times):.2f} ms")
        print(f"Max:     {max(times):.2f} ms")
        print(f"Std:     {statistics.stdev(times):.2f} ms")
        print(f"P95:     {times_sorted[int(0.95 * N_TIMED_RUNS)]:.2f} ms")

        with open("latency_hybrid.txt", "w") as f:
            f.write(f"Hybrid model inference latency (CPU)\n")
            f.write(f"Protocol: model loaded once, 1 warm-up run discarded, "
                    f"{N_TIMED_RUNS} timed runs on {os.path.basename(target)}\n\n")
            f.write(f"Total (sum):  {sum(times):.1f} ms ({sum(times)/1000:.2f} s)\n")
            f.write(f"Mean:   {statistics.mean(times):.2f} ms\n")
            f.write(f"Median: {statistics.median(times):.2f} ms\n")
            f.write(f"Min:    {min(times):.2f} ms\n")
            f.write(f"Max:    {max(times):.2f} ms\n")
            f.write(f"Std:    {statistics.stdev(times):.2f} ms\n")
            f.write(f"P95:    {times_sorted[int(0.95 * N_TIMED_RUNS)]:.2f} ms\n")

        print(f"\nSaved -> latency_hybrid.txt")


if __name__ == "__main__":
    main()
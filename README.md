# Real Photo vs Photo-of-Screen Detector

This project detects whether an input image is:

- **0 → Real Photo**
- **1 → Photo of a Screen**

The final solution uses a hybrid model combining handcrafted computer vision features with EfficientNet-B0 embeddings.

---

## Repository

```
predict.py
requirements.txt
hybrid_screen_classifier.pkl
hybrid_pca.pkl

training/
experiments/
REPORT.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

Single image:

```bash
python predict.py image.jpg
```

Folder:

```bash
python predict.py folder/
```

Output example:

```
0.8734
```

where

- **0 = Real Photo**
- **1 = Photo of Screen**

---

## Files

| File                   | Description                                                   |
|------------------------|---------------------------------------------------------------|
| predict.py             | Final inference script                                        |
| training/hybrid_svm.py | Model training notebook                                       |
| plots/                 | Classifiers comparisons,confusion matrix,misclassified images |
| REPORT.md              | Technical report                                              |
---

For methodology, experiments and evaluation, see **REPORT.md**.
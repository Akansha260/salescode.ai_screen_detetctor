# Real Photo vs Photo-of-Screen Detector

This project detects whether an input image is:

- **0 → Real Photo**
- **1 → Photo of a Screen**

The final solution uses a hybrid model combining handcrafted computer vision features with EfficientNet-B0 embeddings.

---

## Repository

```
predict.py                       Final inference script
benchmark_latency.py             Latency measurement script
requirements.txt                 Dependencies
hybrid_screen_classifier.pkl     Trained model bundle (classifier + scaler + PCA + CNN weights)
README.md                        This file
Report.md                        Technical report

results/
└── latency_hybrid.txt           Latency measurement results

training/
└── hybrid_svm.py                Final model training script

plots/
├── classifier_chart.png         Classifier comparison
├── confusion_matrix.png         Validation confusion matrix
└── misclassified_images.png     Remaining validation errors
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

## Measuring latency

```bash
python benchmark_latency.py image.jpg        
python benchmark_latency.py folder/         
```

For full methodology, experiments, latency, cost, and evaluation, see Report.md.
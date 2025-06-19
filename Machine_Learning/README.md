# Machine Learning Lab Projects

**Course:** Machine Learning (NTUA, Academic Year 2024–25)
**Student:** Michael-Athanasios Peppas (03121026)

---

## Overview

This repository contains two end-to-end lab exercises covering both supervised classification on tabular data and clustering of hyperspectral imagery:

1. **Lab 1:** “RainTomorrow” classification pipeline including data cleaning, feature encoding, scaling, model training (GaussianNB, KNN, Logistic Regression, SVM, MLP, Decision Tree, Random Forest) and evaluation.
2. **Lab 2:** Salinas hyperspectral image analysis using dimensionality reduction (PCA), clustering (KMeans, Fuzzy C-Means), and deep feature extraction via a pretrained CNN.

All notebooks run in Google Colab or a local Jupyter environment, leveraging Python 3.8+, pandas, NumPy, scikit-learn, matplotlib, scikit-fuzzy, and TensorFlow or PyTorch for CNN-based feature extraction.

---

## Repository Structure

```bash
Machine_Learning/
├── ML_Lab1/
│   ├── ML_Lab1.ipynb             # Weather forecasting classification pipeline
│   └── ML_Lab1_utils/
│       ├── train-val.csv         # Combined training and validation data (70% train / 30% validation)
│       ├── test.csv              # Hold-out test dataset for final predictions
├── ML_Lab2/
│   ├── ML_Lab2.ipynb             # Hyperspectral clustering & CNN-based feature analysis
│   └── ML_Lab2_utils/
│       ├── salinas_image.npy     # Hyperspectral image array (512×217×224)
│       ├── salinas_labels.npy    # Ground truth labels (17 classes)
└── README.md                     # Project overview and instructions
```

---

## Lab 1 – RainTomorrow Classification

**Path:** `ML_Lab1/`
**Notebook:** `ML_Lab1.ipynb`
**Utilities:** `ML_Lab1_utils/`

### Description

Predict next-day rainfall (`RainTomorrow`) using historical weather observations.

### Key Concepts & Techniques

- **Data Handling & EDA**
  - `pandas` for DataFrame operations, missing data profiling (`df.isnull()`) and descriptive statistics.
  - Visualization of feature distributions with histograms and boxplots.
- **Preprocessing Pipeline**
  - **Imputation:**
    - Numerical: `SimpleImputer(strategy='mean')` grouped by location or date.
    - Categorical: mode-based filling per feature.
  - **Encoding:**
    - `OneHotEncoder` for low-cardinality features.
    - **Target Encoding** for `Location` to capture spatial rainfall patterns.
  - **Scaling:** `StandardScaler` for numerical features.
- **Feature Selection**
  - Correlation-based filtering to remove redundant features.
  - Creation of engineered features (e.g., humidity difference, pressure gradient).
- **Modeling**
  - Algorithms:
    - **GaussianNB**, **KNeighborsClassifier** (k=5)
    - **LogisticRegression** (L2 regularization)
    - **MLPClassifier** (one hidden layer of 100 units)
    - **SVC** with RBF kernel
    - **DecisionTreeClassifier**
    - **RandomForestClassifier** (n_estimators=100)
  - Pipeline integration with `sklearn.pipeline.Pipeline`.
  - Cross-validation using `StratifiedKFold` (5 folds).
- **Evaluation**
  - Metrics: Precision, Recall, F1-score (`classification_report`), ROC-AUC curves, Confusion Matrix.
  - Train/Validation/Test split: 70% training, 30% validation; final test on `test.csv`.
  - Visualization: bar plots for F1-scores, ROC and PR curves.

### Results Highlights

- **Random Forest:** F1-score ~0.96 on validation.
- **Logistic Regression:** ROC-AUC ~0.92 consistently across folds.
- **Feature Importance:** Wind speed, humidity gradient, and pressure difference are top predictors.
- **Kaggle Submission:** `submission_opt_fin.csv` generated for leaderboard evaluation.

---

## Lab 2 – Salinas Hyperspectral Clustering & Classification

**Path:** `ML_Lab2/`
**Notebook:** `ML_Lab2.ipynb`
**Utilities:** `ML_Lab2_utils/`

### Description

Cluster and analyze Salinas hyperspectral data, comparing raw-pixel clustering with deep CNN feature clustering.

### Key Concepts & Techniques

- **Data Preparation & Visualization**
  - Load hyperspectral cube (`salinas_image.npy`) and label map (`salinas_labels.npy`) via `numpy`.
  - Plot sample spectral bands and true label distribution with `matplotlib`.
- **Spectral Signature Analysis**
  - Compute and plot mean reflectance curves for each class channel.
- **Dimensionality Reduction**
  - PCA (`sklearn.decomposition.PCA`) to reduce to top 3 components capturing >99% variance.
- **Clustering Methods**
  - **KMeans** (`n_clusters=17`, `init='k-means++'`) for hard clustering.
  - **Fuzzy C-Means** via `scikit-fuzzy` (`m=2.0`) for soft clustering.
- **Deep Feature Extraction**
  - Extract features from pretrained **MobileNetV3** (TensorFlow or PyTorch) using global average pooling.
- **Evaluation Metrics**
  - **Adjusted Rand Index (ARI)**, **Silhouette Score** for clustering performance.
  - Comparison of raw pixel vs. PCA vs. CNN features.
- **Visualization**
  - 2D scatter plots of PCA projections colored by cluster.
  - Grid display of representative pixel patches per cluster.

### Results Highlights

- **Raw Pixel KMeans:** ARI ~0.15.
- **PCA + KMeans:** ARI ~0.30, significant speed-up and variance preservation.
- **CNN Features + KMeans:** ARI ~0.55, demonstrating transfer learning benefits.
- **Fuzzy C-Means:** Provides smoother cluster boundaries, ARI ~0.48.

---

## Results & Discussion

Detailed figures, metrics tables, and analysis are provided in the result sections of each notebook, including:

- Classification performance comparisons across algorithms.
- Clustering quality for different feature representations.
- Interpretations of model strengths and limitations.

---

## Prerequisites

- Python 3.8+
- Libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scikit-fuzzy`, `tensorflow` or `torch`

Install via:

```bash
pip install numpy pandas matplotlib scikit-learn scikit-fuzzy tensorflow
```

---

## Usage

1. **Clone the repository**

   ```bash
   git clone <repo_url> Machine_Learning
   cd Machine_Learning
   ```

2. **Install dependencies** as above.
3. **Ensure data files** in `ML_Lab1_utils/` and `ML_Lab2_utils/`.
4. **Open notebooks** (`.ipynb`) in Google Colab or Jupyter and run sequentially.

---

## License

Released under the MIT License. See `LICENSE` for details.

---

### *Prepared by Michael-Athanasios Peppas (03121026)*

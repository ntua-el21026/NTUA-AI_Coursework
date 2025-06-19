# Neural Networks and Deep Learning Lab Projects

**Course:** Neural Networks and Deep Learning (NTUA, Summer 2025)
**Student:** Michael-Athanasios Peppas (03121026)

---

## Overview

This repository contains two end-to-end lab exercises covering both convolutional architectures on image data and transformer-based NLP tasks:

1. **Lab 1:** Wide Residual Networks (WRN) training on CIFAR-10 with Mixup augmentation, calibration measurement, and corruption robustness evaluation using CIFAR-10-C.
2. **Lab 2:** Transformer fine-tuning on benchmark NLP datasets (Yelp Polarity, PiQA, TruthfulQA, Winogrande), with evaluation pipelines for classification, commonsense reasoning, and semantic similarity.

All notebooks run in Google Colab or a local Jupyter environment, leveraging PyTorch, Hugging Face Transformers, and supporting libraries.

---

## Repository Structure

```bash
Neural_Networks_and_Deep_Learning/
├── DL_Lab1/
│   ├── DL_Lab1.ipynb            # Main experiment notebook: WRN on CIFAR-10 & Mixup
│   ├── DL_Lab1_utils/
│   │   ├── wideresnet.py        # Custom WideResNet implementation
│   │   ├── cifar_loader.py      # Data loading & augmentation utilities
│   │   ├── data/                # CIFAR-10 training/validation splits & metadata
│   │   ├── models/              # Checkpoints (.pt) for each run
│   │   ├── runs/                # TensorBoard logs & experiment metrics
│   │   ├── Mixup_Beyond_ERM.pdf # Key paper on Mixup methodology
│   │   └── Wide_Residual_Networks.pdf
├── DL_Lab2/
│   ├── DL_Lab2.ipynb            # Secondary notebook: Custom dataset and robustness tests
└── README.md                    # This overview file
```

---

## Lab 1 – Wide Residual Networks & Mixup

**Notebook:** `DL_Lab1/DL_Lab1.ipynb`
**Utilities:** `DL_Lab1/DL_Lab1_utils/`

### Key Concepts

- **Residual Connections:** Skip-connections that ease gradient flow, enabling the training of very deep networks.
- **Wide Residual Networks (WRN):** Scale model capacity by widening residual blocks (depth × width factor *k*) rather than simply increasing depth.
- **Batch Normalization & Dropout:**
  - **BatchNorm** to stabilize per‐batch activation distributions.
  - **Dropout** (tested at 0.0 vs. 0.3) to regularize wide layers and reduce overfitting.
- **Mixup Augmentation:**
  - Create synthetic examples:
    \[
      \tilde{x} = \lambda x_i + (1-\lambda)x_j,\quad
      \tilde{y} = \lambda y_i + (1-\lambda)y_j,\quad
      \lambda\sim\mathrm{Beta}(\alpha,\alpha).
    \]
  - Smooth decision boundaries and improve calibration.
- **Robustness Evaluation:**
  - **CIFAR-10-C** benchmark with 15 corruption types (noise, blur, weather, digital) at severity levels 1–5.
  - **Mean Corruption Error (mCE)** to quantify performance degradation under distribution shift.

### Implementation Details

- **Environment & Libraries:**
  - Python 3.8+, PyTorch & torchvision, NumPy, pandas, Matplotlib, TensorBoard, tqdm.
- **Data Pipeline:**
  1. **Train/Val Split:** 50 000 CIFAR-10 train images split into 45 000 for training and 5 000 for validation.
  2. **Transforms:**
     - `RandomCrop(32, padding=4)`
     - `RandomHorizontalFlip()`
     - `Normalize(mean=[0.491,0.482,0.447], std=[0.247,0.243,0.262])`
  3. **DataLoader:** Batch size 128, num_workers=4, shuffle=True for training, shuffle=False for val/test.
- **Model & Training:**
  - **Architectures:** WRN-16-8, WRN-28-10, WRN-40-4 (with and without Dropout).
  - **Optimizer:** SGD (lr=0.1), Nesterov momentum=0.9, weight_decay=5e-4.
  - **LR Scheduler:**
    - `CosineAnnealingLR(T_max=200)` for smooth decay, or
    - `StepLR(milestones=[60,120,160], gamma=0.2)`.
  - **Loss Function:**
    - `CrossEntropyLoss` on one-hot or Mixup soft labels.
  - **Epochs:** 200 with optional early stopping on validation loss.
- **Logging & Monitoring:**
  - TensorBoard scalars for loss, accuracy, and ECE each epoch.
  - Checkpoint saving of best–validation models.

### Metrics & Evaluation

- **Accuracy:** Top-1 on validation and CIFAR-10 test split.
- **Calibration:**
  - **Expected Calibration Error (ECE)** computed over 10 confidence bins.
  - **Reliability Diagrams** to visualize predicted vs. true accuracy.
- **Robustness:**
  - **Mean Corruption Error (mCE)** aggregated per corruption category and overall.
  - Severity-level analysis to observe performance drop-off.

### Selected Results

- **WRN-28-10 + Mixup (α=1.0):**
  - **Validation Accuracy:** 94.3%
  - **Test Accuracy:** 94.7%
  - **ECE:** 2.8% (60% lower than baseline model’s 7.0%)
  - **mCE:** 22.5% (vs. 35.2% without Mixup)
- **Depth vs. Width Trade-off:**
  - WRN-16-8 achieves 93.5% (clean) vs. WRN-40-4’s 94.9%, indicating diminishing returns beyond width factor 10.
  - Applying `Dropout(0.3)` yields up to 5% improvement in high-severity corruption accuracy.

---

## Lab 2 – Transformer Fine-Tuning & NLP Tasks

**Notebook:** `DL_Lab2/DL_Lab2.ipynb`

### Key Concepts

- **Fine-Tuning Pipelines:** Leveraging the Hugging Face `Trainer` API to adapt pretrained Transformer models (e.g. DistilBERT, BERT-base) to downstream text classification tasks.
- **Sentiment Analysis:** Framing Yelp Polarity as a binary classification problem, balancing data to mitigate class skew.
- **Commonsense Reasoning:** Evaluating on PiQA (physical reasoning) by recasting the two-choice task into NLI—scoring each choice via cross-encoder entailment models and selecting the higher “margin.”
- **Factual Question Answering:** Tackling TruthfulQA by combining zero-shot NLI entailment probabilities with sentence-transformer embeddings to rank candidate answers.
- **Winogrande Fill-In:** Applying and comparing four modeling strategies:
  1. **Mask-Fill:** Using MLM heads to predict the blank token.
  2. **Zero-Shot NLI:** Framing each choice as a hypothesis and scoring with an NLI model.
  3. **Sequence Classification:** Fine-tuning a text-classification head on Winogrande.
  4. **Text2Text Generation:** Employing encoder–decoder models (Flan-T5) to generate the missing word.

### Implementation Details

1. **Data Preparation:**
   - **Yelp Polarity:** Subsample 300 positive and 300 negative reviews for a balanced set.
   - **PiQA, TruthfulQA & Winogrande:** Use fixed 100-example subsets for rapid iteration; full-dataset runs saved separately.
   - All loaded via the Hugging Face `datasets` library with a fixed random seed.

2. **Tokenization & Collation:**
   - Instantiate `AutoTokenizer` for each model checkpoint.
   - Use `DataCollatorWithPadding` for dynamic padding to max length (e.g., 128 tokens).

3. **Trainer Configuration:**
   - Define `TrainingArguments` (LR 2e-5–5e-5, batch size 8–32, epochs 3–6, warmup, weight decay).
   - Implement `compute_metrics` using `sklearn.metrics` for accuracy, precision, recall, F1.
   - Log metrics every epoch; enable early stopping.

4. **Evaluation Pipelines:**
   - **Yelp:** Evaluate after each epoch; log to TensorBoard.
   - **PiQA:** Compute entailment score for both choices, take score difference (“margin”).
   - **TruthfulQA:** Combine NLI entailment and cosine similarity of sentence embeddings via weighted sum.
   - **Winogrande:**
     - Mask-Fill predictions via top MLM probabilities.
     - Zero-Shot NLI on choice hypotheses.
     - Sequence classification with fine-tuned head.
     - Text2Text generation via `AutoModelForSeq2SeqLM`.

5. **Visualization:**
   - **Tables:** Summarize accuracy, F1, inference latency.
   - **Line Plots:** Training vs. validation loss/accuracy.
   - **Heatmaps:** Grid search results (LR × batch size).
   - **Bar Charts:** Strategy comparisons per dataset.
   - **Confusion Matrices:** Identify error patterns.

### Selected Results

- **Yelp Polarity:** DistilBERT reaches **90.4%** accuracy after 4 epochs.
- **PiQA:** RoBERTa-large-MNLI achieves **78.9%** accuracy; margin distributions highlight confidence gaps.
- **TruthfulQA:** NLI + embedding method peaks at **38.7%** accuracy.
- **Winogrande:** Flan-T5 large attains **91.8%** accuracy; generation outperforms mask-fill (~65%), zero-shot (~70%), classification (~75%).

---

## Results & Discussion

Each notebook wraps up with:

- **Training Curves:** Loss and accuracy over epochs.
- **Calibration:** ECE and reliability diagrams for CNN models.
- **Robustness:** mCE across CIFAR-10-C corruption types.
- **NLP Metrics:** Accuracy and confidence margins for each task.
- **Comparisons:** Mixup vs. baseline and Mask-Fill vs. NLI vs. Classification vs. Generation.

---

## Usage

1. **Clone repository**

   ```bash
   git clone <repo_url>
   cd Neural_Networks_and_Deep_Learning
   ```

2. **Install dependencies**

   ```bash
   pip install torch torchvision torchaudio transformers datasets evaluate  sentence-transformers numpy pandas matplotlib scikit-learn tqdm
   ```

3. **Prepare CIFAR-10-C** (for Lab 1)
   Download from the CIFAR-C website and extract under:
   `DL_Lab1/DL_Lab1_utils/data/CIFAR-10-C/`
4. **Run notebooks**
   Open `.ipynb` files in Colab or Jupyter; ensure `*_utils/` directories exist.

---

## License

This project is licensed under the MIT License. See `LICENSE` for more information.

---

### *Prepared by Michael-Athanasios Peppas (03121026)*

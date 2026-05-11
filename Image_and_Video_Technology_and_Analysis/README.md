# Image and Video Technology and Analysis Lab Projects

**Course:** Image and Video Technology and Analysis (NTUA, Spring 2026)
**Student:** Michael-Athanasios Peppas (03121026)

---

## Overview

This repository contains two end-to-end lab exercises covering both classical image representation methods and modern CNN-based image classification:

1. **Lab 1:** Laplacian pyramid image coding, exact reconstruction, entropy analysis, and quantization experiments on RGB and grayscale images.
2. **Lab 2:** CNN image classification on a 20-class CIFAR-100 subset, comparing LeNet, AlexNet, VGG, a custom MyCNN model, regularization strategies, and transfer learning with VGG19 and EfficientNetB0.

All notebooks run in Google Colab or a local Jupyter environment, leveraging Python 3.8+, NumPy, pandas, matplotlib, Pillow, scikit-image, TensorFlow/Keras, and scikit-learn.

---

## Repository Structure

```bash
Image_and_Video_Technology_and_Analysis/
├── lab1/
│   ├── IVTA_03121026_Peppas_Michael_1.ipynb
│   │                                   # Main notebook: Laplacian pyramid theory, implementation, and experiments
│   ├── 1st_exercise.pdf               # Lab assignment statement
│   └── The Laplacian Pyramid as a Compact Image Code.pdf
│                                       # Reference paper by Burt and Adelson
├── lab2/
│   ├── IVTA_03121026_Peppas_Michael_2.ipynb
│   │                                   # Main notebook: CNN comparison, regularization, and transfer learning
│   ├── 2nd_exercise.pdf               # Lab assignment statement
│   ├── 2nd_exercise_notebook_template.ipynb
│   ├── CNN_Lab_Full_Theory.md         # Theory and oral-exam notes
│   ├── implementations/               # Reference implementation notes for LeNet, AlexNet, and VGG
│   └── papers/                        # Reference CNN papers
├── lab_material/
│   ├── lab1/                          # Instructor notebooks for fundamentals, filters, pyramids, sampling, edges
│   └── lab2/                          # Instructor notebooks for HOG, SIFT, Hough/Otsu, CNNs, ML, transfer learning
└── README.md                          # Project overview and instructions
```

---

## Lab 1 - Laplacian Pyramid Image Coding

**Path:** `lab1/`
**Notebook:** `IVTA_03121026_Peppas_Michael_1.ipynb`
**Supporting files:** `1st_exercise.pdf`, `The Laplacian Pyramid as a Compact Image Code.pdf`

### Description

This lab studies the **Laplacian Pyramid** as both a multiscale image representation and a compact image coding method. The notebook first explains the theory behind Gaussian/Laplacian pyramids and entropy, then implements the full pipeline from scratch, and finally evaluates how kernel shape, pyramid depth, and quantization strength affect reconstruction quality and coding efficiency.

### Key Concepts & Techniques

- **Gaussian and Laplacian Pyramids**
  - Recursive `REDUCE` and `EXPAND` operators.
  - Multiscale decomposition into residual bands and a final coarse Gaussian level.
- **Separable Filtering**
  - Construction of the 5-tap generating kernel parameterized by `a`.
  - Efficient 2D convolution using horizontal and vertical 1D passes.
- **Exact Reconstruction**
  - Expand-and-sum decoding from the coarsest level back to full resolution.
  - Handling of odd image sizes and boundary conditions with reflect padding.
- **Uniform Quantization**
  - Level-wise quantization of Laplacian coefficients.
  - Study of the rate-distortion tradeoff as bin size increases.
- **Entropy-Based Coding Analysis**
  - Shannon entropy estimation per pyramid level.
  - Estimated coding rate in bits per original sample for the full pyramid.
- **Experimental Evaluation**
  - Test images: **Lena (RGB)** and `skimage.data.camera()` (grayscale).
  - Parameter sweeps over:
    - `a in {0.3, 0.4, 0.5, 0.6, 0.7}`
    - `depth in {3, 4, 5, 6}`
    - `bin_size in {4, 16}`

### Implementation Details

- **Languages & Libraries:** Python 3.8+, NumPy, pandas, matplotlib, Pillow, scikit-image.
- **Core Functions:**
  - `GKernel(a)`
  - `_separable_conv2d(I, h)`
  - `GREDUCE(I, h)`
  - `GPyramid(I, a, depth)`
  - `GEXPAND(I, h, out_shape=None)`
  - `LPyramid(I, a, depth)`
  - `L_Pyramid_Decode(L, a)`
  - `L_Quantization(L, bin_size)`
  - `reconstruction_error(original, reconstructed)`
- **Metrics:**
  - Max absolute reconstruction error
  - Mean Squared Error (MSE)
  - PSNR
  - Entropy per level
  - Estimated bits per original sample

### Results Highlights

- **Lossless behavior before quantization**
  - Reconstruction is effectively exact for both images across all tested `a` and `depth` values, with MSE on the order of `10^-31` to `10^-30`.
- **Best global kernel parameter**
  - The lowest estimated coding rate is achieved near **`a = 0.6`**, in agreement with the reference paper.
  - Best combinations found in the notebook:
    - **Lena (RGB):** `a = 0.6`, `depth = 5`, `total_bpp = 5.769710`
    - **camera (grayscale):** `a = 0.6`, `depth = 6`, `total_bpp = 5.484721`
- **Per-level entropy behavior**
  - The finest residual level `L0` is optimized at **`a = 0.6`** for both images.
  - Intermediate levels are mostly optimized near **`a = 0.5`**, showing that entropy-optimal behavior depends on scale.
- **Quantization tradeoff**
  - **Lena (RGB):**
    - `bin_size = 4`: `5.770946 -> 3.178666 bpp`, `PSNR = 39.45 dB`
    - `bin_size = 16`: `5.770946 -> 0.937192 bpp`, `PSNR = 28.77 dB`
  - **camera (grayscale):**
    - `bin_size = 4`: `5.484721 -> 3.122948 bpp`, `PSNR = 40.47 dB`
    - `bin_size = 16`: `5.484721 -> 1.303582 bpp`, `PSNR = 29.27 dB`

---

## Lab 2 - CNN Image Classification and Transfer Learning

**Path:** `lab2/`
**Notebook:** `IVTA_03121026_Peppas_Michael_2.ipynb`
**Supporting files:** `2nd_exercise.pdf`, `CNN_Lab_Full_Theory.md`, `papers/`, `implementations/`

### Description

This lab studies the theory and practice of **Convolutional Neural Networks (CNNs)** for image classification. It first compares the historical design ideas behind LeNet, AlexNet, and VGG, then applies those ideas to a student-specific 20-class CIFAR-100 subset. The implementation evaluates CNNs trained from scratch, regularized versions of a custom model, and ImageNet-pretrained transfer-learning models.

### Key Concepts & Techniques

- **CNN Theory and Architecture Comparison**
  - Local receptive fields, weight sharing, feature maps, and pooling.
  - LeNet, AlexNet, and VGG design principles.
  - Comparative analysis of layers, filter sizes, activations, parameter counts, pooling, dropout, and performance behavior.
- **Dataset Preparation**
  - CIFAR-100 fine-label dataset.
  - Student-specific class subset selected with `team_seed = 26`.
  - Label remapping from original CIFAR-100 IDs to contiguous labels `0..19`.
- **Training from Scratch**
  - Adapted `LeNet`, compact `AlexNet`, compact `VGG`, and custom `MyCNN`.
  - Common hyperparameter search over optimizer, loss, and batch size.
  - Final training and evaluation with accuracy, macro-F1, and weighted-F1.
- **Overfitting Control**
  - Dropout variants for `MyCNN`.
  - Light and medium data augmentation.
  - Train-validation gap analysis.
- **Transfer Learning**
  - ImageNet-pretrained `VGG19` and `EfficientNetB0`.
  - Frozen-base training followed by careful fine-tuning.
  - Model-specific preprocessing and resizing to `128 x 128 x 3`.

### Implementation Details

- **Languages & Libraries:** Python 3.8+, TensorFlow/Keras, NumPy, pandas, matplotlib, scikit-learn.
- **Dataset Subset:** 20 CIFAR-100 classes selected by `team_seed = 26`, including bus, chimpanzee, cloud, dinosaur, forest, fox, girl, hamster, lobster, motorcycle, oak_tree, pine_tree, seal, shrew, skunk, snake, sweet_pepper, telephone, tiger, and worm.
- **Core Components:**
  - `build_lenet`
  - `build_alexnet`
  - `build_vgg`
  - `build_mycnn`
  - `MacroF1Callback`
  - `build_q2_mycnn`
  - `build_q3_transfer_model`
  - `set_q3_frozen_base`
  - `set_q3_finetuning`
- **Training Setup:**
  - Best common setup from grid search: Adam optimizer, KL divergence loss, batch size 32, final training for 50 epochs.
  - Question 2 grid over dropout and augmentation variants.
  - Question 3 frozen-base training followed by fine-tuning of top pretrained layers.

### Results Highlights

- **Question 1 - CNNs trained from scratch**
  - `MyCNN` gives the best scratch-trained result:
    - Test accuracy: `0.6205`
    - Test macro-F1: `0.6309`
  - Ranking by test macro-F1: `MyCNN > VGG > AlexNet > LeNet`.
- **Question 2 - MyCNN regularization**
  - Best regularized model: `MyCNN_Best_Dropout_Augmentation`.
    - Dropout: `0.3`
    - Augmentation: `medium`
    - Test accuracy: `0.6595`
    - Test macro-F1: `0.6607`
    - Overfitting gap: `0.1082`
  - Data augmentation is more effective than dropout alone in this experiment.
- **Question 3 - Transfer learning**
  - Best overall model: fine-tuned `EfficientNetB0`.
    - Trainable parameters: `1,816,436`
    - Test accuracy: `0.8710`
    - Test macro-F1: `0.8702`
    - Test weighted-F1: `0.8702`
  - Transfer learning substantially outperforms all models trained from scratch.

---

## Reference Material

The `lab_material/` folder contains instructor-provided notebooks used as supplementary course material. These cover:

- image fundamentals and sampling,
- spatial and frequency filtering,
- Gaussian and Laplacian pyramids,
- edge detection,
- Hough transform and Otsu thresholding,
- HOG and SIFT descriptors,
- classical machine learning for image classification,
- CNN-based image classification and transfer learning.

---

## Prerequisites

- **Python 3.8+**
- **Libraries:**

  ```bash
  pip install numpy pandas matplotlib pillow scikit-image scikit-learn tensorflow jupyter
  ```

---

## Usage

1. **Clone the repository**

   ```bash
   git clone <repo_url> Image_and_Video_Technology_and_Analysis
   cd Image_and_Video_Technology_and_Analysis
   ```

2. **Install dependencies** listed above.
3. **Open the lab notebooks** with Jupyter or Google Colab:
   - `lab1/IVTA_03121026_Peppas_Michael_1.ipynb`
   - `lab2/IVTA_03121026_Peppas_Michael_2.ipynb`
4. **Run cells sequentially** and keep each lab's supporting PDFs, theory notes, and material folders in their expected relative paths.
5. **Optional for Lab 1:** place a local `lena.png` / `lena.jpg` in the runtime if desired; otherwise the notebook attempts to download Lena automatically.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## *Prepared by Michael-Athanasios Peppas (03121026)*

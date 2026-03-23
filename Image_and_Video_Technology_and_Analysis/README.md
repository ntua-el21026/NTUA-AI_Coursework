# Image and Video Technology and Analysis Lab Projects

**Course:** Image and Video Technology and Analysis (NTUA, Spring 2026)  
**Student:** Michael-Athanasios Peppas (03121026)

---

## Overview

This course folder currently contains one completed end-to-end lab exercise focused on multiscale image representation and image coding with the **Laplacian Pyramid**, along with the supporting course material notebooks for Labs 1 and 2.

The main project in `lab1/` combines:

1. **Theory review** of Gaussian and Laplacian pyramids, entropy, and quantization based on the paper *The Laplacian Pyramid as a Compact Image Code*.
2. **Full algorithm implementation** for pyramid construction, decoding, and uniform quantization.
3. **Experimental evaluation** on both RGB and grayscale images, including entropy analysis, parameter sweeps, and rate-distortion behavior.

The notebook runs in Google Colab or a local Jupyter environment using Python 3.8+, NumPy, pandas, matplotlib, Pillow, and scikit-image.

---

## Repository Structure

```bash
Image_and_Video_Technology_and_Analysis/
├── lab1/
│   ├── IVTA_03121026_Peppas_Michael_1.ipynb        # Main notebook: Laplacian pyramid theory, implementation, and experiments
│   ├── 1st_exercise.pdf                            # Lab assignment statement
│   └── The Laplacian Pyramid as a Compact Image Code.pdf
│                                                   # Reference paper by Burt and Adelson
├── lab_material/
│   ├── lab1/                                       # Instructor notebooks for image fundamentals, filters, pyramids, sampling, edges
│   └── lab2/                                       # Instructor notebooks for HOG, SIFT, Hough, CNNs, transfer learning
└── README.md                                       # Project overview and instructions
```

---

## Project 1 - Laplacian Pyramid Image Coding

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
  - Careful handling of odd image sizes and boundary conditions with reflect padding.
- **Uniform Quantization**
  - Level-wise quantization of Laplacian coefficients.
  - Study of the rate-distortion tradeoff as bin size increases.
- **Entropy-Based Coding Analysis**
  - Shannon entropy estimation per pyramid level.
  - Estimated coding rate in bits per original sample for the full pyramid.
- **Experimental Evaluation**
  - Test images: **Lena (RGB)** and `skimage.data.camera()` (grayscale).
  - Parameter sweeps over:
    - `a ∈ {0.3, 0.4, 0.5, 0.6, 0.7}`
    - `depth ∈ {3, 4, 5, 6}`
    - `bin_size ∈ {4, 16}`

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
    - `bin_size = 4`: `5.770946 → 3.178666 bpp`, `PSNR = 39.45 dB`
    - `bin_size = 16`: `5.770946 → 0.937192 bpp`, `PSNR = 28.77 dB`
  - **camera (grayscale):**
    - `bin_size = 4`: `5.484721 → 3.122948 bpp`, `PSNR = 40.47 dB`
    - `bin_size = 16`: `5.484721 → 1.303582 bpp`, `PSNR = 29.27 dB`
  - Smaller bins preserve detail better, while larger bins yield stronger compression with visibly larger artifacts.

---

## Reference Material

The `lab_material/` folder contains the instructor-provided notebooks used as supplementary course material. These cover topics such as:

- image fundamentals and sampling,
- spatial and frequency filtering,
- Gaussian and Laplacian pyramids,
- edge detection,
- Hough transform,
- HOG and SIFT descriptors,
- classical machine learning for image classification,
- CNN-based image classification and transfer learning.

---

## Prerequisites

- **Python 3.8+**
- **Libraries:**

  ```bash
  pip install numpy pandas matplotlib pillow scikit-image jupyter
  ```

---

## Usage

1. **Clone the repository**

   ```bash
   git clone <repo_url> Image_and_Video_Technology_and_Analysis
   cd Image_and_Video_Technology_and_Analysis
   ```

2. **Install dependencies** listed above.
3. **Open the main notebook** in `lab1/` with Jupyter or Google Colab.
4. **Run cells sequentially** to:
   - build the Gaussian and Laplacian pyramids,
   - verify exact reconstruction,
   - reproduce entropy plots,
   - and reproduce the quantization experiments.
5. **Optional:** place a local `lena.png` / `lena.jpg` in the runtime if desired; otherwise the notebook attempts to download Lena automatically.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## *Prepared by Michael-Athanasios Peppas (03121026)*

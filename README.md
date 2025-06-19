# NTUA AI Coursework

**Student:** Michael-Athanasios Peppas (03121026)
**Institution:** National Technical University of Athens, ECE

This repository includes Google Colab notebooks from three core courses: Artificial Intelligence, Machine Learning, and Neural Networks & Deep Learning. Each course contains two comprehensive lab projects with supporting code and data.

---

## Courses

### Artificial Intelligence

- **AI_Lab1:** Maze generation, search algorithms (BFS, Dijkstra, A*, Greedy), and adversarial planning with Alpha-Beta agents.
- **AI_Lab2:** Hybrid movie recommender combining SWI-Prolog symbolic reasoning and Python for data handling and evaluation.

See `Artificial_Intelligence/README.md` for detailed instructions, key concepts, and results.

### Machine Learning

- **ML_Lab1:** RainTomorrow classification: data preprocessing, feature engineering, model training with scikit-learn, and evaluation.
- **ML_Lab2:** Salinas hyperspectral clustering and classification using KMeans, Fuzzy C-Means, PCA, and CNN feature extraction.

See `Machine_Learning/README.md` for detailed instructions, key concepts, and results.

### Neural Networks & Deep Learning

- **DL_Lab1:** WideResNet architectures on CIFAR-10, Mixup augmentation, calibration measurement, and robustness on CIFAR-10-C.
- **DL_Lab2:** Transformer fine-tuning for sentiment, PiQA, TruthfulQA, and Winogrande tasks using Hugging Face.

See `Neural_Networks_and_Deep_Learning/README.md` for detailed instructions, key concepts, and results.

---

## Repository Structure

```bash
ntua-ai-coursework/
├── Artificial_Intelligence/
│   ├── AI_Lab1/
│   ├── AI_Lab2/
│   └── README.md
├── Machine_Learning/
│   ├── ML_Lab1/
│   ├── ML_Lab2/
│   └── README.md
├── Neural_Networks_and_Deep_Learning/
│   ├── DL_Lab1/
│   ├── DL_Lab2/
│   └── README.md
├── LICENSE
└── README.md
```

---

## Prerequisites

- **Python 3.8+**
- **Libraries:**

  ```python
  pip install numpy pandas matplotlib scikit-learn torch torchvision torchaudio transformers datasets evaluate sentence-transformers tqdm
  ```

- **SWI-Prolog (v8.x):** required for AI_Lab2

  ```python
  sudo apt-get install swi-prolog
  pip install pyswip
  ```

---

## Getting Started

1. **Clone the repository**

   ```bash
   git clone <repo_url> ntua-ai-coursework
   cd ntua-ai-coursework
   ```

2. **Install dependencies** as listed above.
3. **Explore each course** by opening its `README.md` and running the lab notebooks in Google Colab or Jupyter.
4. **For DL labs**, download and extract the CIFAR-10-C dataset into the specified utils/data folders.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## *Prepared by Michael-Athanasios Peppas (03121026)*

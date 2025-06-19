# Artificial Intelligence Lab Projects

**Course:** Artificial Intelligence (NTUA, Summer 2025)
**Student:** Michael-Athanasios Peppas (03121026)

---

## Overview

This repository contains two end-to-end lab exercises covering both classical search techniques and hybrid recommendation systems:

1. **Lab 1:** Maze generation and pathfinding using BFS, Dijkstra, A*, Greedy Best-First, and adversarial planning with Alpha-Beta pruning.
2. **Lab 2:** Prolog-backed movie recommender that combines symbolic similarity reasoning (SWI-Prolog) with Python for data handling, personalization, and evaluation.

All notebooks run in Google Colab or a local Jupyter environment, leveraging Python 3.8+, NumPy, pandas, matplotlib, scikit-learn, and pyswip for Prolog integration.

---

## Repository Structure

```bash
Artificial_Intelligence/
├── AI_Lab1/
│   ├── AI_Lab1.ipynb         # Main notebook: maze generation, search algorithms, adversarial planning
│   └── AI_Lab1_utils/        # Helpers: maze generator, visualization, agent classes, experiment scripts

├── AI_Lab2/
│   ├── AI_Lab2.ipynb         # Main notebook: Prolog-backed movie recommender
│   └── AI_Lab2_utils/        # Helpers: db.pl (rules), CSV metadata and ratings

└── README.md                 # This file
```

---

## Project 1 – Maze Pathfinding & Adversarial Agents

**Notebook:** `AI_Lab1/AI_Lab1.ipynb`
**Utilities:** `AI_Lab1_utils/`

### Key Concepts & Techniques

- **Maze Generation:** Randomized DFS with tunable wall density (threshold parameter).
- **Search Algorithms:**
  - **Breadth‑First Search (BFS)** for unweighted shortest paths.
  - **Dijkstra’s Algorithm** (uniform-cost search).
  - **A\*** with admissible heuristics (Manhattan, Euclidean, Chebyshev, Octile).
  - **Greedy Best‑First Search** (heuristic-only guidance).
- **Heuristic Design:** Impact of g(n) and h(n) choices on optimality and performance.
- **Adversarial Planning:** Minimax with Alpha‑Beta pruning—agent vs. ghost.
- **Visualization:** Step‑by‑step frontier expansion, animated GIFs, reliability of visual insights vs. quantitative metrics.

### Implementation Details

- **Languages & Libraries:** Python 3.8+, NumPy, matplotlib, queue, random.
- **Classes:**
  - `Maze`: grid representation, neighbor queries, rendering.
  - `Pathfinder`: unified interface for g/h combinations, path reconstruction, complexity tracking.
  - `Agent` & `ABagent`: adversarial chaser (A*) and escapee (Alpha‑Beta) implementations.
- **Experiments:**
  - Complexity vs. maze size (N ∈ {11, 25, 41, 51, 61}).
  - Node expansions, path length, execution time logged and plotted.
  - Bonus: interactive maze drawing and custom scenario design.

### Results Highlights

- **Search Scaling:**
  - BFS expansions grow ~O(N²); heuristics reduce expansions by up to 90%.
  - A* with Octile heuristic often yields the best expansion–optimality tradeoff.
- **Adversarial Behavior:**
  - Alpha‑Beta agent with combined goal/ghost heuristic escapes in >80% of sparse mazes (depth=5).
  - Visualization aids intuition but quantitative metrics (win rate, time) confirm robustness.

---

## Project 2 – Hybrid Movie Recommendation System

**Notebook:** `AI_Lab2/AI_Lab2.ipynb`
**Utilities:** `AI_Lab2_utils/`

### Key Concepts & Techniques

- **Symbolic Reasoning:** SWI‑Prolog knowledge base (`db.pl`) encoding movie facts and similarity rules.
- **Data Handling:** pandas for CSV I/O, DataFrame manipulation.
- **Similarity Models:** Six composite schemes (basic_sim_1…adv_sim_3), normalized to 0–100, tiered recommendations (5 levels).
- **Hybrid Pipeline:**
  1. **Prolog Queries:** `find_sim_tier/3` and `similar/3` for content-based recommendations.
  2. **Python Interface:** `recommender.py` to cache similarities and normalize scores.
  3. **User Personalization:** weighted aggregation of neighbor scores based on user ratings.
- **Evaluation Metrics:** precision, recall, F1-score via scikit-learn.
- **Experimentation:** grid search over models, tier ranges, training size, thresholds; diagnostic plots and heatmaps.

### Implementation Details

- **Languages & Libraries:** Python 3.8+, pandas, numpy, pyswip, scikit-learn, matplotlib.
- **Prolog Setup:** `pyswip` embedding, dynamic fact assertion, consult `db.pl`.
- **Similarity Caching:** Preload tiered neighbor sets to avoid repeated Prolog calls.
- **Training Loop:**
  - Sample user ratings (N ∈ {3,5,10,25,50,75,all}).
  - Aggregate weighted votes by tier (min_votes filter).
  - Generate recommendation scores, filter seen movies.
- **Evaluation Loop:**
  - Predict binary likes (threshold auto or fixed).
  - Compute metrics over test set, average across repetitions.
  - Visualize F1 heatmaps, precision–recall scatter, line plots, boxplots.

### Results Highlights

- **Content-Based Only:** `adv_sim_1` model yields the most balanced precision (∼0.60) and recall (∼0.95).
- **Personalization Impact:** using ≥10 ratings achieves >0.65 F1; diminishing returns beyond 25.
- **Model Sensitivity:** `basic_sim_3` excels at production-scale similarity (high precision), `adv_sim_2` at global-popularity filtering.

---

## Prerequisites

- **Python 3.8+**
- **Libraries:**

  ```bash
  pip install numpy pandas matplotlib pyswip scikit-learn
  ```

- **SWI‑Prolog (v8.x):** required for Project 2 Prolog integration.
  - On Colab:

    ```bash
    sudo apt-get install swi-prolog
    pip install pyswip
    ```

---

## Usage

1. **Clone the repository**

   ```bash
   git clone <repo_url> Artificial_Intelligence
   cd Artificial_Intelligence
   ```

2. **Install dependencies** (see Prerequisites).
3. **Run notebooks** in Colab or Jupyter:
   - Ensure `AI_Lab1_utils/` and `AI_Lab2_utils/` folders are alongside their notebooks.
   - For Colab, mount Google Drive if needed and adjust paths accordingly.
4. **Visualize results** and inspect final sections of each notebook for plots, GIFs, and analysis.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## *Prepared by Michael-Athanasios Peppas (03121026)*

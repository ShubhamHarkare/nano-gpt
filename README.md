# nano‑gpt (Shubham Harkare)

**A compact, from‑scratch re‑creation of GPT‑2 in Python.** This project—implemented as both a Jupyter Notebook and modular scripts—focuses on understanding core transformer principles and building a miniature GPT‑2 model.

---

###  Project Overview

- **Objective**: Recreate a smaller-scale GPT‑2 model from scratch to deepen understanding of transformer architectures.
- **Approach**: Implemented model, training pipeline, and dataset handling in Python, using Tiny Shakespeare data for experimentation.

---

### Repository Contents

- `nano-gpt.ipynb`: Fully documented notebook walking through model building and training steps.
- `train.py`: Key training script for model optimization.
- `split_data.py`: Preprocessing utility for slicing and preparing dataset.
- `utils.py`: Helper functions to support data preparation, training, and sampling.
- `tiny-shakeshpere.txt`: Sample dataset based on Shakespeare's works for training experiments.
- `requirements.txt`: Python dependencies necessary to run and experiment with the project.

---

###  Getting Started

1. Clone the repository:  
   ```bash
   git clone https://github.com/ShubhamHarkare/nano-gpt.git
   cd nano-gpt
   pip install -r requirements.txt

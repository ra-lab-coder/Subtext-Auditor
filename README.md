# 🌊 Deepsea-Communication-Orientation-Auditor

> A lightweight NLP system that classifies **interpersonal communication style** as  
**Task-Oriented / Cold** vs **Emotionally Dependent / Hot**, based on a sociological theory of  
**“side-by-side vs face-to-face interaction.”**”

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://subtext-auditor-hfpspk6d9snsnernfcgnws.streamlit.app/)

## 🧠 Project Overview

**DeepSea Communication Orientation Auditor** is an end-to-end NLP project that analyzes short chat conversations and estimates **how the interaction is oriented**, rather than what the relationship *is*.

This project combines:
* 🗂️ Synthetic data generation using structured templates
* 🧩 A custom sociological labeling framework 
* 🔤 TF–IDF feature engineering
* 🤖 Logistic Regression classifier
* 🖥️ Interactive Streamlit web interface 
* 📦 **CI/CD Deployment** on Streamlit Cloud

It demonstrates a full ML workflow from **dataset creation → modeling → evaluation → UI deployment**.

---

## 🔍 Problem Motivation

Most NLP classifiers focus on:
* sentiment
* toxicity
* topic detection

However, **human communication has another important dimension**:

* Are people interacting **side-by-side** (focused on tasks, ideas, or external goals)?  
* Or **face-to-face** (focused on emotional validation, dependency, and relational closeness)?

This project explores that dimension.

---

## 🧠 Communication Orientation Framework

The classification is based on a **communication-style theory**, not relationship labels.

| Class | Orientation | Description |
|-----|------------|-------------|
| 0 | Task-Oriented / Cold | Side-by-side interaction. Focus on ideas, tasks, problem-solving, boundaries, low emotional dependency. |
| 1 | Emotionally Dependent / Hot | Face-to-face interaction. Emotional validation, prioritization, dependency, privacy-blurring, relational focus. |

⚠️ **Important:**  
A conversation labeled “Emotionally Dependent” does **not** imply romance or infidelity.  
It only reflects the *orientation* of communication in that exchange.

---

## 🗂️ Dataset Design

There is no public dataset for communication-orientation classification.  
This project therefore builds a **synthetic dataset** using carefully designed templates.

The generator includes:
- task-focused conversations with explicit boundaries
- emotionally dependent conversations with prioritization and validation
- **hard negatives** that intentionally blur surface cues
- shared topics across classes to prevent topic leakage

Each sample includes:
- `text`
- `label`
- `difficulty` (easy / hard)
- `template_id` (for grouped evaluation)

---

## 🏗️ Project Architecture

```text
deepsea-auditor/
│
├── data/
│   ├── deepsea_conversations.csv   # Raw synthetic dataset
│   ├── train.csv                   # Training split
│   ├── val.csv                     # Validation split
│   └── test.csv                    # Held-out test split
│
├── models/
│   └── deepsea_model.pkl           # Serialized model artifact
│
├── src/
    ├── app.py                          # Streamlit UI entry point
    ├── generate_data.py                # Script: synthetic data creator
    ├── split_data.py                   # Script: train/val/test splitter
    ├── train_model.py                  # Script: Training pipeline (TF-IDF + LogReg)
    ├── test.py               # Script: Performance metrics evaluation
└── requirements.txt                # Python dependencies
```

## 🧪 Evaluation Methodology

Because the data is template-generated, **random train/test splits would cause template leakage.**

To avoid this, the project uses:
* **GroupShuffleSplit**
* Grouped by `template_id`
* Ensuring validation and test sets contain **conversation styles never seen during training**
This produces **honest generalization estimates.**

**Example Validation Performance (group-split)**
* Accuracy: ~0.80
* F1-score: ~0.79
* Errors concentrated in intentionally ambiguous cases
This reflects the **inherent ambiguity of real interpersonal communication**, not model failure.

---

## 🖥️ Streamlit Application
Run locally:
```bash
streamlit run src/app.py
```
The UI allows users to paste a short chat snippet and returns:
* **Communication Orientation Score** (probability of emotionally dependent style)
* Verdict: Task-Oriented / Ambiguous / Emotionally Dependent
* Confidence visualization
* Natural-language explanation

---

## 🔮 Future Work
* Increase linguistic diversity and paraphrasing
* Add contextual metadata (e.g. professional vs personal context)
* Replace TF-IDF with sentence embeddings (SBERT)
* Phrase-level explainability (highlight contributing cues)
* Multi-turn conversation timeline analysis
* Backend deployment with FastAPI + Docker

---

## ⚠️ Disclaimer
This project is a research and educational prototype trained entirely on synthetic data.

It is not:
* a psychological assessment tool
* relationship advice
* a detector of romantic or sexual intent
Its purpose is to explore communication patterns, not personal relationships.

---

## 🙋‍♀️ Author
Designed and built by Ruoxue Wang,
MSc Data Science & Machine Learning (UCL),
exploring human communication through machine learning.

If you found this interesting, feel free to ⭐ the repo!

# 🌊 Pure-Friendship-Auditor

> A lightweight NLP system that classifies interpersonal chat messages as **Platonic/Cold** or **Emotional-Affair/Hot** communication based on a sociological theory of “side-by-side vs. face-to-face interactions.”

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_LINK_HERE)

## 🧠 Project Overview

**Pure-Friendship-Auditor** is an end-to-end NLP application that analyzes chat conversations and estimates the likelihood that the interaction resembles emotional-affair style communication rather than platonic friendship.

This project combines:
* 🗂️ **Synthetic data generation** via LLM prompting
* ✨ **Custom sociological labeling theory** (Platonic vs. Emotional-Affair)
* 🔤 **TF–IDF feature engineering**
* 🤖 **Logistic Regression classifier**
* 📊 **Interactive Streamlit web app**
* 📦 **CI/CD Deployment** on Streamlit Cloud

It demonstrates a full ML workflow from **dataset creation → modeling → evaluation → UI deployment**.

---

## 🔍 Problem Motivation

Most NLP classification tasks focus on sentiment, toxicity, or topic detection—but human relationships contain another dimension:

* 👉 Are two people relating in a platonic, boundary-respecting way?
* 👉 Or is the communication emotionally intimate and boundary-blurring?

This project explores that space using the creator’s sociological theory:

| Class | Meaning | Description |
| :--- | :--- | :--- |
| **0** | **Platonic / Cold** | **Side-by-side communication.** Focus on ideas, tasks, problem-solving, low emotional dependence. |
| **1** | **Emotional-Affair / Hot** | **Face-to-face communication.** Emotional validation, dependency, privacy violations, romantic tone. |

**Gap:** No public dataset exists for this nuance → this project builds one synthetically using high-quality templates.

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
    ├── evaluate_model.py               # Script: Performance metrics evaluation
    └── requirements.txt                # Python dependencies

---

## 🧪 1. Synthetic Data Generation

The dataset is created using structured templates that encode:
* **Platonic communication patterns** (objective tone, boundaries)
* **Emotional-affair patterns** (validation, privacy, dependency, emojis)

**Run:**
```bash
python generate_data.py
python split_data.py

## 📚 2. Model Training

The ML pipeline uses:
* **TF-IDF vectorizer (1–2 n-grams)
* **Logistic Regression with class balancing

**Train:**
```bash
python src/train.py

**Evaluate:**
```bash
python src/test.py

On synthetic data, the model achieves near-perfect separation (expected due to controlled templates).
Realistic performance would decrease once more ambiguous samples are added.

## 🖥️ 3. Streamlit Web Application
Launch locally:
```bash
streamlit run app.py

The UI allows users to paste any chat snippet.
The model outputs:
* **Emotional-Affair Risk Score (0–1)
* **Verdict (Platonic / Ambiguous / High Risk)
* **Explanation text
* **Progress bar visualization


## 🔮 Future Improvements
* **Create more ambiguous and human-like conversation samples
* **Add metadata: gender, sexual orientation, relational context
* **Incorporate Sentence-BERT embeddings
* **Add explainability (highlight which phrases contributed to classification)
* **Build a multi-turn conversational timeline visualizer
* **Deploy a full backend with FastAPI + Docker

## 📢 Disclaimer
This tool is a prototype (v1) trained on synthetic data only.
It is intended for educational, research, and reflective purposes,
not for psychological evaluation or relationship advice.

## 🙋‍♀️ Author
Designed and built by Ruoxue Wang,
MSc Data Science & Machine Learning (UCL),
exploring human communication through machine learning.

If you found this interesting, feel free to ⭐ the repo!

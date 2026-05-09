# Simple Deployment Demo Website

This folder adds a lightweight local website for the optional **simple deployment interface** part of Phase 2.

## What it does
- Lets you choose **Phase 1** or **Phase 2**
- Lets you choose a feature extraction method: **Flatten / PCA / HOG**
- Lets you choose a model:
  - Phase 1: Logistic Regression, KNN, Gaussian Naive Bayes
  - Phase 2: Regularized Logistic Regression, Tuned KNN, Random Forest
- Trains the selected manual model on a small MNIST demo subset
- Lets you test on:
  - a held-out MNIST sample
  - your own uploaded image

## How to run
```bash
pip install -r requirements_demo.txt
streamlit run app.py
```

## Notes
- This is a **local demo website**, not a cloud deployment.
- It reuses the same manual project code in `src/`.
- For speed, the app uses a smaller subset of MNIST suitable for demonstration.
- If your uploaded digit is black on white, enable **Invert colors**.

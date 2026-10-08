# Deep Learning Classifier (PyTorch)

A feed-forward neural network (MLP) that classifies tumors as malignant or
benign using the scikit-learn breast cancer dataset.

## Features
- PyTorch model with dropout and BCE loss
- Train / validation / test split, scaler fit on train only
- Early stopping with best-weights restore
- Training curves and confusion matrix plots
- `predict.py` for loading the saved model

## Run
```bash
pip install -r requirements.txt
python train.py     # trains, evaluates, saves model.pt + scaler.joblib
python predict.py   # runs inference with the saved model
```

## Output
- `model.pt`, `scaler.joblib`
- `training_curves.png`, `confusion_matrix.png`

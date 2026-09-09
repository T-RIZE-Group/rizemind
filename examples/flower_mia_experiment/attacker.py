import torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def extract_attack_features(model, probe_loader):
    model.eval()
    features = []
    with torch.no_grad():
        for x, _ in probe_loader:
            outputs = model(x)
            features.extend(outputs.view(-1).numpy())
    return features

def compute_attack_auc(X_attack, y_attack):
    # Train binary classifier to predict whether A was in the coalition
    # Split into train/test for the attacker, or just use cross val / CV AUC
    # To keep it simple, we use 5-fold CV AUC
    from sklearn.model_selection import cross_val_score
    import numpy as np
    
    if len(np.unique(y_attack)) < 2:
        return 0.5 # Cannot compute AUC if only one class
        
    clf = LogisticRegression()
    # ROC AUC from cross validation
    scores = cross_val_score(clf, X_attack, y_attack, cv=5, scoring='roc_auc')
    return np.mean(scores)

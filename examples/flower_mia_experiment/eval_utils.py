import torch
from sklearn.metrics import roc_auc_score

def evaluate_task_metrics(model, dataloader):
    model.eval()
    y_true = []
    y_scores = []
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            outputs = model(x).view(-1)
            labels = y.view(-1)

            y_scores.extend(outputs.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()

    return {
        "auc": roc_auc_score(y_true, y_scores),
        "accuracy": correct / total,
    }

def evaluate_task_auc(model, dataloader):
    return evaluate_task_metrics(model, dataloader)["auc"]

def evaluate_task_accuracy(model, dataloader):
    return evaluate_task_metrics(model, dataloader)["accuracy"]

import torch

def calculate_accuracy(predictions, targets):
    predicted_labels = (torch.sigmoid(predictions) >= 0.5).float()
    correct = (predicted_labels == targets).sum().item()
    return correct / len(targets)
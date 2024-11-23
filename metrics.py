from typing import List
import torch
from torch.functional import F
from sklearn.metrics import recall_score


def multi_hot_encode(y_pred: List[List[int]], num_classes: int) -> torch.Tensor:
    one_hot = torch.zeros(len(y_pred), num_classes)
    
    for i, pred in enumerate(y_pred):
        one_hot[i, pred] = 1
    
    return one_hot


def calc_recall(y_true: List[int], y_pred: List[List[int]], num_classes: int) -> float:
    y_true = F.one_hot(torch.tensor(y_true, dtype=torch.long), num_classes=num_classes).tolist()

    y_pred = multi_hot_encode(y_pred, num_classes).tolist()

    return recall_score(y_true, y_pred, average="macro").item()


def calc_mrr(y_true: List[int], y_pred: List[List[int]]) -> float:
    mrr = 0.0

    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true in pred:
            mrr += 1.0 / (pred.index(true) + 1)

    mrr /= len(y_true)

    return mrr

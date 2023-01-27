import torch


def pf1beta(predictions, labels, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if labels[idx]:
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta ** 2
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0


class ProbF1Score:
    def __init__(self, predictions: torch.Tensor, labels: torch.Tensor, beta: int = 1):
        self.predictions = predictions
        self.labels = labels
        self.beta = beta

        self.score = pf1beta(self.predictions, self.labels, self.beta)

    def f1_score(self):
        return self.score

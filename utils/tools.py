
import torch
from torch.nn.functional import one_hot as hot


class Tools:
    @staticmethod
    def compute_preds(cls_pred, labels):
        _, max_index = torch.max(cls_pred, 2)
        lb = labels[1]
        one_hot_preds = hot(max_index, num_classes=cls_pred.shape[2]).float()
        lb1 = one_hot_preds
        matches = torch.eq(one_hot_preds, labels[1]).int()
        f = lb - lb1
        final = torch.sum(f == 1).item() / 400
        correct_predictions = matches.prod(dim=2)
        losses = 1 - correct_predictions
        total_loss = losses.sum().item()
        return final
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    x
    Combined loss, choose lambda parameter to shift learning
    on between classification and regression.
    """
    def __init__(self, lam):
        super(CombinedLoss, self).__init__()
        self.CEL = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.lam = lam

    def forward(self, reg_preds, reg_labs, clas_preds, clas_labs):
        reg_loss = self.MSE(reg_preds, reg_labs)
        clas_loss = self.CEL(clas_preds, clas_labs)
        combined_loss = reg_loss * self.lam + (1 - self.lam) * clas_loss

        return combined_loss

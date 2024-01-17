
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
        final = torch.sum(f == 1).item() / (32 * 16)# CALIBRATE THERE
        correct_predictions = matches.prod(dim=2)
        losses = 1 - correct_predictions
        total_loss = losses.sum().item()
        return final

    @staticmethod
    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    @staticmethod
    def concat_two_tensors(tz1: torch.Tensor, tz2: torch.Tensor) -> torch.Tensor:
        """
        Just casually concatenate two tensors with
        respect of saving ids.
        """
        tensor1_without_last = tz1[:-1, :,:]
        tensor2_without_last = tz2[:-1, :,:]
        last_row1 = tz1[-1,:,:]
        last_row2 = tz2[-1,:,:]
        combined_last_row = torch.cat((tz1[-1,:,:], tz2[-1,:,:]), dim=0)
        combined_last_row = torch.unsqueeze(combined_last_row, 0)
        combined_no_last_row = torch.cat((tensor1_without_last, tensor2_without_last), dim=1)
        result = torch.concat((combined_no_last_row, combined_last_row), dim=0)
        x = 2
        return result
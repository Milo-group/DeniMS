import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import Metric, MeanSquaredError, MetricCollection
import time
import wandb
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchMSE, SumExceptBatchKL, CrossEntropyMetric, \
    ProbabilityMetric, NLL



class TrainLossDiscreteEdges(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train):
        super().__init__()
        self.edge_loss = CrossEntropyMetric()

    def forward(self, masked_pred_E, true_E, log: bool):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """

        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0

        if log:
            to_log = {"train_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1}

            if wandb.run:
                wandb.log(to_log, commit=True)

        return loss_E

    def reset(self):
        for metric in [self.edge_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1

        to_log = {"train_epoch/E_CE": epoch_edge_loss}

        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log




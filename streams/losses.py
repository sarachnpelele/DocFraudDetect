"""
This is the Lovász-Softmax loss implementation, used alongside regular binary cross-entropy. It helps with 
the fact that tampered pixels are rare, usually only 1-4% of an image, so a plain loss 
function alone tends to under-value getting those small regions right.

This is a real, published loss function (Berman et al., 2018), not something I built 
myself, the code below is adapted from the standard version used in segmentation 
research, the same one used in Qu et al. (2023) and Wong et al. (2025), the papers 
that led me to use it here.

Actually used in Stage 2's(aka full 4 stream model) real training, combined with BCE (see CombinedLoss below).
"""

import torch
import torch.nn as nn


def lovasz_grad(gt_sorted):
    #Computes gradient of the Lovász extension w.r.t sorted errors.
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    #Binary Lovász hinge loss for flattened predictions/labels.
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(torch.relu(errors_sorted), grad)
    return loss


class LovaszLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):

        batch_size = logits.shape[0]
        losses = []
        for i in range(batch_size):
            losses.append(lovasz_hinge_flat(logits[i].reshape(-1), labels[i].reshape(-1)))
        return torch.stack(losses).mean()


class CombinedLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszLoss()

    def forward(self, logits, labels):
        return self.bce(logits, labels) + self.lovasz(logits, labels)
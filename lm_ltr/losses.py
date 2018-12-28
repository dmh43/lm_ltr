import torch
import torch.nn as nn

def hinge_loss(score_difference, target, margin=1.0):
  maxes = torch.max(torch.tensor(0.0, device=score_difference.device),
                    torch.tensor(margin, device=score_difference.device) - target * score_difference)
  return 1.0 / len(maxes) * maxes.sum()

def bce_loss(score_difference, target):
  y = (target > 0).float()
  return nn.functional.binary_cross_entropy_with_logits(score_difference, y)

def truncated_hinge_loss(score_difference, target, margin=1.0, truncation=-1.0):
  maxes = torch.max(torch.tensor(0.0, device=score_difference.device),
                    torch.tensor(margin, device=score_difference.device) - target * score_difference)
  truncated = torch.min(torch.tensor(margin - truncation, device=score_difference.device),
                        maxes)
  return 1.0 / len(truncated) * truncated.sum()

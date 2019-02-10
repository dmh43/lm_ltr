import torch
import torch.nn as nn
import torch.nn.functional as F

def hinge_loss(score_difference, target, margin=1.0, reduction='elementwise_mean'):
  maxes = torch.max(torch.tensor(0.0, device=score_difference.device),
                    torch.tensor(margin, device=score_difference.device) - target * score_difference)
  if reduction is 'none':
    return maxes
  elif reduction is 'elementwise_mean':
    return 1.0 / len(maxes) * maxes.sum()
  else:
    raise NotImplementedError()

def bce_loss(score_difference, target, weight=None, reduction='elementwise_mean'):
  y = (target > 0).float()
  return nn.functional.binary_cross_entropy_with_logits(score_difference, y, weight=weight, reduction=reduction)

def l1_loss(score_difference, target, reduction='elementwise_mean'):
  y = (target > 0).float()
  return F.l1_loss(score_difference, y, reduction=reduction)

def smoothed_bce_loss(score_difference, target, reduction='elementwise_mean'):
  if reduction is not 'elementwise_mean': raise NotImplementedError()
  device = score_difference.device
  total_loss = torch.tensor(0.0, device=device)
  scale = 0.9
  y_plus = (target > 0).float() * scale
  y_minus = (target > 0).float() * (1 - scale)
  for i, y in enumerate([y_plus, y_minus]):
    cls_idx = torch.full((score_difference.size(0),), i, dtype=torch.long, device=device)
    loss = F.cross_entropy(score_difference, cls_idx, reduction='none')
    total_loss += y.dot(loss)
  return total_loss / score_difference.shape[0]

def truncated_hinge_loss(score_difference, target, margin=1.0, truncation=-1.0, reduction='elementwise_mean'):
  maxes = torch.max(torch.tensor(0.0, device=score_difference.device),
                    torch.tensor(margin, device=score_difference.device) - target * score_difference)
  truncated = torch.min(torch.tensor(margin - truncation, device=score_difference.device),
                        maxes)
  if reduction is 'none':
    return truncated
  elif reduction is 'elementwise_mean':
    return 1.0 / len(truncated) * truncated.sum()
  else:
    raise NotImplementedError()

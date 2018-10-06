import torch
import torch.nn as nn

def hinge_loss(score_difference, target, margin=1.0):
  maxes = torch.max(torch.zeros_like(score_difference), margin - target.float() * score_difference)
  return 1.0 / len(maxes) * maxes.sum()

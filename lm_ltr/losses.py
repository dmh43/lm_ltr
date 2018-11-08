import torch
import torch.nn as nn

def hinge_loss(score_difference, target, margin=1.0):
  maxes = torch.max(torch.zeros_like(score_difference), margin - target.float() * score_difference)
  return 1.0 / len(maxes) * maxes.sum()

def truncated_hinge_loss(score_difference, target, margin=1.0, truncation=-1.0):
  maxes = torch.max(torch.zeros_like(score_difference),
                    margin - target.float() * score_difference)
  truncated = torch.min(torch.ones_like(score_difference) * (margin - truncation),
                        maxes)
  return 1.0 / len(truncated) * truncated.sum()

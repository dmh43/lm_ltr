import lm_ltr.utils as utils

def test_with_negative_samples():
  with_neg = utils.with_negative_samples([{'val':i} for i in range(4)], 2, 100)
  assert len(with_neg) == 4 * 2 + 4

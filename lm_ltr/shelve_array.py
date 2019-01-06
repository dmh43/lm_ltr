class ShelveArray:
  def __init__(self, shelf):
    self.shelf = shelf

  def __iter__(self):
    for key in self.shelf:
      yield self.shelf[key]

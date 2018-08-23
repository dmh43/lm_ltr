from fastai.sgdr import Callback

class EarlyStopping(Callback):
  def __init__(self, learner, save_path, enc_path=None, patience=5):
    super().__init__()
    self.learner=learner
    self.save_path=save_path
    self.enc_path=enc_path
    self.patience=patience
  def on_train_begin(self):
    self.best_val_loss=100
    self.num_epochs_no_improvement=0
  def on_epoch_end(self, metrics):
    val_loss = metrics[0]
    if val_loss < self.best_val_loss:
      self.best_val_loss = val_loss
      self.num_epochs_no_improvement = 0
      self.learner.save(self.save_path)
      if self.enc_path is not None:
        self.learner.save_encoder(self.enc_path)
    else:
      self.num_epochs_no_improvement += 1
    if self.num_epochs_no_improvement > self.patience:
      print(f'Stopping - no improvement after {self.patience+1} epochs')
      return True
  def on_train_end(self):
    print(f'Loading best model from {self.save_path}')
    self.learner.load(self.save_path)

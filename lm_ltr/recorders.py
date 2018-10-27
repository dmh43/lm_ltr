from fastai import Recorder
import matplotlib.pyplot as plt
from time import time
import pickle

class PlottingRecorder(Recorder):
  def __init__(self, experiment, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.experiment = experiment

  def on_train_begin(self, pbar, metrics, **kwargs)->None:
    pass

  def on_batch_begin(self, **kwargs)->None:
    pass

  def on_backward_begin(self, smooth_loss, **kwargs)->None:
    pass

  def on_epoch_end(self, epoch, num_batch, smooth_loss, last_metrics, **kwargs)->bool:
    pass

  def on_train_end(self, **kwargs):
    self.plot_losses()
    model_name = self.experiment.model_name
    plt.savefig(f'./loss_plots_{model_name}.png')
    plt.close()

class LossesRecorder(Recorder):
  def __init__(self, experiment, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.experiment = experiment

  def on_train_begin(self, pbar, metrics, **kwargs)->None:
    pass

  def on_batch_begin(self, **kwargs)->None:
    pass

  def on_backward_begin(self, smooth_loss, **kwargs)->None:
    pass

  def on_epoch_end(self, epoch, num_batch, smooth_loss, last_metrics, **kwargs)->bool:
    pass

  def on_train_end(self, **kwargs):
    model_name = self.experiment.model_name
    with open(f'./losses_{model_name}.pkl', 'wb+') as fh:
      pickle.dump(fh, self.losses)

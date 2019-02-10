from fastai import Recorder
import matplotlib.pyplot as plt
import pickle

class PlottingRecorder(Recorder):
  def __init__(self, experiment, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.experiment = experiment

  def on_train_end(self, **kwargs):
    if hasattr(self, 'losses'):
      self.plot_losses()
      model_name = self.experiment.model_name
      plt.savefig(f'./loss_plots_{model_name}.png')
      plt.close()

class LossesRecorder(Recorder):
  def __init__(self, experiment, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.experiment = experiment

  def on_train_end(self, **kwargs):
    if hasattr(self, 'losses'):
      model_name = self.experiment.model_name
      with open(f'./losses_{model_name}.pkl', 'wb+') as fh:
        pickle.dump(fh, self.losses)

import collections
import numpy as np
from fastai.model import Stepper, Iterable, IterBatch, print_stats, append_stats, batch_sz
from fastai.core import V, VV, no_grad_context, delistify, to_np, datafy
from fastai.swa import fix_batchnorm
from fastai.layer_optimizer import LayerOptimizer
from tqdm import tnrange, tqdm

def validate_next(stepper, metrics, val_iter):
  """Computes the loss on the next minibatch of the validation set."""
  stepper.reset(False)
  with no_grad_context():
    (*x,y) = val_iter.next()
    preds,l = stepper.evaluate(x, VV(y))
    res = [delistify(to_np(l))]
    res += [f(datafy(preds), datafy(y)) for f in metrics]
  stepper.reset(True)
  return res

def validate(stepper, dl, metrics, epoch, seq_first=False, validate_skip = 0):
  if epoch < validate_skip: return [float('nan')] + [float('nan')] * len(metrics)
  batch_cnts,loss,res = [],[],[]
  stepper.reset(False)
  with no_grad_context():
    for (*x,y) in iter(dl):
      y = VV(y)
      preds, l = stepper.evaluate(x, y)
      batch_cnts.append(batch_sz(x, seq_first=seq_first))
      loss.append(to_np(l))
      res.append([f(datafy(preds), datafy(y)) for f in metrics])
  return [np.average(loss, 0, weights=batch_cnts)] + list(np.average(np.stack(res), 0, weights=batch_cnts))

def fit(model, data, n_epochs, opt, crit, metrics=None, callbacks=None, stepper=Stepper,
    swa_model=None, swa_start=None, swa_eval_freq=None, visualize=False, **kwargs):
  """ Fits a model

  Arguments:
     model (model): any pytorch module
       net = to_gpu(net)
     data (ModelData): see ModelData class and subclasses (can be a list)
     opts: an optimizer. Example: optim.Adam.
     If n_epochs is a list, it needs to be the layer_optimizer to get the optimizer as it changes.
     n_epochs(int or list): number of epochs (or list of number of epochs)
     crit: loss function to optimize. Example: F.cross_entropy
  """

  seq_first = kwargs.pop('seq_first', False)
  all_val = kwargs.pop('all_val', False)
  get_ep_vals = kwargs.pop('get_ep_vals', False)
  validate_skip = kwargs.pop('validate_skip', 0)
  metrics = metrics or []
  callbacks = callbacks or []
  avg_mom=0.98
  batch_num,avg_loss=0,0.
  for cb in callbacks: cb.on_train_begin()
  names = ["epoch", "trn_loss", "val_loss"] + [f.__name__ for f in metrics]
  if swa_model is not None:
    swa_names = ['swa_loss'] + [f'swa_{f.__name__}' for f in metrics]
    names += swa_names
    # will use this to call evaluate later
    swa_stepper = stepper(swa_model, None, crit, **kwargs)

  layout = "{!s:10} " * len(names)
  if not isinstance(n_epochs, Iterable): n_epochs=[n_epochs]
  if not isinstance(data, Iterable): data = [data]
  if len(data) == 1: data = data * len(n_epochs)
  for cb in callbacks: cb.on_phase_begin()
  model_stepper = stepper(model, opt.opt if hasattr(opt,'opt') else opt, crit, **kwargs)
  ep_vals = collections.OrderedDict()
  tot_epochs = int(np.ceil(np.array(n_epochs).sum()))
  cnt_phases = np.array([ep * len(dat.trn_dl) for (ep,dat) in zip(n_epochs,data)]).cumsum()
  phase = 0
  for epoch in tnrange(tot_epochs, desc='Epoch'):
    if phase >= len(n_epochs): break #Sometimes cumulated errors make this append.
    model_stepper.reset(True)
    cur_data = data[phase]
    if hasattr(cur_data, 'trn_sampler'): cur_data.trn_sampler.set_epoch(epoch)
    if hasattr(cur_data, 'val_sampler'): cur_data.val_sampler.set_epoch(epoch)
    num_batch = len(cur_data.trn_dl)
    t = tqdm(iter(cur_data.trn_dl), leave=False, total=num_batch, miniters=0)
    if all_val: val_iter = IterBatch(cur_data.val_dl)

    for (*x,y) in t:
      batch_num += 1
      for cb in callbacks: cb.on_batch_begin()
      loss = model_stepper.step(x, V(y), epoch)
      avg_loss = avg_loss * avg_mom + loss * (1-avg_mom)
      debias_loss = avg_loss / (1 - avg_mom**batch_num)
      t.set_postfix(loss=debias_loss, refresh=False)
      stop=False
      los = debias_loss if not all_val else [debias_loss] + validate_next(model_stepper,metrics, val_iter)
      for cb in callbacks: stop = stop or cb.on_batch_end(los)
      if stop: return
      if batch_num >= cnt_phases[phase]:
        for cb in callbacks: cb.on_phase_end()
        phase += 1
        if phase >= len(n_epochs):
          t.close()
          break
        for cb in callbacks: cb.on_phase_begin()
        if isinstance(opt, LayerOptimizer): model_stepper.opt = opt.opt
        if cur_data != data[phase]:
          t.close()
          break

    if not all_val:
      vals = validate(model_stepper, cur_data.val_dl, metrics, epoch, seq_first=seq_first, validate_skip = validate_skip)
      stop=False
      for cb in callbacks: stop = stop or cb.on_epoch_end(vals)
      if swa_model is not None:
        if (epoch + 1) >= swa_start and ((epoch + 1 - swa_start) % swa_eval_freq == 0 or epoch == tot_epochs - 1):
          fix_batchnorm(swa_model, cur_data.trn_dl)
          swa_vals = validate(swa_stepper, cur_data.val_dl, metrics, epoch, validate_skip = validate_skip)
          vals += swa_vals

      if epoch > 0:
        print_stats(epoch, [debias_loss] + vals, visualize, prev_val)
      else:
        print(layout.format(*names))
        print_stats(epoch, [debias_loss] + vals, visualize)
      prev_val = [debias_loss] + vals
      ep_vals = append_stats(ep_vals, epoch, [debias_loss] + vals)
    if stop: break
  for cb in callbacks: cb.on_train_end()
  if get_ep_vals: return vals, ep_vals
  else: return vals

from functools import partial

import fire
from fastai.text import *
from fastai.lm_rnn import *
import torch

from early_stopping import EarlyStopping


def train_lm(dir_path, pretrain_path, cl=25, pretrain_id='wt103', lm_id='', bs=64,
       dropmult=1.0, lr=4e-3, startat=0,
       use_clr=True, use_regular_schedule=False, use_discriminative=True, notrain=False, joined=False,
       train_file_id='', early_stopping=False):

  PRE = 'fwd_'
  IDS = 'ids'
  train_file_id = train_file_id if train_file_id == '' else f'_{train_file_id}'
  joined_id = 'lm_' if joined else ''
  lm_id = lm_id if lm_id == '' else f'{lm_id}_'
  lm_path=f'{PRE}{lm_id}lm'
  enc_path=f'{PRE}{lm_id}lm_enc'

  dir_path = Path(dir_path)
  pretrain_path = Path(pretrain_path)
  pre_lm_path = pretrain_path / 'models' / f'{PRE}{pretrain_id}.h5'

  bptt=70
  em_sz,nh,nl = 400,1150,3
  opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

  trn_lm_path = dir_path / 'tmp' / f'trn_{joined_id}{IDS}{train_file_id}.npy'
  val_lm_path = dir_path / 'tmp' / f'val_{joined_id}{IDS}.npy'

  print(f'Loading {trn_lm_path} and {val_lm_path}')
  trn_lm = np.load(trn_lm_path)
  trn_lm = np.concatenate(trn_lm)
  val_lm = np.load(val_lm_path)
  val_lm = np.concatenate(val_lm)

  itos = pickle.load(open(dir_path / 'tmp' / 'itos.pkl', 'rb'))
  vs = len(itos)

  trn_dl = LanguageModelLoader(trn_lm, bs, bptt)
  val_dl = LanguageModelLoader(val_lm, bs, bptt)
  md = LanguageModelData(dir_path, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

  drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*dropmult

  learner = md.get_model(opt_fn, em_sz, nh, nl,
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])
  learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
  learner.clip=0.3
  learner.metrics = [accuracy]
  wd=1e-7

  lrs = np.array([lr/6,lr/3,lr,lr/2]) if use_discriminative else lr
  if startat == 0:
    wgts = torch.load(pre_lm_path, map_location=lambda storage, loc: storage)
    print(f'Loading pretrained weights...')
    ew = to_np(wgts['0.encoder.weight'])
    row_m = ew.mean(0)

    itos2 = pickle.load(open(pretrain_path / 'tmp' / f'itos_{pretrain_id}.pkl', 'rb'))
    stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})
    nw = np.zeros((vs, em_sz), dtype=np.float32)
    nb = np.zeros((vs,), dtype=np.float32)
    for i,w in enumerate(itos):
      r = stoi2[w]
      if r>=0:
        nw[i] = ew[r]
      else:
        nw[i] = row_m

      wgts['0.encoder.weight'] = T(nw)
      wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(nw))
      wgts['1.decoder.weight'] = T(np.copy(nw))
      learner.model.load_state_dict(wgts)
      #learner.freeze_to(-1)
      #learner.fit(lrs, 1, wds=wd, use_clr=(6,4), cycle_len=1)
  else:
    print('Loading LM that was already fine-tuned on the target data...')
    learner.load(lm_path)

  if not notrain:
    learner.unfreeze()
    if use_regular_schedule:
      print('Using regular schedule. Setting use_clr=None, n_cycles=cl, cycle_len=None.')
      use_clr = None
      n_cycles=cl
      cl=None
    else:
      n_cycles=1
    callbacks = []
    if early_stopping:
      callbacks.append(EarlyStopping(learner, lm_path, enc_path, patience=5))
      print('Using early stopping...')
    learner.fit(lrs, n_cycles, wds=wd, use_clr=(32,10) if use_clr else None, cycle_len=cl,
          callbacks=callbacks)
    learner.save(lm_path)
    learner.save_encoder(enc_path)
  else:
    print('No more fine-tuning used. Saving original LM...')
    learner.save(lm_path)
    learner.save_encoder(enc_path)

if __name__ == '__main__': fire.Fire(train_lm)

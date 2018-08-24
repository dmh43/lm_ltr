import fire
from fastai.text import *
from fastai.lm_rnn import *

from lm_ranker import get_lm_ranker


def freeze_all_but(learner, n):
  c=learner.get_layer_groups()
  for l in c: set_trainable(l, False)
  set_trainable(c[n], True)


def train_ranker(dir_path,
                 ranker_id='',
                 bs=64,
                 cl=1,
                 startat=0,
                 unfreeze=True,
                 lr=0.01,
                 dropmult=1.0,
                 use_clr=True,
                 use_discriminative=True,
                 last=False,
                 chain_thaw=False):
  dir_path = Path(dir_path)
  ranker_id = ranker_id if ranker_id == '' else f'{ranker_id}_'
  intermediate_ranker_file = f'fwd_{ranker_id}ranker_0'
  final_ranker_file = f'fwd_{ranker_id}ranker_1'
  lm_file = f'fwd_lm_enc'
  lm_path = dir_path / 'models' / f'{lm_file}.h5'

  bptt,em_sz,nh,nl = 70,400,1150,3
  opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

  trn_sent = np.load(dir_path / 'tmp' / f'trn_ids.npy')
  val_sent = np.load(dir_path / 'tmp' / 'val_ids.npy')

  trn_lbls = np.load(dir_path / 'tmp' / f'lbl_trn.npy')
  val_lbls = np.load(dir_path / 'tmp' / f'lbl_val.npy')
  trn_lbls = trn_lbls.flatten()
  val_lbls = val_lbls.flatten()
  trn_lbls -= trn_lbls.min()
  val_lbls -= val_lbls.min()
  c=int(trn_lbls.max())+1

  itos = pickle.load(open(dir_path / 'tmp' / 'itos.pkl', 'rb'))
  vs = len(itos)

  trn_ds = TextDataset(trn_sent, trn_lbls)
  val_ds = TextDataset(val_sent, val_lbls)
  trn_samp = SortishSampler(trn_sent, key=lambda x: len(trn_sent[x]), bs=bs//2)
  val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
  trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
  val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
  md = ModelData(dir_path, trn_dl, val_dl)

  dps = np.array([0.4,0.5,0.05,0.3,0.4])*dropmult
  #dps = np.array([0.5, 0.4, 0.04, 0.3, 0.6])*dropmult
  #dps = np.array([0.65,0.48,0.039,0.335,0.34])*dropmult
  #dps = np.array([0.6,0.5,0.04,0.3,0.4])*dropmult

  m = get_lm_ranker(bptt, 20*70, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
        layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
        dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

  learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
  learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
  learn.clip=25.
  learn.metrics = [accuracy]

  lrm = 2.6
  if use_discriminative:
    lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])
  else:
    lrs = lr
  wd = 1e-6
  learn.load_encoder(lm_file)

  if (startat<1) and not last and not chain_thaw:
    learn.freeze_to(-1)
    learn.fit(lrs, 1, wds=wd, cycle_len=1,
          use_clr=None if not use_clr else (8,3))
    learn.freeze_to(-2)
    learn.fit(lrs, 1, wds=wd, cycle_len=1,
          use_clr=None if not use_clr else (8, 3))
    learn.save(intermediate_ranker_file)
  elif startat==1:
    learn.load(intermediate_ranker_file)

  if chain_thaw:
    lrs = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.001])
    print('Using chain-thaw. Unfreezing all layers one at a time...')
    n_layers = len(learn.get_layer_groups())
    print('# of layers:', n_layers)
    # fine-tune last layer
    learn.freeze_to(-1)
    print('Fine-tuning last layer...')
    learn.fit(lrs, 1, wds=wd, cycle_len=1,
          use_clr=None if not use_clr else (8,3))
    n = 0
    # fine-tune all layers up to the second-last one
    while n < n_layers-1:
      print('Fine-tuning layer #%d.' % n)
      freeze_all_but(learn, n)
      learn.fit(lrs, 1, wds=wd, cycle_len=1,
            use_clr=None if not use_clr else (8,3))
      n += 1

  if unfreeze:
    learn.unfreeze()
  else:
    learn.freeze_to(-3)

  if last:
    print('Fine-tuning only the last layer...')
    learn.freeze_to(-1)

  n_cycles = 1
  learn.fit(lrs, n_cycles, wds=wd, cycle_len=cl, use_clr=(8,8) if use_clr else None)
  print('Plotting lrs...')
  learn.sched.plot_lr()
  learn.save(final_ranker_file)

if __name__ == '__main__': fire.Fire(train_ranker)

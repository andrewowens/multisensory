# this was trained using the old version of the loss, which had a separate cls_opt
import os, aolib.util as ut, tfutil as mu, numpy as np

ab = os.path.abspath
pj = ut.pjoin
Params = mu.Params

VidDur = 2.135

def base_path():
  return '../data'

def pretrain_path():
  return '../results/nets/sep'

def base(name, num_gpus = 1, batch_size = 6, vid_dur = None, samp_sr = 21000., resdir = None):
  if vid_dur is None:
    vid_dur = VidDur

  if resdir is None:
    assert name is not None
    resdir = ab(pj(pretrain_path(), name))

  total_dur = 5.
  fps = 29.97
  frame_dur = 1./fps
  pr = Params(resdir = resdir,
              train_iters = 160000,
              step_size = 120000,
              opt_method = 'adam',
              base_lr = 1e-4,
              gamma = 0.1,
              full_model = False, 
              predict_bg = True,
              grad_clip = 10.,
              batch_size = int(batch_size*num_gpus),
              test_batch = 10,
              subsample_frames = None,
              weight_decay = 1e-5,

              train_list = pj(base_path(), 'celeb-tf-v6-full', 'train/tf'),
              val_list = pj(base_path(), 'celeb-tf-v6-full', 'val/tf'),
              test_list = pj(base_path(), 'celeb-tf-v6-full', 'test/tf'),

              init_type = 'shift',
              init_path = '../results/nets/shift/net.tf-650000',
              net_style = 'full',

              im_split = False,
              multi_shift = False,
              num_dbs = None,
              im_type = 'jpeg',
              full_im_dim = 256,
              crop_im_dim = 224,
              dset_seed = None,
              fps = fps,
              show_videos = False,

              samp_sr = samp_sr,
              vid_dur = vid_dur,
              total_frames = int(total_dur*fps),
              sampled_frames = int(vid_dur*fps),
              full_samples_len = int(total_dur * samp_sr),
              samples_per_frame = samp_sr * frame_dur,
              frame_sample_delta = int(total_dur*fps)/2,

              fix_frame = False,
              use_3d = True,
              augment_ims = True,
              augment_audio = False,
              dilate = False,
              cam = False,
              do_shift = False,
              variable_frame_count = False,
              use_sound = True,
              bn_last = True,

              l1_weight = 1.,
              phase_weight = 0.01,
              gan_weight = 0.,
              use_wav_gan = False,
              log_spec = True,
              spec_min = -100.,
              spec_max = 80.,
              normalize_rms = True,

              check_iters = 1000,
              slow_check_iters = 10000,
              print_iters = 10,
              summary_iters = 10,
              profile_iters = None,
              show_iters = None,

              frame_length_ms = 64,
              frame_step_ms = 16,
              sample_len = None,
              freq_len = 1024,
              augment_rms = False,
              loss_types = ['fg-bg'],
              pit_weight = 0.,
              both_videos_in_batch = True,
              bn_scale = True,
              pad_stft = False,
              phase_type = 'pred',
              alg = 'sourcesep',
              mono = False)
  pr.spec_len = 128 * int(2**np.round(np.log2(vid_dur/float(VidDur))))
  pr.num_samples = int(round(pr.samples_per_frame*pr.sampled_frames))
  return pr

def full(num_gpus = 1, vid_dur = VidDur, batch_size = 6, **kwargs):
  pr = base('full', num_gpus, vid_dur = vid_dur, batch_size = batch_size, **kwargs)
  return pr

def unet_pit(num_gpus = 1, vid_dur = VidDur, batch_size = 24, **kwargs):
  pr = base('unet-pit', num_gpus, vid_dur = vid_dur, batch_size = batch_size, **kwargs)
  pr.net_style = 'no-im'
  pr.init_path = None
  pr.loss_types = ['pit']
  pr.pit_weight = 1.
  pr.both_videos_in_batch = False
  return pr

def static(num_gpus = 1, **kwargs):
  pr = base('static', num_gpus, **kwargs)
  pr.net_style = 'static'
  return pr

def scratch(num_gpus = 1, **kwargs):
  pr = base('scratch', num_gpus, **kwargs)
  pr.net_style = 'full'
  pr.init_path = None
  return pr

def mono(num_gpus = 1, **kwargs):
  pr = full(num_gpus).copy()
  pr.mono = True
  pr.name = 'mono'
  return pr

def large(num_gpus = 1, vid_dur = VidDur, batch_size = 6, **kwargs):
  pr = base('large', num_gpus, vid_dur = vid_dur, batch_size = batch_size, **kwargs)
  pr.train_iters = 900000
  return pr

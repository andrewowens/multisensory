# this was trained using the old version of the loss, which had a separate cls_opt
import os, sys, aolib.util as ut, tfutil as mu
import numpy as np
ab = os.path.abspath
pj = ut.pjoin
Params = mu.Params

def shift_lowfps(num_gpus = 1, shift_dur = 4.2):
  total_dur = 10.1
  fps = 29.97
  frame_dur = 1./fps
  samp_sr = 21000.
  spec_sr = 100.
  pr = Params(subsample_frames = 4,
              train_iters = 100000,
              opt_method = 'momentum', 
              base_lr = 1e-2,
              full_model = True, 
              grad_clip = 5.,
              skip_notfound = False,
              augment_ims = True,
              cam = False,
              batch_size = int(15*num_gpus),
              test_batch = 10,
              shift_dur = shift_dur,
              multipass = False,
              both_examples = True,
              small_augment = False,
              resdir = ab('/data/scratch/owens/shift/shift-lowfps'),
              init_path = None,
              weight_decay = 1e-5,
              train_list = '/data/ssd1/owens/audioset-vid-v21/small_train.txt',
              test_list = '/data/ssd1/owens/audioset-vid-v21/small_train.txt',
              num_dbs = None,
              im_type = 'jpeg',
              input_type = 'samples',
              full_im_dim = 256,
              full_flow_dim = 256,
              crop_im_dim = 224,
              sf_pad = int(0.5 * 2**4 * 4),
              use_flow = False,
              #renorm = True,
              renorm = True,
              checkpoint_iters = 1000,
              dset_seed = None,

              samp_sr = samp_sr,
              spec_sr = spec_sr,
              fps = fps,
  
              #neg_frame_buf = -50,
              max_intersection = 30*2,
              specgram_sr = spec_sr,
              num_mel = 64,
              batch_norm = True,
              show_videos = False,
              check_iters = 1000,
              decompress_flow = True,
              print_iters = 10,

              total_frames = int(total_dur*fps),
              sampled_frames = int(shift_dur*fps),
              full_specgram_samples = int(total_dur * spec_sr),
              full_samples_len = int(total_dur * samp_sr),
              sfs_per_frame = spec_sr * frame_dur,
              samples_per_frame = samp_sr * frame_dur,
              frame_sample_delta = int(total_dur*fps)/2,

              fix_frame = False,
              use_3d = True,
              augment = False,

              dilate = False,
              do_shift = True,
              variable_frame_count = False,
              momentum_rate = 0.9,
              use_sound = True,
              bn_last = True,
              summary_iters = 10,
              im_split = True,
              num_splits = 2,
              augment_audio = False,
              multi_shift = False,
              )
  pr.vis_dir = ut.mkdir(pj(pr.resdir, 'vis'))
  return pr


def shift_v1(num_gpus = 1, shift_dur = 4.2):
  total_dur = 10.1
  fps = 29.97
  frame_dur = 1./fps
  samp_sr = 21000.
  spec_sr = 100.
  pr = Params(subsample_frames = None,
              train_iters = 100000,
              opt_method = 'momentum', 
              base_lr = 1e-2,
              full_model = True, 
              grad_clip = 5.,
              skip_notfound = False,
              augment_ims = True,
              init_path = '/data/scratch/owens/shift/shift-lowfps/training/net.tf-30000',
              cam = False,
              batch_size = int(5*num_gpus),
              test_batch = 10,
              shift_dur = shift_dur,
              multipass = False,
              both_examples = True,
              small_augment = False,
              resdir = ab('/data/scratch/owens/shift/shift-v1'),
              #init_path = None,
              weight_decay = 1e-5,
              # train_list = '/data/ssd1/owens/audioset-vid-v21/small_train.txt',
              # test_list = '/data/ssd1/owens/audioset-vid-v21/small_train.txt',
              train_list = '/data/scratch/owens/audioset-vid-v21/train_tfs.txt',
              test_list = '/data/scratch/owens/audioset-vid-v21/test_tfs.txt',
              num_dbs = None,
              im_type = 'jpeg',
              input_type = 'samples',
              full_im_dim = 256,
              full_flow_dim = 256,
              crop_im_dim = 224,
              sf_pad = int(0.5 * 2**4 * 4),
              use_flow = False,
              #renorm = True,
              renorm = True,
              checkpoint_iters = 1000,
              dset_seed = None,

              samp_sr = samp_sr,
              spec_sr = spec_sr,
              fps = fps,
  
              #neg_frame_buf = -50,
              max_intersection = 30*2,
              specgram_sr = spec_sr,
              num_mel = 64,
              batch_norm = True,
              show_videos = False,
              check_iters = 1000,
              decompress_flow = True,
              print_iters = 10,

              total_frames = int(total_dur*fps),
              sampled_frames = int(shift_dur*fps),
              full_specgram_samples = int(total_dur * spec_sr),
              full_samples_len = int(total_dur * samp_sr),
              sfs_per_frame = spec_sr * frame_dur,
              samples_per_frame = samp_sr * frame_dur,
              frame_sample_delta = int(total_dur*fps)/2,

              fix_frame = False,
              use_3d = True,
              augment = False,

              dilate = False,
              do_shift = True,
              variable_frame_count = False,
              momentum_rate = 0.9,
              use_sound = True,
              bn_last = True,
              summary_iters = 10,
              im_split = True,
              num_splits = 2,
              augment_audio = False,
              multi_shift = False,

              model_iter = 650000,
              )
  pr.vid_dur = pr.shift_dur
  pr.num_samples = int(round(pr.samples_per_frame*pr.sampled_frames))
  #pr.vis_dir = ut.mkdir(pj(pr.resdir, 'vis'))
  return pr


def cam_v1(num_gpus = 1, shift_dur = 4.2):
  total_dur = 10.1
  fps = 29.97
  frame_dur = 1./fps
  samp_sr = 21000.
  spec_sr = 100.
  pr = Params(subsample_frames = None,
              train_iters = 100000,
              opt_method = 'momentum', 
              base_lr = 1e-2,
              full_model = True, 
              grad_clip = 5.,
              skip_notfound = False,
              augment_ims = True,
              init_path = '../results/nets/shift/net.tf-650000',
              cam = True,
              batch_size = int(5*num_gpus),
              test_batch = 10,
              shift_dur = shift_dur,
              multipass = False,
              both_examples = True,
              small_augment = False,
              resdir = ab('/data/scratch/owens/shift/cam-v1'),
              weight_decay = 1e-5,
              train_list = '/data/scratch/owens/audioset-vid-v21/train_tfs.txt',
              test_list = '/data/scratch/owens/audioset-vid-v21/test_tfs.txt',
              num_dbs = None,
              im_type = 'jpeg',
              input_type = 'samples',
              full_im_dim = 256,
              full_flow_dim = 256,
              crop_im_dim = 224,
              sf_pad = int(0.5 * 2**4 * 4),
              use_flow = False,
              #renorm = True,
              renorm = True,
              checkpoint_iters = 1000,
              dset_seed = None,

              samp_sr = samp_sr,
              spec_sr = spec_sr,
              fps = fps,
  
              #neg_frame_buf = -50,
              max_intersection = 30*2,
              specgram_sr = spec_sr,
              num_mel = 64,
              batch_norm = True,
              show_videos = False,
              check_iters = 1000,
              decompress_flow = True,
              print_iters = 10,

              total_frames = int(total_dur*fps),
              sampled_frames = int(shift_dur*fps),
              full_specgram_samples = int(total_dur * spec_sr),
              full_samples_len = int(total_dur * samp_sr),
              sfs_per_frame = spec_sr * frame_dur,
              samples_per_frame = samp_sr * frame_dur,
              frame_sample_delta = int(total_dur*fps)/2,

              fix_frame = False,
              use_3d = True,
              augment = False,

              dilate = False,
              do_shift = True,
              variable_frame_count = False,
              momentum_rate = 0.9,
              use_sound = True,
              bn_last = True,
              summary_iters = 10,
              im_split = True,
              num_splits = 2,
              augment_audio = False,
              multi_shift = False,

              model_iter = 675000,
              )
  pr.vid_dur = pr.shift_dur
  pr.num_samples = int(round(pr.samples_per_frame*pr.sampled_frames))
  return pr

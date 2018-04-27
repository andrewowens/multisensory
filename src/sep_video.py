# Separate on- and off-screen sound from video file. See README for usage examples.
import aolib.util as ut, aolib.img as ig, os, numpy as np, tensorflow as tf, tfutil as mu, scipy.io, sys, aolib.imtable as imtable, pylab, argparse, shift_params, shift_net
import sourcesep, sep_params
import aolib.sound as sound
from aolib.sound import Sound
pj = ut.pjoin

class NetClf:
  def __init__(self, pr, sess = None, gpu = None, restore_only_shift = False):
    self.pr = pr
    self.sess = sess
    self.gpu = gpu
    self.restore_only_shift = restore_only_shift

  def init(self, reset = True):
    if self.sess is None:
      print 'Running on:', self.gpu
      with tf.device(self.gpu):
        if reset:
          tf.reset_default_graph()
          tf.Graph().as_default()
        pr = self.pr
        self.sess = tf.Session()
        self.ims_ph = tf.placeholder(
          tf.uint8, [1, pr.sampled_frames, pr.crop_im_dim, pr.crop_im_dim, 3])
        self.samples_ph = tf.placeholder(tf.float32, (1, pr.num_samples, 2))
        
        crop_spec = lambda x : x[:, :pr.spec_len]
        samples_trunc = self.samples_ph[:, :pr.sample_len]

        spec_mix, phase_mix = sourcesep.stft(samples_trunc[:, :, 0], pr)
        print 'Raw spec length:', mu.shape(spec_mix)
        spec_mix = crop_spec(spec_mix)
        phase_mix = crop_spec(phase_mix)
        print 'Truncated spec length:', mu.shape(spec_mix)

        self.specgram_op, phase = map(crop_spec, sourcesep.stft(samples_trunc[:, :, 0], pr))
        self.auto_op = sourcesep.istft(self.specgram_op, phase, pr)

        self.net = sourcesep.make_net(
          self.ims_ph, samples_trunc, spec_mix, phase_mix, 
          pr, reuse = False, train = False)
        self.spec_pred_fg = self.net.pred_spec_fg
        self.spec_pred_bg = self.net.pred_spec_bg
        self.samples_pred_fg = self.net.pred_wav_fg
        self.samples_pred_bg = self.net.pred_wav_bg
        
        print 'Restoring from:', pr.model_path
        if self.restore_only_shift:
          print 'restoring only shift'
          import tensorflow.contrib.slim as slim
          var_list = slim.get_variables_to_restore()
          var_list = [x for x in var_list if x.name.startswith('im/') or x.name.startswith('sf/') or x.name.startswith('joint/')]
          self.sess.run(tf.global_variables_initializer())
          tf.train.Saver(var_list).restore(self.sess, pr.model_path)
        else:
          tf.train.Saver().restore(self.sess, pr.model_path)
        tf.get_default_graph().finalize()

  def predict(self, ims, samples):
    print 'predict'
    print 'samples shape:', samples.shape
    spec_mix = self.sess.run(self.specgram_op, {self.samples_ph : samples})
    spec_pred_fg, spec_pred_bg, samples_pred_fg, samples_pred_bg = self.sess.run(
      [self.spec_pred_fg, self.spec_pred_bg, self.samples_pred_fg, self.samples_pred_bg], 
      {self.ims_ph : ims, self.samples_ph : samples})
    print 'samples pred shape:', samples.shape
    return dict(samples_pred_fg = samples_pred_fg, 
                samples_pred_bg = samples_pred_bg, 
                spec_pred_fg = spec_pred_fg, 
                spec_pred_bg = spec_pred_bg, 
                samples_mix = samples,
                spec_mix = spec_mix)

  def predict_unmixed(self, ims, samples0, samples1):
    # undo mixing
    samples_mix = samples0 + samples1
    spec_pred_fg, samples_pred_fg, spec_pred_bg, samples_pred_bg = self.sess.run(
      [self.spec_pred_fg, self.samples_pred_fg, self.spec_pred_bg, self.samples_pred_bg], 
      {self.ims_ph : ims[None], self.samples_ph : samples_mix[None]})
    spec0 = self.sess.run(self.specgram_op, {self.samples_ph : samples0[None]})
    spec1 = self.sess.run(self.specgram_op, {self.samples_ph : samples1[None]})
    spec_mix = self.sess.run(self.specgram_op, {self.samples_ph : samples_mix[None]})
    auto0 = self.sess.run(self.auto_op, {self.samples_ph : samples0[None]})
    auto1 = self.sess.run(self.auto_op, {self.samples_ph : samples1[None]})
    auto_mix = self.sess.run(self.auto_op, {self.samples_ph : samples_mix[None]})
    return dict(samples_pred_fg = samples_pred_fg[0],
                samples_pred_bg = samples_pred_bg[0],
                spec_pred_fg = spec_pred_fg[0],
                spec_pred_bg = spec_pred_bg[0],
                spec0 = spec0[0],
                spec1 = spec1[0], 
                spec_mix = spec_mix[0],
                auto_mix = auto_mix[0],
                auto0 = auto0[0],
                auto1 = auto1[0])
  
  #def predict_cam(self, ims, samples, n = 3, num_times = 3):
  def predict_cam(self, ims, samples, n = 5, num_times = 3):
    #num_times = 1
    if 1:
      f = min(ims.shape[1:3])
      ims = np.array([ig.scale(im, (f, f)) for im in ims])
      d = int(224./256 * ims.shape[1])
      print 'd =', d, ims.shape
      full = None
      count = None
      if n == 1:
        ys = [ims.shape[1]/2]
        xs = [ims.shape[2]/2]
      else:
        ys = np.linspace(0, ims.shape[1] - d, n).astype('int64')
        xs = np.linspace(0, ims.shape[2] - d, n).astype('int64')

      if num_times == 1:
        print 'Using one time'
        ts = [0.]
      else:
        ts = np.linspace(-2, 2., n)

      for y in ys:
        for x in xs:
          crop = ims[:, y : y + d, x : x + d]
          crop = resize_nd(crop, (crop.shape[0], pr.crop_im_dim, pr.crop_im_dim, 3), order = 1)
          for shift in ts:
            print x, y, t
            snd = sound.Sound(samples, self.pr.samp_sr)
            s0 = int(shift * snd.rate)
            s1 = s0 + snd.samples.shape[0]
            shifted = snd.pad_slice(s0, s1)
            assert shifted.samples.shape[0] == snd.samples.shape[0]

            [cam] = self.sess.run([self.net.vid_net.cam], 
                                  {self.ims_ph : crop[None], 
                                   self.samples_ph : shifted.samples[None]})
            cam = cam[0, ..., 0]
            if full is None:
              full = np.zeros(cam.shape[:1] + ims.shape[1:3])
              count = np.zeros_like(full)
            cam_resized = scipy.ndimage.zoom(
              cam, np.array((full.shape[0], d, d), 'float32') / np.array(cam.shape, 'float32'))
            if 1:
              print 'abs'
              cam_resized = np.abs(cam_resized)
            # print np.abs(cam_resized).max()

            frame0 = int(max(-shift, 0) * self.pr.fps)
            frame1 = cam_resized.shape[0] - int(max(shift, 0) * self.pr.fps)
            ok = np.ones(count.shape[0])
            cam_resized[:frame0] = 0.
            cam_resized[frame1:] = 0.
            ok[:frame0] = 0
            ok[frame1:] = 0

            full[:, y : y + d, x : x + d] += cam_resized
            count[:, y : y + d, x : x + d] += ok[:, None, None]
      assert count.min() > 0
      full /= np.maximum(count, 1e-5)
    #   ut.save('../results/full.pk', full)
    # full = ut.load('../results/full.pk')
    return full

def resize_nd(im, scale, order = 3):
  if np.ndim(scale) == 0:
    new_scale = [scale]*len(im.shape)
  elif type(scale[0]) == type(0):
    dims = scale
    new_scale = (np.array(dims, 'd') + 0.4) / np.array(im.shape, 'd')
    # a test to make sure we set the floating point scale correctly
    result_dims = map(int, new_scale * np.array(im.shape, 'd'))
    assert tuple(result_dims) == tuple(dims)
    scale_param = new_scale
  elif type(scale[0]) == type(0.) and type(scale[1]) == type(0.):
    new_scale = scale
  else:
    raise RuntimeError("don't know how to interpret scale: %s" % (scale,))
  res = scipy.ndimage.zoom(im, scale_param, order = order)
  # verify that zoom() returned an image of the desired size
  if (np.ndim(scale) != 0) and type(scale[0]) == type(0):
    assert res.shape == scale
  return res

def heatmap(frames, cam, lo_frac = 0.5, adapt = True, max_val = 35):
  """ Set heatmap threshold adaptively, to deal with large variation in possible input videos. """
  frames = np.asarray(frames)
  max_prob = 0.35
  if adapt:
    max_val = np.percentile(cam, 97)

  same = np.max(cam) - np.min(cam) <= 0.001
  if same:
    return frames

  outs = []
  for i in xrange(frames.shape[0]):
    lo = lo_frac * max_val
    hi = max_val + 0.001
    im = frames[i]
    f = cam.shape[0] * float(i) / frames.shape[0]
    l = int(f)
    r = min(1 + l, cam.shape[0]-1)
    p = f - l
    frame_cam = ((1-p) * cam[l]) + (p * cam[r])
    frame_cam = ig.scale(frame_cam, im.shape[:2], 1)
    #vis = ut.cmap_im(pylab.cm.hot, np.minimum(frame_cam, hi), lo = lo, hi = hi)
    vis = ut.cmap_im(pylab.cm.jet, frame_cam, lo = lo, hi = hi)
    #p = np.clip((frame_cam - lo)/float(hi - lo), 0, 1.)
    p = np.clip((frame_cam - lo)/float(hi - lo), 0, max_prob)
    p = p[..., None]
    im = np.array(im, 'd')
    vis = np.array(vis, 'd')
    outs.append(np.uint8(im*(1-p) + vis*p))
  return np.array(outs)

def crop_from_cam(ims, cam, pr):
  cam = np.array([ig.blur(x, 2.) for x in cam])
  cam = np.abs(cam)
  cam = cam.mean(0)

  ims = np.asarray(ims)
  y, x = np.nonzero(cam >= cam.max() - 1e-8)
  y, x = y[0], x[0]
  y = int(round((y + 0.5) * ims.shape[1]/float(cam.shape[0])))
  x = int(round((x + 0.5) * ims.shape[2]/float(cam.shape[1])))

  d = np.mean(ims.shape[1:3])
  # h = int(max(224, d//3))
  # w = int(max(224, d//3))
  h = int(max(224, d//2.5))
  w = int(max(224, d//2.5))

  y0 = int(np.clip(y - h/2, 0, ims.shape[1] - h))
  x0 = int(np.clip(x - w/2, 0, ims.shape[2] - w))
  crop = ims[:, y0 : y0 + h, x0 : x0 + w]
  crop = np.array([ig.scale(im, (pr.crop_im_dim, pr.crop_im_dim)) for im in crop])
  return crop

def find_cam(ims, samples, arg):
  clf = shift_net.NetClf(
    shift_params.cam_v1(shift_dur = (0.5+len(ims))/float(pr.fps)), 
    '../results/nets/cam/net.tf-675000', gpu = arg.gpu)
  [cam] = clf.predict_cam_resize(ims[None], samples[None])
  cam = np.abs(cam[0, :, :, :, 0])
  vis = heatmap(ims, cam, adapt = arg.adapt_cam_thresh, 
                max_val = arg.max_cam_thresh)
  return cam, vis

def run(vid_file, start_time, dur, pr, gpu, buf = 0.05, mask = None, arg = None, net = None):
  print pr
  dur = dur + buf
  with ut.TmpDir() as vid_path:
    height_s = '-vf "scale=-2:\'min(%d,ih)\'"' % arg.max_full_height if arg.max_full_height > 0 else ''
    ut.sys_check(ut.frm(
      'ffmpeg -loglevel error -ss %(start_time)s -i "%(vid_file)s" -safe 0  '
      '-t %(dur)s -r %(pr.fps)s -vf scale=256:256 "%(vid_path)s/small_%%04d.png"'))
    ut.sys_check(ut.frm(
      'ffmpeg -loglevel error -ss %(start_time)s -i "%(vid_file)s" -safe 0 '
      '-t %(dur)s -r %(pr.fps)s %(height_s)s "%(vid_path)s/full_%%04d.png"'))
    ut.sys_check(ut.frm(
      'ffmpeg -loglevel error -ss %(start_time)s -i "%(vid_file)s" -safe 0  '
      '-t %(dur)s -ar %(pr.samp_sr)s -ac 2 "%(vid_path)s/sound.wav"'))

    if arg.fullres:
      fulls = map(ig.load, sorted(ut.glob(vid_path, 'full_*.png'))[:pr.sampled_frames])
      fulls = np.array(fulls)

    snd = sound.load_sound(pj(vid_path, 'sound.wav'))
    samples_orig = snd.normalized().samples
    samples_orig = samples_orig[:pr.num_samples]
    samples_src = samples_orig.copy()
    if samples_src.shape[0] < pr.num_samples:
      return None
      
    ims = map(ig.load, sorted(ut.glob(vid_path, 'small_*.png')))
    ims = np.array(ims)
    d = 224
    y = x = ims.shape[1]/2 - d/2
    ims = ims[:, y : y + d, x : x + d]
    ims = ims[:pr.sampled_frames]

    if mask == 'l':
      ims[:, :, :ims.shape[2]/2] = 128
      if arg.fullres:
        fulls[:, :, :fulls.shape[2]/2] = 128
    elif mask == 'r':
      ims[:, :, ims.shape[2]/2:] = 128
      if arg.fullres:
        fulls[:, :, fulls.shape[2]/2:] = 128
    elif mask is None:
      pass
    else: raise RuntimeError()

    samples_src = mu.normalize_rms_np(samples_src[None], pr.input_rms)[0]
    net.init()
    ret = net.predict(ims[None], samples_src[None])
    samples_pred_fg = ret['samples_pred_fg'][0][:, None]
    samples_pred_bg = ret['samples_pred_bg'][0][:, None]
    spec_pred_fg = ret['spec_pred_fg'][0]
    spec_pred_bg = ret['spec_pred_bg'][0]
    print spec_pred_bg.shape
    spec_mix = ret['spec_mix'][0]

    if arg.cam:
      cam, vis = find_cam(fulls, samples_orig, arg)
    else:
      if arg.fullres:
        vis = fulls
      else:
        vis = ims

    return dict(ims = vis, 
                samples_pred_fg = samples_pred_fg, 
                samples_pred_bg = samples_pred_bg, 
                samples_mix = ret['samples_mix'][0],
                samples_src = samples_src, 
                spec_pred_fg = spec_pred_fg, 
                spec_pred_bg = spec_pred_bg, 
                spec_mix = spec_mix)
    
if __name__ == '__main__':
  arg = argparse.ArgumentParser(description='Separate on- and off-screen audio from a video')
  arg.add_argument('vid_file', type = str, help = 'Video file to process')
  arg.add_argument('--duration_mult', type = float, default = None, 
                   help = 'Multiply the default duration of the audio (i.e. %f) by this amount. Should be a power of 2.' % sep_params.VidDur)
  arg.add_argument('--mask', type = str, default = None, 
                   help = "set to 'l' or 'r' to visually mask the left/right half of the video before processing")
  arg.add_argument('--start', type = float, default = 0., help = 'How many seconds into the video to start')
  arg.add_argument('--model', type = str, default = 'full', 
                   help = 'Which variation of othe source separation model to run.')
  arg.add_argument('--gpu', type = int, default = 0, help = 'Set to -1 for no GPU')
  arg.add_argument('--out', type = str, default = None, help = 'Directory to save videos')
  arg.add_argument('--cam', dest = 'cam', default = False, action = 'store_true')
  arg.add_argument('--adapt_cam_thresh', type = int, default = True)
  arg.add_argument('--max_cam_thresh', type = float, default = 35)

  # undocumented/deprecated options
  arg.add_argument('--clip_dur', type = float, default = None)
  arg.add_argument('--duration', type = float, default = None)
  arg.add_argument('--fullres', type = bool, default = True)
  arg.add_argument('--suffix', type = str, default = '')
  arg.add_argument('--max_full_height', type = int, default = 600)

  #arg.set_defaults(cam = False)

  arg = arg.parse_args()
  arg.fullres = arg.fullres or arg.cam

  if arg.gpu < 0:
    arg.gpu = None

  print 'Start time:', arg.start
  print 'GPU =', arg.gpu

  gpus = [arg.gpu]
  gpus = mu.set_gpus(gpus)
  
  if arg.duration_mult is not None:
    pr = sep_params.full()
    step = 0.001 * pr.frame_step_ms
    length = 0.001 * pr.frame_length_ms
    arg.clip_dur = length + step*(0.5+pr.spec_len)*arg.duration_mult
  
  fn = getattr(sep_params, arg.model)
  pr = fn(vid_dur = arg.clip_dur)

  if arg.clip_dur is None:
    arg.clip_dur = pr.vid_dur
  pr.input_rms = np.sqrt(0.1**2 + 0.1**2)
  print 'Spectrogram samples:', pr.spec_len
  pr.model_path = '../results/nets/sep/%s/net.tf-%d' % (pr.name, pr.train_iters)

  if not os.path.exists(arg.vid_file):
    print 'Does not exist:', arg.vid_file
    sys.exit(1)

  if arg.duration is None:
    arg.duration = arg.clip_dur + 0.01

  print arg.duration, arg.clip_dur
  full_dur = arg.duration
  #full_dur = min(arg.duration, ut.video_length(arg.vid_file))
  #full_dur = arg.duration
  step_dur = arg.clip_dur/2.
  filled = np.zeros(int(np.ceil(full_dur * pr.samp_sr)), 'bool')
  full_samples_fg = np.zeros(filled.shape, 'float32')
  full_samples_bg = np.zeros(filled.shape, 'float32')
  full_samples_src = np.zeros(filled.shape, 'float32')
  arg.start = ut.make_mod(arg.start, (1./pr.fps))

  ts = np.arange(arg.start, arg.start + full_dur - arg.clip_dur, step_dur)
  full_ims = [None] * int(np.ceil(full_dur * pr.fps))

  net = NetClf(pr, gpu = gpus[0])

  for t in ut.time_est(ts):
    t = ut.make_mod(t, (1./pr.fps))
    frame_start = int(t*pr.fps - arg.start*pr.fps)
    ret = run(arg.vid_file, t, arg.clip_dur, pr, gpus[0], mask = arg.mask, arg = arg, net = net)
    if ret is None:
      continue
    ims = ret['ims']
    for frame, im in zip(xrange(frame_start, frame_start + len(ims)), ims):
      full_ims[frame] = im
    
    samples_fg = ret['samples_pred_fg'][:, 0]
    samples_bg = ret['samples_pred_bg'][:, 0]
    samples_src = ret['samples_src'][:, 0]
    samples_src = samples_src[:samples_bg.shape[0]]

    sample_start = int(round((t - arg.start) * pr.samp_sr))
    n = samples_src.shape[0]
    inds = np.arange(sample_start, sample_start + n)
    ok = ~filled[inds]
    full_samples_fg[inds[ok]] = samples_fg[ok]
    full_samples_bg[inds[ok]] = samples_bg[ok]
    full_samples_src[inds[ok]] = samples_src[ok]
    filled[inds] = True
  full_samples_fg = np.clip(full_samples_fg, -1., 1.)
  full_samples_bg = np.clip(full_samples_bg, -1., 1.)
  full_samples_src = np.clip(full_samples_src, -1., 1.)
  full_ims = [x for x in full_ims if x is not None]
  table = [['start =', arg.start],
           'fg:', imtable.Video(full_ims, pr.fps, Sound(full_samples_fg, pr.samp_sr)),
           'bg:', imtable.Video(full_ims, pr.fps, Sound(full_samples_bg, pr.samp_sr)),
           'src:', imtable.Video(full_ims, pr.fps, Sound(full_samples_src, pr.samp_sr))]

  if arg.out is not None:
    ut.mkdir(arg.out)
    vid_s = arg.vid_file.split('/')[-1].split('.mp4')[0]
    mask_s = '' if arg.mask is None else '_%s' % arg.mask
    cam_s = '' if not arg.cam else '_cam'
    suffix_s = '' if arg.suffix == '' else '_%s' % arg.suffix
    name = '%s%s%s_%s' % (suffix_s, mask_s, cam_s, vid_s)

    def snd(x): 
      x = Sound(x, pr.samp_sr)
      x.samples = np.clip(x.samples, -1., 1.)
      return x

    print 'Writing to:', arg.out
    ut.save(pj(arg.out, 'ret%s.pk' % name), ret)
    ut.make_video(full_ims, pr.fps, pj(arg.out, 'fg%s.mp4' % name), snd(full_samples_fg))
    ut.make_video(full_ims, pr.fps, pj(arg.out, 'bg%s.mp4' % name), snd(full_samples_bg))
    ut.make_video(full_ims, pr.fps, pj(arg.out, 'src%s.mp4' % name), snd(full_samples_src))
  else:
    print 'Not writing, since --out was not set'

  print 'Video results:'
  ig.show(table)

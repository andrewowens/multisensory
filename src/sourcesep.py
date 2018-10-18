import numpy as np, tfutil as mu, aolib.util as ut, aolib.sound as sound, aolib.img as ig, sep_dset, tensorflow as tf, aolib.imtable as imtable, shift_net, gc, soundrep

import tensorflow.contrib.slim as slim

ed = tf.expand_dims
shape = mu.shape
add_n = mu.maybe_add_n
pj = ut.pjoin
cast_complex = mu.cast_complex
cast_float = mu.cast_float
cast_int = mu.cast_int

def on_cpu(f):
  return mu.on_cpu(f)
  #return f()

class NetClf:
  def __init__(self, pr, model_path, 
               sess = None, gpu = None, 
               restore_only_shift = False,
               input_sr = None):
    self.pr = pr
    self.sess = sess
    self.gpu = gpu
    self.model_path = model_path
    self.restore_only_shift = restore_only_shift
    self.input_sr = input_sr

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

        spec_mix, phase_mix = sep_module(pr).stft(samples_trunc[:, :, 0], pr)
        spec_mix = crop_spec(spec_mix)
        phase_mix = crop_spec(phase_mix)

        self.specgram_op, phase = map(crop_spec, sep_module(pr).stft(samples_trunc[:, :, 0], pr))
        self.auto_op = sep_module(pr).istft(self.specgram_op, phase, pr)

        self.net = sep_module(pr).make_net(
          self.ims_ph, samples_trunc, spec_mix, phase_mix, 
          pr, reuse = False, train = False)
        self.spec_pred_fg = self.net.pred_spec_fg
        self.spec_pred_bg = self.net.pred_spec_bg
        self.samples_pred_fg = self.net.pred_wav_fg
        self.samples_pred_bg = self.net.pred_wav_bg

        print 'Restoring from:', self.model_path
        if self.restore_only_shift:
          print 'restoring only shift'
          import tensorflow.contrib.slim as slim
          var_list = slim.get_variables_to_restore()
          var_list = [x for x in var_list if x.name.startswith('im/') or x.name.startswith('sf/') or x.name.startswith('joint/')]
          self.sess.run(tf.global_variables_initializer())
          tf.train.Saver(var_list).restore(self.sess, self.model_path)
        else:
          tf.train.Saver().restore(self.sess, self.model_path)
        tf.get_default_graph().finalize()

  def predict(self, ims, samples):
    print 'predict'
    print 'samples shape:', samples.shape
    spec_mix = self.sess.run(self.specgram_op, {self.samples_ph : samples})
    spec_pred, spec_pred_bg, samples_pred_fg, samples_pred_bg = self.sess.run(
      [self.spec_pred, self.spec_pred_bg, self.samples_pred_fg, self.samples_pred_bg], 
      {self.ims_ph : ims, self.samples_ph : samples})
    print 'samples pred shape:', samples.shape
    return dict(samples_pred_fg = samples_pred_fg, 
                samples_pred_bg = samples_pred_bg, 
                samples_mix = samples,
                spec_pred = spec_pred, 
                spec_pred_bg = spec_pred_bg, 
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

  def predict_cam(self, ims, samples):
    cam = self.sess.run([self.net.vid_net.cam], {self.ims_ph : ims, self.samples_ph : samples})
    return cam

def read_data(pr, gpus):
  batch = ut.make_mod(pr.batch_size, len(gpus))

  ims, samples, ytids = on_cpu(
    lambda : sep_dset.make_db_reader(
      pr.train_list, pr, batch, ['im', 'samples', 'ytid'],
      num_db_files = pr.num_dbs))
  
  inputs = {'ims' : ims, 'samples' : samples, 'ytids' : ytids}
  splits = [{} for x in xrange(len(gpus))]   
  for k, v in inputs.items():
    if v is None:
      for i in xrange(len(gpus)):
        splits[i][k] = None
    else:
      s = tf.split(v, len(gpus))
      for i in xrange(len(gpus)):
        splits[i][k] = s[i]
  return splits
 
def make_opt(opt_method, lr_val, pr):
  if opt_method == 'adam':
    opt = tf.train.AdamOptimizer(lr_val)
  elif opt_method == 'momentum':
    opt = tf.train.MomentumOptimizer(lr_val, pr.momentum_rate)
  else:
    raise RuntimeError()
  return opt

def make_mono(samples, tile = False):
  samples = tf.reduce_mean(samples, 2)
  if tile:
    samples = tf.tile(ed(samples, 2), (1, 1, 2))
  return samples

def pool3d(x, dim, stride, padding = 'SAME'):
  if np.ndim(stride) == 0:
    stride = [stride, stride, stride]
  if np.ndim(dim) == 0:
    dim = [dim, dim, dim]
  x = tf.nn.max_pool3d(
    x, ksize = [1] + list(dim) + [1],
    strides = [1] + list(stride) + [1], 
    padding = padding)
  print 'pool ->', shape(x)
  return x

def has_prefix(x, prefix):
  if type(prefix) == type(''):
    prefix = [prefix]
  return any(x.startswith(y) for y in prefix)

def slim_losses_with_prefix(prefix, show = True):
  losses = tf.losses.get_regularization_losses()
  losses = [x for x in losses if prefix is None or x.name.startswith(prefix)]
  if show:
    print 'Collecting losses for prefix %s:' % prefix
    for x in losses:
      print x.name
    print
  return mu.maybe_add_n(losses)

def vars_with_prefix(prefix):
  vs = [x for x in tf.trainable_variables() if has_prefix(x.name, prefix)]
  missing_vs = [x for x in tf.trainable_variables() if not has_prefix(x.name, prefix)]
  print
  print 'Variables included (prefix = "%s"):' % prefix
  for x in vs:
    print x.name
  print

  print 'Variables NOT included (prefix = "%s"):' % prefix
  for x in missing_vs:
    print x.name
  print

  return vs

def slim_ups_with_prefix(prefix, show = True):
  ups = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  ups = [x for x in ups if prefix is None or x.name.startswith(prefix)]
  if show:
    print 'Collecting batch norm updates for prefix %s:' % prefix
    for x in ups:
      print x.name
    print
  return ups

def show_results(ims, samples_mix, samples_gt, spec_mix, spec_gt, spec_pred_fg, spec_pred0, 
                 samples_pred, samples_pred0, samples_gt_auto, samples_mix_auto, ytids, 
                 pr = None, table = [], min_before_show = 1, n = None):
  def make_vid(ims, samples):
    samples = np.clip(samples, -1, 1).astype('float64')
    snd = sound.Sound(samples, pr.samp_sr)
    return imtable.Video(ims, pr.fps, snd)

  def vis_spec(spec):
    return ut.jet(spec.T, pr.spec_min, pr.spec_max * 0.75)
  
  for i in range(spec_mix.shape[0])[:n]:
    row = ['mix:', make_vid(ims[i], samples_mix[i]),
           'pred:', make_vid(ims[i], samples_pred[i])]
    # if pr.use_decoder:
    #   row += ['pred (before):', make_vid(ims[i], samples_pred0[i])]
    row += ['gt:', make_vid(ims[i], samples_gt[i]),
            'gt autoencoded:', make_vid(ims[i], samples_gt_auto[i]),
            'mix autoencoded:', make_vid(ims[i], samples_mix_auto[i]),
            ut.link('https://youtube.com/watch?v=%s' % ytids[i], ytids[i])]
    table.append(row)

    row = ['mix:', vis_spec(spec_mix[i]),
           'pred:', vis_spec(spec_pred_fg[i])]
    # if pr.use_decoder:
    #   row += ['pred (before):', make_vid(ims[i], samples_pred0[i])]
    row += ['gt:', vis_spec(spec_gt[i]), 
            'gt autoencoded', vis_spec(spec_gt[i]),
            'mix autoencoded', vis_spec(spec_mix[i]),
            '']
    table.append(row)
  
  if len(table) >= min_before_show*2:
    ig.show(table)
    table[:] = []

  return np.array([1], np.int64)

# def mix_sounds(samples0, pr, quiet_thresh_db = 40., samples1 = None):
#   # todo: for PIT
#   if pr.normalize_rms:
#     samples0 = mu.normalize_rms(samples0)
#     if samples1 is not None:
#       samples1 = mu.normalize_rms(samples1)

#   if samples1 is None:
#     n = shape(samples0, 0)/2
#     samples0 = samples0[:, :pr.sample_len]
#     # samples1 = tf.concat(
#     #   [samples0[n:, :pr.sample_len],
#     #    samples0[:n, :pr.sample_len]], axis = 0)
#     samples1 = samples0[n:]
#     samples0 = samples0[:n]
#   else:
#     samples0 = samples0[:, :pr.sample_len]
#     samples1 = samples1[:, :pr.sample_len]


#   if pr.augment_rms:
#     print 'Augmenting rms'
#     scale0 = tf.random_uniform((shape(samples0, 0), 1, 1), 0.9, 1.1)
#     scale1 = tf.random_uniform((shape(samples1, 0), 1, 1), 0.9, 1.1)
#     samples0 = scale0 * samples0
#     samples1 = scale1 * samples1

#   samples_mix = samples0 + samples1
#   spec_mix, phase_mix = stft(make_mono(samples_mix), pr)
#   spec0, phase0 = stft(make_mono(samples0), pr)
#   spec1, phase1 = stft(make_mono(samples1), pr)

#   print 'Before truncating specgram:', shape(spec_mix)
#   spec_mix = spec_mix[:, :pr.spec_len]
#   print 'After truncating specgram:', shape(spec_mix)
#   phase_mix = phase_mix[:, :pr.spec_len]
#   spec0 = spec0[:, :pr.spec_len]
#   spec1 = spec1[:, :pr.spec_len]
#   phase0 = phase0[:, :pr.spec_len]
#   phase1 = phase1[:, :pr.spec_len]

#   return ut.Struct(
#     samples = samples_mix, 
#     phase = phase_mix,
#     spec = spec_mix,
#     sample_parts = [samples0, samples1], 
#     spec_parts = [spec0, spec1],
#     phase_parts = [phase0, phase1])

def mix_sounds(samples0, pr, quiet_thresh_db = 40., samples1 = None):
  if pr.normalize_rms:
    samples0 = mu.normalize_rms(samples0)
    if samples1 is not None:
      samples1 = mu.normalize_rms(samples1)

  if samples1 is None:
    n = shape(samples0, 0)/2
    samples0 = samples0[:, :pr.sample_len]
    if pr.both_videos_in_batch:
      print 'Using both videos'
      samples1 = tf.concat(
        [samples0[n:, :pr.sample_len],
         samples0[:n, :pr.sample_len]], axis = 0)
    else:
      print 'Only using first videos'
      samples1 = samples0[n:]
      samples0 = samples0[:n]
  else:
    samples0 = samples0[:, :pr.sample_len]
    samples1 = samples1[:, :pr.sample_len]

  if pr.augment_rms:
    print 'Augmenting rms'
    # scale0 = tf.random_uniform((shape(samples0, 0), 1, 1), 0.9, 1.1)
    # scale1 = tf.random_uniform((shape(samples1, 0), 1, 1), 0.9, 1.1)
    db = 0.25
    scale0 = 2.**tf.random_uniform((shape(samples0, 0), 1, 1), -db, db)
    scale1 = 2.**tf.random_uniform((shape(samples1, 0), 1, 1), -db, db)
    samples0 = scale0 * samples0
    samples1 = scale1 * samples1

  samples_mix = samples0 + samples1
  spec_mix, phase_mix = stft(make_mono(samples_mix), pr)
  spec0, phase0 = stft(make_mono(samples0), pr)
  spec1, phase1 = stft(make_mono(samples1), pr)

  print 'Before truncating specgram:', shape(spec_mix)
  spec_mix = spec_mix[:, :pr.spec_len]
  print 'After truncating specgram:', shape(spec_mix)
  phase_mix = phase_mix[:, :pr.spec_len]
  spec0 = spec0[:, :pr.spec_len]
  spec1 = spec1[:, :pr.spec_len]
  phase0 = phase0[:, :pr.spec_len]
  phase1 = phase1[:, :pr.spec_len]

  return ut.Struct(
    samples = samples_mix, 
    phase = phase_mix,
    spec = spec_mix,
    sample_parts = [samples0, samples1], 
    spec_parts = [spec0, spec1],
    phase_parts = [phase0, phase1])

def make_discrim_spec(spec_in, spec_out, phase_in, phase_out, pr, reuse = True, train = True):
  with slim.arg_scope(unet_arg_scope(pr, reuse = reuse, train = train)):
    spec_in = normalize_spec(spec_in, pr)
    spec_out = normalize_spec(spec_out, pr)
    spec_in = ed(spec_in, 3)
    spec_out = ed(spec_out, 3)
    phase_in = ed(phase_in, 3)
    phase_out = ed(phase_out, 3)
    net = tf.concat([spec_in, phase_in, spec_out, phase_out], 3)
    
    net = conv2d_same(net, 32, 4, scope = 'discrim/spec/conv1', stride = 2)
    net = conv2d_same(net, 64, 4, scope = 'discrim/spec/conv2', stride = 2)
    net = conv2d(net, 128, 4, scope = 'discrim/spec/conv3', stride = 2)
    net = conv2d(net, 128, 4, scope = 'discrim/spec/conv4', stride = 2)
    # net = conv2d(net, 256, 4, scope = 'discrim/spec/conv5', stride = 2)
    # net = conv2d(net, 256, 4, scope = 'discrim/spec/conv6', stride = [1, 2])
    logits = conv2d(net, 1, 1, scope = 'discrim/spec/logits', stride = 1, 
                    normalizer_fn = None, activation_fn = None)
    return ut.Struct(logits = logits)


def sigmoid_loss(logits, label):
  loss = tf.nn.sigmoid_cross_entropy_with_logits(
    logits = logits, labels = tf.zeros_like(logits) + label)
  ok = tf.equal(cast_int(logits >= 0.), label)
  acc = tf.stop_gradient(tf.reduce_mean(cast_float(ok)))
  return tf.reduce_mean(loss), acc

def normalize_spec(spec, pr):
  return norm_range(spec, pr.spec_min, pr.spec_max)

def unnormalize_spec(spec, pr):
  return unnorm_range(spec, pr.spec_min, pr.spec_max)

def normalize_phase(phase, pr):
  return norm_range(phase, -np.pi, np.pi)

def unnormalize_phase(phase, pr):
  return unnorm_range(phase, -np.pi, np.pi)

def add_pred_losses(gen_loss, net, snd, pr):
  if 'fg-bg' in pr.loss_types:
    gt = normalize_spec(snd.spec_parts[0], pr)
    pred = normalize_spec(net.pred_spec_fg, pr)

    if 'fg':
      diff = pred - gt
      loss = pr.l1_weight*tf.reduce_mean(tf.abs(diff))
      gen_loss.add_loss(loss, 'diff-fg')

      gt = normalize_phase(snd.phase_parts[0], pr)
      pred = normalize_phase(net.pred_phase_fg, pr)
      diff = pred - gt
      loss = pr.phase_weight*tf.reduce_mean(tf.abs(diff))
      gen_loss.add_loss(loss, 'phase-fg')

    if pr.predict_bg:
      gt = normalize_spec(snd.spec_parts[1], pr)
      pred = normalize_spec(net.pred_spec_bg, pr)
      diff = pred - gt
      loss = pr.l1_weight*tf.reduce_mean(tf.abs(diff))
      gen_loss.add_loss(loss, 'diff-bg')

      gt = normalize_phase(snd.phase_parts[1], pr)
      pred = normalize_phase(net.pred_phase_bg, pr)
      diff = pred - gt
      loss = pr.phase_weight*tf.reduce_mean(tf.abs(diff))
      gen_loss.add_loss(loss, 'phase-bg')

  if 'pit' in pr.loss_types:
    print 'Using permutation loss'
    ns = lambda x : normalize_spec(x, pr)
    np = lambda x : normalize_phase(x, pr)
    gts_ = [[ns(snd.spec_parts[0]), np(snd.phase_parts[0])],
            [ns(snd.spec_parts[1]), np(snd.phase_parts[1])]]
    preds = [[ns(net.pred_spec_fg), np(net.pred_phase_fg)],
             [ns(net.pred_spec_bg), np(net.pred_phase_bg)]]
    l1 = lambda x, y : tf.reduce_mean(tf.abs(x - y), [1, 2])
    losses = []
    for i in xrange(2):
      gt = [gts_[i%2], gts_[(i+1)%2]]
      print 'preds[0][0] shape =', shape(preds[0][0])
      fg_spec = pr.l1_weight * l1(preds[0][0], gt[0][0])
      fg_phase = pr.phase_weight * l1(preds[0][1], gt[0][1])
      
      bg_spec = pr.l1_weight * l1(preds[1][0], gt[1][0])
      bg_phase = pr.phase_weight * l1(preds[1][1], gt[1][1])

      losses.append(fg_spec + fg_phase + bg_spec + bg_phase)
    losses = tf.concat([ed(x, 0) for x in losses], 0)
    print 'losses shape =', shape(losses)
    loss_val = tf.reduce_min(losses, 0)
    print 'losses shape after min =', shape(losses)
    loss_val = pr.pit_weight * tf.reduce_mean(loss_val)
    #loss_val = tf.Print(loss_val, [losses])

    gen_loss.add_loss(loss_val, 'pit')
  # else:
  #   raise RuntimeError()

def make_loss(net, snd, pr, reuse = True, train = True):
  assert set(pr.loss_types).issubset({'pit', 'fg-bg'})
  gen_loss = mu.Loss('gen')
  gen_loss.add_loss(slim_losses_with_prefix('gen'), 'gen:reg')
  add_pred_losses(gen_loss, net, snd, pr)

  n = shape(net.pred_spec_fg, 1)
  if pr.gan_weight > 0:
    discrim_fake_spec = make_discrim_spec(snd.spec[:, :n], net.pred_spec_fg, snd.phase[:, :n], net.pred_phase_fg, pr, reuse = reuse, train = train)
    discrim_real_spec = make_discrim_spec(snd.spec[:, :n], snd.spec_parts[0][:, :n], snd.phase[:, :n], snd.phase_parts[0][:, :n], pr, reuse = True, train = train)

  discrim_loss = mu.Loss('discrim')
  discrim_loss.add_loss(slim_losses_with_prefix('discrim'), 'discrim:reg')

  tasks = []
  if pr.gan_weight > 0:
    tasks.append(('spec', discrim_fake_spec, discrim_real_spec))

  for name, discrim_fake, discrim_real in tasks:
    loss, acc = sigmoid_loss(discrim_fake.logits, 1)
    loss = loss * pr.gan_weight
    gen_loss.add_loss_acc((loss, acc), 'gen:gan_%s' % name)

    loss1, acc1 = sigmoid_loss(discrim_real.logits, 1)
    loss0, acc0 = sigmoid_loss(discrim_fake.logits, 0)
    loss, acc = (0.5*(loss0 + loss1), 0.5*(acc0 + acc1))
    acc = tf.stop_gradient(acc)
    #loss = loss * pr.gan_weight
    discrim_loss.add_loss_acc((loss, acc), 'discrim:%s' % name)

  return gen_loss, discrim_loss

class Model:
  def __init__(self, pr, sess, gpus, is_training = True, profile = False):
    self.pr = pr
    self.sess = sess
    self.gpus = gpus
    self.default_gpu = gpus[0]
    self.is_training = is_training
    self.profile = profile

  def make_model(self):
    with tf.device(self.default_gpu):
      pr = self.pr

      if self.is_training:
        self.make_train_ops()
      else:
        self.make_test_ops(reuse=False)

      self.coord = tf.train.Coordinator()
      self.saver_fast = tf.train.Saver()
      self.saver_slow = tf.train.Saver(max_to_keep = 1000)

      self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
      self.sess.run(self.init_op)
      tf.train.start_queue_runners(sess = self.sess, coord = self.coord)
      print 'Initializing'

      self.merged_summary = tf.summary.merge_all()
      print 'Tensorboard command:'
      summary_dir = ut.mkdir(pj(pr.summary_dir, ut.simple_timestamp()))
      print 'tensorboard --logdir=%s' % summary_dir
      self.sum_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

      if self.profile:
        #self.run_meta = tf.RunMetadata()
        self.profiler = tf.profiler.Profiler(self.sess.graph)

  def make_train_ops(self):
    pr = self.pr
    # steps
    self.step = tf.get_variable(
      'global_step', [], trainable = False,
      initializer = tf.constant_initializer(0), dtype = tf.int64)
    #self.lr = tf.constant(pr.base_lr)

    # model
    scale = pr.gamma ** tf.floor(cast_float(self.step) / float(pr.step_size))
    self.lr = pr.base_lr * scale
    opt = make_opt(pr.opt_method, self.lr, pr)
    self.inputs = read_data(pr, self.gpus)

    gpu_grads, gpu_losses = {}, {}
    for i, gpu in enumerate(self.gpus):
      with tf.device(gpu):
        reuse = (i > 0) 
        ims = self.inputs[i]['ims']
        all_samples = self.inputs[i]['samples']
        ytids = self.inputs[i]['ytids']
        assert not pr.do_shift
        snd = mix_sounds(all_samples, pr)
        net = make_net(ims, snd.samples, snd.spec, snd.phase, 
                       pr, reuse = reuse, train = self.is_training)
        gen_loss, discrim_loss = make_loss(net, snd, pr, reuse = reuse, train = self.is_training)

        if pr.gan_weight <= 0:
          grads = opt.compute_gradients(gen_loss.total_loss())
        else:
          # doesn't work with baselines, such as I3D
          #raise RuntimeError()
          print 'WARNING: DO NOT USE GAN WITH I3D'
          var_list = vars_with_prefix('gen') + vars_with_prefix('im') + vars_with_prefix('sf')
          grads = opt.compute_gradients(gen_loss.total_loss(), var_list = var_list)
        ut.add_dict_list(gpu_grads, 'gen', grads)
        ut.add_dict_list(gpu_losses, 'gen', gen_loss)

        var_list = vars_with_prefix('discrim')
        if pr.gan_weight <= 0:
          grads = []
        else:
          grads = opt.compute_gradients(discrim_loss.total_loss(), var_list = var_list)
        ut.add_dict_list(gpu_grads, 'discrim', grads)
        ut.add_dict_list(gpu_losses, 'discrim', discrim_loss)
        
        if i == 0:
          self.net = net
          self.show_train = self.make_show_op(net, ims, snd, ytids)

    self.gen_loss = gpu_losses['gen'][0]
    self.discrim_loss = gpu_losses['discrim'][0]

    self.train_ops = {}
    self.loss_names = {}
    self.loss_vals = {}
    ops = []
    for name in ['gen', 'discrim']:
      if pr.gan_weight <= 0. and name == 'discrim':
        op = tf.no_op()
      else:
        (gs, vs) = zip(*mu.average_grads(gpu_grads[name]))
        if pr.grad_clip is not None:
          gs, _ = tf.clip_by_global_norm(gs, pr.grad_clip)
        #gs = [mu.print_every(gs[0], 100, ['%s grad norm:' % name, tf.global_norm(gs)])] + list(gs[1:])
        gvs = zip(gs, vs)
        #bn_ups = slim_ups_with_prefix(name)
        #bn_ups = slim_ups_with_prefix(None)
        if name == 'gen':
          bn_ups = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
        else:
          bn_ups = slim_ups_with_prefix('discrim')

        print 'Number of batch norm ups for', name, len(bn_ups)
        with tf.control_dependencies(bn_ups):
          op = opt.apply_gradients(gvs)
        #op = tf.group(opt.apply_gradients(gvs, global_step = (self.step if name == 'discrim' else None)), *bn_ups)
        #op = tf.group(opt.apply_gradients(gvs), *bn_ups)
      ops.append(op)
      self.train_ops[name] = op
      loss = (self.gen_loss if name == 'gen' else self.discrim_loss)
      self.loss_names[name] = loss.get_loss_names()
      self.loss_vals[name] = loss.get_losses()
    self.update_step = self.step.assign(self.step + 1)

    if pr.gan_weight > 0:
      self.train_op = tf.group(*(ops + [self.update_step]))
    else:
      print 'Only using generator, because gan_weight = %.2f' % pr.gan_weight
      self.train_op = tf.group(ops[0], self.update_step)

  def make_show_op(self, net, ims, snd, ytids):
    pr = self.pr
    samples_gt_auto = istft(snd.spec_parts[0], snd.phase, pr)
    samples_mix_auto = istft(snd.spec, snd.phase, pr)
    
    return tf.py_func(
      lambda *args : show_results(*args, pr = pr), 
      [ims, snd.samples, snd.sample_parts[0], 
       snd.spec, snd.spec_parts[0], 
       net.pred_spec_fg, net.pred_spec_fg, net.pred_wav_fg, net.pred_wav_fg,
       samples_gt_auto, samples_mix_auto, ytids], tf.int64)  

  def checkpoint_fast(self):
    check_path = pj(ut.mkdir(self.pr.train_dir), 'net.tf')
    out = self.saver_fast.save(self.sess, check_path, global_step = self.step)
    print 'Checkpoint:', out

  def checkpoint_slow(self):
    check_path = pj(ut.mkdir(pj(self.pr.train_dir, 'slow')), 'net.tf')
    out = self.saver_slow.save(self.sess, check_path, global_step = self.step)
    print 'Checkpoint:', out

  def restore(self, path = None, restore_opt = True, init_type = None):
    if path is None:
      path = tf.train.latest_checkpoint(self.pr.train_dir)      
    print 'Restoring from:', path
    var_list = slim.get_variables_to_restore()
    opt_names = ['Adam', 'beta1_power', 'beta2_power', 'Momentum', 'cache']

    if init_type == 'shift':
      # gamma is reinitialized
      opt_names += ['gen/', 'discrim/', 'global_step', 'gamma']
    elif init_type == 'sep':
      #opt_names += ['global_step']
      opt_names += ['global_step', 'discrim']
    elif init_type is None:
      pass
    else:
      raise RuntimeError()

    if not restore_opt or init_type is not None:
      var_list = [x for x in var_list if not any(name in x.name for name in opt_names)]

    print 'Restoring:'
    for x in var_list:
      print x.name
    print
    
    tf.train.Saver(var_list).restore(self.sess, path)

  def get_step(self):
    return self.sess.run([self.step, self.lr])

  def train(self):
    val_hist = {}
    pr = self.pr
    num_steps = 0
    while True:
      step, lr = self.get_step()
      first = (num_steps == 0)
      if not first and step % pr.check_iters == 0:
        self.checkpoint_fast()
      if not first and step % pr.slow_check_iters == 0:
        self.checkpoint_slow()

      if step >= pr.train_iters:
        break

      if pr.show_iters is not None and (first or step % pr.show_iters == 0):
        self.sess.run(self.show_train)

      loss_ops = self.gen_loss.get_losses() + self.discrim_loss.get_losses()
      loss_names = self.gen_loss.get_loss_names() + self.discrim_loss.get_loss_names()
      start = ut.now_sec()
      if pr.summary_iters is not None and step % pr.summary_iters == 0:
        ret = self.sess.run([self.train_op, self.merged_summary] + loss_ops)
        self.sum_writer.add_summary(ret[1], step)
        loss_vals = ret[2:]
      elif self.profile and (pr.profile_iters is not None and not first and step % pr.profile_iters == 0):
        run_meta = tf.RunMetadata()
        loss_vals = self.sess.run(
          [self.train_op] + loss_ops, 
          options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
          run_metadata = run_meta)[1:]
        opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
        self.profiler.add_step(step, run_meta)
        self.profiler.profile_operations(options = opts)
        self.profiler.profile_graph(options = opts)
        self.profiler.advise(self.sess.graph)
      else:
        loss_vals = self.sess.run([self.train_op] + loss_ops)[1:]

      if step % 100 == 0:
        gc.collect()

      ts = moving_avg('time', ut.now_sec() - start, val_hist)
      out = []
      for name, val in zip(loss_names, loss_vals):
        out.append('%s: %.3f' % (name, moving_avg(name, val, val_hist)))
      out = ' '.join(out)

      if step < 10 or step % pr.print_iters == 0:
        print 'Iteration %d, lr = %.0e, %s, time: %.3f' % (step, lr, out, ts)

      num_steps += 1

def find_best_iter(pr, gpu, num_iters = 10, sample_rate = 10, dset_name = 'val'):
  [gpu] = mu.set_gpus([gpu])
  best_iter = (np.inf, '')
  model_paths = sorted(
    ut.glob(pj(pr.train_dir, 'slow', 'net*.index')), 
    key = lambda x : int(x.split('-')[-1].split('.')[0]))[-5:]
  model_paths = list(reversed(model_paths))
  assert len(model_paths), 'no model paths at %s' % pj(pr.train_dir, 'slow', 'net*.index')
  for model_path in model_paths:
    model_path = model_path.split('.index')[0]
    print model_path
    clf = NetClf(pr, model_path, gpu = gpu)
    clf.init()
    if dset_name == 'train':
      print 'train'
      tf_files = sorted(ut.glob(pj(pr.train_list, '*.tf')))
    elif dset_name == 'val':
      tf_files = sorted(ut.glob(pj(pr.val_list, '*.tf')))
    else:
      raise RuntimeError()

    import sep_eval
    losses = []
    for ims, _, pair in sep_eval.pair_data(tf_files, pr):
      if abs(hash(pair['ytid_gt'])) % sample_rate == 0:
        res = clf.predict_unmixed(ims, pair['samples_gt'], pair['samples_bg'])
        # loss = np.mean(np.abs(res['spec_pred_fg'] - res['spec0']))
        # loss += np.mean(np.abs(res['spec_pred_bg'] - res['spec1']))
        loss = 0.
        if 'pit' in pr.loss_types:
          loss += pit_loss(
            [res['spec0']], [res['spec1']],
            [res['spec_pred_fg']], [res['spec_pred_bg']], pr)
        if 'fg-bg' in pr.loss_types:
          loss += np.mean(np.abs(res['spec_pred_fg'] - res['spec0']))
          loss += np.mean(np.abs(res['spec_pred_bg'] - res['spec1']))
        losses.append(loss)
        print 'running:', np.mean(losses)
        loss = np.mean(losses)
    print model_path, 'Loss:', loss
    best_iter = min(best_iter, (loss, model_path))
  ut.write_lines(pj(pr.resdir, 'model_path.txt'), [best_iter[1]])

def pit_loss(gt0, gt1, pred0, pred1, pr):
  losses = []
  weights = np.array([pr.l1_weight, pr.phase_weight])
  for i in xrange(2):
    gt = [gt0, gt1] if i == 0 else [gt1, gt0]
    loss = 0.
    for j in xrange(1):
      p = np.array([pred0[j], pred1[j]])
      g = np.array([gt[0][j], gt[1][j]])
      w = weights[j]
      loss += w * np.mean(np.abs(p - g))
    losses.append(loss)
  print 'losses =', losses
  return np.min(losses)

# def find_best_iter(pr, gpu, num_iters = 10, sample_rate = 10, dset_name = 'val'):
#   [gpu] = mu.set_gpus([gpu])
#   best_iter = (np.inf, '')
#   model_paths = sorted(
#     ut.glob(pj(pr.train_dir, 'slow', 'net*.index')), 
#     key = lambda x : int(x.split('-')[-1].split('.')[0]))[-5:]
#   model_paths = reversed(model_paths)
#   for model_path in model_paths:
#     model_path = model_path.split('.index')[0]
#     print model_path
#     clf = sep_eval.NetClf(pr, model_path, gpu = gpu)
#     clf.init()
#     if dset_name == 'train':
#       print 'train'
#       tf_files = sorted(ut.glob(pj(pr.train_list, '*.tf')))
#     elif dset_name == 'val':
#       tf_files = sorted(ut.glob(pj(pr.val_list, '*.tf')))
#     else:
#       raise RuntimeError()
    
#     losses = []
#     for ims, _, pair in sep_eval.pair_data(tf_files, pr):
#       if abs(hash(pair['ytid_gt'])) % sample_rate == 0:
#         res = clf.predict_unmixed(ims, pair['samples_gt'], pair['samples_bg'])
#         loss = np.mean(np.abs(res['spec_pred_fg'] - res['spec0']))
#         loss += np.mean(np.abs(res['spec_pred_bg'] - res['spec1']))
#         losses.append(loss)
#         print 'running:', np.mean(losses)
#     loss = np.mean(losses)
#     print model_path, 'Loss:', loss
#     best_iter = min(best_iter, (loss, model_path))
#   ut.write_lines(pj(pr.resdir, 'model_path.txt'), [best_iter[1]])



# def find_best_iter(pr, gpu, num_iters = 10, sample_rate = 10):
#   [gpu] = mu.set_gpus([gpu])
#   def f((model_path, gpu_num)):
#     model_path = model_path.split('.index')[0]
#     print model_path
#     clf = sep_eval.NetClf(pr, model_path, gpu = gpu)
#     clf.init()
#     tf_files = sorted(ut.glob(pj(pr.val_list, '*.tf')))
#     losses = []
#     for ims, _, pair in sep_eval.pair_data(tf_files, pr):
#       if abs(hash(pair['ytid_gt'])) % sample_rate == 0:
#         res = clf.predict_unmixed(ims, pair['samples_gt'], pair['samples_bg'])
#         loss = np.mean(np.abs(res['spec_pred_fg'] - res['spec0']))
#         loss += np.mean(np.abs(res['spec_pred_bg'] - res['spec1']))
#         losses.append(loss)
#     loss = np.mean(losses)
#     print model_path, 'Loss:', loss
#     return (loss, model_path)
#   model_files = sorted(
#     ut.glob(pj(pr.train_dir, 'slow', 'net*.index')), 
#     key = lambda x : int(x.split('-')[-1].split('.')[0]))[-5:]
#   for model_file in ut.model_files
#   ut.write_lines(pj(pr.resdir, 'model_path.txt'), [best_iter[1]])


def moving_avg(name, x, vals, avg_win_size = 100, p = 0.99):
  vals[name] = p*vals.get(name, x) + (1 - p)*x
  return vals[name]

def conv2d(*args, **kwargs):
  out = slim.conv2d(*args, **kwargs)
  print kwargs['scope'], shape(args[0]), '->', shape(out)
  return out

def conv2d_same(*args, **kwargs):
  out = mu.conv2d_same(*args, **kwargs)
  print kwargs['scope'], '->', shape(out)
  return out

def deconv2d(*args, **kwargs):
  out = slim.conv2d_transpose(*args, **kwargs)
  print kwargs['scope'], shape(args[0]), '->', shape(out)
  return out

def unet_arg_scope(pr, 
                   weight_decay = 1e-5,
                   reuse = False, 
                   renorm = True,
                   train = True,
                   scale = True,
                   center = True):
  batch_norm_params = {
    'decay': 0.9997,
    'epsilon': 1e-5,
    'updates_collections': slim.ops.GraphKeys.UPDATE_OPS,
    'scale' : scale,
    'center' : center,
    'is_training' : train,
    'renorm' : renorm,
    'param_initializers' : {'gamma' : tf.random_normal_initializer(1., 0.02)},
  }
  normalizer_fn = slim.batch_norm
  normalizer_params = batch_norm_params
  with slim.arg_scope([slim.batch_norm],
                      **batch_norm_params):
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose],
        weights_regularizer = slim.regularizers.l2_regularizer(weight_decay),
        weights_initializer = tf.random_normal_initializer(0, 0.02),
        activation_fn = tf.nn.relu,
        normalizer_fn = normalizer_fn,
        reuse = reuse,
        normalizer_params = normalizer_params) as sc:
        return sc

def norm_range(x, min_val, max_val):
  return 2.*(x - min_val)/float(max_val - min_val) - 1.

def unnorm_range(y, min_val, max_val):
  return 0.5*float(max_val - min_val) * (y + 1) + min_val

def print_vals(name, x):
  return tf.Print(x, [name, tf.reduce_min(x), tf.reduce_max(x)])

def stft(samples, pr):
  spec_complex = tf.contrib.signal.stft(
    samples, 
    frame_length = soundrep.stft_frame_length(pr),
    frame_step = soundrep.stft_frame_step(pr),
    pad_end = pr.pad_stft)
  
  mag = tf.abs(spec_complex)
  #phase = tf.angle(spec_complex)
  phase = mu.angle(spec_complex)
  if pr.log_spec:
    mag = soundrep.db_from_amp(mag)
  return mag, phase

def make_complex(mag, phase):
  mag = cast_complex(mag)
  phase = cast_complex(phase)
  j = tf.constant(1j, dtype = tf.complex64)
  return mag * (tf.cos(phase) + j*tf.sin(phase))

def istft(mag, phase, pr):
  if pr.log_spec:
    mag = soundrep.amp_from_db(mag)
  samples = tf.contrib.signal.inverse_stft(
    make_complex(mag, phase), 
    frame_length = soundrep.stft_frame_length(pr),
    frame_step = soundrep.stft_frame_step(pr),
    fft_length = soundrep.stft_num_fft(pr))
  return samples

# def griffin_lim(mag, phase, pr):
#   import soundrep
#   if pr.log_spec:
#     mag = soundrep.amp_from_db(mag)
#   samples = soundrep.griffin_lim(
#     make_complex(mag, phase), 
#     frame_length = soundrep.stft_frame_length(pr),
#     frame_step = soundrep.stft_frame_step(pr),
#     num_fft = soundrep.stft_num_fft(pr),
#     num_iters = 5)
#   return samples

def make_net(ims, sfs, spec, phase, pr, 
             reuse = True, train = True, 
             vid_net_full = None):
  if pr.mono:
    print 'Using mono!'
    sfs = make_mono(sfs, tile = True)

  if vid_net_full is None:
    if pr.net_style == 'static':
      n = shape(ims, 1)
      if 0:
        ims_tile = tf.tile(ims[:, n/2:n/2+1], (1, n, 1, 1, 1))
      else:
        ims = tf.cast(ims, tf.float32)
        ims_tile = tf.tile(ims[:, n/2:n/2+1], (1, n, 1, 1, 1))
      vid_net_full = shift_net.make_net(ims_tile, sfs, pr, None, reuse, train)
    elif pr.net_style == 'no-im':
      vid_net_full = None
    elif pr.net_style == 'full':
      vid_net_full = shift_net.make_net(ims, sfs, pr, None, reuse, train)
    elif pr.net_style == 'i3d':
      with tf.variable_scope('RGB', reuse = reuse):
        import sep_i3d
        i3d_net = sep_i3d.InceptionI3d(1)
        vid_net_full = ut.Struct(scales = i3d_net(ims, is_training = train))
    
  with slim.arg_scope(unet_arg_scope(pr, reuse = reuse, train = train)): 
    acts = []
    def conv(*args, **kwargs):
      out = conv2d(*args, activation_fn = None, **kwargs)
      acts.append(out)
      out = mu.lrelu(out, 0.2)
      return out
    
    def deconv(*args, **kwargs):
      args = list(args)
      if kwargs.get('do_pop', True):
        skip_layer = acts.pop()
      else:
        skip_layer = acts[-1]
      if 'do_pop' in kwargs:
        del kwargs['do_pop'] 
      x = args[0]
      if kwargs.get('concat', True):
        x = tf.concat([x, skip_layer], 3)
      if 'concat' in kwargs:
        del kwargs['concat']
      args[0] = tf.nn.relu(x) 
      return deconv2d(*args, activation_fn = None, **kwargs)

    def merge_level(net, n):
      if vid_net_full is None:
        return net
      vid_net = tf.reduce_mean(vid_net_full.scales[n], [2, 3], keep_dims = True)
      vid_net = vid_net[:, :, 0, :, :]; 
      s = shape(vid_net)
      if shape(net, 1) != s[1]:
        vid_net = tf.image.resize_images(vid_net, [shape(net, 1), 1])
        print 'Video net before merge:', s, 'After:', shape(vid_net)
      else:
        print 'No need to resize:', s, shape(net)
      vid_net = tf.tile(vid_net, (1, 1, shape(net, 2), 1))
      net = tf.concat([net, vid_net], 3)
      acts[-1] = net
      return net

    num_freq = shape(spec, 2)
    net = tf.concat(
      [ed(normalize_spec(spec, pr), 3), 
       ed(normalize_phase(phase, pr), 3)], 3)

    net = net[:, :, :pr.freq_len, :]  
    net = conv(net, 64,  4, scope = 'gen/conv1', stride = [1, 2])
    net = conv(net, 128, 4, scope = 'gen/conv2', stride = [1, 2])
    net = conv(net, 256, 4, scope = 'gen/conv3', stride = 2)
    net = merge_level(net, 0)
    net = conv(net, 512, 4, scope = 'gen/conv4', stride = 2)
    net = merge_level(net, 1)
    net = conv(net, 512, 4, scope = 'gen/conv5', stride = 2)
    net = merge_level(net, 2)
    net = conv(net, 512, 4, scope = 'gen/conv6', stride = 2)
    net = conv(net, 512, 4, scope = 'gen/conv7', stride = 2)
    net = conv(net, 512, 4, scope = 'gen/conv8', stride = 2)
    net = conv(net, 512, 4, scope = 'gen/conv9', stride = 2)
 
    net = deconv(net, 512, 4, scope = 'gen/deconv1', stride = 2, concat = False)
    net = deconv(net, 512, 4, scope = 'gen/deconv2', stride = 2)
    net = deconv(net, 512, 4, scope = 'gen/deconv3', stride = 2)
    net = deconv(net, 512, 4, scope = 'gen/deconv4', stride = 2)
    net = deconv(net, 512, 4, scope = 'gen/deconv5', stride = 2)
    net = deconv(net, 256, 4, scope = 'gen/deconv6', stride = 2)
    net = deconv(net, 128, 4, scope = 'gen/deconv7', stride = 2)
    net = deconv(net, 64, 4, scope = 'gen/deconv8', stride = [1, 2])

    out_fg = deconv(net, 2, 4, scope = 'gen/fg', stride = [1, 2], 
                    normalizer_fn = None, do_pop = False)
    out_bg = deconv(net, 2, 4, scope = 'gen/bg', stride = [1, 2], 
                    normalizer_fn = None, do_pop = False)
      
    def process(out):
      pred_spec = out[..., 0]
      pred_spec = tf.tanh(pred_spec)
      pred_spec = unnormalize_spec(pred_spec, pr)

      pred_phase = out[..., 1]
      pred_phase = tf.tanh(pred_phase)
      pred_phase = unnormalize_phase(pred_phase, pr)

      val = soundrep.db_from_amp(0.) if pr.log_spec else 0.
      pred_spec = tf.pad(pred_spec, [(0, 0), (0, 0), (0, num_freq - shape(pred_spec, 2))], constant_values = val)

      if pr.phase_type == 'pred':
        pred_phase = tf.concat([pred_phase, phase[..., -1:]], 2)
      elif pr.phase_type == 'orig':
        pred_phase = phase
      else: 
        raise RuntimeError()

      # if ut.hastrue(pr, 'griffin_lim'):
      #   print 'using griffin-lim'
      #   pred_wav = griffin_lim(pred_spec, pred_phase, pr)
      # else:
      pred_wav = istft(pred_spec, pred_phase, pr)
      return pred_spec, pred_phase, pred_wav

    pred_spec_fg, pred_phase_fg, pred_wav_fg = process(out_fg)
    pred_spec_bg, pred_phase_bg, pred_wav_bg = process(out_bg)

    return ut.Struct(pred_spec_fg = pred_spec_fg, 
                     pred_wav_fg = pred_wav_fg, 
                     pred_phase_fg = pred_phase_fg,
                     
                     pred_spec_bg = pred_spec_bg,
                     pred_phase_bg = pred_phase_bg,
                     pred_wav_bg = pred_wav_bg,
                     
                     vid_net = vid_net_full,
                     )
  
def truncate_min(x, y):
  n = min(shape(x, 1), shape(y, 1))
  x = x[:, :n]
  y = y[:, :n]
  return x, y

def train(pr, gpus, restore = False, restore_opt = True, profile = False):
  print pr
  gpus = mu.set_gpus(gpus)
  with tf.Graph().as_default():
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.InteractiveSession(config = config)
    model = Model(pr, sess, gpus, profile = profile)
    model.make_model()
    if restore:
      model.restore(restore_opt = restore_opt)
    elif pr.init_path is not None:
      if pr.init_type in ['shift', 'sep']:
        model.restore(pr.init_path, restore_opt = False, init_type = pr.init_type)
      elif pr.init_type == 'i3d':
        opt_names = ['Adam', 'beta1_power', 'beta2_power', 'Momentum']
        rgb_variable_map = {}
        for variable in tf.global_variables():
          if any(x in variable.name for x in opt_names):
            print 'Skipping:', variable.name
            continue
          if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')] = variable
            print 'Restoring:', variable.name
        rgb_saver = tf.train.Saver(var_list = rgb_variable_map, reshape=True)
        rgb_saver.restore(sess, pr.init_path)
      elif pr.init_type == 'scratch':
        pass
      else:
        raise RuntimeError()

    tf.get_default_graph().finalize()
    model.train()
 

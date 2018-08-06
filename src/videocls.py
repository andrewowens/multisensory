# Example code for fine-tuning our audio-visual network to solve an
# action-recognition task.  We suggest rewriting this code, reusing
# only the parts that are relevant to your application.
import tfutil as tfu, aolib.util as ut, tensorflow as tf
import shift_net as shift
import tensorflow.contrib.slim as slim

ed = tf.expand_dims
shape = tfu.shape
add_n = tfu.maybe_add_n
pj = ut.pjoin
cast_float = tfu.cast_float
cast_int = tfu.cast_int

def make_net(ims, samples, pr, reuse = True, train = True):
  if pr.net_type == 'i3d':
    import i3d_kinetics
    keep_prob = 0.5 if train else 1.
    if pr.use_i3d_logits:
      with tf.variable_scope('RGB', reuse = reuse):
        net = tfu.normalize_ims(ims)
        i3d_net = i3d_kinetics.InceptionI3d(pr.num_classes, spatial_squeeze = True, final_endpoint = 'Logits')
        logits, _ = i3d_net(net, is_training = train, dropout_keep_prob = keep_prob)
        return ut.Struct(logits = logits, prob = tf.nn.softmax(logits), last_conv = logits)
    else:
      with tf.variable_scope('RGB', reuse = reuse):
        i3d_net = i3d_kinetics.InceptionI3d(pr.num_classes, final_endpoint = 'Mixed_5c')
        net = tfu.normalize_ims(ims)
        net, _ = i3d_net(net, is_training = train, dropout_keep_prob = keep_prob)
      last_conv = net
      net = tf.reduce_mean(last_conv, [1, 2, 3], keep_dims = True)
      with slim.arg_scope(shift.arg_scope(pr, reuse = reuse, train = train)):
        logits = shift.conv3d(
          net, pr.num_classes, [1, 1, 1], scope = 'lb/logits', 
          activation_fn = None, normalizer_fn = None)[:, 0, 0, 0, :]
        return ut.Struct(logits = logits, 
                         prob = tf.nn.softmax(logits), 
                         last_conv = net)

  elif pr.net_type == 'shift':
    with slim.arg_scope(shift.arg_scope(pr, reuse = reuse, train = train)):
      # To train the network without audio, you can set samples to be an all-zero array, and
      # set pr.use_sound = False.
      shift_net = shift.make_net(ims, samples, pr, reuse = reuse, train = train)
      if pr.use_dropout:
        shift_net.last_conv = slim.dropout(shift_net.last_conv, is_training = train)

      net = shift_net.last_conv
      net = tf.reduce_mean(net, [1, 2, 3], keep_dims = True)
      logits = shift.conv3d(
        net, pr.num_classes, [1, 1, 1], scope = 'lb/logits', 
        activation_fn = None, normalizer_fn = None)[:, 0, 0, 0, :]
      return ut.Struct(logits = logits, prob = tf.nn.softmax(logits), last_conv = net)
  elif pr.net_type == 'c3d':
    import c3d
    with slim.arg_scope(shift.arg_scope(reuse = reuse, train = train)):
      net = c3d.make_net(ims, samples, pr, reuse = reuse, train = train)
      net = net.last_conv
      net = tf.reduce_mean(net, [1, 2, 3], keep_dims = True)
      logits = c3d.conv3d(
        net, pr.num_classes, [1, 1, 1], scope = 'lb/logits', 
        activation_fn = None, normalizer_fn = None)[:, 0, 0, 0, :]
      return ut.Struct(logits = logits, prob = tf.nn.softmax(logits), last_conv = net)
  else: 
    raise RuntimeError()

def read_data(pr, gpus):
  """ This is the code for reading data. We suggest rewriting the I/O code for your own applications"""
  if pr.variable_frame_count:
    #import shift_dset
    import ucf_dset as shift_dset
  else:
    import cls_dset as shift_dset

  with tf.device('/cpu:0'):
    batch = ut.make_mod(pr.batch_size, len(gpus))
    ims, samples, labels = tfu.on_cpu(
      lambda : shift_dset.make_db_reader(
        pr.train_list, pr, batch, ['im', 'samples', 'label'],
        num_db_files = pr.num_dbs))
    
    inputs = {'ims' : ims, 'samples' : samples, 'label' : labels}
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

def num_samples(pr):
  return int(round(pr.samples_per_frame*pr.sampled_frames))

def label_loss(logits, labels, smooth = False):
  if smooth:
    nc = shape(logits, 1)
    oh = tf.one_hot(labels, nc)
    p = 0.05
    oh = p*(1./nc) + (1 - p) * oh
    loss = tf.nn.softmax_cross_entropy_with_logits(
      logits = logits, labels = oh)
  else:
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits = logits, labels = labels)
  acc = tf.reduce_mean(tfu.cast_float(tf.equal(tf.argmax(logits, 1), labels)))
  acc = tf.stop_gradient(acc)
  acc.ignore = True
  loss = tf.reduce_mean(loss)
  return loss, acc

class Model:
  def __init__(self, pr, sess, gpus, is_training = True, profile = False):
    self.pr = pr
    self.sess = sess
    self.gpus = gpus
    self.default_gpu = gpus[0]
    self.is_training = is_training
    self.profile = profile

  def make_train_model(self):
    with tf.device(self.default_gpu):
      pr = self.pr
      # steps
      self.step = tf.get_variable(
        'global_step', [], trainable = False,
        initializer = tf.constant_initializer(0), dtype = tf.int64)
      self.lr = tf.constant(pr.base_lr)

      # model
      scale = pr.gamma ** tf.floor(cast_float(self.step) / float(pr.step_size))
      self.lr_step = pr.base_lr * scale
      #lr = tf.Print(lr, [lr, lr*1e3, scale])
      opt = shift.make_opt(pr.opt_method, self.lr_step, pr)
      self.inputs = read_data(pr, self.gpus)

      gpu_grads, gpu_losses = {}, {}
      for i, gpu in enumerate(self.gpus):
        with tf.device(gpu):
          reuse = (i > 0) 
          ims = self.inputs[i]['ims']
          samples = self.inputs[i]['samples']
          labels = self.inputs[i]['label']

          net = make_net(ims, samples, pr, reuse = reuse, train = self.is_training)
          self.loss = tfu.Loss('loss')
          self.loss.add_loss(shift.slim_losses_with_prefix(None), 'reg')
          self.loss.add_loss_acc(label_loss(net.logits, labels), 'label')
          grads = opt.compute_gradients(self.loss.total_loss())

          ut.add_dict_list(gpu_grads, self.loss.name, grads)
          ut.add_dict_list(gpu_losses, self.loss.name, self.loss)

          if i == 0:
            self.net = net
        
      (gs, vs) = zip(*tfu.average_grads(gpu_grads['loss']))
      if pr.grad_clip is not None:
        gs, _ = tf.clip_by_global_norm(gs, pr.grad_clip)
      gs = [tfu.print_every(gs[0], 100, ['grad norm:', tf.global_norm(gs)])] + list(gs[1:])
      gvs = zip(gs, vs)
      #for g, v in zip(grads, vs):
      # if g[0] is not None:
      #   tf.summary.scalar('%s_grad_norm' % v.name, tf.reduce_sum(g[0]**2)**0.5)
      #   tf.summary.scalar('%s_val_norm' % v.name, tf.reduce_sum(v**2)**0.5)
      #self.train_op = opt.apply_gradients(gvs, global_step = self.step)
      
      bn_ups = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      # self.train_op = tf.group(self.train_op, *bn_ups)

      with tf.control_dependencies(bn_ups):
        self.train_op = opt.apply_gradients(gvs, global_step = self.step)
      
      self.coord = tf.train.Coordinator()
      self.saver_fast = tf.train.Saver()
      self.saver_slow = tf.train.Saver(max_to_keep = 1000)

      #self.init_op = tf.global_variables_initializer()
      if self.is_training:
        self.init_op = tf.group(
          tf.global_variables_initializer(), 
          tf.local_variables_initializer())
        self.sess.run(self.init_op)

      tf.train.start_queue_runners(sess = self.sess, coord = self.coord)

      self.merged_summary = tf.summary.merge_all()
      print 'Tensorboard command:'
      summary_dir = ut.mkdir(pj(pr.summary_dir, ut.simple_timestamp()))
      print 'tensorboard --logdir=%s' % summary_dir
      self.sum_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

      if self.profile:
        self.profiler = tf.profiler.Profiler(self.sess.graph)

  def make_test_model(self):
    with tf.device(self.default_gpu):
      pr = self.pr
      print 'test variable frame count'
      if 0 and pr.variable_frame_count:
        self.test_ims_ph = tf.placeholder(tf.uint8, [1, None, pr.crop_im_dim, pr.crop_im_dim, 3])
        self.test_samples_ph = tf.placeholder(tf.float32, [1, None, 2])
      else:
        if hasattr(pr, 'resampled_frames'):
          self.test_ims_ph = tf.placeholder(tf.uint8, [1, pr.resampled_frames, pr.crop_im_dim, pr.crop_im_dim, 3])
        else:
          self.test_ims_ph = tf.placeholder(tf.uint8, [1, pr.sampled_frames, pr.crop_im_dim, pr.crop_im_dim, 3])
        self.test_samples_ph = tf.placeholder(tf.float32, [1, num_samples(pr), 2])

      assert not self.is_training 
      #self.is_training = True
      self.test_net = make_net(
        self.test_ims_ph, self.test_samples_ph, pr, 
        reuse = False, train = self.is_training)

  def checkpoint_fast(self):
    check_path = pj(ut.mkdir(self.pr.train_dir), 'net.tf')
    out = self.saver_fast.save(self.sess, check_path, global_step = self.step)
    print 'Checkpoint:', out

  def checkpoint_slow(self):
    check_path = pj(ut.mkdir(pj(self.pr.train_dir, 'slow')), 'net.tf')
    out = self.saver_slow.save(self.sess, check_path, global_step = self.step)
    print 'Checkpoint:', out

  #def restore(self, path = None, restore_opt = True, ul_only = False):
  def restore(self, path = None, restore_opt = True, ul_only = False):
    if path is None:
      path = tf.train.latest_checkpoint(self.pr.train_dir)      
    print 'Restoring:', path
    var_list = slim.get_variables_to_restore()
    for x in var_list:
      print x.name
    print
    var_list = slim.get_variables_to_restore()
    if not restore_opt:
      opt_names = ['Adam', 'beta1_power', 'beta2_power', 'Momentum'] + ['cls']# + ['renorm_mean_weight', 'renorm_stddev_weight', 'moving_mean', 'renorm']
      print 'removing bn gamma'
      opt_names += ['gamma']
      var_list = [x for x in var_list if not any(name in x.name for name in opt_names)]
    if ul_only:
      var_list = [x for x in var_list if not x.name.startswith('lb/') and ('global_step' not in x.name)]
    #var_list = [x for x in var_list if ('global_step' not in x.name)]
    print 'Restoring variables:'
    for x in var_list:
      print x.name
    tf.train.Saver(var_list).restore(self.sess, path)
    # print 'TEST: restoring all'
    # tf.train.Saver().restore(self.sess, path)

  def get_step(self):
    return self.sess.run([self.step, self.lr_step])

  def train(self):
    val_hist = {}
    pr = self.pr
    i = 0
    while True:
      step, lr = self.get_step()

      if i > 0 and step % pr.check_iters == 0:
        self.checkpoint_fast()
      if i > 0 and step % pr.slow_check_iters == 0:
        self.checkpoint_slow()

      if step >= pr.train_iters:
        break

      start = ut.now_sec()
      if step % 20 == 0:
        ret = self.sess.run([self.train_op, self.merged_summary] + self.loss.get_losses())
        self.sum_writer.add_summary(ret[1], step)
        loss_vals = ret[2:]
      else:
        loss_vals = self.sess.run([self.train_op] + self.loss.get_losses())[1:]
      ts = moving_avg('time', ut.now_sec() - start, val_hist)

      out = []
      for name, val in zip(self.loss.get_loss_names(), loss_vals):
        out.append('%s: %.3f' % (name, moving_avg(name, val, val_hist)))
      out = ' '.join(out)

      if step < 10 or step % pr.print_iters == 0:
        print 'Iteration %d, lr = %.0e, %s, time: %.3f' % (step, lr, out, ts)
      i += 1

def moving_avg(name, x, vals, avg_win_size = 100, p = 0.99):
  vals[name] = p*vals.get(name, x) + (1 - p)*x
  return vals[name]

def train(pr, gpus, restore = False, restore_opt = True, 
          num_gpus = None, profile = False):
  print pr
  gpus = tfu.set_gpus(gpus)
  with tf.Graph().as_default():
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.InteractiveSession(config = config)
    gpus = gpus[:num_gpus]
    model = Model(pr, sess, gpus, profile = profile)
    model.make_train_model()

    if restore:
      model.restore(restore_opt = restore_opt)
    elif pr.init_path is not None:
      init_ops = []
      if pr.net_type == 'i3d':
        opt_names = ['Adam', 'beta1_power', 'beta2_power', 'Momentum']
        rgb_variable_map = {}
        for variable in tf.global_variables():
          if any(x in variable.name for x in opt_names):
            print 'Skipping:', variable.name
            continue
          if pr.init_from_2d:
            if variable.name.split('/')[0] == 'RGB':
              # if 'moving_mean' in variable.name or 'moving_variance' in variable.name:
              #   continue
              cp_name = (
                variable.name
                .replace('RGB/inception_i3d', 'InceptionV1')
                .replace('Conv3d', 'Conv2d')
                .replace('batch_norm', 'BatchNorm')
                .replace('conv_3d/w', 'weights')
                .replace(':0', ''))
              print 'shape of', variable.name, shape(variable)
              v = tf.get_variable(cp_name, shape(variable)[1:], tf.float32)
              #rgb_variable_map[cp_name] = variable
              rgb_variable_map[cp_name] = v
              n = shape(v, 0)
              init_ops.append(variable.assign(1.0/float(n) * tf.tile(ed(v, 0), (n, 1, 1, 1, 1))))
          else:
            if variable.name.split('/')[0] == 'RGB':
              rgb_variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
        rgb_saver.restore(sess, pr.init_path)
        for x in init_ops:
          print 'Running:', x
          sess.run(x)
      else:
        print 'Restoring from init_path:', pr.init_path
        model.restore(pr.init_path, ul_only = True, restore_opt = False)

    tf.get_default_graph().finalize()

    model.train()

# Example parameters for UCF-101
# def shift_base(name, num_gpus):
#   total_dur = 10.
#   fps = 29.97
#   frame_dur = 1./fps
#   samp_sr = 21000.
#   pr = Params(train_iters = TrainIters,
#               gamma = 0.1,
#               step_size = StepSize,
#               subsample_frames = None,
#               cam = False,
#               base_lr = BaseLR,
#               opt_method = OptMethod, 
#               multipass = False,
#               momentum_rate = 0.9,
#               grad_clip = None,
#               batch_size = int(8*num_gpus),
#               val_batch = 1,
#               resdir = pj('../results/ucf-eval', name),
#               weight_decay = 1e-5,
#               train_list = pj(DataPath, 'ucf-tf-train-v5/tf'),
#               val_list = pj(DataPath, 'ucf-tf-train-v5/tf'),
#               test_list = '/data/efros/owens/ucf-test-files-1',
#               init_path = '../results/nets/shift/net.tf-650000',
#               use_sound = True,
#               im_type = 'jpeg',
#               input_type = 'samples',
#               full_im_dim = 256,
#               crop_im_dim = 224,
#               renorm = True,
#               checkpoint_iters = 1000,
#               dset_seed = None,
#               samp_sr = samp_sr,
#               fps = fps,
#               total_frames = int(total_dur*fps),
#               sampled_frames = int(VidDur*fps),
#               full_samples_len = int(total_dur * samp_sr),
#               samples_per_frame = samp_sr * frame_dur,
#               frame_sample_delta = int(total_dur*fps)/2,
  
#               max_intersection = -1,
#               batch_norm = True,
#               show_videos = False,
#               slow_check_iters = 1000,
#               check_iters = 500,
#               decompress_flow = True,
#               print_iters = 10,

#               fix_frame = False,
#               do_shift = False,
#               use_3d = True,
#               augment_ims = True,
#               augment_audio = True,
#               multi_shift = False,
#               num_dbs = None,
#               num_classes = 101,
#               add_top_block = False,
#               variable_frame_count = True,
#               net_type = 'shift',
#               test_size = 3783,
#               pool_frac = None,
#               bn_last = False,
#               im_split = True,
#               num_splits = 4,
#               use_dropout = False,
#               bn_scale = True,
#               )
#   pr.num_samples = int(pr.samples_per_frame * float(pr.sampled_frames))
#   return pr

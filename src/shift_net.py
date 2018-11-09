 #import numpy as np, tfutil as mu, aolib.util as ut, copy, shift_dset, tensorflow as tf
import numpy as np, tfutil as mu, aolib.util as ut, copy, shift_dset, tensorflow as tf
import tensorflow.contrib.slim as slim

ed = tf.expand_dims
shape = mu.shape
add_n = mu.maybe_add_n
pj = ut.pjoin
cast_float = mu.cast_float
cast_int = mu.cast_int

def read_data(pr, gpus):
  with tf.device('/cpu:0'):
    batch = ut.make_mod(pr.batch_size, len(gpus))

    ims, samples = mu.on_cpu(
      lambda : shift_dset.make_db_reader(
        pr.train_list, pr, batch, ['im', 'samples'],
        num_db_files = pr.num_dbs))
    
    inputs = {'ims' : ims, 'samples' : samples}
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

# need to have a separate 2d/3d arg_scopes for compatibility with tensorflow 1.9
def arg_scope_3d(pr,
              #weight_decay = 1e-5,
              reuse = False, 
              renorm = True,
              train = True,
              center = True):
  scale = ut.hastrue(pr, 'bn_scale')
  print 'bn scale:', scale
  weight_decay = pr.weight_decay
  print 'arg_scope train =', train
  bn_prs = {
    'decay': 0.9997,
    'epsilon': 0.001,
    'updates_collections': slim.ops.GraphKeys.UPDATE_OPS,
    'scale' : scale,
    'center' : center,
    'is_training' : train,
    'renorm' : renorm,
  }
  normalizer_fn = slim.batch_norm
  normalizer_params = copy.copy(bn_prs)
  normalizer_params['renorm'] = False
  with slim.arg_scope([slim.batch_norm], **bn_prs):
    with slim.arg_scope(
      [slim.convolution],
      weights_regularizer = slim.regularizers.l2_regularizer(weight_decay),
      weights_initializer = slim.initializers.variance_scaling_initializer(),
      activation_fn = tf.nn.relu,
      normalizer_fn = normalizer_fn,
      reuse = reuse,
      normalizer_params = normalizer_params) as sc:
      return sc

def arg_scope_2d(pr,
              #weight_decay = 1e-5,
              reuse = False, 
              renorm = True,
              train = True,
              center = True):
  scale = ut.hastrue(pr, 'bn_scale')
  print 'bn scale:', scale
  weight_decay = pr.weight_decay
  print 'arg_scope train =', train
  bn_prs = {
    'decay': 0.9997,
    'epsilon': 0.001,
    'updates_collections': slim.ops.GraphKeys.UPDATE_OPS,
    'scale' : scale,
    'center' : center,
    'is_training' : train,
    'renorm' : renorm,
  }
  normalizer_fn = slim.batch_norm
  normalizer_params = bn_prs
  normalizer_params_2d = copy.deepcopy(normalizer_params)
  normalizer_params_2d['renorm'] = False
  with slim.arg_scope([slim.batch_norm], **bn_prs):
    with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer = slim.regularizers.l2_regularizer(weight_decay),
      weights_initializer = slim.initializers.variance_scaling_initializer(),
      #weights_initializer = tf.contrib.layers.xavier_initializer(),
      activation_fn = tf.nn.relu,
      normalizer_fn = normalizer_fn,
      reuse = reuse,
      normalizer_params = normalizer_params_2d) as sc:
      return sc
 
def conv3d(*args, **kwargs):
  out = slim.convolution(*args, **kwargs)
  print kwargs['scope'], '->', shape(out), 'before:', shape(args[0])
  if 0:
    out = tf.Print(out, [kwargs['scope'],
                         tf.reduce_mean(out, [0,1,2,3]),
                         tf.nn.moments(out, axes = [0,1,2,3])], summarize = 20)
  return out

def conv2d(*args, **kwargs):
  out = slim.conv2d(*args, **kwargs)
  print kwargs['scope'], '->', shape(out)
  return out

def make_opt(opt_method, lr_val, pr):
  if opt_method == 'adam':
    opt = tf.train.AdamOptimizer(lr_val)
  elif opt_method == 'momentum':
    opt = tf.train.MomentumOptimizer(lr_val, pr.momentum_rate)
  else:
    raise RuntimeError()
  
  if pr.multipass:
    import multi_pass_optimizer
    opt = multi_pass_optimizer.MultiPassOptimizer(opt, pr.multipass_count)
  return opt

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

def sigmoid_loss(logits, labels):
  loss = tf.nn.sigmoid_cross_entropy_with_logits(
    logits = logits, labels = ed(cast_float(labels), 1))
  acc = tf.reduce_mean(mu.cast_float(tf.equal(cast_int(logits > 0), ed(labels, 1))))
  acc = tf.stop_gradient(acc)
  acc.ignore = True
  loss = tf.reduce_mean(loss)
  return loss, acc

def slim_losses_with_prefix(prefix, show = True):
  losses = tf.losses.get_regularization_losses()
  losses = [x for x in losses if prefix is None or x.name.startswith(prefix)]
  if show:
    print 'Collecting losses for prefix %s:' % prefix
    for x in losses:
      print x.name
    print
  return mu.maybe_add_n(losses)

class Model:
  def __init__(self, pr, sess, gpus, is_training = True, pr_test = None):
    self.pr = pr
    self.sess = sess
    self.gpus = gpus
    self.default_gpu = gpus[0]
    self.is_training = is_training
    self.pr_test = pr if pr_test is None else pr_test

  def make_model(self):
    with tf.device(self.default_gpu):
      pr = self.pr
      # steps
      self.step = tf.get_variable(
        'global_step', [], trainable = False,
        initializer = tf.constant_initializer(0), dtype = tf.int64)
      self.lr = tf.constant(pr.base_lr)

      # model
      opt = make_opt(pr.opt_method, pr.base_lr, pr)
      self.inputs = read_data(pr, self.gpus)

      gpu_grads, gpu_losses = {}, {}
      for i, gpu in enumerate(self.gpus):
        with tf.device(gpu):
          reuse = (i > 0) 
          with tf.device('/cpu:0'):
            ims = self.inputs[i]['ims']
            samples_ex = self.inputs[i]['samples']
            assert pr.both_examples
            assert not pr.small_augment
            labels = tf.random_uniform(
              [shape(ims, 0)], 0, 2, dtype = tf.int64, name = 'labels_sample')
            samples0 = tf.where(tf.equal(labels, 1), samples_ex[:, 1], samples_ex[:, 0])
            samples1 = tf.where(tf.equal(labels, 0), samples_ex[:, 1], samples_ex[:, 0])
            labels1 = 1 - labels

          net0 = make_net(ims, samples0, pr, reuse = reuse, train = self.is_training)
          net1 = make_net(None, samples1, pr, im_net = net0.im_net, reuse = True, train = self.is_training)
          labels = tf.concat([labels, labels1], 0)
          net = ut.Struct(
            logits = tf.concat([net0.logits, net1.logits], 0),
            cam = tf.concat([net0.cam, net1.cam], 0),
            last_conv = tf.concat([net0.last_conv, net1.last_conv], 0))
          
          loss = mu.Loss('loss')
          loss.add_loss(slim_losses_with_prefix(None), 'reg')
          loss.add_loss_acc(sigmoid_loss(net.logits, labels), 'label')
          grads = opt.compute_gradients(loss.total_loss())

          ut.add_dict_list(gpu_grads, loss.name, grads)
          ut.add_dict_list(gpu_losses, loss.name, loss)
          #self.loss = loss

          if i == 0:
            self.net = net

      self.loss = mu.merge_losses(gpu_losses['loss'])
      for name, val in zip(self.loss.get_loss_names(), self.loss.get_losses()):
        tf.summary.scalar(name, val)

      if not self.is_training:
        #pr_test = pr.copy()
        pr_test = self.pr_test.copy()
        pr_test.augment_ims = False
        print 'pr_test ='
        print pr_test

        self.test_ims, self.test_samples, self.test_ytids = mu.on_cpu(
          lambda : shift_dset.make_db_reader(
            pr_test.test_list, pr_test, pr.test_batch, ['im', 'samples', 'ytid'], one_pass = True))

        if pr_test.do_shift:
          self.test_labels = tf.random_uniform([shape(self.test_ims, 0)], 0, 2, dtype = tf.int64)
          self.test_samples = tf.where(tf.equal(self.test_labels, 1), self.test_samples[:, 1], self.test_samples[:, 0])
        else:
          self.test_labels = tf.ones(shape(self.test_ims, 0), dtype = tf.int64)
          #self.test_samples = tf.where(tf.equal(self.test_labels, 1), self.test_samples[:, 1], self.test_samples[:, 0])
          print 'sample shape:', shape(self.test_samples)

        self.test_net = make_net(self.test_ims, self.test_samples, pr_test, reuse = True, train = self.is_training)

      (gs, vs) = zip(*mu.average_grads(gpu_grads['loss']))
      if pr.grad_clip is not None:
        gs, _ = tf.clip_by_global_norm(gs, pr.grad_clip)
      gs = [mu.print_every(gs[0], 100, ['grad norm:', tf.global_norm(gs)])] + list(gs[1:])
      gvs = zip(gs, vs)

      bn_ups = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      if pr.multipass:
        ops = [opt.apply_gradients(gvs, global_step = self.step) for i in xrange(pr.multipass_count)]
        def op_helper(count = [0]):
          op = ops[count[0] % len(ops)]
          count[0] += 1
          return op
        self.train_op = op_helper
      else:
        op = tf.group(opt.apply_gradients(gvs, global_step = self.step), *bn_ups)
        self.train_op = lambda : op
      
      self.coord = tf.train.Coordinator()
      self.saver = tf.train.Saver()

      self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
      self.sess.run(self.init_op)
      tf.train.start_queue_runners(sess = self.sess, coord = self.coord)

      self.merged_summary = tf.summary.merge_all()
      print 'Tensorboard command:'
      summary_dir = ut.mkdir(pj(pr.summary_dir, ut.simple_timestamp()))
      print 'tensorboard --logdir=%s' % summary_dir
      self.sum_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

  def checkpoint(self):
    check_path = pj(ut.mkdir(self.pr.train_dir), 'net.tf')
    out = self.saver.save(self.sess, check_path, global_step = self.step)
    print 'Checkpoint:', out

  def restore(self, path = None, restore_opt = True, restore_resnet18_blocks = True, restore_dilation_blocks = True):
    if path is None:
      path = tf.train.latest_checkpoint(self.pr.train_dir)      
    print 'Restoring from:', path
    var_list = slim.get_variables_to_restore()
    opt_names = ['Adam', 'beta1_power', 'beta2_power', 'Momentum', 'cache']
    if not restore_resnet18_blocks:
      opt_names += ['conv2_2_', 'conv3_2_', 'conv4_2_', 'conv5_2_']

    if not restore_opt:
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
    first = True
    while True:
      step, lr = self.get_step()

      if not first and step % pr.check_iters == 0:
        self.checkpoint()

      start = ut.now_sec()
      if step % pr.summary_iters == 0:
        ret = self.sess.run([self.train_op(), self.merged_summary] + self.loss.get_losses())
        self.sum_writer.add_summary(ret[1], step)
        loss_vals = ret[2:]
      else:
        loss_vals = self.sess.run([self.train_op()] + self.loss.get_losses())[1:]
      ts = moving_avg('time', ut.now_sec() - start, val_hist)

      out = []
      for name, val in zip(self.loss.get_loss_names(), loss_vals):
        out.append('%s: %.3f' % (name, moving_avg(name, val, val_hist)))
      out = ' '.join(out)

      if step < 10 or step % pr.print_iters == 0:
        print 'Iteration %d, lr = %.0e, %s, time: %.3f' % (step, lr, out, ts)

      first = False

def moving_avg(name, x, vals, avg_win_size = 100, p = 0.99):
  vals[name] = p*vals.get(name, x) + (1 - p)*x
  return vals[name]

def train(pr, gpus, restore = False, restore_opt = True):
  print pr
  gpus = mu.set_gpus(gpus)
  with tf.Graph().as_default():
    config = tf.ConfigProto(allow_soft_placement = True)
    #config = tf.ConfigProto()
    sess = tf.InteractiveSession(config = config)
    model = Model(pr, sess, gpus)
    model.make_model()

    if restore:
      model.restore(restore_opt = restore_opt)
    elif pr.init_path is not None:
      model.restore(pr.init_path, restore_resnet18_blocks = False, restore_opt = False)

    tf.get_default_graph().finalize()
    model.train()

def block2(net, nf, dim, scope, stride = 1, reuse = None):
  short = net
  if stride != 1 and shape(net, -1) == nf:
    short = slim.max_pool2d(short, [1, 1], stride)
  elif stride != 1:
    short = conv2d(net, nf, [1, 1], scope = '%s_short' % scope, 
                   stride = stride, activation_fn = None)
    
  net = conv2d(net, nf, dim, scope = '%s_1' % scope, stride = stride)
  net = conv2d(net, nf, dim, scope = '%s_2' % scope, 
               activation_fn = None, normalizer_fn = None)
  net = short + net
  net = slim.batch_norm(
    net, scope = '%s_bn' % scope, 
    activation_fn = tf.nn.relu, reuse = reuse)
  return net

def block3(net, nf, dim, scope, stride = 1, rate = 1, reuse = None, bottleneck = False, use_bn = True):
  short = net
  if stride != 1 and shape(net, -1) == nf:
    short = pool3d(short, [1, 1, 1], stride)
  elif stride != 1 or shape(net, -1) != nf:
    short = conv3d(net, nf, [1, 1, 1], scope = '%s_short' % scope, 
                   stride = stride, activation_fn = None)

  if bottleneck:
    net = conv3d(net, nf, [1, 1, 1], scope = '%s_bottleneck' % scope, rate = rate)
  net = conv3d(net, nf, dim, scope = '%s_1' % scope, rate = rate, stride = stride)
  net = conv3d(net, nf, dim, scope = '%s_2' % scope, activation_fn = None, 
               normalizer_fn = None, rate = rate)
  net = short + net
  if use_bn:
    net = slim.batch_norm(net, scope = '%s_bn' % scope, 
                          activation_fn = tf.nn.relu, reuse = reuse)
  else:
    net = tf.nn.relu(net)

  return net

def normalize_sfs(sfs, scale = 255.):
  return tf.sign(sfs)*(tf.log(1 + scale*tf.abs(sfs)) / tf.log(1 + scale))

def my_fractional_pool(net, frac, n):
  net = tf.nn.max_pool(net, [1, int(round(frac)), 1, 1], [1, 1, 1, 1], 'SAME')
  #return tf.image.resize_nearest_neighbor(net, [n, tf.shape(net)[1]])
  print 'shape =', shape(net)
  s = shape(net)
  net = tf.image.resize_nearest_neighbor(net, [n, 1])
  print 'shape after resize =', shape(net)
  net.set_shape((s[0], None, s[2], s[3]))
  return net


def merge(sf_net, im_net, pr, reuse = None, train = True):
  # fuse image and sound feature nets
  s0 = shape(sf_net)
  print 'frac:', float(shape(sf_net, 1)-1)/shape(im_net, 1)
  if train:
    sf_net = tf.nn.fractional_max_pool(sf_net, [1, float(shape(sf_net, 1)-1)/shape(im_net, 1), 1, 1])[0]
  else:
    # deterministic (note: we didn't do this for the experiments in the paper -- only for the release)
    sf_net = tf.nn.fractional_max_pool(
      sf_net, [1, float(shape(sf_net, 1)-1)/shape(im_net, 1), 1, 1],
      deterministic = (not train), pseudo_random = (not train), 
      seed = (0 if train else 1), seed2 = (0 if train else 2))[0]

  s1 = shape(sf_net)
  with slim.arg_scope(arg_scope_2d(pr = pr, reuse = reuse, train = train)):
    sf_net = conv2d(sf_net, 128, [3, shape(sf_net, 2)], scope = 'sf/conv5_1')
  with slim.arg_scope(arg_scope_3d(pr = pr, reuse = reuse, train = train)):
    sf_net = sf_net[:, :, shape(sf_net, 2)/2]
    sf_net = ed(ed(sf_net, 2), 2)
    sf_net = tf.tile(sf_net, [1, 1, shape(im_net, 2), shape(im_net, 3), 1])
    if not pr.use_sound:
      print 'Not using sound!'
      sf_net = tf.zeros_like(sf_net)
    net = tf.concat([im_net, sf_net], 4)
    print 'sf_net shape before merge: %s, and after merge: %s' % (s0, s1)

    short = tf.concat([net[..., :64], net[..., -64:]], 4)
    net = conv3d(net, 512, [1, 1, 1], scope = 'im/merge1')
    net = conv3d(net, 128, [1, 1, 1], scope = 'im/merge2', 
                 normalizer_fn = None, activation_fn = None)
    net = slim.batch_norm(net + short, scope = 'im/%s_bn' % 'merge_block', 
                          activation_fn = tf.nn.relu, reuse = reuse)

  return net

def make_net(ims, sfs, pr, im_net = None, reuse = True, train = True):
  if pr.subsample_frames and ims is not None:
    ims = ims[:, ::pr.subsample_frames]

  im_scales = []
  scales = []
  with slim.arg_scope(arg_scope_2d(pr = pr, reuse = reuse, train = train)):
    # sound feature subnetwork
    sf_net = normalize_sfs(sfs)
    sf_net = ed(sf_net, 2)
    sf_net = conv2d(sf_net, 64, [65, 1], scope = 'sf/conv1_1', stride = 4, reuse = reuse)
    sf_net = slim.max_pool2d(sf_net, [4, 1], [4, 1])

    sf_net = block2(sf_net, 128, [15, 1], 'sf/conv2_1', stride = [4, 1], reuse = reuse)
    sf_net = block2(sf_net, 128, [15, 1], 'sf/conv3_1', stride = [4, 1], reuse = reuse)
    sf_net = block2(sf_net, 256, [15, 1], 'sf/conv4_1', stride = [4, 1], reuse = reuse)

  with slim.arg_scope(arg_scope_3d(pr = pr, reuse = reuse, train = train)):
    if im_net is None:
      im_net = mu.normalize_ims(ims)
      im_net = conv3d(im_net, 64, [5, 7, 7], scope = 'im/conv1', stride = 2)
      im_net = pool3d(im_net, [1, 3, 3], [1, 2, 2])

      im_net = block3(im_net, 64, [3, 3, 3], 'im/conv2_1', stride = 1, reuse = reuse)
      im_net = block3(im_net, 64, [3, 3, 3], 'im/conv2_2', stride = 2, reuse = reuse)
      scales.append(im_net)
      im_scales.append(im_net)

  net = merge(sf_net, im_net, pr, reuse, train = train)

  with slim.arg_scope(arg_scope_3d(pr = pr, reuse = reuse, train = train)):
    net = block3(net, 128, [3, 3, 3], 'im/conv3_1', stride = 1, reuse = reuse)
    net = block3(net, 128, [3, 3, 3], 'im/conv3_2', stride = 1, reuse = reuse)
    scales.append(net)
    im_scales.append(net)

    #s = (2 if ut.hastrue(pr, 'cam2') else 1)
    s = 2
    net = block3(net, 256, [3, 3, 3], 'im/conv4_1', stride = [2, s, s], reuse = reuse)
    net = block3(net, 256, [3, 3, 3], 'im/conv4_2', stride = 1, reuse = reuse)
    im_scales.append(net)

    time_stride = (2 if ut.hastrue(pr, 'extra_stride') else 1)
    print 'time_stride =', time_stride
    s = (1 if pr.cam else 2)
    net = block3(net, 512, [3, 3, 3], 'im/conv5_1', stride = [time_stride, s, s], reuse = reuse)
    net = block3(net, 512, [3, 3, 3], 'im/conv5_2', stride = 1, reuse = reuse)
    scales.append(net)
    im_scales.append(net)

    last_conv = net
    net = tf.reduce_mean(net, [1, 2, 3], keep_dims = True)

    logits_name = 'joint/logits'
    logits = conv3d(
      net, 1, [1, 1, 1], scope = logits_name,
      activation_fn = None, normalizer_fn = None)[:, 0, 0, 0, :]
    cam = conv3d(
      last_conv, 1, [1, 1, 1], scope = logits_name,
      activation_fn = None, normalizer_fn = None, reuse = True)

    return ut.Struct(logits = logits, cam = cam, last_conv = last_conv, 
                     im_net = im_net, scales = scales, im_scales = im_scales)
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
        self.ims_resize_ph = tf.placeholder(
          tf.uint8, [1, pr.sampled_frames, None, None, 3])
        self.samples_ph = tf.placeholder(tf.float32, (1, pr.num_samples, 2))
        self.net = make_net(self.ims_ph, self.samples_ph, pr, reuse = False, train = False)
        ims_resize = self.ims_resize_ph
        ims_resize = ed(tf.image.resize_images(ims_resize[0], (pr.crop_im_dim, pr.crop_im_dim)), 0)
        ims_resize.set_shape(shape(self.ims_ph))
        self.net_resize = make_net(ims_resize, self.samples_ph, pr, reuse = True, train = False)
        self.sess.run(tf.global_variables_initializer())
        tf.train.Saver().restore(self.sess, self.model_path)
        tf.get_default_graph().finalize()
    return self

  def predict_cam(self, ims, samples):
    self.init()
    cam = self.sess.run([self.net.cam], {self.ims_ph : ims, self.samples_ph : samples})
    return cam

  def predict_cam_resize(self, ims, samples):
    self.init()
    cam = self.sess.run([self.net_resize.cam], {self.ims_resize_ph : ims, self.samples_ph : samples})
    return cam


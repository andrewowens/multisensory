import tensorflow as tf, aolib.util as ut, numpy as np, os, aolib.img as ig, time, sys
import tensorflow.contrib.slim as slim

ab = os.path.abspath
pj = ut.pjoin
rs = tf.reshape

def shape(x, d = None):
  s = x.get_shape().as_list()
  return s if d is None else s[d]

def dropout_prob(pr): return (0.5 if pr.dropout else 1.)

def weight_decay(pr): return slim.l2_regularizer(pr.weight_decay)

def layer_params(pr, train = True, reuse = None):
  if pr.batch_norm:
    bn_fn = slim.batch_norm
    bn_pr = {'is_training': train, 'trainable' : train, 'decay': 0.9997, 'updates_collections': None}
    # print 'TEST!'
    # bn_pr = {'is_training': train, 'trainable' : train, 'decay': 0.9997}
  else:
    bn_fn = None
    bn_pr = None
    
  if pr.init_method == 'xavier':
    init_conv = tf.contrib.layers.xavier_initializer_conv2d()
    init_fc = tf.contrib.layers.xavier_initializer()
  elif pr.init_method == 'gaussian':
    init_conv = tf.truncated_normal_initializer(stddev = 0.01)
    init_fc = tf.truncated_normal_initializer(stddev = 0.01)
    
  weights_regularizer = weight_decay(pr)

  #conv2d_scope = slim.arg_scope([slim.conv2d],
  conv2d_scope = slim.arg_scope([slim.conv2d, slim.convolution],
                                weights_regularizer = weights_regularizer, \
                                normalizer_fn = bn_fn,
                                normalizer_params = bn_pr,
                                activation_fn = tf.nn.relu,
                                weights_initializer = init_conv, \
                                trainable = train,
                                #restore = not train,
                                biases_initializer = tf.constant_initializer(0.1),
                                reuse = reuse)
  dropout_scope = slim.arg_scope([slim.dropout],
                                 keep_prob = dropout_prob(pr),
                                 is_training = train)
  fc_scope = slim.arg_scope([slim.fully_connected],
                            activation_fn = tf.nn.relu,
                            weights_regularizer = weights_regularizer,
                            weights_initializer = init_fc,
                            biases_initializer = tf.constant_initializer(0.1),
                            trainable = train,
                            reuse = reuse)
  return conv2d_scope, dropout_scope, fc_scope



def average_grads(tower_grads):
  average_grads = []
  for ii, grad_and_vars in enumerate(zip(*tower_grads)):
    grads = []
    #print ii, len(grad_and_vars)
    for g, v in grad_and_vars:
      #print g, v.name
      if g is None:
        print 'skipping', v.name
        continue
      else:
        print 'averaging', v.name
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    if len(grads) == 0:
      #print 'no grads for', v.name
      grad = None
    else:
      #grad = tf.concat_v2(grads, 0)
      grad = tf.concat(grads, 0)
      #grad = mean_vals(grad, 0)
      grad = tf.concat(grads, 0)
      grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

# def mean_vals(x, a = None):
#   if a is None:
#     return tf.div(tf.reduce_sum(x), tf.cast(tf.size(x), tf.float32))
#   else:
#     return tf.div(tf.reduce_sum(x, a), tf.cast(tf.shape(x)[a], tf.float32))

def logsumexp(a, axis=None, b=None, keep_dims = False):
  a_max = tf.reduce_max(a, reduction_indices = axis, keep_dims = True)
  if b is not None:
    tmp = b * tf.exp(a - a_max)
  else:
    tmp = tf.exp(a - a_max)
  # suppress warnings about log of zero
  with np.errstate(divide='ignore'):
    out = tf.log(tf.reduce_sum(tmp, reduction_indices = axis, keep_dims = keep_dims))
  if not keep_dims:
    a_max = tf.squeeze(a_max, squeeze_dims = [axis])
  return out + a_max

def multi_softmax(target, axis, name=None):
  # https://gist.github.com/raingo/a5808fe356b8da031837
  with tf.op_scope([target], name, 'softmax'):
    max_axis = tf.reduce_max(target, axis, keep_dims=True)
    target_exp = tf.exp(target-max_axis)
    normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
    softmax = target_exp / normalize
    return softmax

def mdn_loss_indep(mix, mu, log_sigma_sq, x, b, nc, k, nx, smooth = None, mean_loss = True, mean_dim = False):
  cns = tf.constant
  const = tf.to_float(cns(-0.5 * np.log(2*np.pi)))
  C = const + (-0.5) * log_sigma_sq
  sigma_sq = tf.exp(log_sigma_sq)
  
  xt = tf.tile(rs(x, (b, nx, 1, k)), (1, 1, nc, 1))
  mut = tf.tile(rs(mu, (b, 1, nc, k)), (1, nx, 1, 1))
  sigma_sqt = tf.tile(rs(sigma_sq, (b, 1, nc, k)), (1, nx, 1, 1))
  Ct = tf.tile(rs(C, (b, 1, nc, k)), (1, nx, 1, 1))
  mixt = tf.tile(rs(mix, (b, 1, nc, k)), (1, nx, 1, 1))

  diff = xt - mut
  dist = cns(-0.5) * tf.div(diff**2, sigma_sqt)
  #log_mix_prob = Ct + dist
  log_mix_prob = dist
  log_probs = logsumexp(log_mix_prob, b = mixt*tf.exp(Ct), axis = 2)  
  # log_probs = tf.Print(log_probs, ['x =', x, 'Ct = ', Ct, 'dist =', dist, 'mix = ', mix, 'mu = ', mu,
  #                                  'lsq = ', log_sigma_sq, 'log_mix_prob =', log_mix_prob,
  #                                  'mix sum = ', tf.reduce_sum(mix, reduction_indices = 1),
  #                                  'log probs = ', log_probs, tf.shape(log_probs)], summarize = 10)
  print 'log_probs shape =', shape(log_probs)
  combine_dims = (lambda x : tf.reduce_mean(x, 2)) \
                 if mean_dim else (lambda x : tf.reduce_sum(x, 2))
  if mean_loss:
    return -tf.reduce_mean(combine_dims(log_probs))
  else:
    return -tf.reduce_mean(combine_dims(log_probs), 1)
  
  #return -tf.reduce_mean(tf.reduce_mean(log_probs, reduction_indices = 2))

def make_mdn_indep(fc, conv_target, pr, scope, use_gpu = True, mean_loss = True, mean_dim = False):
  scope = scope + '_'
  tni = tf.truncated_normal_initializer
  batch_size, time_dim, w, dim = [x.value for x in conv_target.get_shape()]
  if w > 1:
    conv_target = tf.reshape(conv_target, (batch_size, time_dim*w, 1, dim))
                             
  batch_size, time_dim, w, dim = [x.value for x in conv_target.get_shape()]
  assert w == 1
  nc = pr.num_components

  mix = slim.fully_connected(fc, nc*dim, activation_fn = None,
                             scope = scope + 'mix',
                             biases_initializer = tni(stddev = 0.1),
                             weights_regularizer = weight_decay(pr))
  mix = tf.reshape(mix, (batch_size, nc, dim))
  mix = multi_softmax(mix, axis = 1)

  if pr.mix_only:
    print 'Not doing mixture regression'
    fc_regress = tf.zeros(fc.get_shape(), dtype = 'float32')
  else:
    fc_regress = fc

  if pr.use_stdev:
    log_sigma_sq = slim.fully_connected(fc_regress, nc*dim, activation_fn = None, scope = scope + 'log_sigma_sq',
                                        biases_initializer = tni(mean = np.log(1.**2), stddev = 0.1),
                                        weights_regularizer = weight_decay(pr))
    log_sigma_sq = tf.maximum(log_sigma_sq, np.log(0.001))
    log_sigma_sq = tf.reshape(log_sigma_sq, (batch_size, nc, dim))
  else:
    print 'not using stdev'
    log_sigma_sq = tf.log(tf.ones((batch_size, nc, dim), dtype = tf.float32)**2)

  if pr.mix_only:
    mu = tf.get_variable(scope + 'mu', [batch_size, nc*dim],
                         initializer = tni(mean = 0., stddev = 0.1))
  else:
    mu = slim.fully_connected(fc_regress, nc*dim, scope = scope + 'mu', activation_fn = None,
                              biases_initializer = tni(mean = 0., stddev = 0.1),
                              weights_regularizer = weight_decay(pr))
  mu = tf.reshape(mu, (batch_size, nc, dim))

  xs = conv_target[:, :, 0, :]

  if use_gpu:
    return mdn_loss_indep(mix, mu, log_sigma_sq, xs, batch_size, nc, dim, time_dim, mean_loss = mean_loss, mean_dim = mean_dim)
  else:
    with tf.device('/cpu:0'):
      return mdn_loss_indep(mix, mu, log_sigma_sq, xs, batch_size, nc, dim, time_dim, mean_loss = mean_loss, mean_dim = mean_dim)

def gpu_mask(gpus):
  # if gpus is None or len(gpus) == 0:
  #   return ''
  # else:
  if gpus is None:
    gpus = []
  return 'export CUDA_VISIBLE_DEVICES=' + ','.join(map(str, gpus))

def maybe_add_n(xs):
  return tf.add_n(xs) if len(xs) else tf.constant(0., dtype = tf.float32)

class Loss:
  def __init__(self, base_name = '', prefix = None):
    self.losses = []
    self.loss_names = []
    self.total_loss_ = None
    #self.ema = tf.train.ExponentialMovingAverage(decay=0.99)
    self.prefix = prefix
    # todo
    self.base_name = base_name
    self.name = base_name

  def add_loss(self, x, name, summary = False):
    self.losses.append(x)
    #self.losses_smoothed.append(self.ema.apply
    self.loss_names.append(name)

  def add_loss_acc(self, (loss, acc), base_name, summary = False):
    acc = tf.stop_gradient(acc)
    acc.ignore = True
    self.add_loss(loss, 'loss:%s' % base_name)
    self.add_loss(acc, 'acc:%s' % base_name)

  def total_loss(self):
    if self.total_loss_ is None:
      #print 'Adding:', ' '.join([x.name for x in self.losses if not ut.haseq(x, 'ignore', True)])
      #print 'Skipping:', ' '.join([x.name for x in self.losses if ut.haseq(x, 'ignore', True)])
      print 'adding:'
      for x in [x for x in self.losses if not ut.haseq(x, 'ignore', True)]:
        print x
      self.total_loss_ = maybe_add_n([x for x in self.losses if not ut.haseq(x, 'ignore', True)])
    # with tf.device('/cpu:0'):
    #   self.total_loss_  = tf.Print(self.total_loss_, ['loss0 =', self.losses[0]])
    return self.total_loss_

  def get_losses(self):
    return [self.total_loss()] + self.losses      

  def get_loss_names(self):
    #names = ['loss:' + self.base_name] + self.loss_names
    total_name = 'loss' if self.base_name == '' else ('total:%s' % self.base_name)
    names = [total_name] + self.loss_names
    assert len(set(names)) == len(names), 'repeated names'
    return names

  def clear(self):
    self.loss_names = []
    self.losses = []

def merge_losses(losses):
  loss0 = losses[0]
  merged = Loss(loss0.base_name, loss0.prefix)
  for i in xrange(len(loss0.losses)):
    name = loss0.loss_names[i]
    assert all(name == x.loss_names[i] for x in losses)
    merged.loss_names.append(name)
    merged.losses.append(tf.reduce_mean([x.losses[i] for x in losses]))
  return merged

def clip(x, min_, max_):
  return tf.maximum(tf.minimum(x, max_), min_)

def print_minmax(name, x):
  return tf.Print(x, [name, 'min =', tf.reduce_min(x), 'max =', tf.reduce_max(x)])

def find_model_file(pr):
  print pr.train_dir
  return tf.train.latest_checkpoint(pr.train_dir)

def get_step():
  print 'vars =', [x.name for x in tf.global_variables()]
  step = ut.find_first(lambda x : x.name.startswith('global_step:'),
                       tf.global_variables())
  return step

# def set_gpus(gpus):
#   if gpus is not None:
#     if np.ndim(gpus) == 0:
#       gpus = [gpus]
#     os.putenv('CUDA_VISIBLE_DEVICES', ','.join(map(str, gpus)))

def set_gpus(gpus):
  if gpus is None or gpus == [None]:
    os.putenv('CUDA_VISIBLE_DEVICES', '')
    return ['/cpu:0']
  else:
    if np.ndim(gpus) == 0:
      gpus = [gpus]
    os.putenv('CUDA_VISIBLE_DEVICES', ','.join(map(str, gpus)))
    gpus = range(len(gpus))
    return gpu_strs(gpus)

def normalize_ims(im):
  #return -1. + (1./128) * (tf.cast(im, tf.float32))
  if type(im) == type(np.array([])):
    im = im.astype('float32')
  else:
    im = tf.cast(im, tf.float32)
  return -1. + (2./255) * im 

def unnormalize_ims(im):
  #return -1. + (1./128) * (tf.cast(im, tf.float32))
  # return -0.5 + (1./128) * (tf.cast(im, tf.float32))
  if type(im) == type(np.array([])):
    im = im.astype('float32')
  else:
    im = tf.cast(im, tf.float32)
  #return (im+0.5)*128
  return (im + 1.) * (255./2)

def normalize_sfs(sfs, pr, normalize_rms = True):
  if pr.sf_type == 'specgram':
    return tf.expand_dims(sfs, 3)
  elif pr.sf_type == 'combo':
    return (tf.expand_dims(sfs[0], 3), normalize_ims(sfs[1]))
  else:
    sfs = tf.cast(sfs, 'float32') / np.iinfo(np.dtype(np.int16)).max
    if normalize_rms:
      rms = 1e-5 + tf.sqrt(tf.reduce_mean(sfs**2, reduction_indices = 1))
      desired_rms = 0.01
      sfs *= desired_rms / tf.expand_dims(rms, 1)
    sfs = tf.expand_dims(tf.expand_dims(sfs, 2), 3)
  return sfs

def gpu_strs(gpus):
  if gpus is not None and np.ndim(gpus) == 0:
    gpus = [gpus]
  return ['/cpu:0'] if gpus is None else ['/gpu:%d' % x for x in gpus]

# def print_every(ret, num_iters, vals, **kwargs):
#   step = get_step()
#   if step is not None:
#     return tf.cond(tf.equal(tf.mod(step, num_iters), 0),
#                    lambda : tf.Print(ret, vals, **kwargs), lambda : ret)
#   else:
#     return ret

# def print_every(ret, num_iters, vals, **kwargs):
#   #with tf.device('/cpu:0'):
#   step = get_step()
#   if step is not None:
#     return tf.cond(tf.equal(tf.mod(step, num_iters), 0),
#                    lambda : tf.Print(ret, vals, **kwargs), lambda : ret)
#   else:
#     return ret

def print_every(ret, num_iters, vals, **kwargs):
  with tf.device('/cpu:0'):
    step = get_step()
    if step is not None:
      return tf.cond(tf.equal(tf.mod(step, num_iters), 0),
                   lambda : tf.Print(ret, vals, **kwargs), lambda : ret)
    else:
      return ret

# def run_every(num_iters, fn, args):
#   step = get_step()
#   if step is not None:
#     return tf.cond(tf.equal(tf.mod(step, num_iters), 0),
#                    lambda : ident_fn(fn, *args), lambda : args[0])
#   else:
#     return args[0]

def run_every(num_iters, fn, args):
  step = get_step()
  if step is not None:
    return tf.cond(tf.equal(tf.mod(step, num_iters), 0),
                   lambda : ident_fn(fn, *args), lambda : args[0])
  else:
    return args[0]

def expand_many(x, dims):
  for i in xrange(len(dims)):
    x = tf.expand_dims(x, dims[i])
  return x

def normalize_flows(flows, pr):
  # consider subtracting mean flow, or doing more complex stabilization here
  # (should be moved to the CPU/data-loading code if the latter)
  #return flows
  if pr.flow_norm_type == 'normalize':
    flows = flows - expand_many(tf.reduce_mean(flows, (1, 2)), (1, 1))
    stdev = tf.reduce_mean(tf.sqrt(tf.reduce_sum(1e-2 + flows**2, 3)), (1, 2))
    flows = flows / expand_many(tf.expand_dims(stdev, 1), (1, 1))
  return flows

def ident_fn(fn, *args):
  def wrapper(*args):
    fn(*args)
    return args[0]
  x = args[0]
  s = shape(x)
  x = tf.py_func(wrapper, args, x.dtype)
  x.set_shape(s)
  return x
  
def batch_split_n(xs, n):
  n = int(n)
  assert n > 0
  i = 0
  while i < shape(xs, 1):
    yield xs[:, i : i + n]
    i += n

def norm_pix(im):
  norm = tf.expand_dims(tf.sqrt(tf.reduce_sum(1e-4 + im**2, 3)), 3)
  return im / norm, norm

def normalize_and_append(im):
  return tf.concat(list(norm_pix(im)), 3)

def center_crop(im, full_dim, crop_dim):
  im = ig.rgb_from_gray(im, remove_alpha = True)
  im = ig.scale(im, [full_dim]*2, 1)
  im = ut.crop_center(im, crop_dim)
  return im

class Params(ut.Struct):
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

  @property
  def train_dir(self):
    return ut.mkdir(pj(self.resdir, 'training'))

  @property
  def summary_dir(self):
    return ut.mkdir(pj(self.resdir, 'summary'))

  @property
  def name(self):
    return ut.add_slash(self.resdir).split('/')[-2]

def find_lr_np(pr, step):
  if hasattr(pr, 'lr_schedule'):
    for rng, lr in pr.lr_schedule:
      if step in rng:
        return lr
  else:
    gamma = 0.1 if not hasattr(pr, 'gamma') else pr.gamma
    return pr.base_lr * gamma**np.floor(step / pr.step_size)

def find_lr(pr, step):
  assert not hasattr(pr, 'lr_schedule')
  gamma = 0.1 if not hasattr(pr, 'gamma') else pr.gamma
  scale = gamma ** tf.floor(cast_float(step) / float(pr.step_size))
  return scale * pr.base_lr
            
def normalize(x, eps = 1e-6):
  return x / tf.expand_dims(tf.sqrt(tf.reduce_sum(eps + x**2, 1)), 1)

def find_var(name):
  return ut.find_first(lambda x : x.name.startswith('%s:' % name),
                       tf.global_variables())



def my_softmax_cross_entropy_loss(logits, targets):
  assert shape(logits) == shape(targets)
  loss = -tf.reduce_sum(logits * targets, 1) + tf.reduce_logsumexp(logits, 1)
  #loss = tf.Print(loss, [loss, tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = targets)])
  return loss

#UseSummary = True
UseSummary = False

def scalar_summary(name, val):
  if UseSummary:
    tf.summary.scalar(name, val)

def cast_float(x): return tf.cast(x, tf.float32)
def cast_int(x): return tf.cast(x, tf.int64)
def cast_complex(x): return tf.cast(x, tf.complex64)

def on_cpu(f, gpu = '/cpu:0'):
  with tf.device(gpu):
    return f()

def normalize_l2(x, eps = 1e-6):
  return x / (eps + tf.expand_dims(tf.sqrt(tf.reduce_sum(x**2, 1)), 1))

def normalize_l1(x, eps = 1e-6):
  return x / (eps + tf.expand_dims(tf.reduce_sum(tf.abs(x), 1), 1))

def label_loss(logits, labels, smooth = False, smooth_prob = 0.05):
  if smooth:
    nc = shape(logits, 1)
    oh = tf.one_hot(labels, nc)
    oh = smooth_prob*(1./nc) + (1 - smooth_prob) * oh
    loss = tf.nn.softmax_cross_entropy_with_logits(
      logits = logits, labels = oh)
  else:
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits = logits, labels = labels)
  acc = tf.reduce_mean(cast_float(tf.equal(tf.argmax(logits, 1), labels)))
  acc = tf.stop_gradient(acc)
  acc.ignore = True
  loss = tf.reduce_mean(loss)
  return loss, acc

def make_opt(opt_method, lr_val):
  if opt_method == 'adam':
    return tf.train.AdamOptimizer(lr_val)
  elif opt_method == 'momentum':
    return tf.train.MomentumOptimizer(lr_val, 0.9)
  else:
    raise RuntimeError()

def tf_file_ok(fname):
  if not os.path.exists(fname):
    raise RuntimeError('does not exists: %s' % fname)
  else:
    try:
      for rec in tf.python_io.tf_record_iterator(fname):
        pass
      return True
    except:
      return False

def ph_like(x):
  return tf.placeholder(x.dtype, shape = shape(x))


# from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/resnet_utils.py
def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None, **kwargs):
  """Strided 2-D convolution with 'SAME' padding.
  When stride > 1, then we do explicit zero-padding, followed by conv2d with
  'VALID' padding.
  Note that
     net = conv2d_same(inputs, num_outputs, 3, stride=stride)
  is equivalent to
     net = tf.contrib.layers.conv2d(inputs, num_outputs, 3, stride=1,
     padding='SAME')
     net = subsample(net, factor=stride)
  whereas
     net = tf.contrib.layers.conv2d(inputs, num_outputs, 3, stride=stride,
     padding='SAME')
  is different when the input's height or width is even, which is why we add the
  current function. For more details, see ResnetUtilsTest.testConv2DSameEven().
  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    scope: Scope.
  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """
  if stride == 1:
    return slim.conv2d(
        inputs,
        num_outputs,
        kernel_size,
        stride=1,
        rate=rate,
        padding='SAME',
        scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.conv2d(
        inputs,
        num_outputs,
        kernel_size,
        stride=stride,
        rate=rate,
        padding='VALID',
        scope=scope, 
        **kwargs)


def lrelu(x, a):
  # https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
  x = tf.identity(x)
  return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def lrelu_fn(a): return lambda x : lrelu(x, a)

def angle(z):
  # from https://github.com/tensorflow/tensorflow/issues/483
  """
  Returns the elementwise arctan of z, choosing the quadrant correctly.

  Quadrant I: arctan(y/x)
  Qaudrant II: \pi + arctan(y/x) (phase of x<0, y=0 is \pi)
  Quadrant III: -\pi + arctan(y/x)
  Quadrant IV: arctan(y/x)

  Inputs:
      z: tf.complex64 or tf.complex128 tensor
  Retunrs:
      Angle of z
  """
  return tf.atan2(tf.imag(z), tf.real(z))
  # if z.dtype == tf.complex128:
  #     dtype = tf.float64
  # else:
  #     dtype = tf.float32
  # x = tf.real(z)
  # y = tf.imag(z)
  # xneg = tf.cast(x < 0.0, dtype)
  # yneg = tf.cast(y < 0.0, dtype)
  # ypos = tf.cast(y >= 0.0, dtype)

  # offset = xneg * (ypos - yneg) * np.pi

  # return tf.atan(y / x) + offset


def normalize_rms(samples, desired_rms = 0.1, eps = 1e-4):
  rms = tf.maximum(eps, tf.sqrt(tf.reduce_mean(samples**2, reduction_indices = 1)))
  samples = samples * desired_rms / tf.expand_dims(rms, 1)
  return samples

def normalize_rms_np(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2, 1)))
  samples = samples * (desired_rms / rms)
  return samples

def normalize_rms_np2(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2, 1)))
  samples = samples * (desired_rms / rms)[..., None]
  return samples

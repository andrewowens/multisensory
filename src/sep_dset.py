import tensorflow as tf, aolib.util as ut, numpy as np, os, time, sys, tfutil
import tensorflow.contrib.slim as slim
import tensorflow.contrib.ffmpeg as ffmpeg

cifar_path = '../data/cifar-10'

ab = os.path.abspath
pj = ut.pjoin
rs = tf.reshape
shape = tfutil.shape
ed = tf.expand_dims

def round_int(x): return tf.cast(tf.round(x), tf.int64)
def cast_int(x): return tf.cast(x, tf.int64)
def cast_float(x): return tf.cast(x, tf.float32)

def read_example(rec_queue, pr, input_types):
  reader = tf.TFRecordReader()
  k, serialized_example = reader.read(rec_queue)
  full = pr.full_im_dim

  feats = {}
  if not pr.im_split:
    feats['im'] = tf.FixedLenFeature([], dtype=tf.string)
  else:
    if pr.num_splits == 2:
      feats['im_0'] = tf.FixedLenFeature([], dtype=tf.string)
      feats['im_1'] = tf.FixedLenFeature([], dtype=tf.string)
    else:
      for i in xrange(pr.num_splits):
        feats['im_%d' % i] = tf.FixedLenFeature([], dtype=tf.string)
        feats['num_frames_%d' % i] = tf.FixedLenFeature([1], dtype=tf.int64)
  if 'sfs' in input_types:
    feats['sfs'] = tf.FixedLenFeature([pr.full_specgram_samples * pr.num_mel], dtype=tf.float32)
  if 'samples' in input_types:
    feats['sound'] = tf.FixedLenFeature([], dtype=tf.string)
  if 'ytid' in input_types:
    feats['ytid'] = tf.FixedLenFeature([], dtype=tf.string)
  if 'flow' in input_types:
    feats['flow'] = tf.FixedLenFeature([], dtype=tf.string)
  if 'label' in input_types:
    feats['label'] = tf.FixedLenFeature([1], dtype=tf.int64)
  if pr.variable_frame_count:
    feats['num_frames'] = tf.FixedLenFeature([1], dtype=tf.int64)
  #serialized_example = tf.Print(serialized_example, ['reading'])
  example = tf.parse_single_example(serialized_example, features = feats)

  if not pr.im_split:
    ims = tf.image.decode_jpeg(example['im'], channels = 3, name = 'decode_im')
  else:
    if pr.num_splits == 2 and (not hasattr(pr, 'new_split') or not pr.new_split):
      total_frames = pr.total_frames if not pr.variable_frame_count \
                     else tf.cast(example['num_frames'][0], tf.int32)
      def f(x): 
        x.set_shape((full*total_frames/2, full, 3))
        return x
      ims = tf.concat(map(f, [tf.image.decode_jpeg(example['im_0'], channels = 3, name = 'decode_im1'),
                              tf.image.decode_jpeg(example['im_1'], channels = 3, name = 'decode_im2')]), 0)
    else:
      groups = []
      for i in xrange(pr.num_splits):
        im = tf.image.decode_jpeg(example['im_%d' % i], channels = 3, name = 'decode_im%d' % i)
        frames = tf.cast(example['num_frames_%d' % i][0], tf.int32)
        #im.set_shape((full * frames, full, 3))
        im = tf.reshape(im, tf.concat([[frames], [full], [full], [3]], 0), name = 'reshape_ims')
        groups.append(im)
      ims = tf.concat(groups, 0)

  #ims = tf.Print(ims, ['jpeg'])

  total_frames = pr.total_frames if not pr.variable_frame_count \
                 else tf.cast(example['num_frames'][0], tf.int32)

  if pr.variable_frame_count:
    ims = tf.reshape(ims, tf.concat([[total_frames], [full], [full], [3]], 0), name = 'reshape_ims')
  else:
    ims.set_shape((full*total_frames, full, 3))
    ims = tf.reshape(ims, (total_frames, full, full, 3))

  if 'flow' in input_types:
    flows = tf.image.decode_jpeg(example['flow'], channels = 3, name = 'decode_flow')[:, :, :2]
    flows.set_shape((pr.full_flow_dim * total_frames, pr.full_flow_dim, 2))
    flows = tf.reshape(
      flows, (total_frames, pr.full_flow_dim, pr.full_flow_dim, 2))
  else:
    #print 'total frames =', total_frames
    # flows = tf.zeros((total_frames, pr.full_flow_dim, 
    #                   pr.full_flow_dim, 2), dtype = tf.float32)
    #flows = cast_float(tf.zeros_like(ims)[..., :2])
    flows = tf.constant(0.)
    
  if 'samples' in input_types:
    samples = tf.decode_raw(example['sound'], tf.int16)
  else:
    samples = tf.zeros(pr.full_samples_len*2, dtype = tf.int16)

  if pr.variable_frame_count:
    samples = tf.reshape(samples, (tf.shape(samples)[0]/2, 2), name = 'reshape_samples')
  else:
    samples.set_shape((pr.full_samples_len*2))
    samples = tf.reshape(samples, (pr.full_samples_len, 2))
  
  samples = tf.cast(samples, 'float32') / np.iinfo(np.dtype(np.int16)).max
  #samples = tf.Print(samples, ['samples'])

  if 'sfs' in input_types:
    #sfs = tf.zeros((pr.num_mel, pr.full_specgram_samples, 1), dtype = tf.float32)
    sfs = tf.reshape(example['sfs'], (pr.num_mel, pr.full_specgram_samples, 1))
    sfs = sfs[:, :, 0]
    sfs = tf.transpose(sfs)

  if 'label' in input_types:
    label = example['label'][0]
  else:
    label = tf.constant(-1, tf.int64)

  num_slice_frames = pr.sampled_frames
  num_samples = int(pr.samples_per_frame * float(num_slice_frames))

  if pr.do_shift:
    choices = []
    max_frame = total_frames - num_slice_frames
    frames1 = ([0] if pr.fix_frame else xrange(max_frame))
    for frame1 in frames1:
      found = False
      for frame2 in reversed(range(max_frame)):
        inv1 = xrange(frame1, frame1 + num_slice_frames)
        inv2 = xrange(frame2, frame2 + num_slice_frames)
        if len(set(inv1).intersection(inv2)) <= pr.max_intersection:
          found = True
          choices.append([frame1, frame2])
          if pr.fix_frame:
            break
      if pr.skip_notfound:
        pass
      else:
        assert found
    print 'Number of frame choices:', len(choices)
    choices = tf.constant(np.array(choices), dtype = tf.int32)
    idx = tf.random_uniform([1], 0, shape(choices, 0), dtype = tf.int64)[0]
    start_frame_gt = choices[idx, 0]
    shift_frame = choices[idx, 1]
  elif ut.hastrue(pr, 'use_first_frame'):
    shift_frame = start_frame_gt = tf.constant(0, dtype = tf.int32)
  else:
    shift_frame = start_frame_gt = tf.random_uniform(
      [1], 0, total_frames - num_slice_frames, dtype = tf.int32)[0]

  #shift_frame = tf.Print(shift_frame, [shift_frame])
  #start_frame_gt = tf.Print(start_frame_gt, [start_frame_gt])

  if hasattr(pr, 'resize_dims'):
    ims = tf.image.resize_images(ims, pr.resize_dims)
  if pr.augment_ims:
    print 'Augment:', pr.augment_ims
    # todo: handle flow by concatenating ims with flow
    r = tf.random_uniform(
      [2], 0, shape(ims, 1) - pr.crop_im_dim, dtype = tf.int32)
    x, y = r[0], r[1]
    #x = tf.Print(x, [x, y])
  else:
    if hasattr(pr, 'resize_dims'):
      y = pr.resize_dims[0]/2 - pr.crop_im_dim/2
      x = pr.resize_dims[1]/2 - pr.crop_im_dim/2
      #print 'y =', y, 'x = ', x
    else:
      y = x = pr.full_im_dim/2 - pr.crop_im_dim/2
  
  offset = [start_frame_gt, y, x, 0]
  #print shape(ims), pr.crop_im_dim
  d = pr.crop_im_dim
  #print 'inputs:', offset, num_slice_frames, d, d, 3
  size_im = [num_slice_frames, d, d, 3]
  size_fl = [num_slice_frames, d, d, 2]
  ims = tf.slice(ims, offset, size_im)
  if 'flows' in input_types:
    flows = tf.slice(flows, offset, size_fl)
  ims.set_shape([num_slice_frames] + list(shape(ims)[1:]))
  if 'flows' in input_types:
    flows.set_shape([num_slice_frames] + list(shape(flows)[1:]))

  # moved
  if pr.augment_ims:
    ims = tf.cond(tf.cast(tf.random_uniform([1], 0, 2, dtype = tf.int64)[0], tf.bool),
                  lambda : tf.map_fn(tf.image.flip_left_right, ims), 
                  lambda : ims)

  # slice audio
  def slice_samples(frame):
    start = round_int(pr.samples_per_frame * cast_float(frame))
    if pr.augment_audio:
      print 'samples shape:', shape(samples)
      start0 = start# (+ 168659 41341) 210000
      dist = 3#int(0.05*pr.fps)
      delta = tf.random_uniform(
        [1], long(-dist*pr.samples_per_frame), long(dist*pr.samples_per_frame), dtype = tf.int64)[0]
      start = start + delta
      #start = tf.maximum(cast_int(0), cast_int(start))
      #start = tf.minimum(start, long(shape(samples, 0) - num_samples))
      start = tf.maximum(start, cast_int(0))
      start = tf.minimum(start, tf.cast(cast_int(cast_float(total_frames)*pr.samples_per_frame) - num_samples - 1, tf.int64))
      #start = tf.Print(start, [start0, start, pr.full_samples_len, num_samples, dist*pr.samples_per_frame]) 
    offset = [start, 0]
    size = [num_samples, 2]
    r = tf.slice(samples, offset, size, name = 'slice_sample')
    r.set_shape([num_samples] + list(shape(r)[1:]))
    return r

  if 'samples' in input_types:
    samples_gt = slice_samples(start_frame_gt)
    samples_shift = slice_samples(shift_frame)
  else:
    samples_gt = samples_shift = tf.zeros((1, 1), dtype = tf.int16)  

  sfs_gt = sfs_shift = tf.zeros((1, 1))

  if 'ytid' in input_types:
    ytid = example['ytid']
  else:
    ytid = ''

  # 0 -> fake, 1 -> real
  samples_exs = tf.concat([ed(samples_shift, 0), ed(samples_gt, 0)], 0)
  sfs_exs = tf.concat([ed(sfs_shift, 0), ed(sfs_gt, 0)], 0)

  if not pr.do_shift:
    # samples_exs = samples_exs[0]
    # sfs_exs = sfs_exs[0]
    samples_exs = samples_exs[1]
    sfs_exs = sfs_exs[1]

  if pr.multi_shift:
    new_samples = [samples_exs[0], samples_exs[1]]
    for i in xrange(pr.num_shifts):
      idx = tf.random_uniform([1], 0, shape(choices, 0), dtype = tf.int64)[0]
      # doesn't necessarily match gt frame in terms of suppression, but that's prob ok...
      frame = choices[idx, 1]
      new_samples.append(slice_samples(frame))
    samples_exs = tf.concat([ed(xx, 0) for xx in new_samples], 0)
    print 'samples_exs shape:', shape(samples_exs)

  # samples_exs = tf.Print(samples_exs, ['done samples_exs', shape(samples_exs)])
  # samples_exs = tf.Print(samples_exs, ['done ims', shape(ims)])
  # samples_exs = tf.Print(samples_exs, ['ytid', ytid])
  return ims, flows, samples_exs, sfs_exs, label, ytid

def get_rec_files(path, needs_done_file):
  if type(path) != type(''):
    rec_files = ut.flatten(ut.glob(pj(x, '*.tf')) for x in path)
  else:
    rec_files = ut.glob(pj(path, '*.tf')) + ut.glob(pj(path, '*.tfrecords'))

  if needs_done_file:
    rec_files = [x for x in rec_files if os.path.exists(x + '_done.txt')]
  rec_files = sorted(rec_files)

  return rec_files

def rec_files_from_path(path, num_db_files = None):
  print 'Path:', path
  if path.endswith('.txt'):
    rec_files = ut.read_lines(path)
    rec_files = filter(os.path.exists, rec_files)[:num_db_files]
  else:
    rec_files = sorted(ut.glob(path, '*.tf'))
  return rec_files
  
def make_db_reader(path, pr, batch_size, input_types, db_start = None,
                   num_db_files = None, num_threads = 12, one_pass = False):
  print 'one pass =', one_pass
  if not os.path.exists(path):
    raise RuntimeError('Data path does not exist: %s' % path)

  if pr.dset_seed is not None or one_pass:
    num_threads = 1
  rec_files = rec_files_from_path(path, num_db_files = num_db_files)

  if not one_pass:
    rec_files = ut.shuffled(rec_files)
  
  if hasattr(pr, 'alt_tf_path') and pr.alt_tf_path is not None:
    assert pr.alt_tf_path[0] in rec_files[0]
    rec_files = [(x.replace(pr.alt_tf_path[0], pr.alt_tf_path[1]) if i % 2 == 0 else x) for i, x in enumerate(rec_files)]
  #ut.printlns(rec_files)

  file_groups = ut.split_into(rec_files, num_threads)
  num_threads = min(len(file_groups), num_threads)
  queues = [tf.train.string_input_producer(
    group, seed = pr.dset_seed, 
    shuffle = (pr.dset_seed is None),
    num_epochs = (1 if one_pass else None)) for group in file_groups]
  example_list =  [read_example(queue, pr, input_types) for queue in queues]

  if not one_pass and (pr.dset_seed is None):
    # ims, flows, samples, sfs = tf.train.shuffle_batch_join(
    #   example_list, batch_size = batch_size, capacity = 1200, 
    #   min_after_dequeue = 50, seed = pr.dset_seed)
    # ims, flows, samples, sfs, labels, ytids = tf.train.shuffle_batch_join(
    #   example_list, batch_size = batch_size, capacity = 250, 
    #   min_after_dequeue = 50, seed = pr.dset_seed)
    ims, flows, samples, sfs, labels, ytids = tf.train.shuffle_batch_join(
      example_list, batch_size = batch_size, capacity = 200, 
      min_after_dequeue = 20, seed = pr.dset_seed)
  else:
    ims, flows, samples, sfs, labels, ytids = tf.train.batch(example_list[0], batch_size)

  rets = {'im' : ims,
          'flow' : flows,
          'samples' : samples,
          'sfs' : sfs,
          'label' : labels,
          'ytid' : ytids}

  return [rets[k] for k in input_types]

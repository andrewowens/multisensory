import numpy as np, tfutil as mu, aolib.util as ut, tensorflow as tf

ed = tf.expand_dims
shape = mu.shape
add_n = mu.maybe_add_n
pj = ut.pjoin
cast_float = mu.cast_float
cast_int = mu.cast_int
cast_complex = lambda x : tf.cast(x, tf.complex64)

# def griffin_lim(spec, 
#                 frame_length,
#                 frame_step,
#                 n_fft,
#                 n_iter = 1):
#                 #n_iter = 5):
#                 #n_iter = 60):
#   # https://github.com/candlewill/Griffin_lim
#   def invert_spec(spec):
#     return tf.contrib.signal.inverse_stft(spec, frame_length, frame_step, n_fft)

#   spec = tf.cast(spec, dtype=tf.complex64)  # [t, f]
#   X_best = tf.identity(spec)
#   for i in range(n_iter):
#     X_t = invert_spec(X_best)
#     est = tf.contrib.signal.stft(X_t, frame_length, frame_step, n_fft, pad_end = False)  # (1, T, n_fft/2+1)
#     phase = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)  # [t, f]
#     X_best = spec * phase  # [t, t]
#   X_t = invert_spec(X_best)
#   y = tf.real(X_t)
#   y = cast_float(y)
#   return y



def griffin_lim(spec, 
                frame_length,
                frame_step,
                num_fft,
                num_iters = 1):
                #num_iters = 20):
                #num_iters = 10):
  invert_spec = lambda spec : tf.contrib.signal.inverse_stft(spec, frame_length, frame_step, num_fft)

  spec_mag = tf.cast(tf.abs(spec), dtype=tf.complex64)
  best = tf.identity(spec)
  for i in range(num_iters):
    samples = invert_spec(best)
    est = tf.contrib.signal.stft(samples, frame_length, frame_step, num_fft, pad_end = False)  # (1, T, n_fft/2+1)
    phase = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64) 
    best = spec_mag * phase
  X_t = invert_spec(best)
  y = tf.real(X_t)
  y = cast_float(y)
  return y

def pack_spec(spec, pr):
  assert spec.shape[-1] % 2 == 1
  spec = spec[..., :-1]
  spec_mag = db_from_amp(tf.abs(spec))
  spec_phase = mu.angle(spec)
  return spec, spec_mag, spec_phase

def unpack_spec(spec_mag, spec_phase, pr):
  spec_mag = cast_complex(amp_from_db(spec_mag))
  spec_phase = cast_complex(spec_phase)
  j = tf.constant(1j, dtype = tf.complex64)
  spec = spec_mag * (tf.cos(spec_phase) + tf.sin(spec_phase)*j)
  assert spec.shape[-1] % 2 == 0
  spec = tf.pad(spec, [(0, 0), (0, 0), (0, 1)])
  return spec

log10 = lambda x : tf.log(x)/tf.log(cast_float(10))

def db_from_amp(x):
  return 20. * log10(tf.maximum(1e-5, x))

def amp_from_db(x):
  return tf.pow(10., x / 20.)

def stft_frame_length(pr): return int(pr.frame_length_ms * pr.samp_sr * 0.001)
def stft_frame_step(pr): return int(pr.frame_step_ms * pr.samp_sr * 0.001)
def stft_num_fft(pr): return int(2**np.ceil(np.log2(stft_frame_length(pr))))

def stft(samples, pr):
  tracks = []
  for i in xrange(shape(samples, -1)):
    spec = tf.contrib.signal.stft(samples[..., i], 
                                 frame_length = stft_frame_length(pr),
                                 frame_step = stft_frame_step(pr))
    spec, spec_mag, spec_phase = pack_spec(spec, pr)
    tracks.append(spec_mag)
    tracks.append(spec_phase)
  return tf.concat([ed(x, 3) for x in tracks], 3)

def samples_from_spec(spec, pr):
  return griffin_lim(spec, stft_frame_length(pr), stft_frame_step(pr), stft_num_fft(pr))

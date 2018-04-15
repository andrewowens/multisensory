import wave, util as ut, os, numpy as np, img as ig, imtable, pylab as pl, copy
import scipy.io.wavfile

class converted_wav:
  def __init__(self, in_fname):
    self.in_fname = in_fname
    self.out_fname = ut.make_temp('.wav')
    os.remove(self.out_fname)

  def __enter__(self):
    #if 0 != os.system('ffmpeg -loglevel warning -i "%s" -acodec pcm_s16le "%s"' % (self.in_fname, self.out_fname)):
    if 0 != os.system('ffmpeg -loglevel warning -i "%s" -acodec pcm_s32le "%s"' % (self.in_fname, self.out_fname)):
      raise RuntimeError('Could not convert wav file')
    return self.out_fname

  def __exit__(self, type, value, traceback):
    assert self.in_fname != self.out_fname
    os.remove(self.out_fname)

def load_wav(wav_fname):
  # deprecated
  with converted_wav(wav_fname) as fname:
    return scipy.io.wavfile.read(fname)

def load_sound(wav_fname):
  rate, samples = scipy.io.wavfile.read(wav_fname)
  times = (1./rate) * np.arange(len(samples))
  return Sound(times, rate, samples)

def load_sound_uint8(wav_fname):
  rate, samples = scipy.io.wavfile.read(wav_fname)
  assert samples.dtype == np.uint8
  samples = np.array(samples, 'float32') / 128. - 0.5
  times = (1./rate) * np.arange(len(samples))
  return Sound(times, rate, samples)

def time_idx_interval(times, t1, t2):
  ti1 = np.searchsorted(times, t1)
  ti2 = np.searchsorted(times, t2)
  return ti1, ti2

def db(x):
  return 20 * np.log10(x)
 
def vis_specgram(freqs, sft, times, lo_db = -90., hi_db = -10., fig_ = None,
                 freq_lo_hz = 1., freq_hi_hz = 20000., 
                 width_in = 4., time_lo = None, time_hi = None,
                 plot_fn = (lambda fig, ax : None)):

  #fig = pl.figure(frameon=False) if fig_ is None else fig_
  fig = pl.gcf() if fig_ is None else fig_
  fig.clf()
  fig.set_facecolor((1,1,1,1))
  ax = fig.gca()

  #f1, f2 = np.min(freqs), np.max(freqs)
  # f1 = freq_lo_hz
  # f2 = np.max(freqs) if freq_hi_hz is None else freq_hi_hz
  
  # t1 = times[0] if time_lo is None else time_lo
  # t2 = times[-1] if time_hi is None else time_hi
  t1 = times[0]
  t2 = times[-1]

  sft = sft.T
  #lsp = (sft[(f1 <= freqs) & (freqs <= f2)][:, (t1 <= times) & (times <= t2)]).copy()
  # fi1 = np.searchsorted(freqs, f1)
  # fi2 = np.searchsorted(freqs, f2)
  ti1 = np.searchsorted(times, t1)
  ti2 = np.searchsorted(times, t2)
  #asdf
  #lsp = (sft[fi1 : fi2][:, ti1 : ti2]).copy()
  lsp = (sft[:, ti1 : ti2]).copy()
  ok = (lsp > 0)
  lsp[ok] = db(lsp[ok])
  lsp[-ok] = db(0.0000000001)

  #vis_ok = ut.clip_rescale(lsp, lo_db, hi_db)
  vis_ok = lsp
  #print np.min(lsp), np.max(lsp)
  f1, f2 = freq_lo_hz, freq_hi_hz
  #print f1, f2
  #ax.imshow(vis_ok, vmin = 0., vmax = 1., cmap = pl.cm.jet, extent = (t1, t2, f1, f2), aspect = 'auto')
  #ax.imshow(vis_ok, vmin = 0., vmax = 1., cmap = pl.cm.gray_r, extent = (t1, t2, f1, f2), aspect = 'auto')
  # ax.imshow(vis_ok, vmin = lo_db, vmax = hi_db, cmap = pl.cm.gray_r,
  #           extent = (t1, t2, f1, f2), aspect = 'auto')
  ax.imshow(vis_ok, vmin = lo_db, vmax = hi_db, cmap = pl.cm.gray_r,
            extent = (t1, t2, freqs[-1], freqs[0]), aspect = 'auto')
  #ax.set_yscale('log')
  ax.set_yscale('linear')
  ax.set_xscale('linear')
  ax.set_axis_bgcolor((1,1,1,1))
  ax.set_ylim([0, 20000])
  ax.set_xlim([time_lo, time_hi])
  
  plot_fn(fig, ax)

  fig.tight_layout(pad = 0)
  #ret = ig.from_fig(ax)

  ret = ig.from_fig(fig)
  # if fig_ is None:
  #   pl.close(fig)

  #w, h = fig.get_size_inches()
  #fig.set_size_inches(float(width_in), float(width_in)/w * h)
  
  return ret[:, :, :3] #ret[:, :, :3]

def specgram_max():
  sft_hi = 0.1
  # This number was computed roughly using the following procedure:
  # snd = scan.sound().normalized()
  # specgram = sound.make_specgram(snd.samples, snd.rate)      
  # np.percentile(specgram[1], 99.5)
  return sft_hi

def centered_sound(sound, ts, duration = 1.5):
  tidx1, tidx2 = time_idx_interval(sound.times, ts - duration/2, ts + duration/2)
  return sound[tidx1 : tidx2].normalized()
  
# def vis_specgram_at_time(sound, ts, duration = 0.5):
#   sound_sub = centered_sound(sound, ts, duration)
#   spec = make_specgram(sound_sub.samples, sound_sub.rate, sample_times = sound_sub.times, noverlap = 2000)
#   def plot_fn(fig, ax, ts = ts):
#     ax.plot([ts, ts], ax.get_ylim(), alpha = 0.25, color = 'w', lw = 2)

#   return vis_specgram(spec[0], spec[1], spec[2], hi = specgram_max(), plot_fn = plot_fn)

# def vis_specgram_at_times(snd, times, window_sec, par = 0):
#   snd = snd.normalized().to_mono()
#   freqs, sft, sft_times = make_specgram(snd.samples, snd.rate, sample_times = snd.times, noverlap = 2000)
#   ims = []
#   for t in times:
#     def plot_fn(fig, ax, t = t):
#       ax.plot([t, t], ax.get_ylim(), alpha = 0.25, color = 'blue', lw = 2)
#     time_lo = t - window_sec/2.
#     time_hi = t + window_sec/2.
#     ims.append(vis_specgram(freqs, sft, sft_times, time_lo = time_lo, time_hi = time_hi, plot_fn = plot_fn))
#   return ims

def vis_specgram_at_times(snd, times, window_sec, par = 0, im_shape = (500, 888), compress_ims = False):
  snd = snd.normalized().to_mono()
  spec = make_specgram(snd.samples, snd.rate, sample_times = snd.times, noverlap = 2000)

  def f(t, spec = spec, window_sec = window_sec, im_shape = im_shape):
    freqs, sft, sft_times = spec
    def plot_fn(fig, ax, t = t):
      ax.plot([t, t], ax.get_ylim(), alpha = 0.25, color = 'blue', lw = 2)
    time_lo = t - window_sec/2.
    time_hi = t + window_sec/2.
    vsc = vis_specgram(freqs, sft, sft_times, time_lo = time_lo,
                       time_hi = time_hi, plot_fn = plot_fn)
    return ig.compress(ig.scale(vsc, im_shape))
  if not compress_ims:
    ims = map(ig.uncompress, ip.map(par, f, times))
  return ims

# def make_mono(x):
#   if np.ndim(x) > 1:
#     assert np.ndim(x) == 2
#     return np.sum(x, axis = 1, dtype = 'double')
#   else:
#     return np.array(x, dtype = 'double')

def make_mono(x):
  if np.ndim(x) > 1:
    assert np.ndim(x) == 2
    return np.mean(x, axis = 1, dtype = 'double')
  else:
    return np.array(x, dtype = 'double')

def stfft(x, nfft, noverlap, win, par = 0):
  x = make_mono(x)
  step = nfft - noverlap
  n = win.shape[0]
  res = []
  win_starts = np.arange(0, x.shape[0] - nfft, step)
  for i in win_starts:
    sub = x[i : i + n]
    res.append(np.fft.fft(sub * win))
  return np.array(res, 'complex'), win_starts

def make_specgram(sound, rate, shift_fft = True, sample_times = None,
                  nfft = None, noverlap = 2000, par = 0):
  assert rate > 1 # probably should have multiple samples per second
  if nfft is None:
    nfft = int(np.ceil(0.05 * rate))

  nfft += (nfft % 2)
    
  win = np.hamming(nfft)
  sft, time_idx = stfft(sound, nfft, noverlap, win, par = par)
  sft = np.real(sft * np.conjugate(sft))
  sft /= np.sum(np.abs(win)**2)
  freqs = np.fft.fftfreq(sft.shape[1], 1./rate)
  
  # Since the input is real, the result will be symmetric, and thus we can throw away
  # the negative frequencies.
  nfreq = nfft // 2
  assert (freqs[nfreq-1] > 0) and (freqs[nfreq] < 0)
  freqs = freqs[nfreq - 1 : 0 : -1]
  sft = sft[:, nfreq - 1 : 0 : -1]

  if sample_times is None:
    times = time_idx * (1./rate)
  else:
    times = sample_times[time_idx]
  
  return freqs, np.asarray(sft, dtype = 'float32'), times
  
def test_spectrogram():
  # http://matplotlib.org/examples/pylab_examples/specgram_demo.html
  dt = 1./0.0005
  t = np.arange(0., 20., dt)
  #t = np.arange(0., 3., dt)
  s1 = np.sin((2*np.pi)*100*t)
  s2 = 2 * np.sin((2*np.pi)*400*t)
  s2[-((10 < t) & (t < 12))] = 0
  nse = 0.01 * np.random.randn(len(t))
  if 0:
    x = s1
  else:
    x = s1 + s2 + nse
  freqs, spec, spec_times = make_specgram(x, dt)

  pl.clf()

  ax1 = pl.subplot(211)
  ax1.plot(t, x)

  if 1:
    lsp = spec.copy()
    lsp[spec > 0] = np.log(spec[spec > 0])
    lsp = ut.clip_rescale(lsp, -10, np.percentile(lsp, 99))
  else:
    lsp = spec.copy()
    lsp = ut.clip_rescale(lsp, 0, np.percentile(lsp, 99))

  ax2 = pl.subplot(212, sharex = ax1)
  ax2.imshow(lsp.T, cmap = pl.cm.jet, 
             extent = (0., t[-1], np.min(freqs), np.max(freqs)), 
             aspect = 'auto')

  ig.show(vis_specgram(freqs, spec, spec_times))
  ut.toplevel_locals()

def pink_noise(n, scale = 1., alpha = 1.):
  # not exactly pink
  if n <= 1:
    return np.randn(n)
  spec = np.random.randn(n)
  # power \prop 1/f^alpha ==> sqrt(power) \prop 1/sqrt(f^alpha)
  spec = 1./(scale * np.sqrt(np.arange(1, n+1)**alpha)) * spec
  return np.fft.irfft(spec)[:n]
  
class Sound:
  def __init__(self, times, rate, samples = None):
    # Allow Sound(samples, sr)
    if samples is None:
      samples = times
      times = None
    if samples.dtype == np.float32:
      samples = samples.astype('float64')
      
    self.rate = rate
    self.samples = ut.atleast_2d_col(samples)
    self.length = samples.shape[0]
    if times is None:
      self.times = np.arange(len(self.samples)) / float(self.rate)
    else:
      self.times = times

  def copy(self):
    return copy.deepcopy(self)
  
  def parts(self):
    return (self.times, self.rate, self.samples)

  def __getslice__(self, *args):
    return Sound(self.times.__getslice__(*args), self.rate,
                 self.samples.__getslice__(*args))

  def duration(self):
    return self.samples.shape[0] / float(self.rate)
  
  def normalized(self, check = True):
    if self.samples.dtype == np.double:
      assert (not check) or np.max(np.abs(self.samples)) <= 4.
      x = copy.deepcopy(self)
      x.samples = np.clip(x.samples, -1., 1.)
      return x
    else:
      s = copy.deepcopy(self)
      s.samples = np.array(s.samples, 'double') / np.iinfo(s.samples.dtype).max
      s.samples[s.samples < -1] = -1
      s.samples[s.samples > 1] = 1
      return s

  def unnormalized(self, dtype_name = 'int32'):
    s = self.normalized()
    inf = np.iinfo(np.dtype(dtype_name))
    samples = np.clip(s.samples, -1., 1.)
    samples = inf.max * samples
    samples = np.array(np.clip(samples, inf.min, inf.max), dtype_name)
    s.samples = samples
    return s
    
  def sample_from_time(self, t, bound = False):
    if bound:
      return min(max(0, int(np.round(t * self.rate))), self.samples.shape[0]-1)
    else:
      return int(np.round(t * self.rate))
    # if self.times[0] != 0:
    #   return int(np.argmin(np.abs(self.times - t)))
    # else:
    #   return min(max(0, int(np.round(t * self.rate))), self.samples.shape[0]-1)
    

  st = sample_from_time

  def shift_zero(self):
    s = copy.deepcopy(self)
    s.times -= s.times[0]
    return s

  def select_channel(self, c):
    s = copy.deepcopy(self)
    s.samples = s.samples[:, c]
    return s

  def left_pad_silence(self, n):
    if n == 0:
      return self.shift_zero()
    else:
      if np.ndim(self.samples) == 1:
        samples = np.concatenate([[0] * n, self.samples])
      else:
        samples = np.vstack([np.zeros((n, self.samples.shape[1]), self.samples.dtype), self.samples])
    return Sound(None, self.rate, samples)

  def right_pad_silence(self, n):
    if n == 0:
      return self.shift_zero()
    else:
      if np.ndim(self.samples) == 1:
        samples = np.concatenate([self.samples, [0] * n])
      else:
        samples = np.vstack([self.samples, np.zeros((n, self.samples.shape[1]), self.samples.dtype)])
    return Sound(None, self.rate, samples)

  def pad_slice(self, s1, s2):
    assert s1 < self.samples.shape[0] and s2 >= 0
    s = self[max(0, s1) : min(s2, self.samples.shape[0])]
    s = s.left_pad_silence(max(0, -s1))
    s = s.right_pad_silence(max(0, s2 - self.samples.shape[0]))
    return s

  def to_mono(self, force_copy = True):
    s = copy.deepcopy(self)
    s.samples = make_mono(s.samples)
    return s

  def slice_time(self, t1, t2):
    return self[self.st(t1) : self.st(t2)]

  @property
  def nchannels(self):
    return 1 if np.ndim(self.samples) == 1 else self.samples.shape[1]

  def save(self, fname):
    s = self.unnormalized('int16')
    scipy.io.wavfile.write(fname, s.rate, s.samples) 

  def resampled(self, new_rate, clip = True):
    import subband
    if new_rate == self.rate:
      return copy.deepcopy(self)
    else:
      #assert self.samples.shape[1] == 1
      return Sound(None, new_rate, subband.resample(self.samples, float(new_rate)/self.rate, clip = clip))

  def trim_to_size(self, n):
    return Sound(None, self.rate, self.samples[:n])

def play(samples_or_snd, sr = None, compress = True):
  if sr is None:
    samples = samples_or_snd.samples
    sr = samples_or_snd.rate
  else:
    samples = samples_or_snd
  snd = Sound(None, sr, samples)#.unnormalized('int16')
  
  path = ut.pjoin(imtable.get_www_path(), 'sounds')
  if compress:
    with ut.temp_file('.wav') as wav_fname:
      fname = ut.make_temp('.mp3', dir = path)
      #scipy.io.wavfile.write(wav_fname, snd.rate, snd.samples)
      snd.save(wav_fname)
      os.system('ffmpeg -loglevel error -y -i "%s" "%s"' % (wav_fname, fname))
  else:
    fname = ut.make_temp('.wav', dir = path)
    scipy.io.wavfile.write(fname, snd.rate, snd.samples) 

  os.system('chmod a+rwx %s' % fname)
  #url = ut.pjoin(imtable.PUBLIC_URL, 'sounds', os.path.split(fname)[1])
  url = ut.pjoin(imtable.get_url(), 'sounds', os.path.split(fname)[1])
  print url
  return url

class LongWav:
  def __init__(self, fname):
    self.wav = wave.open(fname)
    self.rate = int(self.wav.getframerate())
    # 16 bit
    if self.wav.getsampwidth() != 2:
      raise RuntimeError('Expected 16-bit wave file!')
    self.length = self.wav.getnframes()
    
  def __getslice__(self, i, j):
    self.wav.setpos(i)
    #print i, j
    data = self.wav.readframes(j - i)
    data = np.fromstring(data, dtype = np.int16)
    return Sound(None, self.rate, data)

  def sound(self):
    return self[:self.length]
    
  def duration(self):
    return self.length/float(self.rate)

  def sample_from_time(self, t, bound = False):
    if bound:
      return int(min(max(0, int(np.round(t * self.rate))), self.length-1))
    else:
      return int(np.round(t * self.rate))

  def slice_time(self, t1, t2):
    return self[self.st(t1) : self.st(t2)]

  def pad_slice(self, s1, s2):
    assert s1 < self.length and s2 >= 0
    s = self[max(0, s1) : min(s2, self.length)]
    s = s.left_pad_silence(max(0, -s1))
    s = s.right_pad_silence(max(0, s2 - self.length))
    return s

  st = sample_from_time

def resample_snd((snd, sr)):
  return snd.resampled(sr)

def concat_sounds(snds, par = 1):
  #snds = [snd.to_mono() for snd in snds]
  if ut.ndistinct(ut.mapattr(snds).rate) == 1:
    return Sound(None, snds[0].rate, np.concatenate(ut.mapattr(snds).samples, axis = 0))
  else:
    sr = max(ut.mapattr(snds).rate)
    #new_snds = [snd.resampled(sr) for snd in snds]

    if par:
      new_snds = ut.parmap(resample_snd, [(snd, sr) for snd in snds])
    else:
      new_snds = map(resample_snd, [(snd, sr) for snd in snds])
      
    assert ut.ndistinct(ut.mapattr(new_snds).rate) == 1
    return concat_sounds(new_snds)


def convert_sound_compat(in_fname, out_fname, duration = None, rate = None, codec = 'pcm_s32le', ffmpeg_flags = ''):
  """ Probably there is a better way to do this... """
  duration_flag = ('-t %.7f' % duration if duration is not None else '')
  assert out_fname.endswith('.wav')
  # with ut.temp_file('.mp3') as tmp_file:
  #   print tmp_file
  #   ok = (0 == ut.sys_print('ffmpeg -i "%s" %s -y "%s"' % (in_fname, duration_flag, tmp_file)))
  #   ok = ok and (0 == os.system('ffmpeg -i "%s" -y -ac 1 -acodec pcm_s32le "%s"' % (tmp_file, out_fname)))
  #   print 'writing', out_fname, ok
  #   #ok = ok and (0 == ut.sys_print('ffmpeg -i "%s" -y -f s16le -acodec pcm_s16le -ac 1 "%s"' % (tmp_file, out_fname)))
  #asdf
  #with ut.temp_file('.mp3') as mp3_tmp, ut.temp_file('.wav') as wav_tmp:
  with ut.temp_file('.mp3') as mp3_tmp, ut.temp_file('.wav') as wav_tmp:
    ok = (0 == ut.sys_print('ffmpeg -i "%s" %s -y %s "%s"' % (in_fname, duration_flag, ffmpeg_flags, mp3_tmp)))
    #ok = ok and (0 == os.system('ffmpeg -i "%s" -y -ac 1 -acodec pcm_s32le "%s"' % (mp3_tmp, wav_tmp)))
    ar_str = '-ar %d' % rate if rate is not None else ''
    ok = ok and (0 == os.system('ffmpeg -i "%s" -y -ac 1 -acodec %s %s "%s"' % (mp3_tmp, codec, ar_str, wav_tmp)))
    if ok:
      load_sound(wav_tmp).save(out_fname)
    print 'writing', out_fname, ok
    #ok = ok and (0 == ut.sys_print('ffmpeg -i "%s" -y -f s16le -acodec pcm_s16le -ac 1 "%s"' % (tmp_file, out_fname)))
  return ok

# def convert_sound_compat(in_fname, out_fname, duration = None, rate = None, codec = 'pcm_s32le', ffmpeg_flags = ''):
#   """ Probably there is a better way to do this... """
#   duration_flag = ('-t %.7f' % duration if duration is not None else '')
#   assert out_fname.endswith('.wav')
#   # with ut.temp_file('.mp3') as tmp_file:
#   #   print tmp_file
#   #   ok = (0 == ut.sys_print('ffmpeg -i "%s" %s -y "%s"' % (in_fname, duration_flag, tmp_file)))
#   #   ok = ok and (0 == os.system('ffmpeg -i "%s" -y -ac 1 -acodec pcm_s32le "%s"' % (tmp_file, out_fname)))
#   #   print 'writing', out_fname, ok
#   #   #ok = ok and (0 == ut.sys_print('ffmpeg -i "%s" -y -f s16le -acodec pcm_s16le -ac 1 "%s"' % (tmp_file, out_fname)))
#   #asdf
#   #with ut.temp_file('.mp3') as mp3_tmp, ut.temp_file('.wav') as wav_tmp:
#   with ut.temp_file('.mp3') as mp3_tmp, ut.temp_file('.wav') as wav_tmp:
#     ok = (0 == ut.sys_print('ffmpeg -i "%s" %s -y %s "%s"' % (in_fname, duration_flag, ffmpeg_flags, mp3_tmp)))
#     #ok = ok and (0 == os.system('ffmpeg -i "%s" -y -ac 1 -acodec pcm_s32le "%s"' % (mp3_tmp, wav_tmp)))
#     ar_str = '-ar %d' % rate if rate is not None else ''
#     ok = ok and (0 == os.system('ffmpeg -i "%s" -y -ac 1 -acodec %s %s "%s"' % (mp3_tmp, codec, ar_str, out_fname)))
#     # if ok:
#     #   load_sound(wav_tmp).save(out_fname)
#     print 'writing', out_fname, ok
#     #ok = ok and (0 == ut.sys_print('ffmpeg -i "%s" -y -f s16le -acodec pcm_s16le -ac 1 "%s"' % (tmp_file, out_fname)))
#   return ok

def audio_sampling_rate(fname):
  return int(ut.sys_with_stdout('ffprobe -show_streams %s | grep sample_rate' % fname).split('sample_rate=')[1])

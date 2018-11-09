import numpy as np, sys, datetime, tempfile, glob
import tempfile, webbrowser, os, pylab, random
import util as ut, img as ig

def get_url():
  # set this to be the URL of your web server (if you have one)
  return None

def get_www_path():
  # set this to be the directory hosted by a public web server (if you have one)
  return None

def show_table(table, title = None, rows_per_page = 100, show = False,
               base_dir = None, output_path = None, archive = False,
               use_www = True):
  """ Display a table of images and text in a web browser. A table is
  represented as a list of lists, e.g. show_table([[im1, im2], ['hello', im3]]).
  Handles pagination (so you can input a table with thousands of images).

  Possible table values:
  - Arrays are assumed to be images.
  - Strings
  - A tuple ('img', filename) is assumed to be an image and its filename
  - A tuple ('img_mv', filename) is an image file that is moved into the output path
  """
  # empty html page for empty table
  # if len(table) == 0:
  #   table = ['']

  use_www = use_www and get_www_path() is not None and os.path.exists(get_www_path())
  
  if (base_dir is not None) and (not os.path.exists(base_dir)):
    print >>sys.stderr, 'base directory', base_dir, 'does not exist'
    base_dir = None

  # figure out the output directory from the arguments
  if output_path is None:
    # make a temporary subdirectory in base_dir if no output_path is specified
    if base_dir is None:
      # hard-coded path: since /tmp doesn't have much space on the vision machines
      if os.path.exists('/data/vision/billf/aho-vis/tmp'):
        base_dir = '/data/vision/billf/aho-vis/tmp'
      else:
        base_dir = None

    output_dir = ut.make_temp_dir(dir = base_dir)
  else:
    if base_dir is not None:
      print 'Ignoring base_dir since output_path is set'
    # clear output_path and use it
    if os.path.exists(output_path):
      # todo: make this quiet when if fails
      #os.system('rm %s/*.html %s/*.png' % (output_path, output_path))
      pass
    else:
      os.system('mkdir %s' % output_path)
    output_dir = output_path
    
  # hard-coded path to web server
  if use_www:
    os.system('rmdir %s' % output_dir)
    symlink_dir = output_dir
    # output_dir = ut.make_temp_dir(dir = '/csail/vision-billf5/aho/www/tables')
    # public_url = 'http://quickstep.csail.mit.edu/aho/tables'
    #output_dir = ut.make_temp_dir(dir = '/csail/vision-billf5/aho/www/tab')
    output_dir = ut.make_temp_dir(dir = get_www_path())

  if title is None:
    title = datetime.datetime.today().strftime('%I:%M %p %a')

  html_rows = html_from_rows(table, output_dir)
  if use_www:
    # not sure what the minimal permissions to set are
    os.system('chmod -R a+rwx %s' % output_dir)
    os.system('ln -s %s %s' % (output_dir, symlink_dir))
    os.system('chmod -R a+rwx %s' % symlink_dir)
    
  fnames = paginate_table(html_rows, rows_per_page, title, output_dir)
  # adding the file:// prefix makes it possible to open the file in
  # a browser using find-file-at-point in emacs
  if use_www:
    url = os.path.join(get_url(), os.path.split(output_dir)[1])
    #symlink_fname = 'file://%s' % os.path.abspath(os.path.join(symlink_dir, 'index.html'))
    print 'showing', url, '->', output_dir
    # if show:
    #   show_page(symlink_fname)
    os.system('chmod -R a+rwx %s' % output_dir)
    return url
  else:
    print 'showing', ('file://%s' % os.path.abspath(fnames[0]))
    if show:
      show_page(fnames[0])
    return fnames[0]

  if 0:
    # Save table and return HTML; don't paginate
    fname = paginate_table(html_rows, None, title, output_dir)[0]
    print 'showing', ('file://%s' % os.path.abspath(fname))

  if archive:
    dirname = os.path.split(output_dir.rstrip('/'))[1]
    zip_name = dirname + '.zip'
    # make a zip file with a single directory containing files, not a long path by changing cwd temporarily
    os.system('cd %s/..; zip -qr %s %s' % (output_dir, zip_name, dirname))
    print 'saved', os.path.abspath(output_dir.rstrip('/')  + '.zip')

  
    
def table_js():
  return """
<script>
var cycle_idx = {};
var cycle_groups = {};
var cycle_ims = {};
var vid_focus = null;

// http://stackoverflow.com/questions/164397/javascript-how-do-i-print-a-message-to-the-error-console
function log(msg) {
  setTimeout(function() {
    throw new Error(msg);
  }, 0);
}
                
function mod(x, modulus) {
   if (x < 0) {
      x += Math.ceil(-x/modulus)*modulus;
   }
   return x % modulus;
}

function cycle(group, dim, delta, fast, ask) {
//  alert("group = " + group);
//  alert("in cycle_groups " + (group in cycle_groups));
//log('cycle called');
  var exact = -1;
  if (ask) {
    exact = parseInt(prompt("Frame number", "0"))
  } else if (fast) {
    delta *= 10;
  }
  for (var i = 0; i < cycle_groups[group].length; i++) {
    var id = cycle_groups[group][i];
    var ims = cycle_ims[id];
    cycle_idx[id][dim] = mod((exact < 0 ? cycle_idx[id][dim] + delta : exact), (dim == 0 ? ims.length : ims[cycle_idx[id][0]].length));
    //log(cycle_idx[id]);
    // clamp the other index
    cycle_idx[id][1-dim] = Math.max(0, Math.min(cycle_idx[id][1-dim], (dim == 1 ? ims.length : ims[cycle_idx[id][0]].length)));
    document.getElementById(id).src = ims[cycle_idx[id][0]][cycle_idx[id][1]];
  }
}


// the row is controlled by the number keys; column by clicking
function switch_row(e, group) {
//log('sr');
  for (var i = 0; i < cycle_groups[group].length; i++) {
     var id = cycle_groups[group][i];
     var ims = cycle_ims[id];
     // 1 - 9 ==> 0 - 8;
     var s = 48; // ascii '1'
     var n = e.charCode - s - 1;
     if (0 <= n && n < ims.length) {
        if (!(id in cycle_idx)) {
           cycle_idx[id] = [0, 0];
        }
        cycle_idx[id][0] = n;
        cycle_idx[id][1] = mod(cycle_idx[id][1], ims[n].length);
        document.getElementById(id).src = ims[cycle_idx[id][0]][cycle_idx[id][1]];
     }
   }
}

var curr_cycle_group = null;
//document.onkeypress = function(e) {
//  if (curr_cycle_group != null) {
//    switch_row(e, curr_cycle_group);
//  }
//}

document.onkeydown = function (e) {
 if (vid_focus != null && vid_focus.paused) {
    // rough estimate
    var frame_duration = 1./30;
    var vf = vid_focus;
     // 37 39 are left/right
    if (String.fromCharCode(e.keyCode) == 'O') {
      vf.currentTime = Math.max(vf.currentTime - frame_duration, 0);
    } else if (String.fromCharCode(e.keyCode) == 'P') {
      vf.currentTime = Math.min(vf.currentTime + frame_duration, vf.duration);
    }
    return true;
  } else if (curr_cycle_group != null) {
    switch_row(e, curr_cycle_group);
  }
};


function register_cycle(group, id, ims, start) {
  if (!(id in cycle_ims)) {
    if (!(group in cycle_groups)) {
      cycle_groups[group] = [];
    }
    cycle_groups[group].push(id);
    cycle_ims[id] = ims;
    cycle_idx[id] = start;
  }
}
function getParameterByName(name) {
  name = name.replace(/[\[]/, "\\\[").replace(/[\]]/, "\\\]");
  var regex = new RegExp("[\\?&]" + name + "=([^&#]*)");
  var results = regex.exec(location.search);
  return results == null ? "" : decodeURIComponent(results[1].replace(/\+/g, " "));
}

</script>
"""

def transpose_table(lsts, pad = ''):
  return map(list, ut.zip_pad(pad, *lsts))

def show_page(fname):
  if sys.platform == 'darwin':
    # Chrome on MacOS doesn't seem to get the focus with webbrowser.open
    os.system('open %s' % fname)
  else:
    webbrowser.open(fname)

def js_list(xs):
  return '[' + ', '.join(xs) + ']'
  
def cycle_html(im_fnames_table, group_name, start):
  im_list_js = js_list([js_list(map(ut.quote, row)) for row in im_fnames_table])
  id = str(random.random())
  group_name = str(random.random()) if group_name is None else group_name
  default = im_fnames_table[start[0]][start[1]] if len(im_fnames_table) and len(im_fnames_table[0]) else ''
  #return '<img src = "%s" id = "%s" onkeydown = \'\' onclick = \'cycle("%s", %s, (event.shiftKey ? 0 : 1), 1)\', oncontextmenu = \'cycle("%s", %s, (event.shiftKey ? 0 : 1), -1); return false;\'>' % (default, id, id, im_list_js, id, im_list_js)
  return ut.frm('<img src = "%(default)s" id = "%(id)s" onmouseover = \'curr_cycle_group = "%(group_name)s";\'' \
                ' onclick = \'cycle("%(group_name)s", 1, 1, event.shiftKey, event.ctrlKey)\', oncontextmenu = \'cycle("%(group_name)s", 1, -1, event.shiftKey, event.ctrlKey); return false;\'' \
                ' onload = \'register_cycle("%(group_name)s", "%(id)s", %(im_list_js)s, %(repr(list(start)))s);\'>')
      
def paginate_table(table_rows, rows_per_page, title, output_dir):
  if rows_per_page <= 0:
    rows_per_page = int(1e100)
  split_table = list(ut.split_n(table_rows, rows_per_page))
  #html_pages = [ut.make_temp('.html', dir = output_dir) for x in split_table]
  page_names = ['index.html'] + ['page_%d.html' % i for i in xrange(2, 1+len(split_table))]
  page_paths = [os.path.join(output_dir, fname) for fname in page_names]
  for i in xrange(len(split_table)):
    table_html = '<table border = 1><tr>' + '\n<tr>'.join(split_table[i]) + '</table>'
    footer = None
    if len(split_table) == 1:
      footer = ''
    else:
      footer = ''
      footer += ("Back " if i == 0 else "<a href = '%s'>Back</a> " % page_names[i-1])
      footer += ("Next " if i == -1 + len(page_names) else "<a href = '%s'>Next</a> " % page_names[i+1])
      for j in xrange(len(split_table)):
        s = '<b>%d</b>' % (1+j) if (i == j) else str(1+j)
        footer += ("<a href = '%s'>%s</a> " % (page_names[j], s))
      footer += '<br><br><br><br>'
    ut.make_file(page_paths[i], "<html><head>%s<title>%s</title></head><body>%s<br>%s</html>" % (table_js(), title, table_html, footer))
  return page_paths

def path_from_im(x, output_dir, img_encoding = '.png'):
  if type(x) == type((1,)) and (x[0] in ('img', 'img_mv')):
    x = list(x)
    mv_im = (x[0] == 'img_mv')
    if type(x[1]) == type(''):
      x[1] = os.path.abspath(x[1])
    im = x[1]
  elif type(x) == type((1,)) and (x[0] in ('img-png',)):
    assert type(x[1]) == type('')
    path = ut.make_temp('.png', dir = output_dir)
    f = open(path, 'wb')
    f.write(x[1])
    f.close()
    mv_im = False
    im = os.path.split(path)[1]
  else:
    mv_im = False
    im = x
  
  if type(im) == type(''):
    if mv_im:
      new_path = ut.make_temp(os.path.splitext(im)[1], dir = output_dir)
      #new_path = ut.make_temp(img_encoding, dir = output_dir)
      #os.rename(im, new_path)
      ut.sys_check_silent('mv', im, new_path)
      return os.path.split(new_path)[1]
    else:
      return im
  elif type(im) == type(np.array([])):
    path = ut.make_temp(img_encoding, dir = output_dir)
    ig.save(path, im)
    # use relative path in webpage so the directory can be moved
    return os.path.split(path)[1]
  else:
    ut.fail("Don't know how to handle image type: %s" % str(type(im)))
   
def html_from_cell(x, output_dir):
  if type(x) == type(''):
    # text
    return x
  elif type(x) == type(np.array([])):
    return html_from_cell(('img', x), output_dir)
  elif hasattr(x, '__video__') and x.__video__:
    return x.make_html(output_dir)
  elif hasattr(x, '__cycle__') and x.__cycle__:
      return x.make_html(output_dir)  
  elif type(x) == type((1,)):
    if x[0] in ('img', 'img_mv', 'img-png'):
      path = path_from_im(x, output_dir)
      if len(x) >= 3 and type(x[2]) == type({}):
        opts = x[2]
      else:
        opts = dict(ut.split_n(x[2:], 2))
      # for tool-tips
      maybe_title = ('title = "%s"' % opts['title']) if 'title' in opts else ''
      html = "<img src = '%s' %s>" % (path, maybe_title)
      if 'link' in opts:
        html = ut.link(opts['link'], html)
      return html
    elif x[0] == 'animation':
      # make an animated gif
      duration = (0.5 if len(x) < 3 else float(x[2]))
      seq_fname = make_temp_animated_gif(x[1], duration = duration, dir = output_dir)
      return html_from_cell(('img_mv', seq_fname), output_dir)
    elif x[0] == 'cycle':
      # clicking: advance column; number keys: change row; shift-click: move by 10 frames; control-click: query user for column; 
      im_table = x[1]
      group = None if len(x) < 3 else x[2]

      # format: [0, 0] (default) | col | [row, col]
      if len(x) < 4:
        start = [0, 0]
      elif np.ndim(x[3]) == 0:
        start = [0, x[3]]
      else:
        start = x[3]

      if len(im_table) == 0 or type(im_table[0]) != type([]):
        im_table = [im_table]

      return cycle_html([[path_from_im(im, output_dir) for im in row] for row in im_table], group, start)

    elif x[0] == 'table':
      return '<table border = 1>%s</table>' % ('<tr>'.join(html_from_rows(x[1], output_dir)))

      #ut.fail("Invalid tuple in table.")
  elif hasattr(x, '__call__'):
    # call function; make a cell from the result
    return html_from_cell(x(), output_dir)
  # # allow table of tables
  # elif type(x) == type([]):
  #   return '<table border = 1>%s</table>' % ('<tr>'.join(html_from_rows(x, output_dir)))
  else:
    return str(x)
    #ut.fail('Invalid element in table: got element of type: %s' % type(x))

# def make_temp_animated_gif(seq, duration = 0.5, dir = None):
#   from external import images2gif
#   seq_fname = ut.make_temp('.gif', dir = dir)
#   images2gif.writeGif(seq_fname, seq, duration=duration, dither=0)
#   #images2gif.writeGif(seq_fname, map(ig.to_pil, seq), duration=duration, nq = 15, subRectangles = False, dither=0)
  
#   return seq_fname

def save_helper((fname, x)):
  ig.save(fname, x)
  
# note: I think duration = #ticks before image changes, and 0.5 probably gets mapped to 0? I'm not sure.  Setting it to 1 makes it very fast, 100 much slower.
def make_temp_animated_gif_cmdline(seq, duration = 0.5, dir = None, tmp_ext = '.ppm', seq_fname = None):
  if seq_fname is None:
    seq_fname = ut.make_temp('.gif', dir = dir)
  # avoid ffmpeg prompt
  os.system('rm %s' % seq_fname)
  base = ut.make_temp('')
  #fnames = ['%s_%d.jpg' % (base, i) for i in xrange(len(seq))]
  fnames = ['%s_%04d%s' % (base, i, tmp_ext) for i in xrange(len(seq))]
  
  # for fname, x in zip(fnames, seq):
  #   ig.save(fname, x)

  # def f((fname, x)):
  #   ig.save(fname, x)
    
  ut.parmap(save_helper, zip(fnames, seq))
  
  ut.prn_sys('convert -layers OptimizePlus -delay %f -loop 0 %s_*%s %s' % (duration, base, tmp_ext, seq_fname))
             
  #fps = 1./duration
  # ut.prn_sys('ffmpeg -r %f -i %s_%%d.jpg %s.avi' % (fps, base, base))
  # ut.prn_sys('ffmpeg -i %s.avi -pix_fmt rgb24 %s' % (base, seq_fname))
  return seq_fname
  #os.system('rm %s*.png %s*.avi' % (base, base))  

make_temp_animated_gif = make_temp_animated_gif_cmdline

def html_from_rows(table, output_dir):
  # allow single-entry table without lists
  if type(table) != type([]):
    table = [table]
  html_rows = []
  for row in table:
    # a total hack; allows for multiple tables
    if type(row) == type((1,)) and row[0] == 'break':
      html_rows.append('</table><br><table border = 1>')
    else:
      # allow single column without nesting lists
      if type(row) != type([]):
        row = [row]
      html_rows.append("<td>" + "<td>".join(html_from_cell(x, output_dir) for x in row))
      
  return html_rows

show = show_table

# def show_links(urls):
#   table = []
#   for url in urls:
#     if ut.istup(url):
#       table.append
  

def test():
  # show([[['hello', 'world'],
  #        'asdf', 'fdsa',
  #        ['a', 'b', 'c', 'd', 'e', 'f']],
  #       [['foo', 'bar'], ['x']]])
  show([('table',
         [['hello', 'world'],
          'asdf', 'fdsa',
          ['a', 'b', 'c', 'd', 'e', 'f']]),
        ('table',
         [['foo', 'bar'],
          ['x']])])

  # animated gif test
  show(('animation', [ig.make(50, 50), ig.make(50, 50, (255, 0, 0))], 0.1))

def pad_im_even(x):
  if x.shape[0] % 2 == 1 or x.shape[1] % 2 == 1:
    x = ig.pad_corner(x, x.shape[1] % 2, x.shape[0] % 2)
  return x

def make_video_helper((i, x, in_dir, tmp_ext)):
  out_fname = ut.pjoin(in_dir, 'ao-video-frame%05d%s' % (i, tmp_ext))
  if type(x) == type(''):
    f = open(out_fname, 'wb')
    f.write(x)
    f.close()
  else:
    # pad to be even
    x = pad_im_even(x)
    ig.save(out_fname, x)

# def make_video(out_fname, ims, fps, tmp_ext = '.ppm', sound = None):
#   # Compressed images
#   if type(ims[0]) == type(''):
#     #tmp_ext = '.jpg'
#     tmp_ext = '.png'
    
#   assert tmp_ext.startswith('.')
#   in_dir = ut.make_temp_dir_big()
#   ut.parmap(make_video_helper, 
#             [(i, x, in_dir, tmp_ext) for i, x in enumerate(ims)])

#   with ut.temp_file('.wav') as tmp_wav:
#     if sound is None:
#       sound_flags = ''
#     else:
#       if type(sound) != type(''):
#         sound.save(tmp_wav)
#         sound = tmp_wav
#       sound_flags = '-i "%s" -shortest' % sound

#     cmd = ('ffmpeg -loglevel warning -f image2 -r %f -i %s/ao-video-frame%%05d%s %s '
#            '-pix_fmt yuv420p -vcodec h264 -acodec aac -strict -2 -y %s') % (fps, in_dir, tmp_ext, sound_flags, out_fname)
  
#     print cmd
#     if 0 == os.system(cmd):
#       for x in glob.glob(ut.pjoin(in_dir, 'ao-video-frame*%s' % tmp_ext)):
#         os.remove(x)
#       os.rmdir(in_dir)
#     else:
#       raise RuntimeError()

def make_video(out_fname, ims, fps, tmp_ext = '.ppm', sound = None):
  # Compressed images
  if type(ims[0]) == type(''):
    #tmp_ext = '.jpg'
    tmp_ext = '.png'
    
  assert tmp_ext.startswith('.')
  in_dir = ut.make_temp_dir_big()
  ut.parmap(make_video_helper, 
            [(i, x, in_dir, tmp_ext) for i, x in enumerate(ims)])

  with ut.temp_file('.wav') as tmp_wav:
    if sound is None:
      sound_flags = ''
    else:
      if type(sound) != type(''):
        sound.save(tmp_wav)
        sound = tmp_wav
      sound_flags = '-i "%s" -shortest' % sound

    # removed aac flag from above, for compatibility with some wav files
    # cmd = ('ffmpeg -loglevel warning -f image2 -r %f -i %s/ao-video-frame%%05d%s %s '
    #        '-pix_fmt yuv420p -vcodec h264 -strict -2 -y %s') % (fps, in_dir, tmp_ext, sound_flags, out_fname)
    cmd = ('ffmpeg -loglevel error -f image2 -r %f -i %s/ao-video-frame%%05d%s %s '
           '-pix_fmt yuv420p -ar 22050 -vcodec h264 -strict -2 -y %s') % (fps, in_dir, tmp_ext, sound_flags, out_fname)
  
    print cmd
    if 0 == os.system(cmd):
      for x in glob.glob(ut.pjoin(in_dir, 'ao-video-frame*%s' % tmp_ext)):
        os.remove(x)
      os.rmdir(in_dir)
    else:
      raise RuntimeError()

def convert_video(in_fname, out_fname):
  # cmd = ('ffmpeg -loglevel warning -i "%s" '
  #        '-pix_fmt yuv420p -vcodec h264 -acodec aac -strict -2 -y %s') % (in_fname, out_fname)
  cmd = ('ffmpeg -loglevel fatal -i "%s" '
         '-pix_fmt yuv420p -vcodec h264  -strict -2 -y %s') % (in_fname, out_fname)
  print cmd
  os.system(cmd)
  
def test_video_cycle():
  ims = ['/afs/csail.mit.edu/u/a/aho/bear.jpg',
         '/afs/csail.mit.edu/u/a/aho/test.jpg',
         '/afs/csail.mit.edu/u/a/aho/me_cropped.jpg']
  ims = [ig.load(x)[:120,:120] for x in ims]
  show([Video(ims)])
  print 'cycle'
  show([Cycle(ims, None)])
  #show([Video(map(ig.load, ims))])

def sync_with_fps(ims, timestamps, target_fps, max_lag_sec = 0.06):
  vid = []
  timestamps = np.array(timestamps, 'd')
  #timestamps -= timestamps[0] 
  i = 0
  while i < len(ims):
    vid.append(ims[i])
    video_time = lambda : (len(vid) - 1) * (1./target_fps)
    #print video_time(), timestamps[i]
    # We're over time; skip frames until we catch up with the timestamps
    while i < len(timestamps) and video_time() > timestamps[i] + max_lag_sec:
      print 'Over time. Skipping frames to catch up.'
      i += 1

    # We're far ahead of the true time.  Keep showing the same image until we reach it again.
    while video_time() < timestamps[i] - max_lag_sec:
      print 'Under time. Repeating the last frame to slow down.'
      vid.append(ims[i])

    i += 1
  
  return vid

class Video:
  def __init__(self, ims = None, fps = 1, sound = None, timestamps = None,
               video_fname = None, copy_method = 'copy', video_url = None):
    self.__video__ = True
    self.video_fname = video_fname
    self.video_url = video_url
    if video_fname is not None:
      self.copy_method = copy_method
      #return
    #print 'init video'
    if timestamps is None:
      self.ims = ims
    else:
      self.ims = sync_with_fps(ims, timestamps, fps)

    self.fps = fps
    self.sound = sound

  def make_html(self, out_dir, ext = '.mp4'):
    if self.video_url is None:
      if self.video_fname is None:
        fname = ut.make_temp(ext, dir = out_dir)
        os.remove(fname)
        make_video(fname, self.ims, self.fps, sound = self.sound)
      else:
        fname = ut.make_temp(ext, dir = out_dir)
        os.remove(fname)
        if self.copy_method == 'copy':
          os.system('cp "%s" %s' % (self.video_fname, fname))
        elif self.copy_method == 'symlink':
          os.system('ln -s "%s" %s' % (self.video_fname, fname))
        else:
          raise RuntimeError()
      rel_fname = fname.split('/')[-1]
    else:
      rel_fname = self.video_url
      
    vid_id = str(random.random())

    return """
    <table>
      <tr>
        <td>
          <video controls id = "%(vid_id)s" onplay = "vid_focus = null;" onpause = "vid_focus = document.getElementById('%(vid_id)s');" onloadedmetadata = "var t = getParameterByName('videoStart'); if (t != null) { this.currentTime = t; } if (getParameterByName('videoAutoplay')) { this.play(); }">
          <source src = "%(rel_fname)s" type = "video/mp4" preload = "none"> </video>
        </td>
        <td>
          Speed: <input type = "text" size = "5" value = "1" oninput = "document.getElementById('%(vid_id)s').playbackRate = parseFloat(event.target.value);">
        </td>
      </tr>
    </table>
          """ % locals()

  def save(self, fname):
    assert fname.endswith('.mp4')
    if os.path.exists(fname):
      os.remove(fname)
    make_video(fname, self.ims, self.fps, sound = self.sound)
    
class Cycle:
  def __init__(self, ims, group = None):
    self.ims = ims
    self.group = group
    self.start = [0,0]
    self.__cycle__ = True

  def make_html(self, output_dir):
    if len(self.ims) == 0 or type(self.ims[0]) != type([]):
      ims = [self.ims]
      
    return cycle_html([[path_from_im(im, output_dir) for im in row] for row in ims], self.group, self.start)

# def url_from_path(path):
#   path.split('/'
#   path.replace(get_www_path(), 


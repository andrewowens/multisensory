import numpy as np
import os, pylab
import itertools as itl
from PIL import Image, ImageDraw, ImageFont
import util as ut
import scipy.misc, scipy.misc.pilutil # not sure if this is necessary
import scipy.ndimage
from StringIO import StringIO
#import cv

def show(*args, **kwargs):
  import imtable
  return imtable.show(*args, **kwargs)

# Functional code for drawing on images:
def draw_on(f, im):
  pil = to_pil(im)
  draw = ImageDraw.ImageDraw(pil)
  f(draw)
  return from_pil(pil)

def color_from_string(s):
  """ todo: add more, see matplotlib.colors.cnames """
  colors = {'r' : (255, 0, 0), 'g' : (0, 255, 0), 'b' : (0, 0, 255)}
  if s in colors:
    return colors[s]
  else:
    ut.fail('unknown color: %s' % s)

def parse_color(c):
  if type(c) == type((0,)) or type(c) == type(np.array([1])):
    return c
  elif type(c) == type(''):
    return color_from_string(c)
  
def colors_from_input(color_input, default, n):
  """ Parse color given as input argument; gives user several options """
  # todo: generalize this to non-colors
  expanded = None
  if color_input is None:
    expanded = [default] * n
  elif (type(color_input) == type((1,))) and map(type, color_input) == [int, int, int]:
    # expand (r, g, b) -> [(r, g, b), (r, g, b), ..]
    expanded = [color_input] * n
  else:
    # general case: [(r1, g1, b1), (r2, g2, b2), ...]
    expanded = color_input

  expanded = map(parse_color, expanded)
  return expanded
  
def draw_rects(im, rects, outlines = None, fills = None, texts = None, text_colors = None, line_widths = None, as_oval = False):
  rects = list(rects)
  outlines = colors_from_input(outlines, (0, 0, 255), len(rects))
  text_colors = colors_from_input(text_colors, (255, 255, 255), len(rects))
  fills = colors_from_input(fills, None, len(rects))
  
  if texts is None: texts = [None] * len(rects)
  if line_widths is None: line_widths = [None] * len(rects)
  
  def check_size(x, s): ut.check(x is None or len(x) == len(rects), "%s different size from rects" % s)
  check_size(outlines, 'outlines')
  check_size(fills, 'fills')
  check_size(texts, 'texts')
  check_size(text_colors, 'texts')
  
  def f(draw):
    for (x, y, w, h), outline, fill, text, text_color, lw in itl.izip(rects, outlines, fills, texts, text_colors, line_widths):
      if lw is None:
        if as_oval:
          draw.ellipse((x, y, x + w, y + h), outline = outline, fill = fill)
        else:
          draw.rectangle((x, y, x + w, y + h), outline = outline, fill = fill)
      else:
        # TODO: to do this right, we need to find where PIL draws the corners
        # x -= lw
        # y -= lw
        # w += 2*lw
        # h += 2*lw
        # pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        # for i in xrange(len(pts)):
        #   #draw.line(pts[i] + pts[(i+1)%4], fill = outline, width = lw)
        #   draw.rectangle(pts[i] + pts[(i+1)%4], fill = outline, width = lw)
        d = int(np.ceil(lw/2))
        draw.rectangle((x-d, y-d, x+w+d, y+d), fill = outline)
        draw.rectangle((x-d, y-d, x+d, y+h+d), fill = outline)
        
        draw.rectangle((x+w+d, y+h+d, x-d, y+h-d), fill = outline)
        draw.rectangle((x+w+d, y+h+d, x+w-d, y-d), fill = outline)
          
      if text is not None:
        # draw text inside rectangle outline
        border_width = 2
        draw.text((border_width + x, y), text, fill = text_color)
  return draw_on(f, im)


def draw_rects_scale(sc, im, rects, outlines = None, fills = None, texts = None, text_colors = None):
  scaled_rects = []
  for r in rects:
    r = np.array(r)
    sr = r * sc
    if r[2] >= 1 and r[3] >= 1:
      sr[2:] = np.maximum(sr[2:], 1.)
    scaled_rects.append(sr)
    
  return draw_rects(scale(im, sc), scaled_rects, outlines, fills, texts, text_colors)
  
def draw_pts(im, points, colors = None, width = 1, texts = None):
  #ut.check(colors is None or len(colors) == len(points))
  points = list(points)
  colors = colors_from_input(colors, (255, 0, 0), len(points))
  rects = [(p[0] - width/2, p[1] - width/2, width, width) for p in points]
  return draw_rects(im, rects, fills = colors, outlines = [None]*len(points), texts = texts)

def draw_lines(im, pts1, pts2, colors = None, width = 0):
  ut.check(len(pts1) == len(pts2), 'Line endpoints different sizes')
  colors = colors_from_input(colors, None, len(pts1))
  def f(draw):
    for p1, p2, c in itl.izip(pts1, pts2, colors):
      draw.line(ut.int_tuple(p1) + ut.int_tuple(p2), fill = c, width = width)
  return draw_on(f, im)

def draw_text(im, texts, pts, colors, font_size = None, bold = False):
  im = rgb_from_gray(im)
  # todo: add fonts, call from draw_rects
  ut.check(len(pts) == len(texts))
  #ut.check((colors is None) or len(colors) == len(texts))
  colors = colors_from_input(colors, (0, 0, 0), len(texts))
  def f(draw):
    if font_size is None:
      font = None
    else:
      #font_name = '/usr/share/fonts/truetype/ttf-liberation/LiberationMono-Regular.ttf'
      font_choices = ['/usr/share/fonts/truetype/freefont/FreeMono%s.ttf' % ('Bold' if bold else ''), '/Library/Fonts/PTMono.ttc']
      for font_name in font_choices:
        if os.path.exists(font_name):
          break
      else:
        raise RuntimeError('could not find a suitable font on this machine (please edit paths in img.py)')

      font = ImageFont.truetype(font_name, size = font_size)  

    for pt, text, color in itl.izip(pts, texts, colors):
      draw.text(ut.int_tuple(pt), text, fill = color, font = font)
  return draw_on(f, im)

def draw_text_ul(im, text, color = (0, 255, 0), font_size = 25):
  return draw_text(im, [text], [(0, 0)], [color], font_size = font_size)
  
def luminance(im):
  if len(im.shape) == 2:
    return im
  else:
    # see http://www.mathworks.com/help/toolbox/images/ref/rgb2gray.html
    return np.uint8(np.round(0.2989 * im[:,:,0] + 0.587 * im[:,:,1] + 0.114 * im[:,:,2]))

#def sub_img(im, x_or_rect, y = None, w = None, h = None):
def sub_img(im, x_or_rect, y = None, w = None, h = None):
  if x_or_rect is None:
    return im
  elif y is None:
    x, y, w, h = x_or_rect
  else:
    x = x_or_rect
  return im[y : y + h, x : x + w]

def sub_img_frac(im, x_or_rect, y = None, w = None, h = None):
  if y is None:
    x, y, w, h = x_or_rect
  else:
    x = x_or_rect
  x = int(x*im.shape[1])
  y = int(y*im.shape[0])
  w = int(w*im.shape[1])
  h = int(h*im.shape[0])
  return im[y : y + h, x : x + w]

# def stack_img_pair(im1, im2):
#   h1, w1 = im1.shape[:2]
#   h2, w2 = im2.shape[:2]
#   im3 = np.zeros((max(h1, h2), w1 + w2, 3), dtype = im1.dtype)
#   im3[:h1, :w1, :] = rgb_from_gray(im1)
#   im3[:h2, w1:, :] = rgb_from_gray(im2)
#   return im3

# def stack_imgs(ims):
#   """ slow, should rewrite """
#   assert len(ims) > 0
#   res = ims[0]
#   for im in ims[1:]:
#     res = stack_img_pair(res, im)
#   return res

# def hstack_ims(ims):
#   max_h = max(im.shape[0] for im in ims)
#   result = []
#   for im in ims:
#     frame = np.zeros((max_h, im.shape[1], 3))
#     frame[:im.shape[0],:im.shape[1]] = rgb_from_gray(im)
#     result.append(frame)
#   return np.hstack(result)

def hstack_ims(ims, bg_color = (0, 0, 0)):
  max_h = max([im.shape[0] for im in ims])
  result = []
  for im in ims:
    #frame = np.zeros((max_h, im.shape[1], 3))
    frame = make(im.shape[1], max_h, bg_color)
    frame[:im.shape[0],:im.shape[1]] = rgb_from_gray(im)
    result.append(frame)
  return np.hstack(result)

# def hstack_ims_mult(*all_ims):
#   max_h = max(max(im.shape[0] for im in ims) for ims in all_ims)
  
#   result = []
#   for im in ims:
#     frame = np.zeros((max_h, im.shape[1], 3))
#     frame[:im.shape[0],:im.shape[1]] = rgb_from_gray(im)
#     result.append(frame)
#   return np.hstack(result)

def vstack_ims(ims, bg_color = (0, 0, 0)):
  if len(ims) == 0:
    return make(0, 0)
  
  max_w = max([im.shape[1] for im in ims])
  result = []
  for im in ims:
    #frame = np.zeros((im.shape[0], max_w, 3))
    frame = make(max_w, im.shape[0], bg_color)
    frame[:im.shape[0],:im.shape[1]] = rgb_from_gray(im)
    result.append(frame)
  return np.vstack(result)

def make_rgb(im):
  im = rgb_from_gray(im, False)
  if im.shape[2] < 3:
    raise RuntimeError()
  elif im.shape[2] > 3:
    im = im[:, :, :3]
  return im
  
def rgb_from_gray(img, copy = True, remove_alpha = True):
  if img.ndim == 3 and img.shape[2] == 3:
    return img.copy() if copy else img
  elif img.ndim == 3 and img.shape[2] == 4:
    return (img.copy() if copy else img)[..., :3]
  elif img.ndim == 3 and img.shape[2] == 1:
    return np.tile(img, (1,1,3))
  elif img.ndim == 2:
    return np.tile(img[:,:,np.newaxis], (1,1,3))
  else:
    raise RuntimeError('Cannot convert to rgb. Shape: ' + str(img.shape))

def load(im_fname, gray = False):
  if im_fname.endswith('.gif'):
    print "GIFs don't load correctly for some reason"
    ut.fail('fail')
  im = from_pil(Image.open(im_fname))
  # use imread, then flip upside down
  #im = np.array(list(reversed(pylab.imread(im_fname)[:,:,:3])))
  if gray:
    return luminance(im)
  elif not gray and np.ndim(im) == 2:
    return rgb_from_gray(im)
  else:
    return im
imread = load

def loadsc(fname, scale, gray = False):
  return resize(load(fname, gray = gray), scale)
  
def save(img_fname, a):
  if img_fname.endswith('jpg'):
    return Image.fromarray(np.uint8(a)).save(img_fname, quality = 100)
  else:
    #return Image.fromarray(np.uint8(a)).save(img_fname)
    return Image.fromarray(np.uint8(a)).save(img_fname, quality = 100)

# def make_temp_file(ext):
#   fd, fname = tempfile.mkstemp(ext)
#   # shouldn't delete file
#   os.close(fd)
#   return fname

# def make_pretty(img):
#   if img.dtype == 'bool':
#     return img * 255
#   elif (0 <= np.min(img)) and (np.max(img) <= 1.0):
#     return img*255
#   return img

def show_html(html):
  page = ut.make_temp('.html')
  ut.make_file(page, html)
  print 'opening', page
  webbrowser.open(page)

# # http://opencv.willowgarage.com/wiki/PythonInterface
# def cv2array(im):
#   depth2dtype = {
#       cv.IPL_DEPTH_8U: 'uint8',
#       cv.IPL_DEPTH_8S: 'int8',
#       cv.IPL_DEPTH_16U: 'uint16',
#       cv.IPL_DEPTH_16S: 'int16',
#       cv.IPL_DEPTH_32S: 'int32',
#       cv.IPL_DEPTH_32F: 'float32',
#       cv.IPL_DEPTH_64F: 'float64',
#     }

#   arrdtype=im.depth
#   a = np.fromstring(
#        im.tostring(),
#        dtype=depth2dtype[im.depth],
#        count=im.width*im.height*im.nChannels)
#   a.shape = (im.height,im.width,im.nChannels)
#   return a

# def to_cv(a):
#   dtype2depth = {
#       'uint8':   cv.IPL_DEPTH_8U,
#       'int8':  cv.IPL_DEPTH_8S,
#       'uint16':  cv.IPL_DEPTH_16U,
#       'int16':   cv.IPL_DEPTH_16S,
#       'int32':   cv.IPL_DEPTH_32S,
#       'float32': cv.IPL_DEPTH_32F,
#       'float64': cv.IPL_DEPTH_64F,
#     }
#   try:
#     nChannels = a.shape[2]
#   except:
#     nChannels = 1
#   cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]),
#       dtype2depth[str(a.dtype)],
#       nChannels)
#   cv.SetData(cv_im, a.tostring(),
#          a.dtype.itemsize*nChannels*a.shape[1])
#   return cv_im

#def to_pil(im): return Image.fromarray(np.uint8(im))
def to_pil(im): 
  #print im.dtype
  return Image.fromarray(np.uint8(im))

def from_pil(pil): 
  #print pil
  return np.array(pil)

def to_pylab(a): return np.uint8(a)

def test_draw_text():
  im = 255 + np.zeros((300, 300, 3))
  show([draw_text(im, ['hello', 'world'], [(100, 200), (0, 0)], [(255, 0, 0), (0, 255, 0)]),
        draw_text(im, ['hello', 'world'], [(100, 100), (0, 0)], [(255, 0, 0), (0, 255, 0)], font_size = 12)])
  
      
def save_tmp(im, encoding = '.png', dir = None):
  fname = ut.make_temp(encoding, dir = dir)
  save(fname, im)
  return fname

def save_tmp_nfs(im, encoding = '.png'):
  return save_tmp(im, encoding, '/csail/vision-billf5/aho/tmp')
  
# def resize(im, size):
#   if type(size) == type(1):
#     size = float(size)
#   #return scipy.misc.pilutil.imresize(im, size)
#   return scipy.misc.imresize(im, size)


#def resize(im, scale, order = 3, hires = 'auto'):

def resize(im, scale, order = 3, hires = False):
  if hires == 'auto':
    hires = (im.dtype == np.uint8)

  if np.ndim(scale) == 0:
    new_scale = [scale, scale]
  # interpret scale as dimensions; convert integer size to a fractional scale
  elif ((scale[0] is None) or type(scale[0]) == type(0)) \
           and ((scale[1] is None) or type(scale[1]) == type(0)) \
           and (not (scale[0] is None and scale[1] is None)):
    # if the size of only one dimension is provided, scale the other to maintain the right aspect ratio
    if scale[0] is None:
      dims = (int(float(im.shape[0])/im.shape[1]*scale[1]),  scale[1])
    elif scale[1] is None:
      dims = (scale[0], int(float(im.shape[1])/im.shape[0]*scale[0]))
    else:
      dims = scale[:2]
      
    new_scale = [float(dims[0] + 0.4)/im.shape[0], float(dims[1] + 0.4)/im.shape[1]]
    # a test to make sure we set the floating point scale correctly
    result_dims = [int(new_scale[0]*im.shape[0]), int(new_scale[1]*im.shape[1])]
    assert tuple(result_dims) == tuple(dims)
  elif type(scale[0]) == type(0.) and type(scale[1]) == type(0.):
    new_scale = scale
    #new_scale = scale[1], scale[0]
  else:
    raise RuntimeError("don't know how to interpret scale: %s" % (scale,))
    # want new scale' to be such that
    # int(scale'[0]*im.shape[0]) = scale[0], etc. (that's how zoom computes the new shape)
    # todo: any more numerical issues?
    #print 'scale before', im.shape, scale
    # print 'scale after', scale
    # print 'new image size', [int(scale[0]*im.shape[0]),int(scale[1]*im.shape[1])]
  #scale_param = new_scale if im.ndim == 2 else (new_scale[0], new_scale[1], 1)
  scale_param = new_scale if im.ndim == 2 else (new_scale[0], new_scale[1], 1)

  if hires:
    #sz = map(int, (scale_param*im.shape[1], scale_param*im.shape[0]))
    sz = map(int, (scale_param[1]*im.shape[1], scale_param[0]*im.shape[0]))
    return from_pil(to_pil(im).resize(sz, Image.ANTIALIAS))
  else:
    res = scipy.ndimage.zoom(im, scale_param, order = order)
    # verify that zoom() returned an image of the desired size
    if (np.ndim(scale) != 0) and type(scale[0]) == type(0) and type(scale[1]) == type(0):
      assert res.shape[:2] == (scale[0], scale[1])
    return res

scale = resize

# import skimage
# resize = skimage.imresize

def test_resize():
  im = make(44, 44)
  assert resize(im, (121, 120, 't')).shape[:2] == (121, 120)
  assert resize(im, (2., 0.5, 't')).shape[:2] == (88, 22)
  
def show_file(fname):
  show(load(fname))

def img_extensions():
    return ['png', 'gif', 'jpg', 'jpeg', 'bmp', 'ppm', 'pgm']

def is_img_file(fname):
    return any(fname.lower().endswith(ext) for ext in img_extensions())

def blur(im, sigma):
  if np.ndim(im) == 2:
    return scipy.ndimage.filters.gaussian_filter(im, sigma)
  else:
    return np.concatenate([scipy.ndimage.filters.gaussian_filter(im[:, :, i], sigma)[:, :, np.newaxis] for i in xrange(im.shape[2])], axis = 2)

def blit(src, dst, x, y, opt = None):
  if opt == 'center':
    x -= src.shape[1]/2
    y -= src.shape[0]/2
  # crop intersecting 
  dx, dy, dw, dh = ut.crop_rect_to_img((x, y, src.shape[1], src.shape[0]), dst)
  sx = dx - x
  sy = dy - y
  
  dst[dy : dy + dh, dx : dx + dw] = src[sy : sy + dh, sx : sx + dw]

def weighted_add(src, dst, x, y, src_weight, dst_weight, opt = None):
  if opt == 'center':
    x -= src.shape[1]/2
    y -= src.shape[0]/2
  # crop intersecting 
  dx, dy, dw, dh = ut.crop_rect_to_img((x, y, src.shape[1], src.shape[0]), dst)
  sx = dx - x
  sy = dy - y
  dst[dy : dy + dh, dx : dx + dw] = dst[dy : dy + dh, dx : dx + dw]*dst_weight + src[sy : sy + dh, sx : sx + dw]*src_weight

def make(w, h, fill = (0,0,0)):
  return np.uint8(np.tile([[fill]], (h, w, 1)))

def luminance_rgb(im): return rgb_from_gray(luminance(im))

def rotate(img, angle, fill = 0):
    """ Rotate image around its center by the given angle (in
    radians).  No interpolation is used; indices are rounded.  The
    returned image may be larger than the original, but the middle
    pixel corresponds to the middle of the original.  Pixels with no
    correspondence are filled as 'fill'.

    Also returns mapping from original image to rotated. """
    r = int(np.ceil(np.sqrt(img.shape[0]**2 + img.shape[1]**2)))
    X, Y = np.mgrid[0:r, 0:r]
    X = X.flatten()
    Y = Y.flatten()
    X2 = np.array(np.round(img.shape[1]/2 + np.cos(angle) * (X - r/2) - np.sin(angle) * (Y - r/2)), dtype = int)
    Y2 = np.array(np.round(img.shape[0]/2 + np.sin(angle) * (X - r/2) + np.cos(angle) * (Y - r/2)), dtype = int)
    good = ut.logical_and_many(X2 >= 0, X2 < img.shape[1], Y2 >= 0, Y2 < img.shape[0])
    out = fill + np.zeros((r, r) if img.ndim == 2 else (r, r, img.shape[2]), dtype = img.dtype)
    out[Y[good], X[good]] = img[Y2[good], X2[good]]
    T = np.dot(np.dot(ut.rigid_transform(np.eye(2), [img.shape[1]/2, img.shape[0]/2]),
                      ut.rigid_transform(ut.rotation_matrix2(angle))),
               ut.rigid_transform(np.eye(2), [-r/2, -r/2]))
    return out, np.linalg.inv(T)

def map_img(f, im, dtype = None, components = None):
  new_im = np.zeros(im.shape if components is None else im.shape + (components,), \
                    dtype = im.dtype if dtype is None else dtype)
  for y in xrange(im.shape[0]):
    for x in xrange(im.shape[1]):
      new_im[y,x] = f(im[y,x])
  return new_im

def add_border(img, w, h, color = (0, 0, 0)):
  assert 0 <= w
  assert 0 <= h
  out = make(img.shape[1] + 2*w, img.shape[0] + 2*h, color)
  out[h:(h + img.shape[0]), w : (w + img.shape[1])] = img
  return out

def pad_corner(im, pw, ph, color = (0, 0, 0)):
  out = make(im.shape[1] + pw, im.shape[0] + ph, color)
  out[:im.shape[0], :im.shape[1]] = im
  return out
  
def expand(im, new_shape, opt = 'center'):
  if type(new_shape) == type(0.):
    new_w = int(im.shape[1]*new_shape)
    new_h = int(im.shape[0]*new_shape)
  elif type(new_shape) == type((1,)):
    new_shape = new_shape[:2]
    new_h, new_w = new_shape
  else:
    raise RuntimeError("Don't know how to interpret shape")
    
  if im.shape[0] >= new_h and im.shape[1] >= new_w:
    return im.copy()
  else:
    im = rgb_from_gray(im)
    r = make(new_w, new_h)
    if opt == 'center':
      blit(im, r, im.shape[1]/2, im.shape[0]/2, opt = 'center')
    elif opt == 'corner':
      r[:im.shape[0], :im.shape[1]] = im
    return r

def combine_rgb(r, g, b):
  a = np.zeros(r.shape + (3,))
  a[:,:,0] = r
  a[:,:,1] = g
  a[:,:,2] = b
  return a

def compute_pyramid(ptm, interval, min_size):
  # based on pff's featpyramid.m
  # todo: upsample one level
  sc = 2**(1.0/interval)
  imsize = im.shape[:2]
  max_scale = int(1 + np.floor(np.log(np.min(imsize)/min_size)/np.log(sc)))
  ims = [None]*max_scale
  scale = [None]*len(ims)

  # skipping 2x scale
  for i in xrange(1, interval+1):
    im_scaled = resize(ptm, 1/sc**(i-1))
    ims[-1 + i] = im_scaled
    scale[-1 + i] = 1/sc**(i-1)
    for j in xrange(i+interval, max_scale+1, interval):
      im_scaled = resize(im_scaled, 0.5)
      ims[-1 + j] = im_scaled
      scale[-1 + j] = 0.5*scale[-1 + j - interval]
  assert None not in ims
  return ims, scale

#imrotate = scipy.misc.imrotate

def imrotate(*args):
  import warnings
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    return scipy.misc.imrotate(*args)
  
def from_fig_slow(fig = None, tight = True):
  ext = 'png'
  if fig is None:
    fig = pylab.gcf()
  IO = StringIO()
  if tight:
    pylab.savefig(IO, format = ext, bbox_inches = 'tight')
  else:
    pylab.savefig(IO, format = ext)
  IO.seek(0)
  return from_pil(Image.open(IO))

# def from_fig_fast(fig = None, tight = True):
#   ext = 'raw'
#   if fig is None:
#     fig = pylab.gcf()
#   IO = StringIO()
#   pylab.savefig(IO, format = ext)
#   IO.seek(0)
#   w, h = fig.canvas.get_width_height()
#   return np.fromstring(IO.buf, dtype = np.uint8).reshape((600, -1, 4))

# def from_fig(fig = None):
#   """
#   http://www.icare.univ-lille1.fr/wiki/index.php/How_to_convert_a_matplotlib_figure_to_a_numpy_array_or_a_PIL_image
#   @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
#   @param fig a matplotlib figure
#   @return a numpy 3D array of RGBA values
#   """

#   if fig is None:
#     fig = pylab.gcf()
    
#   # draw the renderer
#   fig.canvas.draw()
  
#   # Get the RGBA buffer from the figure
#   w,h = fig.canvas.get_width_height()
#   buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
#   buf.shape = (h, w, 4)
  
#   # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
#   buf = np.roll(buf, 3, axis = 2)
#   return buf
#   #return buf[..., 1:]

# def from_fig(fig = None):
#   """
#   @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
#   @param fig a matplotlib figure
#   @return a numpy 3D array of RGBA values
#   http://www.icare.univ-lille1.fr/wiki/index.php/How_to_convert_a_matplotlib_figure_to_a_numpy_array_or_a_PIL_image  
#   """
#   if fig is None:
#     fig = pylab.gcf()
    
#   # draw the renderer
#   fig.canvas.draw()
  
#   # Get the RGBA buffer from the figure
#   w,h = fig.canvas.get_width_height()
#   buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
#   buf.shape = (h, w, 4)
  
#   # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
#   buf = np.roll(buf, 3, axis = 2)

#   # not sure how to set the background to white
#   p =  buf[:, :, 3] / 255.
#   buf = np.array(buf[:, :, :3] * p[:, :, np.newaxis] + (1 - p)[:, :, np.newaxis]*255, 'uint8')
#   return buf

def from_fig(fig = None, size_inches = None):
  """
  @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
  @param fig a matplotlib figure
  @return a numpy 3D array of RGBA values
  http://www.icare.univ-lille1.fr/wiki/index.php/How_to_convert_a_matplotlib_figure_to_a_numpy_array_or_a_PIL_image  
  """
  if fig is None:
    fig = pylab.gcf()
  if size_inches is not None:
    fig.set_size_inches(*size_inches)
  # draw the renderer
  fig.canvas.draw()
  
  # Get the RGBA buffer from the figure
  w,h = fig.canvas.get_width_height()
  buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
  buf.shape = (h, w, 4)
  
  # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
  buf = np.roll(buf, 3, axis = 2)

  # not sure how to set the background to white
  p =  buf[:, :, 3] / 255.
  buf = np.array(buf[:, :, :3] * p[:, :, np.newaxis] + (1 - p)[:, :, np.newaxis]*255, 'uint8')
  return buf

def show_fig():
  show(from_fig())
  
def scale_vals(A, lo, hi):
  return np.uint8(255*(np.clip(A, lo, hi) - lo) / float(hi - lo))

def merge_ims(srcs, pts_or_rects, bg, opt = None):
  """ Makes a new image where each image in patches is copied at a
  corresponding pixel location.  Overlapping images are averaged
  together. """
  dst = rgb_from_gray(bg)
  layer = np.zeros(dst.shape)
  #counts = np.zeros(dst.shape[:2], 'l')
  counts = np.zeros(dst.shape[:2], 'd')
  for src, r in itl.izip(srcs, pts_or_rects):
    r = ut.int_tuple(r)
    x, y = r[:2]

    # rescale if we're given a rectangle, and it has a different size
    if len(r) > 2:
      assert len(r) == 4
      assert opt != 'center'
      if src.shape[:2] != (r[3], r[2]):
        src = resize(src, (r[3], r[2]))
    elif opt == 'center':
      x -= src.shape[1]/2
      y -= src.shape[0]/2

    # crop intersecting 
    dx, dy, dw, dh = ut.crop_rect_to_img((x, y, src.shape[1], src.shape[0]), dst)
    sx = dx - x
    sy = dy - y
    layer[dy : dy + dh, dx : dx + dw] += src[sy : sy + dh, sx : sx + dw, :3]
    if np.ndim(src) == 3 and src.shape[2] == 4:
      counts[dy : dy + dh, dx : dx + dw] += np.array(src[sy : sy + dh, sx : sx + dw, 3],'d')/255.
    else:
     counts[dy : dy + dh, dx : dx + dw] += 1
  dst[counts > 0] = layer[counts > 0] / counts[counts > 0][:, np.newaxis]
  return dst

def label_im(im, text, color = (0, 255, 0)):
  return draw_text(im, [text], [(25, im.shape[0] - 25)], [color])

def remap_color(im, xy):
  assert im.shape[:2] == xy.shape[:2]
  assert xy.shape[2] == 2
  vals = []
  for i in xrange(im.shape[2]):
    dx = xy[..., 0].flatten()[np.newaxis, :]
    dy = xy[..., 1].flatten()[np.newaxis, :]
    v = scipy.ndimage.map_coordinates(im[..., i], np.concatenate([dy, dx]))
    vals.append(v.reshape(im.shape[:2] + (1,)))
  return np.concatenate(vals, axis = 2)
    
  
def stack_meshgrid(xs, ys, dtype = 'l'):
  x, y = np.meshgrid(xs, ys)
  return np.array(np.concatenate([x[..., np.newaxis], y[..., np.newaxis]], axis = 2), dtype = dtype)

def sub_img_pad(im, (x, y, w, h), oob = 0):
  if len(im.shape) == 2:
    dst = np.zeros((h, w))
  else:
    dst = np.zeros((h, w, im.shape[2]))

  dst[:] = oob
  sx, sy, sw, sh = ut.crop_rect_to_img((x, y, w, h), im)
  dst[(sy - y) : (sy - y) + sh,
      (sx - x) : (sx - x) + sw] = im[sy : sy + sh, sx : sx + sw]
  return dst

def sub_img_reflect(im, (x, y, w, h)):
  x, y, w, h = map(ut.iround, [x, y, w, h])
  yy, xx = np.mgrid[y : y + h, x : x + w]
  vals = np.uint8(lookup_bilinear(im, xx.flatten(), yy.flatten(), order = 0, mode = 'reflect'))
  return vals.reshape((h, w, im.shape[2]))
  
def compress(im, format = 'png'):
  out = StringIO()
  im = to_pil(im)
  im.save(out, format = format)
  c = out.getvalue()
  out.close()
  return c

def compress_jpeg(im, format = 'jpeg'):
  return compress(im, format)

def uncompress(s):
  return from_pil(Image.open(StringIO(s)))

# def cv_uncompress(s):
#   import cv2
#   #return cv2.imdecode(s)

#   a = cv2.imdecode(np.fromstring(s, np.uint8), cv2.IMREAD_COLOR)#
#   b = uncompress(s)
#   print a.shape, b.shape
#   print np.mean(np.abs(a.astype('float32')-b.astype('float32')))
#   return a
  

def test_compress():
  im = load('/afs/csail.mit.edu/u/a/aho/bear.jpg')
  print 'orig', ut.guess_bytes(im)
  s = compress(im)
  print 'comp', ut.guess_bytes(s)
  assert(np.all(im == uncompress(s)))
  
def mix_ims(im1, im2, mask, alpha = 0.5):
  im1 = im1.copy()
  im2 = np.asarray(im2)
  if len(im2) == 3:
    # single color
    im1[mask] = im1[mask]*alpha + im2*(1-alpha)
  else:
    im1[mask] = im1[mask]*alpha + im2[mask]*(1-alpha)
  return im1

#def lookup_bilinear(im, x, y, order = 3, mode = 'constant', cval = 0.0):
def lookup_bilinear(im, x, y, order = 1, mode = 'constant', cval = 0.0):
  yx = np.array([y, x])
  if np.ndim(im) == 2:
    return scipy.ndimage.map_coordinates(im, yx, order = order, mode = mode, cval = cval)
  else:
    return np.concatenate([scipy.ndimage.map_coordinates(im[:, :, i], yx, order = order, mode = mode)[:, np.newaxis] \
                           for i in xrange(im.shape[2])], axis = 1)

def map_helper((xs, i, order, mode)):
  return scipy.ndimage.map_coordinates(xs, np.array([i]), order = order, mode = mode)[:, np.newaxis]

def lookup_bilinear1d(xs, i, order = 4, mode = 'constant', cval = 0.0, par = 0):
  if np.ndim(xs) == 1:
    return scipy.ndimage.map_coordinates(xs, i, order = order, mode = mode, cval = cval)
  else:
    if par:
      vals = [(xs[:, j], i, order, mode) for j in xrange(xs.shape[1])]
      return np.concatenate(ut.parmap(map_helper, vals), axis = 1)
    else:
      return np.concatenate([scipy.ndimage.map_coordinates(xs[:, j], np.array([i]), order = order, mode = mode)[:, np.newaxis] \
                             for j in xrange(xs.shape[1])], axis = 1)
      
  
#def pixels_in_bounds(im, xs, ys):
def pixels_in_bounds(im_shape, xs, ys):
  return ut.land(0 <= xs, xs < im_shape[1],
                 0 <= ys, ys < im_shape[0])

def im2float(im):
  im = np.array(im, 'float32')
  im /= 255.
  return im

def try_load_img(fname, default_size = (256, 256)):
  try:
    return load(fname)
  except IOError:
    print 'Failed to load:', fname
    return make(default_size[0], default_size[1])
  

def exif(fname, use_jhead = True):
  if use_jhead:
    pass
  else:
    with Image.open(fname) as img:
      if not hasattr(img, '_getexif'):
        return None
      return img._getexif()

def exif_shape(fname):
  ex = exif(fname)
  yid = 40963
  xid = 40962
  if ex is None or xid not in ex or yid not in ex:
    return None
  return ex[yid], ex[xid]

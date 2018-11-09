import time, random, math, numpy, os, sys, tempfile, pylab, subprocess, matplotlib, datetime, \
       itertools as itl, copy, StringIO, cPickle as pickle, gc, collections, bisect, traceback, multiprocessing
import numpy as np
import img as ig
import scipy.sparse
import inspect
import glob as _glob

#import networkx as nx
#graph = nx

def fail(s = ''): raise RuntimeError(s)

def invert_injective(d):
  """ invert a one-to-one map d """
  inv = {}
  for k in d:
    if d[k] in inv:
      raise RuntimeError('not an injective map')
    inv[d[k]] = k
  return inv

def invert_non_injective(d):
  """ returns d such that d[k] is the list of values v for which
  d[v] = k"""
  inv = {}
  for k in d:
    if d[k] not in inv:
      inv[d[k]] = []
    inv[d[k]].append(k)
  return inv

def negf(f): return lambda x: -f(x)

# def max_orelse(lst, x):
#   try:
#     return max(lst)
#   except ValueError:
#     return 0

def mindef(xs, default, key = None):
  try:
    if key is None:
      return min(xs)
    else:
      return min(xs, key = key)
  except ValueError:
    return default

def maxdef(xs, default, key = None):
  try:
    if key is None:
      return max(xs)
    else:
      return max(xs, key = key)
  except ValueError:
    return default

# see also itertools.join
def flatten(lists):
  newlist = []
  for x in lists:
    for y in x:
      newlist.append(y)
  return newlist

def with_file(fname, f):
  r = file(fname)
  try:
    ret = f(r)
    r.close()
  except:
    r.close()
  return ret

def add_dict_list(m, k, v):
  if k in m:
    m[k].append(v)
  else:
    m[k] = [v]

def concat_dict_list(m, k, v):
  if k in m:
    m[k] += v
  else:
    m[k] = v

def add_dict_set(m, k, v):
  if k in m:
    m[k].add(v)
  else:
    m[k] = set()
    m[k].add(v)


def add_dict_dict(m, k1, k2, v):
  if k1 not in m:
    m[k1] = dict()
  m[k1][k2] = v


def add_dict_dict_list(m, k1, k2, v):
  if k1 not in m:
    m[k1] = dict()
  if k2 not in m[k1]:
    m[k1][k2] = []
  m[k1][k2].append(v)

def increment_key(m, k, v = 1):
  m[k] = m.get(k, 0) + v
  
# uses networkx graph
def filter_edges(f, graph):
  newgraph = graph.copy()
  for e in graph.edges_iter():
    if not f(e[0],e[1],graph.get_edge(*e)):
      newgraph.remove_edge(*e)
  return newgraph

def filter_edges_imp(f, graph):
  for e in graph.edges():
    if not f(e[0],e[1],graph.get_edge(*e)):
      graph.remove_edge(*e)

def filter_nodes(f, graph):
  newgraph = graph.copy()
  for v in graph.nodes_iter():
    if not f(v):
      newgraph.remove_node(v)
  return newgraph

def filter_nodes_imp(f, graph):
  for v in graph.nodes_iter():
    if not f(v):
      graph.remove_node(v)

def map_edges(f, graph):
  newgraph = graph.copy()
  for e in graph.edges_iter():
    newgraph.add_edge(e[0], e[1], f(e[0], e[1], graph.get_edge(*e)))
  return newgraph

def filter_ccs(f, graph):
  import networkx as nx
  newgraph = graph.copy()
  for cc in nx.connected_components(graph):
    if not f(cc):
      newgraph.remove_nodes_from(cc)
  return newgraph

def weighted_deg(graph, u):
  return sum(graph[u][v] for v in graph.neighbors(u))

def ordered(u,v): return (min(u,v), max(u,v))

def make_file(fname, contents = ''):
  f = file(fname, 'w')
  f.write(contents)
  f.close()

def ident(x): return x

# http://norvig.com/python-iaq.html
class Struct:
  def __init__(self, *dicts, **fields):
    for d in dicts:
      for k, v in d.iteritems():
        setattr(self, k, v)
    self.__dict__.update(fields)

  def to_dict(self):
    return {a : getattr(self, a) for a in self.attrs()}

  def attrs(self):
    #return sorted(set(dir(self)) - set(dir(Struct)))
    xs = set(dir(self)) - set(dir(Struct))
    xs = [x for x in xs if ((not (hasattr(self.__class__, x) and isinstance(getattr(self.__class__, x), property))) \
          and (not inspect.ismethod(getattr(self, x))))]
    return sorted(xs)

             
  def updated(self, other_struct_ = None, **kwargs):
    s = copy.deepcopy(self)
    if other_struct_ is not None:
      s.__dict__.update(other_struct_.to_dict())
    s.__dict__.update(kwargs)
    return s

  def copy(self):
    return copy.deepcopy(self)
  
  def __str__(self):
    attrs = ', '.join('%s=%s' % (a, getattr(self, a)) for a in self.attrs())
    return 'Struct(%s)' % attrs

def constrain(x, minimum, maximum):
  return max(minimum, min(maximum, x))

def number_values(col):
  number = {}
  next = 0
  for x in col:
    number[x] = next
    next += 1
  return number

def dict_getfn(d, k, f):
  if k not in d:
    d[k] = f()
  return d[k]

#def dict_get(d): return d.__getitem__
def dict_get(d): return lambda k: d[k]

def union(sets):
  u = set()
  for s in sets:
    u.update(s)
  return u

# def vertex_cover(graph):
#   # note sure if this was tested!
#   graph = graph.copy()
#   nodes = graph.nodes()
#   vc = []
#   for u in nodes:
#     if graph.has_node(u) and graph.degree(u) > 0:
#       v = random.sample(graph.neighbors(u), 1)[0]
#       vc.append(u)
#       vc.append(v)
#       graph.remove_nodes_from([u,v])
#   assert graph.number_of_edges() == 0
#   return vc


def in_fun(x): return lambda k: k in x

def max_n(vals, n, key = None): return list(reversed(sorted(vals, key=key)))[:n]

def min_n(vals, n, key = None): return list(sorted(vals, key=key))[:n]

def flip_pair(t): return (t[1], t[0])

# def curry(f): return lambda x: lambda y: f(x,y)

# def curry_arg(f, x): return lambda y: f(x, y)

def tupargs(f): return (lambda t: f(*t))

# returns new dict d' such that d'[k] = d[k]/sum_j(d[j])
def normalize_dict(d):
  total = float(sum(d.values()))
  return dict((k, d[k]/total) for k in d)

class WeightedSampler:
  # not sure if this is tested!
  def __init__(self, f, vals, parallel_weights = None):
    """ requires: vals is a list """
    fxs = None
    if parallel_weights is None:
      fxs = [f(x) for x in vals]
    else:
      fxs = parallel_weights
    fx_sum = float(sum(fxs))
    self.cumulative = []
    c = 0.0
    for fx in fxs:
      self.cumulative.append(c)
      c += fx/fx_sum
    self.vals = vals

  def sample(self):
    s = random.random()
    return self.vals[bisect.bisect_right(self.cumulative, s) - 1]

def max_span_tree(graph):
  neg_graph = map_edges(lambda x,y,w: -w, graph)
  edges = mst.mst(neg_graph)
  t = graph.Graph()
  t.add_nodes_from(graph.nodes_iter())
  t.add_edges_from(edges)
  return map_edges(lambda x,y,w: -w, t)

tic_start_g = []
def tic(msg = None, show = True):
  """ start a timer """
  global tic_start_g
  if (msg is not None) and show:
    print >>sys.stderr, msg
  tic_start_g.append((msg, time.time()))

def toc(s = None, show = True):
  """ end a timer and print the elapsed time """
  global tic_start_g
  if len(tic_start_g) == 0:
    if show:
      print >>sys.stderr, 'toc() called without tic!'
    return
  msg, t = tic_start_g.pop()
  elapsed = time.time() - t
  if s is not None:
    msg = s
  if show:
    if msg is not None:
      print >>sys.stderr, msg, ('%.3f seconds' % elapsed)
    else:
      print >>sys.stderr, ('%.3f seconds' % elapsed)
  return elapsed

def unweighted(graph):
  return map_edges(lambda x, y, w: 1, graph)

def f_eq(a,b): return abs(np.asarray(a) - np.asarray(b)) <= 0.0001
def feq(a, b): return f_eq(a, b) 
# def a_eq(A, B):
#   A = np.asarray(A)
#   B = np.asarray(B)
#   if A.shape != B.shape:
#     return False
#   else:
#     diff = (A - B).flatten()
#     return np.sum(diff**2) <= 0.0001
  
# aeq = a_eq

def line_seg_dist_sq(x1, y1, x2, y2, pt_x, pt_y):
  # http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/
  def dist_sq(x1, y1, x2, y2): return (x1-x2)**2 + (y1-y2)**2
  eps = 0.00001
  if x1 == x2 and y1 == y2:
    return dist_sq(x1, y1, pt_x, pt_y)
  else:
    u = ((pt_x - x1)*(x2 - x1) + (pt_y - y1)*(y2 - y1))/dist_sq(x1,y1,x2,y2)
    if 0 <= u <= 1:
      x = x1 + u*(x2 - x1)
      y = y1 + u*(y2 - y1)
      return dist_sq(x, y,pt_x, pt_y)
    else:
      return min(dist_sq(pt_x, pt_y, x1, y1), dist_sq(pt_x, pt_y, x2,y2))

def test_line_seg_dist_sq():
  assert(f_eq(25., line_seg_dist_sq(0., 0., 1., 1., 4., 5.)))
  assert(f_eq(2., line_seg_dist_sq(0., 0., 1., 1., -1., -1.)))
  assert(f_eq(0.5, line_seg_dist_sq(1., 1., 2., 2., 2., 1.0)))

def remove_neighborhood(graph, u):
  for v in graph.neighbors(u):
    graph.remove_node(v)
  graph.remove_node(u)

def remove_cc(graph, u):
  graph.remove_nodes_from(nx.node_connected_component(graph, u))

def sample1(f, g):
  """ samples one value x from a generator with probability proportional to f(x) """
  pass

def sample_n(n, gen):
  if n < 0:
    raise RuntimeError('n must be at least 0')
  if n == 0:
    return []
  seen = 0
  sample = []
  for x in gen:
    seen += 1
    if len(sample) < n:
      sample.append(x)
    elif random.random() < n/float(seen):
      sample[random.randint(0, n-1)] = x
  return sample

# def dict_map(dict, vals):
#   return [dict[k] for k in vals]

def dict_f(dict):
  return lambda k: dict[k]

def methods(x):
  for m in dir(x):
    print m

day_seconds = 24*60*60

def thresh(g1, t):
  """ returns subgraph of g1 with weights >= t (e.g. removes edges < t
  and nodes with no incident edge >= t """
  g2 = graph.Graph()
  for e in g1.edges_iter():
    v = g1.get_edge(*e)
    if t <= v:
      g2.add_edge(e[0], e[1], v)
  return g2

# def bbox2d(pts):
#   if len(pts) == 0:
#     raise RuntimeError("no pts")
#   xs = np.array([x for x, _ in pts], 'd')
#   ys = np.array([y for _, y in pts], 'd')
#   max_x = np.max(xs)
#   max_y = np.max(ys)
#   min_x = np.min(xs)
#   min_y = np.min(ys)  
#   return (min_x, min_y, 1 + max_x - min_x, 1 + max_y - min_y)


def bbox2d(pts):
  pts = np.asarray(pts)
  if len(pts) == 0:
    raise RuntimeError("no pts")
  return rect_from_pts(np.min(pts[:,0]),
                       np.min(pts[:,1]),
                       np.max(pts[:,0]),
                       np.max(pts[:,1]))

def pts_in_bbox(bbox, pts):
  pts = np.asarray(pts)
  return land(pts[:, 0] >= bbox[0],
              pts[:, 1] >= bbox[1],
              pts[:, 0] < bbox[0]+bbox[2],
              pts[:, 1] < bbox[1]+bbox[3])
              

class yield_list:
  """ Decorator that turns a generator into a function that returns a list """
  def __init__(self, f):
    self.f = f
  def __call__(self, *args, **kwargs):
    return [x for x in self.f(*args, **kwargs)]

@yield_list
def read_lines(fname, keep_empty_lines = True, strip_newline = True, max_lines = None):
  f = file(fname)
  for i, line in enumerate(f):
    if max_lines is not None and i >= max_lines:
      break
    if keep_empty_lines or line != "":
      yield (line.rstrip('\n') if strip_newline else line)
  f.close()

def write_lines(fname, lines):
  assert type(lines) != type('')
  f = file(fname, 'w')
  for line in lines:
    f.write(line)
    f.write("\n")
  f.close()

def blank_file(fname):
  f = file(fname, 'w')
  f.close()

def radians_from_degs(x): return x/180.0*math.pi

def xyz_from_latlng(lat, lng, r1 = 6378.137e3, r2 = 6356.7523142e3):
  lat = radians_of_degs(lat)
  lng = radians_of_degs(lng)
  x = r1*math.cos(lng)*math.cos(lat)
  y = r1*math.sin(lng)*math.cos(lat)
  z = r2*math.sin(lat)
  return (x, y, z)

def rotation_matrix2(angle):
  """ angle in radians; clockwise rotation """
  return np.array([[np.cos(angle), -np.sin(angle)],
          [np.sin(angle), np.cos(angle)]])

def rotation_matrix3(x, y, z):
  """ returns a rotation matrix rotating x radians around the x axis, etc. """
  cx, sx = np.cos(x), np.sin(x)
  cy, sy = np.cos(y), np.sin(y)
  cz, sz = np.cos(z), np.sin(z)
  Rx = numpy.array([[1.0, 0.0, 0.0],
                    [0.0, cx, -sx],
                    [0.0, sx, cx]])
  Ry = numpy.array([[cy, 0.0, -sy],
                    [0.0, 1.0, 0.0],
                    [sy, 0.0, cy]])
  Rz = numpy.array([[cz, -sz, 0.0],
                    [sz, cz, 0.0],
                    [0.0, 0.0, 1.0]])
  return numpy.dot(numpy.dot(Rx, Ry), Rz)

def mult(nums):
  if len(nums) == 0:
    raise TypeError('mult() takes 1 argument')
  t = nums[0]
  for i in xrange(1, len(nums)):
    t *= nums[i]
  return t

def matrix_rank(A, tol = 1e-8):
  # http://mail.scipy.org/pipermail/numpy-discussion/2008-February/031218.html
  return sum(np.where(np.linalg.svd(A, compute_uv = 0) > tol, 1, 0))

def show_profile(prof_fname):
  import pstats
  pstats.Stats(prof_fname).sort_stats(-1).print_stats()

def padded(num, digits):
  return str(num).zfill(digits)

def pad_list(x, n, fill):
  return list(x) + [fill] * max(0, n - len(x))

def hyperplane_pt_dist(w, x):
  #b = w[-1]
  #w = w[:-1]
  #return abs(dot(w, (x - b/n**2))/n)
  return abs((w[-1] + numpy.dot(w[:-1], x))/linalg.norm(w[:-1]))

def test_hyperplane_pt_dist():
  assert(feq(10.0, hyperplane_pt_dist(vec(0, 1, 0), vec(20, 10))))
  assert(feq(7.0, hyperplane_pt_dist(vec(0, 1, -3), vec(20, 10))))
  # todo: double check
  assert(feq(9.5399809, hyperplane_pt_dist(vec(-5./8, -1, 5), vec(10, 10))))

def fst(x):
  return x[0]

def snd(x):
  return x[1]

def last(x):
  return x[-1]

def sortlst(lst):
  return list(sorted(lst))

def angle_from_radians(x):
  return x/math.pi*180.

def trues(xs, f = None):
  if f is None:
    return [x for x in xs if x]
  else:
    return [x for x in xs if f(x)]

def somes(xs):
  return [x for x in xs if x is not None]

def truesi(xs):
  """ indices for which x[i] is True """
  return [i for i, x in enumerate(xs) if x]

def pretty_seconds(sec):
  h = int(sec / 3600)
  sec -= h*3600
  m = int(sec / 60)
  sec -= m*60
  return '%d:%02d:%02d' % (h, m, sec)


def substring_between(s, delim1, delim2):
  """ returns the substring between the first occurrence of delim1
  and the first occurrence of delim2 after delim2"""
  ind1 = s.index(delim1)
  ind2 = s.index(delim2, 1 + ind1)
  if ind1 < ind2:
    return s[ind1 + 1 : ind2]
  else:
    raise RuntimeError("delim2 doesn't follow delim1")

def map_split(f, s, delim = None):
  return map(f, s.split(delim))

def maptup(f, col):
  return tuple(f(x) for x in col)

def do_with_seed(f, seed = 0):
  old_seed = stash_seed(seed)
  res = f()
  unstash_seed(old_seed)
  return res

def stash_seed(new_seed = 0):
  """ Sets the random seed to new_seed. Returns the old seed. """
  if type(new_seed) == type(''):
    new_seed = hash(new_seed) % 2**32

  py_state = random.getstate()
  random.seed(new_seed)

  np_state = np.random.get_state()
  np.random.seed(new_seed)
  return (py_state, np_state)

class constant_seed:
  def __init__(self, s = 0):
    self.new_seed = s

  def __enter__(self):
    self.old_seed = stash_seed(self.new_seed)

  def __exit__(self, type, value, traceback):
    unstash_seed(self.old_seed)


def test_constant_seed():
  n = 5
  #a = [random.randint(0, 1000) for x in xrange(n)]
  with constant_seed():
    a = [random.randint(0, 1000) for x in xrange(n)]

  with constant_seed():
    b = [random.randint(0, 1000) for x in xrange(n)]

  c = [random.randint(0, 1000) for x in xrange(n)]
  assert a == b and c != a


def unstash_seed((py_state, np_state)):
  random.setstate(py_state)
  np.random.set_state(np_state)

def sample_at_most(xs, bound):
  return random.sample(xs, min(bound, len(xs)))

sample_most = sample_at_most

def sample_at_most_each(xss, bound):
  assert len(set(map(len, xss))) <= 1
  inds = sample_at_most(range(len(xss[0])), bound)
  return [take_inds(xs, inds) for xs in xss]


def compose(f, g):
  return lambda *args, **kwargs: f(g(*args, **kwargs))

def pil_from_cv(src):
  return Image.fromstring("L", cv.GetSize(src), src.tostring())

def pygame_from_cv(src):
  import cv
  """ return pygame image from opencv image """
  src_rgb = cv.CreateMat(src.height, src.width, cv.CV_8UC3)
  cv.CvtColor(src, src_rgb, cv.CV_BGR2RGB)
  return pygame.image.frombuffer(src_rgb.tostring(), cv.GetSize(src_rgb), "RGB")

def pygame_from_img(img):
  # filename
  if type(img) == type(''):
    return pygame.image.load(img)
  # OpenCV image
  elif hasattr(img, 'nChannels') or (hasattr(img, 'tostring') and hasattr(img, 'channels')):
    return pygame_from_cv(img)
  # PIL
  elif hasattr(img, 'getpixel'):
    return pygame_from_pil(img)
  # Pygame
  elif hasattr(img, 'unmap_rgb'):
    return img
  else:
    fail('unknown image type; cannot convert to pygame image')

def find_firsti(f, lst):
  for i, x in enumerate(lst):
    if f(x):
      return i
  return -1

def find_first(f, lst):
  for x in lst:
    if f(x):
      return x

def find_only(f, xs):
  found = False
  c = None
  for x in xs:
    if f(x):
      if found:
        raise RuntimeError('Multiple instances found')
      found = True
      c = x
  if found:
    return c
  else:
    raise RuntimeError('No instances found')

def find_split(f, xs):
  found = False
  c = None
  others = []
  for x in xs:
    if f(x):
      if found:
        raise RuntimeError('Multiple instances found')
      found = True
      c = x
    else:
      others.append(x)
  if found:
    return c, others
  else:
    raise RuntimeError('No instances found')  

def count_params(f):
  return f.func_code.co_argcount

def parse_space_delim(s, parse_item = lambda x: x):
  return [map(parse_item, x.split()) for x in s.split('\n') if x != '']

def read_file(fname):
  f = file(fname)
  s = f.read()
  f.close()
  return s

def ignore(x):
  return 

def percentile(lst, p):
  return sortlst(lst)[int(len(lst)*p)]

def normalized(v):
  n = np.sqrt(np.sum(v**2))
  return v.copy() if n == 0 else v/n

def rect(x1, y1, w, h):
  return ((x1 + x, y1 + y) for x in xrange(w) for y in xrange(h))

def zrect(w, h = None, alias = 1):
  if h is None:
    h = w
  return ((x, y) for x in xrange(0, w, alias) for y in xrange(0, h, alias))

def vec(*args):
  return numpy.array(args, dtype = 'd')

pjoin = os.path.join

def int_tuple(x): 
  return tuple([int(v) for v in x])

itup = int_tuple

def map_map(f, x):
  return [map(f, y) for y in x]

def rand_color():
  return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def rand_color_f():
  return (random.random(), random.random(), random.random())

# def homog(pt):
#   a = zeros(1+len(pt))
#   a[:len(pt)] = pt
#   a[len(pt)] = 1.0
#   return a

# def inhomog(pt):
#   return pt[:-1]/pt[-1]

# def transform_pt(A, x):
#   return inhomog(np.dot(A, homog(x)))

def bound(x, min_v, max_v):
  return max(min_v, min(x, max_v))

def take_inds(a, inds):
  return [a[i] for i in inds]

def remove_inds(a, inds):
  new_a = []
  inds = set(inds)
  for i in xrange(len(a)):
    if i not in inds:
      new_a.append(a[i])
  return new_a

# def sample_each(count, lsts):
#   assert(all(x is None or (len(lsts[0]) == len(x))) for x in lsts[1:])
#   inds = random.sample(range(len(lsts[0])), count)
#   return [(None if x is None else take_inds(x, inds)) for x in lsts]

def unzip(lst):
  return zip(*lst)

def iunzip(lst):
  return itl.izip(*lst)

def unziplst(lst):
  return map(list, zip(*lst))

def eigs_sorted(A, enforce_positive = False):
  """ sorted in ascending order
  enforce_positive means make eigs positive if the sum is negative """
  S, U = numpy.linalg.eig(A)
  if enforce_positive and sum(S) < 0:
    S = -S
  inds = list(np.argsort(S))
  inds.reverse()
  return S[inds], U[:, inds]

def add_prefix(pre, suffixes):
  return [pre + s for s in suffixes]

def add_suffix(prefixes, suffix):
  return [p + suffix for p in prefixes]

def deg_from_rad(r):
  return 180. * r / math.pi

def A(s, dtype = 'd'):
  """ matlab-like array literals ; A('1 2 3; 4 5 6; 7 8 9') = [1. 2. 3.; 4. 5. 6.; 7. 8. 9.] """
  check(not (s.startswith('[') or s.endswith(']')), 'omit [ and ]')
  return numpy.array(numpy.matrix('[' + s + ']', dtype = dtype))

def V(s):
  return numpy.array(map(float, s.split()))

def sample_stream(s):
  n = 1.0
  choice = None
  for x in s:
    if random.random() < 1.0/n:
      choice = x
    n += 1
  return choice

def remove_val(x, lst):
  return [y for y in lst if y != x]

def rigid_transform(R, t = None):
  if t is None:
    t = np.zeros(R.shape[0])
  n = len(t)
  T = np.zeros((1+n, 1+n))
  T[:n,:n] = R
  T[:n, -1] = t
  T[-1,-1] = 1.
  return T

def in_bounds2d(a, i, j):
  return (0 <= i < a.shape[0]) and (0 <= j < a.shape[1])

def neighbors(i, j, conn = 4):
  yield (i-1, j)
  yield (i, j+1)
  yield (i+1, j)
  yield (i, j-1)
  if conn == 8:
    yield (i-1, j-1)
    yield (i+1, j-1)
    yield (i+1, j+1)
    yield (i-1, j+1)

def in_bounds_shape(shape, *inds):
  return all(0 <= inds[i] < shape[i] for i in xrange(len(inds)))

def in_bound_neighbors(a, i, j, conn = 4):
  return [(ii, jj) for ii, jj in neighbors(i, j, conn) if in_bounds2d(a, ii, jj)]

def double_array(x):
  return np.array(x, dtype = 'double')

def sorted_desc(vals, key = None):
  return reversed(sorted(vals, key = key))

# def angle_between(u, v):
#   return np.arccos(np.clip(np.dot(normalized(u), normalized(v)), -1., 1.))

def angle_between(u, v):
  n1 = np.linalg.norm(u)
  n2 = np.linalg.norm(v)
  if n1 == 0 or n2 == 0:
    return np.arccos(0)
  else:
    return np.arccos(np.clip(np.dot(u, v)/(n1*n2), -1., 1.))

def normalize_color(c):
  return np.array(c)/255.0

def extract_img_rect(img, r):
  """ r takes the form (x, y, w, h) """
  x, y, w, h = r
  assert in_bounds2d(img, y, x)
  assert in_bounds2d(img, y+h-1, x+w-1)
  return img[y : y+h, x : x+w]

def collect_dict(k_vs):
  d = {}
  for k, v in k_vs:
    if k not in d:
      d[k] = []
    d[k].append(v)
  return d

def shuffled(xs):
  cp = list(xs)
  random.shuffle(cp)
  return cp

def prn_sys(cmd):
  print cmd
  return os.system(cmd)

# def sys(*args):
#   cmd = ' '.join(args)
#   print cmd
#   return os.system(cmd)

# def sys_check(cmd):
#   print cmd
#   if 0 != os.system(cmd):
#     fail('Command failed!')
#   return 0

def sys_check(*args):
  cmd = ' '.join(args)
  print cmd
  if 0 != os.system(cmd):
    fail('Command failed! %s' % cmd)
  return 0

def sys_check_silent(*args):
  cmd = ' '.join(args)
  if 0 != os.system(cmd):
    fail('Command failed: %s' % cmd)
  return 0

def sys_print(s):
  print s
  return os.system(s)

# def homog(x):
#   return np.concatenate([x, [1.]])

# def inhomog(x):
#   y = x[:-1]
#   y /= x[-1]
#   return y

def homog(X):
  # assumes X is either a vector or a matrix of n vectors of the shape (d by n)
  if X.ndim == 1:
    return np.concatenate([X, [1.]])
  else:
    return np.vstack([X, np.ones(X.shape[1])])

def inhomog(x):
  y = x[:-1]
  y /= x[-1]
  return y

# def transform_pt(T, x):
#   return inhomog(np.dot(T, homog(x)))

def homog_transform(H, X):
  return inhomog(np.dot(H, homog(X)))

def array_values(v):
  return set(v.flat)

def pca(X, mean_pt = None):
  """ rows of X are individual pts """
  if mean_pt is None:
    mean_pt = np.mean(X, axis = 0)

  centered = X - mean_pt[np.newaxis, :]
  [U, S, Vt] = np.linalg.svd(np.dot(centered.T,centered))
  return U.T, S, mean_pt

def indicator(n, i):
  a = np.zeros(n)
  if 0 <= i < n:
    a[i] = 1.
  return a

def check(cond, str = 'Check failed!'):
  if not cond:
    fail(str)

def rect_area((x, y, w, h)): return w*h

# def rect_center((x,y,w,h)):
#   return (x + w/2, y + h/2)

def perm(n):
  a = range(n)
  random.shuffle(a)
  return a

def coin_flip(p = 0.5):
  return random.random() < p

def constrain_angle(angle):
  """ returns equivalent angle theta such that 0 <= theta <= 2*pi"""
  return np.mod(angle, 2*math.pi)

def test_constrain_angle():
  assert f_eq(constrain_angle(math.pi), math.pi)
  assert f_eq(constrain_angle(2*math.pi + 1), 1)
  assert f_eq(constrain_angle(4*math.pi + 1), 1)
  assert f_eq(constrain_angle(-4*math.pi + 1), 1)
  assert f_eq(constrain_angle(-0.01), 2*math.pi - 0.01)

def make_angle_positive(angle):
  """ mirror angles that point towards negative y axis """
  angle = constrain_angle(angle)
  return angle if angle < math.pi else angle - math.pi

def make_temp(ext, contents = None, dir = None):
  fd, fname = tempfile.mkstemp(ext, prefix = 'ao_', dir = dir)
  os.close(fd)
  if contents is not None:
    make_file(fname, contents)
  return os.path.abspath(fname)

#def nfs_dir(): return '/csail/vision-billf5/aho/tmp'
#def nfs_dir(): return '/data/vision/billf/aho-vis/tmp'
#def nfs_dir(): return '/data/vision/scratch/billf/aho/tmp'
def nfs_dir(): return '/tmp'

def make_temp_nfs(ext, contents = None):
  return make_temp(ext, contents, dir = nfs_dir())

def make_temp_dir(dir = None):
  return tempfile.mkdtemp(dir = dir)

def make_temp_dir_nfs(dir = None):
  return make_temp_dir(nfs_dir())

make_temp_dir_big = make_temp_dir_nfs

# def make_temp_dir_big():
#   base_dirname = '/data/scratch/aho/'
#   return make_temp_dir(base_dirname)
#   # else:
#   #   dir_base = '/scratch/aho/stuff'
#   #   if not os.path.exists(dir_base):
#   #     mkdir(dir_base)
#   #   return make_temp_dir(dir = dir_base)


class toplevel:
  """ Decorator that places all of a function's local variables into the local variables of the caller """
  # not sure if it's possible to implement this!
  pass

# def sorted_by_key(lst, keys):
#   inds = range(len(lst))
#   inds.sort(key = lambda i: keys[i])
#   return take_inds(lst, inds)

def sorted_by_key(keys, lsts, reverse = False):
  # inds = range(len(keys))
  # inds.sort(key = lambda i: keys[i], reverse = reverse)
  inds = np.argsort(keys)
  if reverse:
    inds = inds[::-1]
  return [take_inds(lst, inds) for lst in lsts]

def crop_rect_to_img((x, y, w, h), im):
  # redundant and bad code
  x1, y1 = x, y
  x2, y2 = x+w-1, y+h-1
  x1 = max(x1, 0)
  x2 = min(x2, im.shape[1]-1)
  y1 = max(y1, 0)
  y2 = min(y2, im.shape[0]-1)
  x, y = x1, y1
  w = 1 + x2 - x1
  h = 1 + y2 - y1
  return (x, y, max(0, w), max(h, 0))

def test_crop_rect():
  assert crop_rect_to_img((10, 10, 1, 1), np.zeros((3,3)))[2] == 0
  assert crop_rect_to_img((0, 0, 1, 1), np.zeros((3,3))) == (0, 0, 1, 1)
  assert crop_rect_to_img((0, 0, 3, 3), np.zeros((3,3))) == (0, 0, 3, 3)
  assert crop_rect_to_img((-1, 0, 2, 1), np.zeros((3,3))) == (0, 0, 1, 1)

def closest_pt_i(pts, X):
  X = np.array(X)
  return argmini([np.dot((y - X), (y - X)) for y in map(np.array, pts)])

def alternate(*args):
  return flatten(zip(*args))

def pts_from_rect_outside(r):
  """ returns start_pt, end_pt where end_pt is _outside_ the rectangle """
  return (r[0], r[1]), ((r[0] + r[2]), (r[1] + r[3]))

def pts_from_rect_inside(r):
  """ returns start_pt, end_pt where end_pt is _inside_ the rectangle """
  return (r[0], r[1]), ((r[0] + r[2] - 1), (r[1] + r[3] - 1))

def rect_corners_inside(r):
  (x1, y1), (x2, y2) = pts_from_rect_inside(r)
  return (x1, y1), (x2, y1), (x2, y2), (x1, y2)

pts_from_rect = pts_from_rect_inside


def rect_from_pts(x1, y1, x2, y2):
  return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)

def center_point(pts):
  # """ Choose the point closest to the center of the point set, as
  # measured by the median distance to every other point. O(n^2)."""
  best_err = np.inf
  best_pt = pts[0]
  for p1 in pts:
    err = np.median([pylab.dist(p1, p2) for p2 in pts])
    #err = np.sum([np.sum((p1 - p2)**2)**0.5 for p2 in pts])
    if err < best_err:
      best_err = err
      best_pt = p1
  return best_pt

def rect_contains_pt((rx, ry, w, h), x, y):
  return rx <= x < rx + w and ry <= y < ry + h

def list_of_lists(n): return [[] for x in xrange(n)]
def list_of_sets(n): return [set() for x in xrange(n)]
def list_of_dicts(n): return [{} for x in xrange(n)]
def repf(f, n): return [f() for x in xrange(n)]

def argmini(lst):
  least = numpy.inf
  leasti = None
  for i, x in enumerate(lst):
    if x < least:
      least = x
      leasti = i
  return leasti

def argmaxi(lst):
  return argmini(-x for x in lst)

# # if multiple equal choices, picks one at random
def argminf(f, lst):
  least = numpy.inf
  leastx = None
  seen_eq = 1
  for x in lst:
    fx = f(x)
    if fx < least:
      least = fx
      leastx = x
      seen_eq = 1
    elif fx == least:
      seen_eq += 1
      if random.random() < 1.0/seen_eq:
        leastx = x
  return leastx

def argmaxf(f, lst):
  return argminf(lambda x: -f(x), lst)

# todo: sample if unknown
def argmini2(lst):
  if len(lst) < 2:
    raise 'list too small'
  if lst[0] < lst[1]:
    min1i = 0
    min2i = 1
  else:
    min1i = 1
    min2i = 0
  for i, x in enumerate(lst):
    if 1 < i:
      if x < lst[min1i]:
        min2i = min1i
        min1i = i
      elif x < lst[min2i]:
        min2i = i
  assert(min1i != min2i)
  return min2i

def min2(lst):
  return lst[argmini2(lst)]


def pad_rect((x, y, w, h), pad):
  return (x - pad, y - pad, w+2*pad, h+2*pad)

def num_splits(xs, chunk_size):
  return (0 if len(xs) % chunk_size == 0 else 1) + (len(xs) / chunk_size)

# @yield_list
# def split_n(xs, n):
#   assert n > 0
#   while len(xs) > 0:
#     yield xs[:n]
#     xs = xs[n:]

@yield_list
def split_n(xs, n):
  """ split into groups of size <= n """
  n = int(n)
  assert n > 0
  i = 0
  while i < len(xs):
    yield xs[i : i + n]
    i += n

def array_split_n(xs, n):
  i = 0
  while i < len(xs):
    yield xs[i : i + n]
    i += n

@yield_list
def split_n_pad(xs, n, pad):
  assert n > 0
  while len(xs) > 0:
    if len(xs) < n:
      yield (xs[:n] + [pad] * (n - len(xs)))
    else:
      yield xs[:n]
    xs = xs[n:]

@yield_list
def extract_subseqs(xs, n, step):
  assert n > 0
  while len(xs) >= n:
    yield xs[:n]
    xs = xs[step:]

@yield_list
def extract_subseqs_pad(xs, n, step, pad):
  assert n > 0
  while len(xs) > 0:
    if len(xs) < n:
      yield (xs[:n] + [pad] * (n - len(xs)))
    else:
      yield xs[:n]
    xs = xs[step:]


def split_into(xs, pieces):
  assert pieces > 0
  return split_n(xs, int(math.ceil(float(len(xs))/pieces)))

def logical_and_many(*args):
  return reduce(np.logical_and, args[1:], args[0])

def logical_or_many(*args):
  return reduce(np.logical_or, args[1:], args[0])

def rect_centered_at(x, y, w, h):
  return (x - w/2, y - h/2, w, h)

def roll_img(im, dx, dy):
  return np.roll(np.roll(im, dy, axis = 1), dx, axis = 0)

def rect_intersect(r1, r2):
  x1 = max(r1[0], r2[0])
  y1 = max(r1[1], r2[1])
  x2 = min(r1[0] + r1[2], r2[0] + r2[2])
  y2 = min(r1[1] + r1[3], r2[1] + r2[3])
  return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))

def rect_im_intersect(im, rect):
  return rect_intersect((0, 0, im.shape[1], im.shape[0]), rect)

def rect_shape_intersect(shape, rect):
  return rect_intersect((0, 0, shape[1], shape[0]), rect)

def rect_empty((x, y, w, h)): return w <= 0 or h <= 0

def scale_rect(r, s):
  w, h = (r[2]*s, r[3]*s)
  x, y = rect_center(r)
  return (x - w/2, y - h/2, w, h)

def scale_rect_coords(r, s):
  return (r[0]*s, r[1]*s, r[2]*s, r[3]*s)

def mutual_overlap(r1, r2):
  ro = rect_intersect(r1, r2)
  a = float(ro[2]*ro[3])
  return min(a/(r1[2]*r1[3]), a/(r2[2]*r2[3]))

def intersection_mask(r1, r2):
  ins = rect_intersect(r1, r2)
  mask1 = np.zeros((r1[3], r1[2]), dtype = np.int32)
  mask2 = np.zeros((r2[3], r2[2]), dtype = np.int32)
  ins1 = (ins[0] - r1[0], ins[1] - r1[1], ins[2], ins[3])
  ins2 = (ins[0] - r2[0], ins[1] - r2[1], ins[2], ins[3])
  mask1[ins1[1] : ins1[1] + ins1[3], ins1[0] : ins1[0] + ins1[2]] = 1
  mask2[ins2[1] : ins2[1] + ins2[3], ins2[0] : ins2[0] + ins2[2]] = 1
  return mask1, mask2

def rects_overlap(r1, r2):
  ins = rect_intersect(r1, r2)
  return ins[2] > 0 and ins[3] > 0

# def rect_jaccard((x1, y1, w1, h1), (x2, y2, w2, h2)):
#   ix1 = max(x1, x2)
#   iy1 = max(y1, y2)
#   ix2 = min(x1 + w1, x2 + w2)
#   iy2 = min(y1 + h1, y2 + h2)
#   w = ix2 - ix1 + 1
#   h = iy2 - iy1 + 1
#   if w <= 0 or h <= 0:
#     return 0

def test_rect_intersect():
  r1 = (0, 0, 100, 100)
  r2 = (50, 50, 25, 25)
  assert rect_intersect(r1, r2) == (50, 50, 25, 25)
  r3 = (-1, 2, 100, 102)
  assert rect_intersect(r1, r3) == (0, 2, 99, 98)

def test_intersection_mask():
  r1 = (0, 0, 5, 5)
  r2 = (3, 3, 6, 6)
  mask1, mask2 = intersection_mask(r1, r2)
  assert np.all(mask1 == np.int32(A('0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 1 1; 0 0 0 1 1')))
  # print mask2
  # print np.int32(A('1 1 0 0 0 0; 1 1 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0'))
  assert np.all(mask2 == np.int32(A('1 1 0 0 0 0; 1 1 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0')))

def rect_center(r):
  return np.array([r[0] + r[2]/2., r[1] + r[3]/2.])

def dist_sq(pt1, pt2):
  return np.sum((pt1 - pt2)**2)

def sortlst(lst, key = None): return list(sorted(lst, key = key))
def revlst(lst): return list(reversed(lst))

def acenter(a):
  ind = tuple([i/2 for i in a.shape])
  return a[ind]

inch_in_meters = 0.0254

def iter_axis2(a):
  for y in xrange(a.shape[0]):
    for x in xrange(a.shape[1]):
      yield a[y, x]

def fun_or_attr(f):
  if type(f) == type(''):
    return lambda x: getattr(x, f)
  else:
    return f

def group_dict(f, vals):
  f = fun_or_attr(f)
  d = {}
  for x in vals:
    k = f(x)
    if k not in d:
      d[k] = []
    d[k].append(x)
  return d

def group_dict_1to1(f, vals):
  d = group_dict(f, vals)
  new_d = {}
  for k, v in d.iteritems():
    if len(v) == 1:
      new_d[k] = v[0]
    else:
      fail('mapping is not 1-to-1')
  return new_d


def dict_many(kvs):
  d = {}
  for k, v in kvs:
    if k not in d:
      d[k] = []
    d[k].append(v)
  return d

def dict_map(f, d):
  return dict((k, f(v)) for k, v in d.iteritems())

def normalize_im(a):
  norms = np.sqrt(np.sum(a ** 2, axis = 2))
  # avoid divide-by-0 warning
  norms[norms == 0] = 1.
  return a / np.tile(norms[:, :, np.newaxis], (1,1,3))

def pr(s):
  print s
  return s

def xy_from_angle(a):
  return np.array([np.cos(a), np.sin(a)])

def img_center(im):
  return np.array([im.shape[1]/2, im.shape[0]/2])

def mkdir(path, make_all = True):
  # replacement for os.mkdir that does not throw an exception of the directory exists
  if not os.path.exists(path):
    if make_all:
      os.system('mkdir -p "%s"' % path)
    else:
      os.system('mkdir "%s"' % path)
  return path

def random_rect(im, w, h):
  y = random.randint(0, max(0, im.shape[0] - h))
  x = random.randint(0, max(0, im.shape[1] - w))
  return (x, y, w, h)

eps = 2.**(-52)

def replace_nan(im, fill):
  im[np.isnan(im)] = fill
  return im

def arr_replace(arr, v1, v2):
  arr[arr == v1] = v2

def now_sec(): return time.time()
def now_ms(): return time.time()*1000.

class TimeEstimator:
  def __init__(self, total_elements, seconds_between_updates = 30, name = None):
    self.name = name
    self.n = 0
    self.total_elements = total_elements
    self.seconds_between_updates = seconds_between_updates
    self.last_update = self.init_time = now_sec()
    self.count_error = False

  def update(self, msg_fun = None):
    self.n += 1
    if (not self.count_error) and (self.n > self.total_elements):
      print >>sys.stderr, "TimeEstimator: Can't estimate time remaining. Invalid element count"
      self.count_error = True
    elif not self.count_error:
      t = now_sec()
      since_last_update = t - self.last_update
      if (since_last_update > self.seconds_between_updates) or (self.n == self.total_elements):
        msg = self.name + ': ' if self.name else ''
        if msg_fun:
          msg += msg_fun() + ', '
        self.last_update = t
        elapsed = t - self.init_time
        p = float(self.n)/self.total_elements
        remaining_time = (1 - p)/p*elapsed
        per_iter_sec = round(float(elapsed)/self.n)
        if self.n == self.total_elements:
          rest_msg = 'total time: %s.' % pretty_seconds(elapsed)
        else:
          rest_msg = '%s remaining.' % pretty_seconds(remaining_time)
        print >>sys.stderr, ('%s%2.1f%% complete, %s %s per iteration. (%s)' \
                             % (msg, 100*p, rest_msg,
                                pretty_seconds(per_iter_sec), readable_timestamp()))
class DurationTester:
  def __init__(self, duration):
    self.duration = duration
    self.last_update = now_sec()

  def test(self):
    if now_sec() - self.last_update > self.duration:
      self.last_update = now_sec()
      return True
    return False

  def reset(self):
    self.last_update = now_sec()

def test_time_estimator():
  n = 60
  te = TimeEstimator(n, 2, 'test_time_estimator')
  for x in xrange(n):
    te.update()
    time.sleep(1)

def test_duration_tester():
  dt = DurationTester(10)
  n = 60
  for x in xrange(n):
    if dt.test():
      print 'yes'
    time.sleep(1)
#def quote(s): return '"%s"' % s

def grep_str(text, query):
  # does not handle regexp
  for line in text.split('\n'):
    if query in line:
      yield line

def listdir_full(dir):
  return (os.path.join(dir, fname) for fname in os.listdir(dir))

def save(fname, x):
  if x.__class__.__name__ == 'AsyncMapResult':
    raise RuntimeError('Error: tried to save AsyncMapResult')
  f = file(fname, 'w')
  pickle.dump(x, f, 2)
  f.close()

def load(fname):
  f = file(fname, 'r')
  x = pickle.load(f)
  f.close()
  return x

def sys_with_stdout(cmd):
  """ Execute shell command and return string containing stdout result. """
  return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()

sys_stdout = sys_with_stdout

def sys_stderr(cmd):
  """ Execute shell command and return string containing stdout result. """
  return subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE).stderr.read()

def one_arg(f):
  return lambda x : f(*x)

def zip_pad(pad, *lsts):
  max_len = max(map(len, lsts))
  with_pad = (lst + [pad] * (max_len - len(lst)) for lst in lsts)
  return zip(*with_pad)

def rect_in_bounds(im, (x, y, w, h)):
  return 0 <= x and 0 <= y and x + w <= im.shape[1] and y + h <= im.shape[0]

def rect_in_bounds_shape(shape, (x, y, w, h)):
  return 0 <= x and 0 <= y and x + w <= shape[1] and y + h <= shape[0]

def img_hessian(im):
  dy, dx = np.gradient(im)
  dyy, dyx = np.gradient(dy)
  _, dxx = np.gradient(dx)
  return dyy, dxx, dyx

def img_laplacian(im):
  dyy, dxx, _ = img_hessian(im)
  return dyy**2 + dxx**2

def strip_dir(dir_name):
  return dir_name[:-1] if dir_name.endswith('/') else dir_name

def ocount(xs):
  return sum(1 for x in xs)

def add_missing(d, ks, val):
  for k in ks:
    if k not in d:
      d[k] = val

def remove_keys(d, ks):
  for k in ks:
    del d[k]

def filter_keys(d, ks):
  ks = set(ks)
  return dict((k, v) for k, v in d.iteritems() if k in ks)

def unique_filename(fname):
  if not os.path.exists(fname):
    return fname
  fname = strip_dir(fname)
  i = 1
  while True:
    base, ext = os.path.splitext(fname)
    new_fname = '%s-%d%s' % (base, i, ext)
    if not os.path.exists(new_fname):
      return new_fname
    i += 1

def angle_diff(a, b):
  diff = abs(a - b) % (2*np.pi)
  return diff if diff < np.pi else 2*np.pi - diff

def test_angle_diff():
  a1 = 0.1
  a2 = 2*np.pi - 0.2
  assert f_eq(angle_diff(a1, a2), 0.3)
  assert f_eq(angle_diff(a2, a1), 0.3)

  a3 = 1.1
  a4 = 4.1
  assert f_eq(angle_diff(a3, a4), 3)
  assert f_eq(angle_diff(a4, a3), 3)

def str_fields(o):
  fields = '\n'.join('  %s = %s' % (f, getattr(o, f)) for f in dir(o))
  return "%s:\n%s" % (o, fields)

def print_fields(o):
  print str_fields(o)

class DictEval:
  def __init__(self, locals = None, globals = None):
    self.locals = locals
    self.globals = globals

  def __getitem__(self, k):
    return eval(k, self.globals, self.locals)

def caller_locals():
  """ local vars for the function that called the function that called caller_locals()"""
  frame = sys._getframe(2)
  return frame.f_locals

def caller_globals():
  """ local vars for the function that called the function that called caller_locals()"""
  frame = sys._getframe(2)
  return frame.f_globals

def vars_from_dict(d):
  lc = caller_locals()
  for varname, val in d.iteritems():
    lc[varname] = val

def is_frame_toplevel(f):
  #return all(x in f.f_locals for x in ('help', 'In', 'Out')) and (f.f_locals['help'].__module__ == 'site')
  return all(x in f.f_locals for x in ('_sh', 'In', 'Out'))

def toplevel_vars(var_dict):
  # Get toplevel frameis
  f = sys._getframe(1)
  while f is not None:
    # Do the locals look like the default IPython locals?
    if is_frame_toplevel(f):
      break
    if hasattr(f, 'f_back'):
      f = f.f_back
    else:
      return
  if f is None:
    print >>sys.stderr, 'Could not find toplevel'
    return
  f.f_locals.update(var_dict)

def get_toplevel_env():
  f = sys._getframe(1)
  while f is not None:
    # Do the locals look like the default IPython locals?
    #if all(x in f.f_locals for x in ('help', 'In', 'Out')) \
    if is_frame_toplevel(f):
      break
    if hasattr(f, 'f_back'):
      f = f.f_back
    else:
      return
  if f is None:
    return
  return f.f_locals

def toplevel_locals(namespace = None):
  """ Adds the caller's local variables to the IPython toplevel locals.
  Prints a warning message on failure. """
  if namespace is None:
    toplevel_vars(caller_locals())
  else:
    toplevel_vars({namespace : caller_locals()})


# def toplevel_locals(*var_names):
#   """ Adds the caller's local variables to the IPython toplevel locals.
#   Fails silently in case it's being run outside of IPython. """
#   # Get toplevel frame
#   f = sys._getframe(1)
#   while f is not None:
#     # Do the locals look like the default IPython locals?
#     if all(x in f.f_locals for x in ('help', 'In', 'Out')) \
#            and (f.f_locals['help'].__module__ == 'site'):
#       break
#     if hasattr(f, 'f_back'):
#       f = f.f_back
#     else:
#       return
#   if f is None:
#     print >>sys.stderr, 'Could not find toplevel'
#     return
#   # Add caller's locals to toplevel
#   caller_locals = sys._getframe(1).f_locals
#   if f is not caller_locals:
#     if len(var_names) == 0:
#       f.f_locals.update(caller_locals)
#     else:
#       for k in var_names:
#         if type(k) == type(('',)):
#           f.f_locals[k[1]] = caller_locals[k[0]]
#         else:
#           f.f_locals[k] = caller_locals[k]
#   else:
#     print >>sys.stderr, 'Could not find toplevel'

# def add_to_toplevel(var_names):
#   # Get toplevel frame
#   f = sys._getframe(1)
#   while f is not None:
#     # Do the locals look like the default IPython locals?
#     if all(x in f.f_locals for x in ('help', 'In', 'Out')) \
#            and (f.f_locals['help'].__module__ == 'site'):
#       break
#     if hasattr(f, 'f_back'):
#       f = f.f_back
#     else:
#       return
#   if f is None:
#     return

#   # Add caller's locals to toplevel
#   caller_locals = sys._getframe(1).f_locals
#   if f is not caller_locals:
#     f.f_locals.update(caller_locals)

# def toplevel_locals():
#   sys._getframe(2).f_locals.update(sys._getframe(1).f_locals)

# def toplevel_locals():
#   sys._getframe(2).f_locals = sys._getframe(1).f_locals

def frm(s):
  """ Magic version of format() that evaluates arbitrary expressions
  inside %(...)s in the format string"""
  frame = sys._getframe(1)
  return s % DictEval(frame.f_locals)

def sub_img_with_pad(A, rect):
  """ Sub-image of A padded with boundary values at invalid
  locations. Note that this means the gradient is 0 at the boundaries."""
  i1, j1 = rect[1], rect[0]
  i2, j2 = rect[1] + rect[3] - 1, rect[0] + rect[2] - 1

  i = np.arange(i1, i2 + 1)
  i = np.where(i < 0, 0, i)
  i = np.where(i > A.shape[0] - 1, A.shape[0] - 1, i)

  j = np.arange(j1, j2 + 1)
  j = np.where(j < 0, 0, j)
  j = np.where(j > A.shape[1] - 1, A.shape[1] - 1, j)

  return A[i[:, np.newaxis], j[np.newaxis, :]]

# def sub_img_with_pad(A, x1, y1, x2, y2):
#   """ Inclusive sub-image of A padded with boundary values at invalid
#   locations. Note that this means the gradient is 0 at the boundaries."""
#   i1, j1 = y1, x1
#   i2, j2 = y2, x2
#   B = np.zeros((i2 - i1 + 1, j2 - j1 + 1) + tuple(A.shape[2:]))
#   for i in xrange(i1, 1 + i2):
#     for j in xrange(j1, 1 + j2):
#       ii = min(max(i, 0), A.shape[0]-1)
#       jj = min(max(j, 0), A.shape[1]-1)
#       B[i - i1, j - j1] = A[ii, jj]
#   return B

def test_sub_img_with_pad():
  A = np.ones((10, 10))
  A[:, 0] = 0
  A[0, :] = 0
  B = sub_img_with_pad(A, -2, -2, 2, 2)
  print B
  assert np.all(B[:, :2] == 0)
  assert np.all(B[:2, :] == 0)
  assert np.all(B[3:, 3:] == 1)

def red_green_spectrum(x):
  assert (0 <= x <= 1)
  m = pylab.get_cmap('RdYlGn')
  #matplotlib.pyplot.hsv()
  return int_tuple(255*np.array(m(x)[:3]))

def is_sorted(xs):
  prev = None
  for x in xs:
    if prev > x:
      return False
    prev = x
  return True

def take_last(xs):
  for x in xs:
    pass
  return x

def relative_module(m):
  return hasattr(m, '__file__') \
         and ((not m.__file__.startswith('/')) \
              or m.__file__.startswith(os.getcwd()))

def safe_reload(m):
  if not relative_module(m):
    reload(m)

def invert_perm(inds):
  res = [None] * len(inds)
  for ord, i in enumerate(inds):
    res[i] = ord
  return res

def assert_eq(a, b):
  if a != b:
    print >>sys.stderr, 'assert_eq failed:', a, '!=', b
  assert a == b

def quote(x): return '"%s"' % x

class ColorChooser:
  def __init__(self, dist_thresh = 500, attempts = 500, init_colors = [], init_pts = []):
    self.pts = init_pts
    self.colors = init_colors
    self.attempts = attempts
    self.dist_thresh = dist_thresh

  def choose(self, new_pt = (0, 0)):
    new_pt = np.array(new_pt)
    nearby_colors = []
    for pt, c in zip(self.pts, self.colors):
      if np.sum((pt - new_pt)**2) <= self.dist_thresh**2:
        nearby_colors.append(c)

    # nearby_colors = np.array(nearby_colors)
    # dist_best = -1
    # color_best = None
    # for i in xrange(self.attempts):
    #   new_color = rand_color()
    #   dist_sum = 0 if len(nearby_colors) == 0 else np.min(np.sqrt(np.sum((nearby_colors - np.array(new_color))**2, axis = 1)))
    #   if dist_sum > dist_best:
    #     dist_best = dist_sum
    #     color_best = new_color

    if len(nearby_colors) == 0:
      color_best = rand_color()
    else:
      nearby_colors = np.array(sample_at_most(nearby_colors, 100), 'l')
      choices = np.array(np.random.rand(self.attempts, 3)*256, 'l')
      dists = np.sqrt(np.sum((choices[:, np.newaxis, :] - nearby_colors[np.newaxis, :, :])**2, axis = 2))
      costs = np.min(dists, axis = 1)
      assert costs.shape == (len(choices),)
      color_best = itup(choices[np.argmax(costs)])

    self.pts.append(new_pt)
    self.colors.append(color_best)
    return color_best

# def distinct_colors(n):
#   cc = ColorChooser(attempts = 10)
#   return do_with_seed(lambda : [cc.choose((0,0)) for x in xrange(n)])

def distinct_colors(n):
  #cc = ColorChooser(attempts = 10, init_colors = [red, green, blue, yellow, purple, cyan], init_pts = [(0, 0)]*6)
  cc = ColorChooser(attempts = 100, init_colors = [red, green, blue, yellow, purple, cyan], init_pts = [(0, 0)]*6)
  do_with_seed(lambda : [cc.choose((0,0)) for x in xrange(n)])
  return cc.colors[:n]

def notnan(x):
  return np.logical_not(np.isnan(x))

def notnan_vals(x):
  return x[notnan(x)]

def debug_numpy_warnings(b):
  if b:
    np.seterr(all='raise')
  else:
    np.seterr(all='warn')

def readable_timestamp():
  return datetime.datetime.today().strftime('%I:%M %p %a')

def readable_datestamp():
  return datetime.datetime.today().strftime('%I:%M %p %a %m/%d/%y')

def simple_datestamp():
  return datetime.datetime.today().strftime('%y-%m-%d')

def simple_timestamp():
  return datetime.datetime.today().strftime('%y-%m-%d_%I-%M%p')


def take_inds_each(vars, inds):
  return [take_inds(x, inds) for x in vars]

def sample_replace(xs, n):
  if len(xs) == 0 and n > 0:
    fail("Can't sample from empty collection")
  ys = []
  for i in xrange(n):
    ys.append(xs[random.randint(0, len(xs)-1)])
  return ys

def concat_lists(heads, tails):
  for hd, tl in itl.izip(heads, tails):
    hd += tl

# def update_class(o):
#   o.__class__ = eval(str(o.__class__))

# def update_class(o):
#   o.__class__ = getattr(sys.modules[o.__module__], str(o.__class__).split('.')[-1])
#   return o

def update_class(o):
  if hasattr(o, '__module__') and o.__module__ in sys.modules:
    m = sys.modules[o.__module__]
    if hasattr(o, '__class__'):
      cname = str(o.__class__).split('.')[-1]
      if hasattr(m, cname):
        o.__class__ = getattr(m, cname)
  return o

def check_print(b, msg = None):
  """ Prints the assert message to stderr if it failed """
  if not b:
    if msg is not None:
      print >>sys.stderr, msg
    fail('Check failed %s' % ('' if msg is None else msg))

def lazy_get(dict, k, f):
  if k not in dict:
    dict[k] = f()
  return dict[k]

def list2d(n, m, v = None):
  return [[None] * m for x in xrange(n)]

def list3d(n, m, d, v = None):
  return [list2d(m, d) for x in xrange(n)]

def transpose_list2d(lsts):
  return map(list, zip(*lsts))

def dup(n, x):
  return [copy.deepcopy(x) for i in xrange(n)]

# def guess_bytes(x):
#   return len(pickle.dumps(x))

def guess_bytes(x):
  return len(pickle.dumps(x, 2))
  #return len(pickle.dumps(x, -1))

def kb(bytes): return bytes/(2.**10)
def mb(bytes): return bytes/(2.**20)
def gb(bytes): return bytes/(2.**30)

def hours(sec): return sec/60./60.
def minutes(sec): return sec/60.

def sec_from_ms(s): return s / 1000.
#def funval(x): return lambda : copy.deepcopy(x)

def indsort(xs, key = None):
  return sortlst(range(len(xs)),
                 key = ((lambda i : xs[i]) if key is None else (lambda i : key(xs[i]))))

def lnot(x): return np.logical_not(x)
def land(*x): return logical_and_many(*x)
def lor(*x): return logical_or_many(*x)

class MapAttr:
  def __init__(self, xs):
    self.xs = xs

  def __getattr__(self, k):
    print 'k=',k
    return [getattr(x, k) for x in self.xs]

class MapAttr:
  def __init__(self, xs):
    self.xs = xs

  def __getattr__(self, k):
    return [getattr(x, k) for x in self.xs]

class MapMeth:
  def __init__(self, xs):
    self.xs = xs

  def __getattr__(self, k):
    return (lambda *args, **kwargs : [getattr(x, k)(*args, **kwargs) for x in self.xs]) 

# def imapattr(xs):
#   return MapAttr(xs)

#def mapattr(xs): return list(imapattr(xs))
# should rename, say factor?
def mapattr(xs): return MapAttr(xs)
def mapmeth(xs): return MapMeth(xs)

def np_bytes(A): return A.dtype.itemsize*A.size

def np_bytes_compress(A):
  f = StringIO.StringIO()
  np.savez_compressed(f, A)
  return len(f.getvalue())

def np_compress(A):
  f = StringIO.StringIO()
  np.savez_compressed(f, A)
  return f.getvalue()

def np_uncompress(s):
  f = StringIO.StringIO(s)
  return np.load(f)['arr_0']

def eat(xs):
  x = None
  for x in xs:
    pass
  return x

# def set_fields(o, *dicts, **fields):
#   for d in dicts:
#     o.__dict__.update(d)
#   o.__dict__.update(fields)


def labval(**kwargs):
  """ Sometimes values, when passed as arguments, aren't descriptive enough without more context. This
  lets you label a value, e.g. foo(labval(reset=True))"""
  assert len(kwargs) == 1;
  return kwargs.values()[0]

def dict_count(xs):
  d = {}
  for x in xs:
    if x not in d:
      d[x] = 1
    else:
      d[x] += 1
  return d

def prnerr():
  import traceback
  traceback.print_exception(sys.last_type, sys.last_value, sys.last_traceback)

izip = itl.izip

# from http://shallowsky.com/blog/programming/python-tee.html
# note: has problems, don't use
class tee :
  def __init__(self, _fd1, _fd2) :
    self.fd1 = _fd1
    self.fd2 = _fd2

  def __del__(self) :
    if self.fd1 != sys.stdout and self.fd1 != sys.stderr :
      self.fd1.close()
    if self.fd2 != sys.stdout and self.fd2 != sys.stderr :
      self.fd2.close()

  def write(self, text) :
    self.fd1.write(text)
    self.fd2.write(text)

  def flush(self) :
    self.fd1.flush()
    self.fd2.flush()

def set_inscomp(s1, s2):
  s1 = set(s1)
  s2 = set(s2)
  return (s1 - s2).union(s2 - s1)

def anynot(xs): return any(not x for x in xs)

# stderrsav = sys.stderr
# outputlog = open(logfilename, "w")
# sys.stderr = tee(stderrsav, outputlog)

# def update_class(o):
#   """ Updates the class to the latest version. Useful for testing
#   methods in the toplevel and for pickling between module reloads. """
#   o.__class__ = eval(str(o.__class__))
#   return o

upcls = update_class

def iceil(x): return int(np.ceil(x))
def ifloor(x): return int(np.floor(x))

def fig_im(f):
  pylab.clf()
  f()
  return ig.from_fig(pylab.gcf())

# def normalize_l1(xs):
#   xs = np.array(xs, 'd')
#   xs[xs != 0] /= np.sum(np.abs(xs))
#   return xs

def normalize_l1(xs):
  xs = np.asarray(xs, 'd')
  s = np.sum(np.abs(xs))
  if s == 0:
    return xs
  else:
    return xs / s



def f1(x): return '%.1f' % x
def f2(x): return '%.2f' % x
def f3(x): return '%.3f' % x
def f4(x): return '%.4f' % x
def f5(x): return '%.5f' % x
def f6(x): return '%.6f' % x

def git_commit_hash():
  return sys_with_stdout('git rev-parse HEAD').rstrip()

def co_occur(vals1, vals2, n1, n2 = None):
  assert vals1.ndim == 1
  assert vals1.shape == vals2.shape
  if n2 is None:
    n2 = n1
  return np.array(scipy.sparse.csr_matrix((np.ones(vals1.shape), [vals1, vals2]), shape = (n1, n2)).todense())

def occur(vals, n):
  if vals.ndim > 1:
    vals = vals.flatten()
  return np.array(scipy.sparse.csr_matrix((np.ones(vals.shape), np.array([vals, np.zeros_like(vals)])), shape = (n, 1)).todense(), dtype = 'l').flatten()

def prob_rows(A):
  return np.double(A) / np.maximum(0.0000001, np.sum(A, axis = 1))[:, np.newaxis]

def prn_lines(lines):
  for x in lines:
    print x

def rect_shape(r):
  return (int(r[3]), int(r[2]))

def ls_full(dir):
  return (os.path.join(dir, fname) for fname in os.listdir(dir))

def wait_for_file(fname, timeout_sec):
  start = now_sec()
  while not os.path.exists(fname):
    if now_sec() - start > timeout_sec:
      fail('File does not exist after %f seconds: %s' % (timeout_sec, fname))
    else:
      time.sleep(1)

def host_file(src_fname, dst_fname = None, server_dir = '/csail/vision-billf5/aho/www/', subdir = 'hosted', public_url = 'http://bozo.csail.mit.edu/aho'):
  server_dir = os.path.join(server_dir, subdir)
  if dst_fname is None:
    dst_fname = make_temp(os.path.splitext(src_fname)[-1], server_dir).split('/')[-1]
  dst_server = os.path.join(server_dir, dst_fname)
  os.system('cp %s %s' % (src_fname, dst_server))
  os.system('chmod a+rwx %s' % dst_server)
  url = os.path.join(public_url, subdir, dst_fname)
  print url
  return dst_server, url

def concat_files(fnames):
  return ''.join(itl.imap(read_file, fnames))

def check_feq(x, y):
  if not feq(x, y):
    fail('check_feq failed: %f != %f' % (x, y))

def check_feqs(*args):
  assert len(args) % 2 == 0
  for i, (x, y) in enumerate(split_n(args, 2)):
    if not feq(x, y):
      fail('check_feqs (%d) failed: %f != %f' % (i, x, y))

def remove_nan(xs):
  return (x for x in xs if not np.isnan(x))

def remove_inf(xs):
  return (x for x in xs if not np.isinf(x))

rmnan = remove_nan
rminf = remove_inf

def strict_np_errs(enable):
  if enable:
    np.seterr(all = 'raise')
  else:
    np.seterr(divide = 'warn', invalid = 'warn', over = 'warn', under = 'ignore')

def afill(val, *args):
  A = np.zeros(*args)
  A[:] = val
  return A

def sample_with_seed(xs, n, seed = 0):
  return do_with_seed(lambda : random.sample(xs, n))

def sample_at_most_with_seed(xs, n, seed = 0):
  return do_with_seed(lambda : sample_at_most(xs, n), seed)

def shuffled_with_seed(xs, seed = 0):
  return do_with_seed(lambda : shuffled(xs), seed)

def asample_most(a, n):
  if len(a) == 0:
    return a.copy()
  else:
    inds = sample_at_most(range(len(a)), n)
    return a[inds]

def choose2(xs):
  for i in xrange(len(xs)):
    for j in xrange(1+i, len(xs)):
      yield xs[i], xs[j]

def preval(code):
  print code, '=', eval(code, caller_globals(), caller_locals())

def bitstrings(n):
  bs = [[]]
  for k in xrange(n):
    bs = [b+[0] for b in bs] + [b+[1] for b in bs]
  return np.array(bs)

def dict_from_list(lst):
  return {i : x for i, x in enumerate(lst)}

def axnorm(a, axis = 0):
  return np.sqrt(np.sum(a**2, axis = axis))

def gaussian_impulse(sigma, size = None):
  import scipy.ndimage
  if size is None:
    size = 4*np.ceil(sigma) + 1
  a = np.zeros((size, size))
  if size > 0:
    a[size/2, size/2] = 1.
  G = scipy.ndimage.gaussian_filter(a, sigma)
  G /= np.sum(G)
  return G

# def catnew(xs, axis = -1):
#   return np.concatenate([np.array(x)[np.newaxis,...] for x in xs], axis = axis)

def istup(x):
  return (type(x) == type((1,)))

def structargs(names):
  if type(names) == type(''):
    names = names.split()
  # def f(tup):
  #   if len(tup) != len(names):
  #     raise RuntimeError('Input tuple should have %d values; got %d' % (len(names), len(tup)))
  #   return ut.Struct(**{k : v for k, v in itl.izip(names, tup)})
  def f(tup):
    d = {}
    if len(tup) != len(names):
      raise RuntimeError('Input tuple should have size %d; got one with size %d' % (len(names), len(tup)))
    for i in xrange(len(tup)):
      if type(names[i]) == type(''):
        d[names[i]] = tup[i]
      elif type(names[i]) == type((1,)):
        if len(names[i]) != len(tup[i]):
          raise RuntimeError('Tuple %d should have size %d; got one with size %d' % (i, len(names[i]), len(tup[i])))
        for j in xrange(len(names[i])):
          if type(names[i][j]) != type(''):
            raise RuntimeError('Field name tuple %d should only contain strings' % i)
          d[names[i][j]] = tup[i][j]
      else:
        raise RuntimeError('Bad input field: %s' % names[i])

    return Struct(**d)

  return f

def structprod(*name_val_pairs, **kwargs):
  if len(name_val_pairs) % 2 != 0:
    raise RuntimeError('Bad input, should alternate name and value')

  names = list(name_val_pairs[::2])
  vals = list(name_val_pairs[1::2])

  for k, v in kwargs.iteritems():
    names.append(k)
    vals.append(v)

  return map(structargs(names), itl.product(*vals))

def test_structargs():
  f = structargs([('foo', 'bar'), 'baz'])
  x = f([(1, 2), 3])
  assert x.foo == 1 and x.bar == 2 and x.baz == 3

def kd_query(kd, *args, **kwargs):
  # cKDTree query returns only one value when k = 1.  This ensures that the result always has the right shape.
  dist, idx = kd.query(*args, **kwargs)
  if np.ndim(idx) == 0:
    return np.array([dist], 'd'), np.array([idx])
  elif np.ndim(idx) == 1:
    return dist[:, np.newaxis], idx[:, np.newaxis]
  else:
    return dist, idx

def alist(col_shape, a, dtype = None):
  """ Return a as an array or if a is empty, return an array with 0 rows and col_shape
  for the other dimensions """
  if type(col_shape) == type(1):
    col_shape = (col_shape,)
  if len(a) == 0:
    return np.zeros((0,) + col_shape, dtype = dtype)
  else:
    a = np.asarray(a, dtype = dtype)
    assert a.shape[1:] == col_shape
    return a

def vals_keysort(d):
  return [d[k] for k in sorted(d)]

# I use this pretty often, and the scipy name for it is clunky
def kd(*args, **kwargs):
  from scipy.spatial import cKDTree
  return cKDTree(*args, **kwargs)

def knnsearch(N, X, k = 1, method = 'brute', p = 2.):
  #if p != 2: assert method == 'kd'

  if method == 'kd':
    kd_ = kd(N)
    return kd_query(kd_, X, k = k, p = p)
  elif method == 'brute':
    import scipy.spatial.distance
    if p == 2:
      D = scipy.spatial.distance.cdist(X, N)
    else:
      D = scipy.spatial.distance.cdist(X, N, p)

    if k == 1:
      I = np.argmin(D, 1)[:, np.newaxis]
    else:
      I = np.argsort(D)[:, :k]
    return D[np.arange(D.shape[0])[:, np.newaxis], I], I 
  else:
    fail('Unknown search method: %s' % method)

def test_knnsearch(N, X, k, method):
  D1, I1 = knnsearch(N, X, k, 'kd')
  D2, I2 = knnsearch(N, X, k, 'brute')
  assert np.max(np.abs(D1 - D2)) <= 0.01
  assert np.max(np.abs(I1 - I2)) <= 0.001
  return D2, I2

# adapted from dave crandall's code
def rot_matrix_abstwist(view_dir, abstwist = 0.):
  # rotate the viewing direction from vec(0,0,1) to view_dir
  """ >>> a_eq(ut.A('0 0 1; 0 1 0; -1 0 0').T, match.rot_matrix_abstwist(ut.vec(1, 0, 0))) """
  # todo: handle case with view_dir looking straight up or down
  # abstwist != 0 untested!
  X, Y, Z = -view_dir
  QUANTUM = 1e-10
  # last column of R is just -(X,Y,Z), (normalized just in case)
  len = np.sqrt(X*X + Y*Y + Z*Z);
  X /= len; Y /= len; Z /= len;

  # there is an ambiguity when looking straight up or straight down; handle it by choosing one
  # arbitrarily
  if np.abs(np.abs(Y) - 1) <= 0.0001:
    return np.array([[1.,0,0], 
                     [0.,0.,-np.sign(Y)],
                     [0.,Y,0.]])

  # # if X and Z are both 0, then the camera is pointing straight up or down.
  # # for now, just handle this as if abstwist were 0.
  #if abs(X) < QUANTUM and abs(Z) < QUANTUM:
  #return rot_matrix_abstwist(X, Y, Z, 0)
  # first column is (A, B, C), where B=asin(twist), |(A,B,C)=1|, and (A,B,C) . (X,Y,Z) = 0.
  # solve for C with quadratic equation:
  # (Z^2/X^2 + 1) * C^2 + 2*Z*B*Y/X^2*C + (B^2*Y^2/X^2 - 1 + B^2)
  B = abstwist
  A, C, A1, C1, A2, C2 = 0, 0, 0, 0, 0, 0
  if (np.abs(X) < QUANTUM):
    C2 = C1 = B*Y/Z;
    A1 = np.sqrt(1-B*B*(1+Y*Y/(Z*Z)));
    A2 = -np.sqrt(1-B*B*(1+Y*Y/(Z*Z)));
  else:
    X_sqr_inv = 1.0/(X*X);
    _a = (Z*Z)*X_sqr_inv + 1
    _b = 2*Z*B*Y*X_sqr_inv
    _c = B*B*Y*Y*X_sqr_inv - 1 + B*B
    _sqrt = np.sqrt(_b*_b - 4 * _a * _c)
    C1 = (-_b + _sqrt)/(2*_a)
    C2 = (-_b - _sqrt)/(2*_a)
    A1 = (-B*Y-C1*Z) / X
    A2 = (-B*Y-C2*Z) / X

  C = C1; A = A1;

  # choose solution that is "90 left" of viewing direction (assumes twist angle 
  #  relatively small, definitely smaller than 90 degrees)
  signmag = np.fabs(A2)+np.fabs(Z) - (np.fabs(C2)+np.fabs(X))
  if ((np.sign(A2) != np.sign(Z) and signmag >= 0) or (np.sign(C2) == np.sign(X) and signmag < 0)):
    C = C2
    A = A2

  # second column is -(A,B,C) x (X, Y, Z)
  return np.array([[A, B*Z - C*Y, -X],
                   [B, C*X - A*Z, -Y],
                   [C, A*Y - B*X, -Z]]).T

def save_ply(fname, pts, colors = None, show_cmd = False, nsample = None):
  if nsample is not None:
    pts, colors = sample_at_most_each([pts, colors], nsample)

  if colors is None:
    colors = [(255, 255, 255)] * len(pts)
  info = "Date: %s" % datetime.datetime.today().strftime('%I:%M %p %a')

  with open(fname, 'w') as f:
    header = """\
ply
format ascii 1.0
comment %s
element vertex %d
property float x
property float y
property float z
property uchar diffuse_red
property uchar diffuse_green
property uchar diffuse_blue
end_header
""" % (info, len(pts))
  f.write(header)
  for (x, y, z), (r, g, b) in zip(pts, colors):
    f.write("%f %f %f %d %d %d\n" % (x, y, z, r, g, b))

  if show_cmd:
    os.system('ply_cmd %s' % fname) 

def show_ply_cmd(fname):
  os.system('ply_cmd %s' % fname)
  
def tupsort(*args):
  if len(args) == 1:
    return tuple(sorted(args[0]))
  else:
    return tuple(sorted(args))
    
def normax(A, axis = -1):
  return np.sqrt(np.sum(A**2, axis = axis))

def normax_padzero(A, axis = -1):
  n = np.sqrt(np.sum(A**2, axis = axis))
  n[n == 0] = np.finfo(np.double).eps
  return n

def make_todo(todo, choices, check_valid = 1):
  if todo == 'all': return choices
  if type(todo) == type(''):
    return todo.split()
  else:
    if check_valid:
      assert choices is None or all((x in choices) for x in todo)
    return todo

def make_todo_cp(cp, todo, choices):
  if todo == 'all':
    todo = choices
  elif type(todo) == type(''):
    todo = todo.split()

  new_todo = set()
  for t in todo:
    if t == 'missing':
      new_todo.update([c for c in choices if not cp.exists(c)])
    else:
      new_todo.add(t)

  #print new_todo, choices
  #assert all((x in choices) for x in new_todo)
  return sorted(new_todo)

def is_todo(name, todo, fname):
  test_missing = ('missing' in todo) or (('missing-' + name) in todo)
  return (name in todo) or (test_missing and (not os.path.exists(fname)))

def rlstrip(file):
  return file.readline().rstrip()

def iround(x): return np.array(np.round(x), 'l')
roundi = iround

def temp_www_url():
  if 0:
    output_root = '/csail/vision-billf5/aho/www/results'
    html_root = 'http://quickstep.csail.mit.edu/aho/results'
    
  output_root = '/csail/vision-billf5/aho/www/results2'
  html_root = 'http://quickstep.csail.mit.edu/aho/results2'    
  dir = make_temp_dir(dir = output_root).split('/')[-1]
  output_dir = pjoin(output_root, dir)
  public_url = pjoin(html_root, dir)
  os.system('chmod -R a+rwx %s' % output_dir)
  return output_dir, public_url

class Checkpoint:
  def __init__(self, resdir):
    self.resdir = resdir
    mkdir(resdir)

  def fname(self, part): return pjoin(self.resdir, '%s.pk' % part)
  def save(self, part, x): save(self.fname(part), x)
  def load(self, part): return load(self.fname(part))
  def exists(self, part): return os.path.exists(self.fname(part))

  def cache(self, reset, part, f):
    if reset or not self.exists(part):
      self.save(part, f())
    return self.load(part)
  
def add_opts(pr, opts, allow_new_fields = True):
  for k in opts.attrs():
    if not k.startswith('__'):
      if not allow_new_fields:
        assert hasattr(pr, k)
      setattr(pr, k, getattr(opts, k))

def add_fields(pr, opts):
  for k in opts.attrs():
    if not k.startswith('__'):
      setattr(pr, k, getattr(opts, k))


def ndistinct(xs):
  return len(set(xs))

def all_same(xs):
  return ndistinct(xs) <= 1

def choose_distinct(f, xs):
  s = set()
  new_xs = []
  for x in xs:
    k = f(x)
    if k not in s:
      s.add(k)
      new_xs.append(x)
  return new_xs

def choose_n_distinct(f, xs, max = np.inf):
  s = {}
  new_xs = []
  for x in xs:
    k = f(x)
    if k not in s:
      s[k] = 0
    if s[k] < max:
      s[k] += 1
      new_xs.append(x)
  return new_xs

def ignore_exc(f, show = True):
  try:
    f()
  except Exception as e:
    if show:
      print sys.exc_info()[0]
    return e
  except:
    if show:
      print sys.exc_info()[0]

def update_local_classes():
  for o in gc.get_objects():
    if hasattr(o, '__module__') and o.__module__ in sys.modules \
           and relative_module(sys.modules[o.__module__]):
      update_class(o)

def rec_update_class(o):
  # todo: rewrite this in C
  q = collections.deque([o])
  seen = set()
  while len(q):
    x = q.pop()
    # list of things to avoid is somewhat arbitrary!
    if (id(x) not in seen):
      seen.add(id(x))
      if (not inspect.ismodule(x)) and (not inspect.isclass(x)):
        update_class(x)
        for y in gc.get_referents(x):
          q.append(y)
  return o

uc = rec_update_class

red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
yellow = (255, 255, 0)
purple = (255, 0, 255)
cyan = (0, 255, 255)

def link(x, txt = None):
  return '<a href = "%s">%s</a>' % (x, (x if (txt is None) else txt))

def take_bool(xs, bs):
  return [x for x, b in itl.izip(xs, bs) if b]

def same_check(xs):
  first, res = True, None
  for x in xs:
    if first:
      first = False
      res = x
    elif res != x:
      raise RuntimeError('All elements should be the same:  %s != %s' % (res, x))
  if first:
    raise RuntimeError('Empty')
  else:
    return res
    
def pylab_normhist(counts, bins):
  h, b = np.histogram(counts, bins)
  pylab.bar(b[:-1], np.array(h, 'd')/float(np.sum(h)), np.diff(b))
  #pylab.bar(bins[:-1], np.array(counts)/float(np.sum(counts)), np.diff(bins))

def ensure_col(x, m, dtype = 'd'):
  if len(x) == 0:
    return np.zeros((0, m), dtype = dtype)
  else:
    return np.array(x)

def mult_ptm(A, ptm):
  new_ptm = np.dot(A, np.vstack([ptm[:, :, 0].flatten(), ptm[:, :, 1].flatten(), ptm[:, :, 2].flatten()]))
  def r(x): return x.reshape(ptm.shape[:2])[:, :, np.newaxis]
  return np.concatenate(map(r, [new_ptm[0, :], new_ptm[1, :], new_ptm[2, :], ptm[:, :, 3].flatten()]), axis = 2)

def mult_im(A, im):
  new_im = np.dot(A, np.vstack([im[:, :, 0].flatten(), im[:, :, 1].flatten(), im[:, :, 2].flatten()]))
  new_im = new_im.astype('d')
  def r(x): return x.reshape(im.shape[:2])[:, :, np.newaxis]
  return np.concatenate(map(r, [new_im[0, :], new_im[1, :], new_im[2, :]]), axis = 2)

#def wait_for_nfs(max_wait = 60*30, test_file = '/data/vision/torralba/sun3d/sfm/scene/lounge/wg_1lounge/s3/RGBDsfm3477_cameraRt_rectify.mat'):
def wait_for_nfs(max_wait = 60*30, test_file = '/data/vision/billf/aho-stuff/vis/data/flickr2/v4-dset/3906219160/feat/0010_sf_tex.pk'):
  # todo: pick a more general file
  if not os.path.exists(test_file):
    print 'NFS down!'
    start = time.time()
    while (not os.path.exists(test_file)) and (time.time() < start + max_wait):
      time.sleep(1)
    print 'Waited for', time.time() - start, 'seconds'

  return os.path.exists(test_file)

@yield_list  
def dir_vals(x):
  for f in dir(x):
    v = getattr(x, f)
    if (not inspect.isroutine(v)) and (not inspect.ismodule(v)) and (not inspect.isclass(v)):
      yield f

double_eps = np.finfo(np.double).eps
float_eps = np.finfo(np.float).eps

def hostname(*args):
  return sys_with_stdout('hostname').rstrip()

def append_ret(lst, x):
  lst.append(x)
  return x

def sjoin(*args):
  return ' '.join(map(str, args))

def number_distinct(xs):
  xs = sorted(set(xs))
  return dict(zip(xs, range(len(xs))))


def minmax(xs):
  return np.min(xs), np.max(xs)

class LocalCheckpoint:
  def __init__(self, name):
    self.name = name

  def _get(self):
    env = get_toplevel_env()
    if env is None:
      return {}

    if '_localcp' not in env:
      env['_localcp'] = {}
      
    if self.name not in env['_localcp']:
      env['_localcp'][self.name] = {}
      
    return env['_localcp'][self.name]
    
  def save(self, part, x):
    self._get()[part] = x
    
  def load(self, part):
    return self._get()[part]
  
  def exists(self, part):
    return (part in self._get())

  def cache(self, reset, part, f):
    if reset or not self.exists(part):
      self.save(part, f())
    return self.load(part)
    
  # def load_locals(self, part):
  #   vars = self.load(part)
  #   assert type(vars) == type({})
  #   print caller_locals()
  #   print 'vars', vars
  #   caller_locals()['actions'] = 123
  #   print 'caller locals after', caller_locals()



def parmap(f, xs, nproc = None):
  if nproc is None:
    nproc = 12

  if 1:
    pool = multiprocessing.Pool(processes = nproc)
    try:
      ret = pool.map_async(f, xs).get(100000)
    finally:
      pool.close()
    return ret
  
  if 0:
    import billiard
    pool = billiard.Pool(processes = nproc)
    try:
      ret = pool.map(f, xs)
    finally:
      pool.close()
  return ret

def parmap_dill(f, xs, nproc = None):
   import pathos.multiprocessing
   if nproc is None:
     nproc = 12
   pool = pathos.multiprocessing.Pool(processes = nproc)
   try:
     ret = pool.map_async(f, xs).get(10000000)
   finally:
     pool.close()
   return ret
     
def aparmap(*args, **kwargs):
  return np.array(parmap(*args, **kwargs))

def aparmapm(par, *args, **kwargs):
  return np.array(maybe_parmap(par, *args, **kwargs))

def parfilter(fn, xs):
  ok = parmap(fn, xs)
  return [x for i, x in enumerate(xs) if ok[i]]
  
def maybe_parmap(par, f, xs, nproc = None):
  if par:
    return parmap(f, xs, nproc)
  else:
    return map(f, xs)
  
def norms(xs):
  xs = np.asarray(xs)
  return np.sqrt(np.sum(xs**2, axis = 1))

def interpolate_line(pts, n = 500):
  pts = np.asarray(pts)
  interp = []
  for i in xrange(len(pts)-1):
    p = np.linspace(0, 1., n)[:, np.newaxis]
    interp += list(p*pts[i][np.newaxis, :] + (1-p)*pts[i+1][np.newaxis, :])
  return ensure_col(interp, pts.shape[1])

def max_diff(a, b):
  return np.max(np.abs(a-b))

def sample_part(xs, n):
  rn = range(len(xs))
  inds = random.sample(rn, n)
  others = sorted(set(rn) - set(inds))
  return take_inds(xs, inds), take_inds(xs, others)

def clip_rescale(x, lo = None, hi = None):
  if lo is None:
    lo = np.min(x)
  if hi is None:
    hi = np.max(x)
  return np.clip((x - lo)/(hi - lo), 0., 1.)

# def clip_rescale_im(x, *args, **kwargs):
#   return np.uint8(255*clip_rescale(x, *args, **kwargs))

def clip_rescale_im(x, *args, **kwargs):
  return ig.rgb_from_gray(np.uint8(255*clip_rescale(x, *args, **kwargs)))

class temp_file:
  def __init__(self, ext, fname_only = False, delete_on_exit = True):
    self.fname = make_temp(ext)
    self.delete_on_exit = delete_on_exit
    if fname_only:
      os.remove(self.fname)

  def __enter__(self):
    return self.fname

  def __exit__(self, type, value, tb):
    if self.delete_on_exit and os.path.exists(self.fname):
      os.remove(self.fname)

class temp_files:
  def __init__(self, ext, count, path = None, fname_only = False, delete_on_exit = True):
    self.fnames = [make_temp(ext, dir = path) for i in xrange(count)]
    self.delete_on_exit = delete_on_exit
    if fname_only:
      for fname in self.fnames:
        os.remove(fname)

  def __enter__(self):
    return self.fnames

  def __exit__(self, type, value, tb):
    if self.delete_on_exit:
      for fname in self.fnames:
        if os.path.exists(fname):
          os.remove(fname)


def imagesc(im, lo = None, hi = None, cmap = None):
  if cmap is None:
    cmap = pylab.cm.jet
  pylab.imshow(clip_rescale(im, lo, hi), cmap)
  return ig.from_fig()

def apply_cmap(im, cmap = pylab.cm.jet, lo = None, hi = None):
  return cmap(clip_rescale(im, lo, hi).flatten()).reshape(im.shape[:2] + (-1,))[:, :, :3]

def jet(im, lo = None, hi = None):
  return np.uint8(255*apply_cmap(im, pylab.cm.jet, lo, hi))

def parula(im, lo = None, hi = None):
  import parula
  return np.uint8(255*apply_cmap(im, parula.parula_map, lo, hi))

def hot(im, lo = None, hi = None):
  return np.uint8(255*apply_cmap(im, pylab.cm.hot, lo, hi))

def cmap_im(cmap, im, lo = None, hi = None):
  return np.uint8(255*apply_cmap(im, cmap, lo, hi))

def bool_from_inds(n, inds):
  a = np.zeros(n, 'bool')
  a[inds] = True
  return a

def merge_maps(maps):
  res = {}
  for x in maps:
    for k, v in x.iteritems():
      add_dict_list(res, k, v)
  return res

# Classes for lazily computing a value and storing the result on a filesystem or in memory
class LazyFS:
  def __init__(self, v):
    self.fname = make_temp_nfs('.pk')
    self.loaded = False
    print self.fname
    save(self.fname, v)
    
  def get(self):
    if not self.loaded:
      self.loaded = True
      wait_for_nfs(test_file = self.fname)
      self.cache = load(self.fname)
    return self.cache

  def clear(self):
    self.loaded = self.cache = None
    if os.path.exists(self.fname):
      print 'Removing', self.fname
      os.remove(self.fname)

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    self.clear()

  def __del__(self):
    pass
    #self.clear()
    
class LazyMem:
  def __init__(self, v):
    self.v = v

  def get(self):
    return self.v

  def clear(self):
    pass

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    self.clear()

class LazyLoader:
  def __init__(self, fname, use_joblib = False):
    self.x = None
    self.loaded = False
    self.fname = fname
    self.use_joblib = use_joblib

  def get(self):
    if not self.loaded:
      load_fn = jload if self.use_joblib else load
      self.x = load_fn(self.fname)
      self.loaded = True
    return self.x
  
def lazy_from_parallel(parallel, x):
  return LazyFS(x) if parallel else LazyMem(x)

def first_nonzero(xs):
  return np.nonzero(xs)[0][0]

def glob(*args):
  return _glob.glob(os.path.join(*args))

def sortglob(*args):
  return sorted(glob(*args))

def cast_types(xs, types):
  return tuple([np.array([x], t)[0] for x, t in zip(xs, types)])

def path_subdir(path, n = 1):
  for i in xrange(n):
    path = os.path.split(path)[0]
  return path

def n_if_none(xs, n, fill = None):
  if xs is None:
    return [fill]*n
  elif len(xs) != n:
    raise RuntimeError('Expected input with %d elements. Got input with %d.' % (n, len(xs)))
  else:
    return xs

def accum_dict(kvs):
  d = {}
  for k, v in kvs:
    add_dict_list(d, k, v)
  return d

def middle(xs):
  return xs[len(xs)/2]

def norm(x):
  return np.linalg.norm(x)


# http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
import contextlib
@contextlib.contextmanager
def printoptions(*args, **kwargs):
  original = np.get_printoptions()
  np.set_printoptions(*args, **kwargs)
  yield
  np.set_printoptions(**original)

def print_prec(x, prec):
  with printoptions(precision=prec, suppress=True):
    print x

def print1f(x): print_prec(x, 1)
def print2f(x): print_prec(x, 2)
def print3f(x): print_prec(x, 3)
def print4f(x): print_prec(x, 4)


def atleast_2d_col(x):
  x = np.asarray(x)
  if np.ndim(x) == 0:
    return x[np.newaxis, np.newaxis]
  if np.ndim(x) == 1:
    return x[:, np.newaxis]
  else:
    return x

def atleast_2d_row(x):
  x = np.asarray(x)
  if np.ndim(x) == 0:
    return x[np.newaxis, np.newaxis]
  if np.ndim(x) == 1:
    return x[np.newaxis, :]
  else:
    return x


def load_mat(fname):
  import scipy.io
  return scipy.io.loadmat(fname)

def load_h5(fname):
  import h5py
  return h5py.File(fname, 'r')

# def round_to_digit(x, d):
  
def add_separator(lst, x):
  out = []
  for i, y in enumerate(lst):
    if i != 0:
      out.append(x)
    out.append(y)
  return out


def loadmat(fname):
  import scipy.io
  return scipy.io.loadmat(fname)

def foo(): return 'baz'

def test(x, cache = {}):
  if 'x' not in cache:
    import iputil as ip
    ip.run_reload()
    cache['x'] = 1
  return foo()
          
  

class Reloader:
  def __init__(self):
    self.done = False

  def run(self):
    if not self.done:
      import iputil as ip
      ip.run_reload()
      self.done = True
    

def add_slash(path):
  return path if path.endswith('/') else (path + '/')

def num_strs(s, n):
  if np.ndim(n) == 0:
    n = range(n)
  return [s % i for i in n]


# l-\infty norm
def absmax(x):
  return np.max(np.abs(x))


class temp_mat:
  def __init__(self, kvs, delete_on_exit = True):
    self.fname = make_temp('.mat')
    self.delete_on_exit = delete_on_exit
    scipy.io.savemat(self.fname, kvs)

  def __enter__(self):
    return self.fname

  def __exit__(self, type, value, traceback):
    if self.delete_on_exit and os.path.exists(self.fname):
      os.remove(self.fname)

def haseq(x, name, val):
  return hasattr(x, name) and getattr(x, name) == val

def getattr_else(x, name, val):
  return getattr(x, name) if hasattr(x, name) else val


def np_serialize(x):
  import msgpack
  import msgpack_numpy as m
  return msgpack.packb(np.array(x, 'float32'), default=m.encode)
  #return msgpack.packb(x, default=m.encode)
  # f = StringIO.StringIO()
  # np.save(f, x)
  # return f.getvalue()

def np_deserialize(s):
  import msgpack
  import msgpack_numpy as m
  return msgpack.unpackb(s, object_hook=m.decode)
#  return np.load(StringIO.StringIO(s))

def make_symlink(src, dst):
  if os.path.exists(dst) and os.path.islink(dst):
    os.remove(dst)
  os.symlink(src, dst)

# def maybe_make_symlink(src, dst):
#   if not os.path.exists(dst):
#     os.symlink(src, dst)

def maybe_make_symlink(src, dst):
  if not os.path.exists(dst) and not os.path.islink(dst):
    os.symlink(src, dst)

def replace_symlink(src, dst):
  if os.path.islink(dst):
    os.remove(dst)
  os.symlink(src, dst)

def updated_dict(x, kvs):
  x = copy.deepcopy(x)
  x.update(kvs)
  return x

def plot_im(*args, **kwargs):
  pylab.clf()
  pylab.plot(*args)
  if kwargs.get('ylim') is not None:
    pylab.ylim(kwargs['ylim'])
  return ig.from_fig()

def swaptup((x,y)): return (y,x)

def append_to_each(xs, y):
  return [(x, y) for x in xs]

def append_to_each_list(xs, ys):
  for x, y in zip(xs, ys):
    x.append(y)
  
def jsave(path, x):
  from sklearn.externals import joblib
  joblib.dump(x, path)

def jload(path):
  #if path.endswith('.jpk'):
  from sklearn.externals import joblib
  return joblib.load(path)
  # else:
  #   return load(path)

def make_even(x): return x - (x % 2)

def if_exists_rm(fname):
  if os.path.exists(fname):
    os.remove(fname)
    
def make_video(im_fnames, fps, out_fname, sound_fname = None, flags = ''):
  if type(sound_fname) != type(''):
    tmp_wav = make_temp('.wav')
    sound_fname.save(tmp_wav)
    sound_fname = tmp_wav
  else:
    tmp_wav = None

  write_ims = (type(im_fnames[0]) != type(''))
  num_ims = len(im_fnames) if write_ims else 0
  with temp_file('.txt') as input_file, temp_files('.ppm', num_ims) as tmp_ims:
    if write_ims:
      for fname, x in zip(tmp_ims, im_fnames):
        ig.save(fname, x)
      im_fnames = tmp_ims

    write_lines(input_file, ['file %s' % fname for fname in im_fnames])
    sound_flags_in = ('-i "%s"' % sound_fname) if sound_fname is not None else ''
    sound_flags_out =  '-acodec aac' if sound_fname is not None else ''
    #os.system('echo input file; cat %s' % input_file)
    sys_check('ffmpeg %s -r %f -loglevel warning -safe 0 -f concat -i "%s" -pix_fmt yuv420p -vcodec h264 -strict -2 -y %s %s "%s"' \
              % (sound_flags_in, fps, input_file, sound_flags_out, flags, out_fname))

  if tmp_wav is not None:
    os.remove(tmp_wav)
    
def make_mod(x, mod):
  return x - (x % mod)


def sort_date(fnames, reverse = False):
  return sorted(fnames, key = os.path.getctime)
  
def argsort_confint(pos_totals):
  """ Return indicies in *decreasing* order using confidence interval. """
  from statsmodels.stats.proportion import proportion_confint
  vals = []
  for pos, total in pos_totals:
    assert pos <= total
    l, u = proportion_confint(count = pos, nobs = total)
    vals.append(l)
    print pos, total, l, u, (u-l)/2.
  return np.argsort(np.array(vals))[::-1]

def afs_readable():
  try:
    f = file('/afs/csail.mit.edu/u/a/aho/test.jpg', 'rb')
    f.close()
    return True
  except:
    return False

def time_est(xs, n = None):
  # import progressbar
  # progress = progressbar.ProgressBar(maxval=n)
  # return progress(xs)
  if n is None:
    n = len(xs)
  te = TimeEstimator(n)
  for x in xs:
    te.update()
    yield x

def rms(x):
  return np.sqrt(np.mean(np.asarray(x)**2))

def normvec(x):
  x = np.asarray(x, dtype = np.float64)
  return x / norm(x)

def mean_pm(p, n):
  import scipy.stats
  if not (n*p > 5 and n*(1-p) > 5):
    print 'Not enough samples for normal approximation (doing anyway)'
  alpha = 0.05
  z = scipy.stats.norm.ppf(1-0.5*alpha)
  edge = z*np.sqrt(1./n*p*(1-p))
  return edge

def mean_pm_normal(stdev, n):
  import scipy.stats
  alpha = 0.05
  z = scipy.stats.norm.ppf(1-0.5*alpha)
  edge = z*np.sqrt(1./n)*stdev
  return edge

def mean_pm_str(p, n):
  return '%.2f%% \\pm %.2f' % (100*p, 100*mean_pm(p, n))

def mean_pm_normal_str(mu, stdev, n):
  return '%.4f \\pm %.4f' % (mu, mean_pm_normal(stdev, n))

# def rand_crop(im, h, w):
#   y = np.random.randint(im.shape[0] - h)
#   x = np.random.randint(im.shape[1] - w)
#   return im[y : y + h, x : x + w]

  
def rand_crop(im, h, w):
  y = np.random.randint(im.shape[0] - h + 1)
  x = np.random.randint(im.shape[1] - w + 1)
  return im[y : y + h, x : x + w]

def choose_crop(im_h, im_w, crop_h, crop_w):
  y = np.random.randint(im_h - crop_h + 1)
  x = np.random.randint(im_w - crop_w + 1)
  return y, x

def sigmoid(x):
  return 1./(1 + np.exp(-x))

# def crop_center(im, dim):
#   assert dim <= im.shape[0] and dim <= im.shape[1]
#   y = im.shape[0]/2
#   x = im.shape[1]/2
#   d = int(dim/2)
#   return im[y - d : y + d + 1, x - d : x + d + 1]

#def crop_center(im, dim):
def crop_center(im, dim1, dim2=None):
  if dim2 is None:
    dim2 = dim1
  assert dim1 <= im.shape[0] and dim2 <= im.shape[1]
  y0 = -dim1/2 + im.shape[0]/2
  x0 = -dim2/2 + im.shape[1]/2
  return im[y0 : y0 + dim1, x0 : x0 + dim2]
  
def take_oks(xs, oks):
  return [x for x, ok in zip(xs, oks) if ok]

def amap(f, xs):
  return np.array(map(f, xs))

def softmax(xs):
  ys = np.exp(xs)
  return ys / np.maximum(float_eps, np.sum(ys))

class wget_temp_file:
  def __init__(self, url, ext):
    self.url = url
    self.tmp_file = temp_file(ext)

  def __enter__(self):
    fname = self.tmp_file.__enter__()
    if 0 == os.system('wget "%s" -O "%s"' % (self.url, fname)):
      return fname
    else:
      return None

  def __exit__(self, type, value, tb):
    return self.tmp_file.__exit__(type, value, tb)


def fps_from_fname(fname):
  import re
  ffmpeg_out = sys_with_stdout('ffmpeg -i "%s" 2>&1' % fname)
  lines = list(grep_str(ffmpeg_out, ': Video: '))
  assert len(lines)
  m = re.match(r".* (\d+\.?\d*) tbr", lines[0])
  return (None if (m is None) else float(m.group(1)))

def video_has_audio(fname):
  ffmpeg_out = sys_with_stdout('ffmpeg -i "%s" 2>&1' % fname)
  matches = list(grep_str(ffmpeg_out, 'Stream #0:1'))
  if len(matches) > 0:
    return 'Audio: none' not in matches[0]
  else:
    return False

def dict_union(*ds):
  u = {}
  for x in ds:
    u.update(x)
  return u

def printlns(xs, last_blank = False):
  for x in xs:
    print x
  if last_blank:
    print

def read_lines_split(fname):
  return [x.split() for x in read_lines(fname)]

def hastrue(x, a):
  return hasattr(x, a) and bool(getattr(x, a))

def hasfalse(x, a):
  return hasattr(x, a) and not bool(getattr(x, a))

def log_softmax(xs):
  assert len(xs.shape) == 2
  import scipy.misc
  return xs - scipy.misc.logsumexp(xs, 1)[:, None]

def one_hot(i, n, dtype = np.float32):
  z = np.zeros(n)
  z[i] = 1
  return z

def video_length(fname):
  if not os.path.exists(fname):
    raise RuntimeError(fname + ' does not exist')
  try:
    #s = sys_with_stdout('ff=$(ffmpeg -i "%s" 2>&1); d="${ff#*Duration: }"; echo "${d%%%%,*}"' % fname).rstrip()
    s = sys_with_stdout('ffmpeg-length "%s"' % fname).rstrip()
    length_sec = (datetime.datetime.strptime(s, '%H:%M:%S.%f') \
                  - datetime.datetime.strptime('00:00:00.0', '%H:%M:%S.%f')).total_seconds()
    return length_sec
  except:
    return None


class TmpDir:
  def __enter__(self):
    self.path = tempfile.mkdtemp()
    return self.path
  
  def __exit__(self, type, value, tb):
    import shutil
    shutil.rmtree(self.path)

class VidFrames:
  def __init__(self, input_vid_path, sound = False, 
               start_time = None, end_time = None, 
               dims = None, sr = 21000,
               fps = None):
    self.input_vid_path = input_vid_path
    self.start_time = start_time
    self.end_time = end_time
    self.sound = sound
    self.dims = dims
    self.sr = sr
    self.fps = fps

  def __enter__(self):
    if not os.path.exists(self.input_vid_path):
      raise RuntimeError('Video does not exist:' + self.input_vid_path)
    self.path = tempfile.mkdtemp()
    start_str = '-ss %f' % self.start_time if self.start_time is not None else ''
    dur_str = '-t %f' % (self.end_time - max(0, self.start_time)) if self.end_time is not None else ''
    dim_str = "-vf 'scale=%d:%d'" % self.dims if self.dims is not None else ''
    fps_str = '-r %f' % self.fps if self.fps is not None else ''
    sys_check_silent('ffmpeg -loglevel fatal %s -i "%s" %s %s %s "%s/%%07d.png"' % 
                     (start_str, self.input_vid_path, dim_str, dur_str, fps_str, self.path))
    if self.sound:
      sound_file = pjoin(self.path, 'sound.wav')
      sys_check_silent('ffmpeg -loglevel fatal %s -i "%s" %s -ac 2 -ar %d "%s"' % 
                       (start_str, self.input_vid_path, dur_str, self.sr, sound_file))
    else:
      sound_file = None

    return sortglob(pjoin(self.path, '*.png')), sound_file
  
  def __exit__(self, type, value, tb):
    import shutil
    shutil.rmtree(self.path)

def centered(half, dim):
  s = half - dim/2.
  return s, s + dim
